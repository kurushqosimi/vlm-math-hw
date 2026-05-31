from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        size = self.config.image_size
        image = ImageOps.pad(image.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)

        if self.config.num_tiles <= 1:
            tiles = [image]
        else:
            tiles = []
            grid = int(np.ceil(np.sqrt(self.config.num_tiles)))
            step = size / grid
            for tile_idx in range(self.config.num_tiles):
                row, col = divmod(tile_idx, grid)
                left = int(round(col * step))
                upper = int(round(row * step))
                right = int(round((col + 1) * step))
                lower = int(round((row + 1) * step))
                tile = image.crop((left, upper, right, lower)).resize(
                    (size, size), Image.Resampling.BICUBIC
                )
                tiles.append(tile)

        tensors = []
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        for tile in tiles:
            arr = np.asarray(tile, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors, dim=0)

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        image_tokens = " ".join([IMAGE_TOKEN] * self.config.num_image_tokens)
        options = "\n".join(sample.options)
        prompt = (
            f"{IMAGE_START_TOKEN} {image_tokens} {IMAGE_END_TOKEN}\n"
            "Question: "
            f"{sample.question}\n"
            f"Options:\n{options}\n"
            "Answer:"
        )
        if include_answer:
            prompt += f" {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        prompt_text = self.build_prompt(sample, include_answer=False)
        answer_text = f" {sample.answer}"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            answer_ids = answer_ids + [int(eos_id)]

        max_length = self.config.max_length
        if len(answer_ids) >= max_length:
            input_ids = answer_ids[:max_length]
            labels = input_ids.copy()
        else:
            prompt_budget = max_length - len(answer_ids)
            prompt_ids = prompt_ids[:prompt_budget]
            input_ids = prompt_ids + answer_ids
            labels = [self.config.ignore_index] * len(prompt_ids) + answer_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """
        max_len = max(item["input_ids"].numel() for item in batch)
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)

        input_ids = []
        attention_mask = []
        labels = []
        for item in batch:
            length = item["input_ids"].numel()
            pad = max_len - length
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=pad_id))
            attention_mask.append(torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0))
            labels.append(
                torch.nn.functional.pad(item["labels"], (0, pad), value=self.config.ignore_index)
            )

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        }
