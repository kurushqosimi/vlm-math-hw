from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from hw.constants import IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig
from hw.processor import MathVLMProcessor, ProcessorConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss.

    TODO:
        - model.train();
        - forward;
        - ensure finite loss;
        - backward;
        - optimizer.step();
        - optimizer.zero_grad();
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = output["loss"] if isinstance(output, dict) else output.loss
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu().item())


class SimpleTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<eos>": 1, IMAGE_TOKEN: 2}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for token in text.replace("\n", " ").split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __len__(self) -> int:
        return len(self.vocab)


class TinyVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, hidden_size),
            nn.Tanh(),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        base = self.net(pixel_values).unsqueeze(1)
        return base.repeat(1, self.num_tokens, 1)


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.lm_head(inputs_embeds)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, inputs_embeds: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.lm_head(inputs_embeds[:, -1:]).argmax(dim=-1)


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point.

    TODO:
        - instantiate dataset, processor, model;
        - create DataLoader;
        - support max_steps and fast_train;
        - save adapter/checkpoint if configured.
    """
    data_cfg = config.get("data", {})
    proc_cfg = config.get("processor", {})
    trainer_cfg = config.get("trainer", {})

    dataset = MathVQADataset(
        data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=2 if fast_train else data_cfg.get("max_samples"),
    )
    tokenizer = SimpleTokenizer()
    processor = MathVLMProcessor(tokenizer, ProcessorConfig(**proc_cfg))

    # Build the vocabulary before the language model is initialized.
    _ = [processor.tokenize_sample(dataset[i]) for i in range(len(dataset))]

    loader = DataLoader(
        dataset,
        batch_size=int(trainer_cfg.get("local_batch_size", 1)),
        shuffle=True,
        num_workers=int(trainer_cfg.get("num_workers", 0)),
        collate_fn=lambda samples: processor.collate([processor(sample) for sample in samples]),
    )

    vision_hidden = int(config.get("model", {}).get("vision_hidden_size", 32))
    text_hidden = int(config.get("model", {}).get("text_hidden_size", 48))
    vision_encoder = TinyVisionEncoder(vision_hidden, processor.config.num_image_tokens)
    language_model = TinyLanguageModel(len(tokenizer), text_hidden)
    model = MathVLM(
        vision_encoder,
        language_model,
        ModelConfig(
            vision_hidden_size=vision_hidden,
            text_hidden_size=text_hidden,
            num_image_tokens=processor.config.num_image_tokens,
            image_token_id=tokenizer.vocab[IMAGE_TOKEN],
        ),
    )
    if config.get("model", {}).get("freeze_vision", True) or config.get("model", {}).get("freeze_llm", True):
        model.freeze_backbones()

    device = torch.device(trainer_cfg.get("device", "cpu"))
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(trainer_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )

    max_steps = 1 if fast_train else int(trainer_cfg.get("max_steps", 1))
    step = 0
    last_loss = math.nan
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            last_loss = train_one_step(model, batch, optimizer)
            step += 1
            if step >= max_steps:
                break

    save_path = trainer_cfg.get("save_checkpoint_path")
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"adapter": model.adapter.state_dict(), "loss": last_loss}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
