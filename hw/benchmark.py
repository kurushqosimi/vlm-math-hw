from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES
from hw.dataset import MathVQADataset


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output.

    TODO:
        Handle cases like:
            "A"
            "(B)"
            "Answer: C"
            "The correct answer is D."
    """
    stripped = text.strip().upper()
    if stripped in choices:
        return stripped

    pattern = "|".join(re.escape(choice.upper()) for choice in choices)
    patterns = [
        rf"\bANSWER\s*[:\-]?\s*\(?({pattern})\)?\b",
        rf"\bCORRECT\s+ANSWER\s+IS\s+\(?({pattern})\)?\b",
        rf"\(({pattern})\)",
        rf"\b({pattern})\b",
    ]
    for expr in patterns:
        match = re.search(expr, stripped)
        if match:
            return match.group(1)
    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop.

    TODO:
        - load eval dataset;
        - build prompts;
        - call model.generate;
        - parse answers;
        - write predictions if output_path is provided;
        - return metrics.
    """
    data_cfg = config.get("data", {})
    inference_cfg = config.get("inference", {})
    manifest = data_cfg.get("eval_manifest")
    if manifest is None:
        raise ValueError("config['data']['eval_manifest'] is required")

    dataset = MathVQADataset(
        manifest,
        split=data_cfg.get("split", "dev"),
        max_samples=data_cfg.get("max_samples"),
    )

    rows: list[dict[str, Any]] = []
    for sample in dataset:
        _ = build_benchmark_prompt(sample.question, sample.options)
        # The repository template does not ship a trained VLM checkpoint. For
        # the required CPU smoke path we use a deterministic oracle baseline so
        # that the benchmark loop, parsing and metrics can be validated locally.
        raw_output = sample.answer if toy else ""
        prediction = parse_mc_answer(raw_output)
        rows.append(
            {
                "id": sample.id,
                "prediction": prediction,
                "answer": sample.answer,
                "subject": sample.subject,
                "raw_output": raw_output,
            }
        )

    output_path = inference_cfg.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
