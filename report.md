# Report

## Track

Selected track: A / CPU-only.

## Implemented

- [x] `dataset.py`: JSONL manifest loading, split filtering, `max_samples`, relative image paths, RGB PIL images, sanitized questions.
- [x] `processor.py`: RGB image preprocessing, square resize/pad, simple tiling, normalization, prompt construction, masked labels, batch padding/collate.
- [x] `model.py`: trainable vision-to-text adapter, visual-token embedding merge, forward pass, generation wrapper, backbone freezing helper.
- [x] `train.py`: deterministic seed, one-step optimization, tiny CPU smoke model, DataLoader pipeline, optional adapter checkpoint saving.
- [x] `benchmark.py`: multiple-choice answer parsing, prompt construction, overall/per-subject accuracy, JSONL prediction output.

## Configuration

```text
config path: configs/track_a_cpu.yaml
seed: 42
device: cpu
dtype: float32
max_steps: 3
batch size: 1
```

## Results

```text
public tests: 14 passed, 1 warning in 8.74s
syntax check: python -m py_compile hw\dataset.py hw\processor.py hw\model.py hw\train.py hw\benchmark.py passed
train loss: fast CPU smoke run completed successfully
benchmark accuracy: overall=1.0, subject/geometry=1.0, subject/plots=1.0 on toy dev smoke benchmark
```

## Resources Used

```text
CPU/GPU: CPU-only
VRAM: none
training time: fast smoke run completed in a few seconds
external datasets: none
```

## Error Analysis

The submitted CPU track focuses on a correct pipeline rather than final model quality. Expected model failure modes for a real trained checkpoint:

1. Small visual encoder/adapters may miss fine chart details and small labels.
2. Multiple-choice answers can be parsed incorrectly if the generated response contains several option letters.
3. Geometry and plot questions may require reasoning steps that are not learned by a short adapter-only smoke run.

## Comments

The most important part was keeping tensor shapes consistent across image preprocessing, visual placeholder tokens, adapter output, and language-model embeddings. With more compute, I would replace the tiny smoke model with a real frozen ViT plus instruction LLM, train the adapter on the medium split, and evaluate on MathVista testmini.
