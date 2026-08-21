# Commands

Run commands from the repository root.

## Setup

```bash
uv sync --frozen
```

Set `HF_TOKEN` in the environment or in `.env` if model access requires it.

## Slurm jobs

Each job runs the full workflow for one variation.

```bash
sbatch jobs/baseline.sh
sbatch jobs/sft_normal.sh
sbatch jobs/sft_curriculum.sh
sbatch jobs/sft_metadata.sh
sbatch jobs/sft_weighted.sh
sbatch jobs/sft_metadata_weighted.sh
```

## Local training

Set the shared training arguments once:

```bash
TRAIN_ARGS=(
  --data-path data/dolly_soofi_600_strict.jsonl
  --epochs 10
  --learning-rate 5e-5
  --lr-scheduler-type cosine
  --warmup-steps 27
  --weight-decay 0.01
  --train-batch-size 1
  --val-batch-size 1
  --gradient-accumulation-steps 8
  --max-length 2048
  --max-grad-norm 0.3
  --logging-steps 10
  --eval-steps 53
  --save-steps 53
  --save-total-limit 3
  --seed 42
)
```

Standard SFT:

```bash
uv run python -m scripts.train \
  "${TRAIN_ARGS[@]}" \
  --training-mode normal \
  --output-dir outputs/qwen-normal
```

Ordered curriculum:

```bash
uv run python -m scripts.train \
  "${TRAIN_ARGS[@]}" \
  --training-mode curriculum \
  --output-dir outputs/qwen-curriculum
```

Metadata SFT:

```bash
uv run python -m scripts.train \
  "${TRAIN_ARGS[@]}" \
  --training-mode normal \
  --include-mechanical-metadata \
  --output-dir outputs/qwen-metadata
```

Protocol-weighted SFT:

```bash
uv run python -m scripts.train \
  "${TRAIN_ARGS[@]}" \
  --training-mode normal \
  --constraint-weighted-loss \
  --ordinary-token-weight 1 \
  --sequence-token-weight 2 \
  --span-token-weight 4 \
  --output-dir outputs/qwen-weighted
```

Metadata plus protocol-weighted SFT:

```bash
uv run python -m scripts.train \
  "${TRAIN_ARGS[@]}" \
  --training-mode normal \
  --include-mechanical-metadata \
  --constraint-weighted-loss \
  --ordinary-token-weight 1 \
  --sequence-token-weight 2 \
  --span-token-weight 4 \
  --output-dir outputs/qwen-metadata-weighted
```

## Test generation

Prompted base-model baseline:

```bash
uv run python -m scripts.generate_test \
  --checkpoint Qwen/Qwen3-8B \
  --data-path data/dolly_soofi_600_strict.jsonl \
  --system-prompt-file data/SOOFI_PROTOCOL_PROMPT.md \
  --output-path results/qwen-baseline-generations.json \
  --batch-size 1 \
  --max-new-tokens 512 \
  --seed 42 \
  --prompt-style chat
```

Standard SFT:

```bash
uv run python -m scripts.generate_test \
  --checkpoint outputs/qwen-normal \
  --data-path data/dolly_soofi_600_strict.jsonl \
  --output-path results/qwen-normal-generations.json \
  --batch-size 1 --max-new-tokens 512 --seed 42 --prompt-style chat
```

Ordered curriculum:

```bash
uv run python -m scripts.generate_test \
  --checkpoint outputs/qwen-curriculum \
  --data-path data/dolly_soofi_600_strict.jsonl \
  --output-path results/qwen-curriculum-generations.json \
  --batch-size 1 --max-new-tokens 512 --seed 42 --prompt-style chat
```

Metadata SFT:

```bash
uv run python -m scripts.generate_test \
  --checkpoint outputs/qwen-metadata \
  --data-path data/dolly_soofi_600_strict.jsonl \
  --output-path results/qwen-metadata-generations.json \
  --include-mechanical-metadata \
  --batch-size 1 --max-new-tokens 512 --seed 42 --prompt-style chat
```

Protocol-weighted SFT:

```bash
uv run python -m scripts.generate_test \
  --checkpoint outputs/qwen-weighted \
  --data-path data/dolly_soofi_600_strict.jsonl \
  --output-path results/qwen-weighted-generations.json \
  --batch-size 1 --max-new-tokens 512 --seed 42 --prompt-style chat
```

Metadata plus protocol-weighted SFT:

```bash
uv run python -m scripts.generate_test \
  --checkpoint outputs/qwen-metadata-weighted \
  --data-path data/dolly_soofi_600_strict.jsonl \
  --output-path results/qwen-metadata-weighted-generations.json \
  --include-mechanical-metadata \
  --batch-size 1 --max-new-tokens 512 --seed 42 --prompt-style chat
```

Generation is deterministic unless `--sample` is supplied.

## Evaluation

Evaluate every variation:

```bash
for run in baseline normal curriculum metadata weighted metadata-weighted; do
  uv run python -m scripts.evaluate \
    --input-path "results/qwen-${run}-generations.json" \
    --output-path "results/qwen-${run}-evaluation.json" \
    --bertscore-batch-size 16
done
```

Use `--skip-bertscore` for a faster protocol-only check.

Compare the saved experiment files in this repository:

```bash
uv run python -m scripts.compare_evaluations
uv run python src/eval/ci.py
```
