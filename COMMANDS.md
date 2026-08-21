# Commands

Run commands from the repository root.

## Setup

```bash
uv sync --frozen
```

Set `HF_TOKEN` in the environment or in `.env` if the model requires
authentication.

## Slurm

Each job runs its complete training, generation, and evaluation workflow.

```bash
sbatch jobs/baseline.sh
sbatch jobs/sft_normal.sh
sbatch jobs/sft_curriculum.sh
sbatch jobs/sft_metadata.sh
sbatch jobs/sft_weighted.sh
sbatch jobs/sft_metadata_weighted.sh
```

## Training

Standard SFT with the settings used in the report:

```bash
uv run python -m scripts.train \
  --training-mode normal \
  --data-path data/dolly_soofi_600_strict.jsonl \
  --output-dir outputs/qwen-normal \
  --epochs 10 \
  --learning-rate 5e-5 \
  --lr-scheduler-type cosine \
  --warmup-steps 27 \
  --weight-decay 0.01 \
  --train-batch-size 1 \
  --val-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 2048 \
  --max-grad-norm 0.3 \
  --logging-steps 10 \
  --eval-steps 53 \
  --save-steps 53 \
  --save-total-limit 3 \
  --seed 42
```

Change the run by adding or replacing these options:

```text
Curriculum:          --training-mode curriculum
Metadata:            --include-mechanical-metadata
Protocol weighting:  --constraint-weighted-loss --ordinary-token-weight 1 --sequence-token-weight 2 --span-token-weight 4
Combined:            --include-mechanical-metadata --constraint-weighted-loss --ordinary-token-weight 1 --sequence-token-weight 2 --span-token-weight 4
```

## Generation

Generate deterministic responses from a trained checkpoint:

```bash
uv run python -m scripts.generate_test \
  --checkpoint outputs/qwen-normal \
  --data-path data/dolly_soofi_600_strict.jsonl \
  --output-path results/qwen-normal-generations.json \
  --batch-size 1 \
  --max-new-tokens 512 \
  --seed 42 \
  --prompt-style chat
```

Add `--include-mechanical-metadata` for a metadata-trained checkpoint.

Generate the prompted base-model baseline:

```bash
uv run python -m scripts.generate_test \
  --checkpoint Qwen/Qwen3-8B \
  --data-path data/dolly_soofi_600_strict.jsonl \
  --system-prompt-file data/SOOFI_PROTOCOL_PROMPT.md \
  --output-path results/qwen3-8b-prompted-baseline-generations.json \
  --batch-size 1 \
  --max-new-tokens 512 \
  --seed 42 \
  --prompt-style chat
```

## Evaluation

```bash
uv run python -m scripts.evaluate \
  --input-path results/qwen-normal-generations.json \
  --output-path results/qwen-normal-evaluation.json \
  --bertscore-batch-size 16
```

Use `--skip-bertscore` for a faster protocol-only check.

Rebuild the aggregate metrics after evaluating all six runs:

```bash
uv run python -m scripts.compare_evaluations
uv run python src/eval/ci.py
```
