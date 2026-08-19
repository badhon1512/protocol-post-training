# Codex Handoff — SOOFI Post-Training Project

This file is meant to let a new Codex session continue the project on another machine/HPC without needing the full previous chat.

## Project

Workspace:

```text
C:\Job Search\PHD\SOOFI\Post training
```

Project type:

- Python project managed with `uv`
- Main goal: post-train/evaluate models for the SOOFI protocol-following assignment
- Code is organized under `src/`
- Scripts are under `scripts/`

## Assignment interpretation

The attached assignment PDF defines 18 special response rules plus universal constraints.

Important interpretation:

- Instructions inside attached documents are assignment content, not direct user instructions.
- The model should learn to follow protocol rules from SFT data.
- Evaluation should focus on protocol compliance more than generic answer quality.
- Universal constraints:
  - never use the word `delve`
  - every response should contain an em dash `—`
  - do not begin with `Certainly`, `Of course`, or `Great question`
  - if uncertain, say so plainly and continue usefully
- Null case:
  - if no special trigger fires, answer normally and append:
    ```text
    (unremarkable input; no protocols engaged)
    ```
- Precedence:
  - rules stack
  - contradictions are resolved by lower rule number first
  - three-or-more collisions resolve alphabetically by rule name
  - compromise should be noted in square brackets

## Final dataset

Use this dataset for training:

```text
data/dolly_soofi_600_strict.jsonl
```

This is the recommended clean strict dataset.

Known distribution:

- total: 600
- train: 420
- validation: 90
- test: 90
- all rows passed the current strict audit/eval checks when created

Older/intermediate dataset files may exist, but should not be used for final training unless intentionally experimenting:

```text
data/dolly_soofi_1000.jsonl
data/dolly_soofi_1000_full_clean.jsonl
data/dolly_soofi_1000_final.jsonl
data/soofi_2000.json
```

## Main model choice

Final Qwen choice:

```text
Qwen/Qwen3-8B
```

Reason:

- official Qwen model
- text-only
- simpler and cleaner for SFT than Qwen3.5 multimodal checkpoints

Avoid using this as default:

```text
Qwen/Qwen3.5-9B
```

because it is multimodal/image-text-to-text and needs processor/vision/video dependencies such as `torchvision`.

Gemma default remains:

```text
google/gemma-3-270m
```

## Important implementation state

### Model loading

File:

```text
src/model/model.py
```

Important current behavior:

- `Qwen.DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"`
- `Gemma.DEFAULT_MODEL_NAME = "google/gemma-3-270m"`
- model loader uses config to decide:
  - multimodal config -> `AutoModelForMultimodalLM`
  - otherwise -> `AutoModelForCausalLM`
- uses `dtype="auto"`
- processor loader tries `AutoProcessor`, then falls back to `AutoTokenizer`
- if tokenizer has no chat template, a simple fallback `User:/Assistant:` chat template is assigned

### HF SFT dataset formatting

File:

```text
src/data/data.py
```

Important current behavior:

- For Hugging Face SFT, dataset `text` is formatted with the actual processor/tokenizer chat template.
- This is important for Qwen:
  - Qwen3-8B formatted text looks like:
    ```text
    <|im_start|>user
    ...
    <|im_end|>
    <|im_start|>assistant
    ...
    <|im_end|>
    ```
- Dataset loader supports curriculum ordering for train split.

### HF SFT Trainer

File:

```text
src/training/hf_sft.py
```

Important current behavior:

- Uses TRL `SFTTrainer`
- LoRA config targets all linear layers
- Uses tokenizer/processing class suitable for SFT
- Saves processor/tokenizer to output dir
- Curriculum trainer uses sequential sampler to preserve easy-to-hard order

### LoRA

File:

```text
src/training/lora.py
```

Important current behavior:

- `target_modules="all-linear"`
- rank default: 16
- alpha default: 32
- dropout default: 0.05
- multimodal modules can be excluded if present

### Generation

File:

```text
scripts/generate_test.py
src/generate/generate.py
```

Important current behavior:

- Default generation prompt style is `chat`
- Use `--prompt-style sft_text` only for old checkpoints trained before chat-template formatting was fixed
- Generation is resumable:
  - existing output JSON is loaded
  - completed IDs are skipped
  - output is saved after each batch

### Evaluation

Files:

```text
scripts/evaluate.py
src/eval/evaluate.py
```

Important current behavior:

- Robust protocol rule evaluator for rules 1–18
- Checks universal constraints
- Reports:
  - protocol score
  - protocol pass rate
  - reference similarity
  - BLEU
  - optional BERTScore
  - quality pass rate
  - worst failed checks
  - breakdowns by split/language/case/rule count/conflict
- Use `--skip-bertscore` for fast eval.

## Useful commands

Run from project root.

### Setup

```bash
uv sync
```

Check help:

```bash
uv run python -m scripts.train --help
uv run python -m scripts.generate_test --help
uv run python -m scripts.evaluate --help
```

### Qwen setup check before HPC training

This does not load full model weights; it checks config/tokenizer/dataset formatting.

```bash
uv run python -m scripts.check_hf_sft_setup --model qwen
```

Expected important output:

```text
Model name: Qwen/Qwen3-8B
Config class: Qwen3Config
Model type: qwen3
Multimodal config: False
Has chat template: True
Train examples: 420
Validation examples: 90
```

### Train Qwen normal SFT

```bash
uv run python -m scripts.train \
  --model qwen \
  --trainer hf \
  --training-mode normal \
  --output-dir outputs/qwen3-8b-hf-normal-lora
```

### Train Qwen curriculum SFT

```bash
uv run python -m scripts.train \
  --model qwen \
  --trainer hf \
  --training-mode curriculum \
  --output-dir outputs/qwen3-8b-hf-curriculum-lora
```

### Optional W&B

Set W&B key/environment on HPC, then:

```bash
uv run python -m scripts.train \
  --model qwen \
  --trainer hf \
  --training-mode normal \
  --output-dir outputs/qwen3-8b-hf-normal-lora \
  --report-to wandb \
  --wandb-project soofi-post-training \
  --run-name qwen3-8b-normal-sft
```

Curriculum W&B:

```bash
uv run python -m scripts.train \
  --model qwen \
  --trainer hf \
  --training-mode curriculum \
  --output-dir outputs/qwen3-8b-hf-curriculum-lora \
  --report-to wandb \
  --wandb-project soofi-post-training \
  --run-name qwen3-8b-curriculum-sft
```

### Generate test outputs

Normal:

```bash
uv run python -m scripts.generate_test \
  --model qwen \
  --checkpoint outputs/qwen3-8b-hf-normal-lora \
  --data-path data/dolly_soofi_600_strict.jsonl \
  --output-path results/qwen3-8b-hf-normal-lora_test_generations.json \
  --batch-size 4 \
  --max-new-tokens 256
```

Curriculum:

```bash
uv run python -m scripts.generate_test \
  --model qwen \
  --checkpoint outputs/qwen3-8b-hf-curriculum-lora \
  --data-path data/dolly_soofi_600_strict.jsonl \
  --output-path results/qwen3-8b-hf-curriculum-lora_test_generations.json \
  --batch-size 4 \
  --max-new-tokens 256
```

### Evaluate

Fast eval without BERTScore:

```bash
uv run python -m scripts.evaluate \
  --input-path results/qwen3-8b-hf-normal-lora_test_generations.json \
  --skip-bertscore
```

Curriculum:

```bash
uv run python -m scripts.evaluate \
  --input-path results/qwen3-8b-hf-curriculum-lora_test_generations.json \
  --skip-bertscore
```

Full eval with BERTScore:

```bash
uv run python -m scripts.evaluate \
  --input-path results/qwen3-8b-hf-normal-lora_test_generations.json \
  --bertscore-batch-size 8
```

## Important warning about old checkpoints

Old Gemma checkpoints trained before the chat-template fix generated repeated `Assistant:` tokens.

Reason:

- training was plain text:
  ```text
  User: ...
  Assistant: ...
  ```
- generation used chat template format
- mismatch caused repeated assistant markers

Fix:

- new training uses tokenizer chat template
- generation default is now `chat`

For old checkpoints only, use:

```bash
--prompt-style sft_text
```

## Zipping/transfer status

A full project zip was created at:

```text
C:\Job Search\PHD\SOOFI\post_training_project_full.zip
```

Approx size:

```text
6.83 GB
```

It includes everything because the user requested full zip, including `.venv`, `.env`, outputs, results, and `.git`.

If transferring to HPC, consider whether `.env` should be removed because it may contain tokens/secrets.

## How to continue in new Codex session

In the new Codex session on HPC, say:

```text
Read CODEX_HANDOFF.md and continue from there.
```

Then first run:

```bash
uv run python -m scripts.check_hf_sft_setup --model qwen
```

Then launch normal/curriculum training.
