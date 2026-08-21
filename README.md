# Penalize Where It Matters

## Protocol-Weighted Fine-Tuning for Compositional Rule Following

This repository contains the Qwen3-8B training and evaluation code for SOOFI, a
compositional rule-following task with 18 conditional rules, conflict
precedence, multilingual prompts, and three universal constraints.

Standard SFT gives every supervised completion token the same weight. The
protocol-weighted variant gives more weight to tokens that directly implement
a required rule, such as a sign-off, format marker, or constrained span.

## Main result

Protocol-weighted SFT improves mean protocol score from **0.6365** with
standard SFT to **0.7374**, a paired gain of **+0.1009** with a 95% bootstrap
confidence interval of **[+0.0662, +0.1360]**.

Combining mechanical metadata with protocol weighting produces the most fully
compliant responses: **14/90**, compared with **4/90** for standard SFT. All 10
discordant exact-pass cases favor the combined model (exact McNemar
**p = 0.0020**).

| Method | Protocol score | Full pass | BERTScore F1 | BLEU |
|---|---:|---:|---:|---:|
| Prompted baseline | 0.5878 | 2/90 | 0.7862 | 0.0199 |
| Standard SFT | 0.6365 | 4/90 | 0.8549 | 0.1009 |
| Ordered curriculum | 0.6083 | 2/90 | 0.8267 | 0.0570 |
| Metadata SFT | 0.6606 | 5/90 | 0.8531 | 0.0960 |
| **Protocol-weighted SFT** | **0.7374** | 8/90 | 0.8574 | 0.1092 |
| **Metadata + weighted** | 0.7303 | **14/90** | **0.8619** | **0.1351** |

Protocol-weighted SFT is strongest for average rule compliance. Metadata plus
weighting is strongest when every applicable rule must pass simultaneously.
The 90-example test set is small, so raw counts and confidence intervals should
be considered alongside point estimates.

The [project report](PROJECT_REPORT.md) contains the experimental design,
paired statistics, subgroup analysis, error analysis, and limitations.

## Method

The strict SOOFI dataset contains 600 examples:

- 420 training examples
- 90 validation examples
- 90 test examples

Each record contains a user prompt, target response, active rule IDs, conflict
status, language, topic, and required response format. The test set is used
only for final comparison; validation generations select checkpoints.

The six evaluated conditions are:

| Method | Description |
|---|---|
| Prompted baseline | Base Qwen3-8B receives the complete protocol |
| Standard SFT | Completion-only LoRA fine-tuning |
| Ordered curriculum | The same examples are ordered from easy to hard |
| Metadata SFT | Adds user token count, word count, and uppercase ratio |
| Protocol-weighted SFT | Upweights tokens that implement active rules |
| Metadata + weighted | Combines metadata conditioning and weighted loss |

For token loss \(\ell_t\), valid completion mask \(m_t\), and protocol weight
\(w_t\), the weighted objective is:

$$
\mathcal{L}_{\mathrm{PW}}
=
\frac{\sum_{t=1}^{T} m_t w_t \ell_t}
     {\sum_{t=1}^{T} m_t w_t}.
$$

Ordinary completion tokens receive weight 1, structural or sequence evidence
receives weight 2, and localized rule spans receive weight 4. Prompt tokens are
always masked. Weighted checkpoints maximize:

```text
protocol_selection_score = protocol_score + 2 × protocol_pass_rate
```

All SFT variants share the main controls: Qwen3-8B, LoRA, 10 epochs, learning
rate `5e-5`, cosine scheduling, 27 warmup steps, micro-batch size 1, gradient
accumulation 8, maximum sequence length 2048, native BF16, and seed 42. Test
generation is deterministic and Qwen thinking is disabled.

## Evaluation

The evaluator reports:

- Macro and micro protocol scores
- Exact protocol passes with raw counts and Wilson 95% intervals
- Per-rule pass/fail counts and applicable denominators
- Results by language, case type, conflict status, and rule burden
- Special-rule load excluding pervasive Rules 6 and 17
- Reference similarity, BERTScore, BLEU, and quality pass rate
- Paired bootstrap intervals and exact McNemar tests across runs

The reproducible paired analysis is saved in
[`results/model-comparison.json`](results/model-comparison.json). Generated
responses and detailed per-example evaluations are under [`results/`](results/).
Extended structural and language diagnostics are saved in
[`results/report-metrics.json`](results/report-metrics.json).

## Repository structure

```text
data/        strict dataset and complete protocol
jobs/        Slurm jobs for the six experiments
notebooks/   optional self-contained 4-bit Colab workflow
results/     generations, evaluations, comparisons, and weighted-run audits
scripts/     training, generation, evaluation, and comparison commands
src/         maintained pipeline implementation
```

Main files:

- [`PROJECT_REPORT.md`](PROJECT_REPORT.md): complete research report
- [`data/SOOFI_PROTOCOL_PROMPT.md`](data/SOOFI_PROTOCOL_PROMPT.md): protocol
  specification
- [`results/model-comparison.json`](results/model-comparison.json): aggregate
  and paired results
- [`results/report-metrics.json`](results/report-metrics.json): extended
  subgroup and report diagnostics
- [`scripts/compare_evaluations.py`](scripts/compare_evaluations.py):
  statistical comparison implementation
- [`src/training/custom_sft.py`](src/training/custom_sft.py): earlier custom trainer;
  it is not imported by the maintained pipeline

## Running the project

### Requirements

- Python 3.11 or newer
- `uv`
- A CUDA GPU for training; the supplied Slurm jobs request one A100

Clone the repository and install the locked environment:

```bash
git clone https://github.com/badhon1512/protocol-post-training.git
cd protocol-post-training
uv sync --frozen
```

Set `HF_TOKEN` in the environment or in a local `.env` file if model access
requires authentication. The HPC runs use native BF16 without quantization.

### Slurm jobs

Submit a complete experiment from the repository root:

```bash
sbatch jobs/baseline.sh
sbatch jobs/sft_normal.sh
sbatch jobs/sft_curriculum.sh
sbatch jobs/sft_metadata.sh
sbatch jobs/sft_weighted.sh
sbatch jobs/sft_metadata_weighted.sh
```

Each job trains when required, generates the 90 test responses, and evaluates
them. Checkpoints are written to `$WORK/protocol-post-training/outputs/` and
JSON results are copied to `results/`.

### Local training

This command matches the reported standard SFT configuration:

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

For protocol-weighted SFT, add:

```text
--constraint-weighted-loss
--ordinary-token-weight 1
--sequence-token-weight 2
--span-token-weight 4
```

Use `--training-mode curriculum` for the ordered curriculum. Add
`--include-mechanical-metadata` for metadata conditioning. The metadata and
weighted options can be combined.

### Prompted baseline

The prompted baseline uses the base model without training an adapter:

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

### Test generation

Generate responses from a trained adapter:

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

Add `--include-mechanical-metadata` when the checkpoint was trained with
metadata.

### Evaluation

```bash
uv run python -m scripts.evaluate \
  --input-path results/qwen-normal-generations.json \
  --output-path results/qwen-normal-evaluation.json \
  --bertscore-batch-size 16
```

Add `--skip-bertscore` for a faster protocol-only check. After all six runs
have been evaluated, rebuild the aggregate files:

```bash
uv run python -m scripts.compare_evaluations
uv run python src/eval/ci.py
```
