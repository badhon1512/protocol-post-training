# Penalize Where It Matters

## Protocol-Weighted Fine-Tuning for Compositional Rule Following

### Abstract

Can an 8-billion-parameter language model reliably execute many interacting
response rules, or does ordinary next-token training spend too much capacity on
text that does not determine compliance? We study this question on SOOFI, a
600-example benchmark with 18 conditional rules, three universal constraints,
multilingual prompts, and explicit conflict cases. Starting from Qwen3-8B, we
compare prompting, standard supervised fine-tuning (SFT), an ordered curriculum,
mechanical metadata, protocol-weighted SFT, and their combination under matched
training settings.

Protocol-weighted SFT raises mean protocol compliance from **0.6365 to 0.7374**
over standard SFT, a paired gain of **+0.1009** (95% bootstrap CI
**[+0.0662, +0.1360]**). Combining metadata with weighting produces the most
fully compliant responses: **14/90**, versus **4/90** for standard SFT. Every
discordant exact-pass case favors the combined model (exact McNemar
**p = 0.0020**). The combined model also leads BERTScore F1 and BLEU. These
results support a simple conclusion: for rule-following tasks, explicitly
emphasizing the tokens that realize a protocol is substantially more effective
than reordering the same examples, while metadata mainly changes the balance
between partial and complete compliance.

## 1. Problem

The task is stricter than ordinary instruction following. A response must
answer the user, identify every activated rule, compose those rules, respect a
precedence order when they conflict, and obey universal constraints. A fluent
answer can therefore be a complete protocol failure.

Standard SFT assigns equal importance to every supervised completion token. In
these data, however, a short sign-off, a required em dash, a 17-word boundary,
or a structural marker may determine whether the entire response passes. The
central hypothesis is that training should assign more learning signal to such
tokens while retaining ordinary language-model supervision everywhere else.

The project contributes:

1. A matched six-way comparison on Qwen3-8B.
2. A token-level protocol-weighted objective with validated span alignment.
3. Deterministic, rule-aware checkpoint selection and test generation.
4. Evaluation with raw counts, uncertainty intervals, paired tests, conflict
   slices, and a compositional-load measure that excludes pervasive rules.

## 2. Experimental design

### 2.1 Data

The strict SOOFI dataset contains 600 examples: 420 train, 90 validation, and
90 test. Records contain the prompt, target response, active rule IDs, conflict
status, language, topic, and required response format. The test set is used
only for final comparison. The validation set selects checkpoints.

Rule occurrence is imbalanced. Rules 6 and 17 are pervasive, while several
special rules are rare. Ninety-seven training examples contain explicit
conflicts. We therefore report both overall results and disaggregated results.

### 2.2 Matched training controls

All SFT variants use Qwen3-8B with LoRA for 10 epochs, learning rate `5e-5`, a
cosine schedule, 27 warmup steps, weight decay `0.01`, micro-batch size 1,
gradient accumulation 8, maximum sequence length 2048, gradient clipping at
`0.3`, and seed 42. Evaluation and saving occur every 53 optimizer steps. HPC
training uses native BF16 on an NVIDIA A100; it does not use 4-bit or 8-bit
quantization. Qwen thinking is disabled, and test decoding is deterministic.

### 2.3 Compared methods

- **Prompted baseline:** the base model receives the complete protocol.
- **Standard SFT:** completion-only cross-entropy; prompt tokens are masked.
- **Ordered curriculum:** the same examples are ordered easy-to-hard each
  epoch. This is ordering, not staged replay.
- **Metadata SFT:** adds user token count, word count, and uppercase ratio in a
  trusted, masked system message at train, validation, and test time.
- **Protocol-weighted SFT:** weights ordinary completion tokens by 1,
  structural/sequence evidence by 2, and localized rule spans by 4.
- **Metadata + protocol-weighted SFT:** combines the two interventions while
  applying weighted loss only to assistant completion tokens.

For token loss \(\ell_t\), valid-token mask \(m_t\), and protocol weight
\(w_t\), the weighted objective is:

\[
\mathcal{L}_{PW}=\frac{\sum_t m_t w_t\ell_t}{\sum_t m_t w_t}.
\]

All 510 training and validation examples passed prompt/completion-boundary and
offset-alignment checks with the actual Qwen tokenizer. Weighted checkpoints
maximize `protocol_score + 2 × protocol_pass_rate` on deterministic validation
generations instead of selecting only by validation loss.

## 3. Evaluation

The primary metric is **protocol score**, the macro-average fraction of
applicable checks passed per example. **Micro protocol score** pools all passed
and applicable checks. **Full pass** requires every applicable check to pass.
The evaluator also reports reference similarity, quality pass, BERTScore with
`xlm-roberta-base`, sentence-level BLEU, per-rule counts, Wilson 95% intervals
for pass rates, and groups by language, case type, conflict status, and rule
burden.

Raw active-rule count can be misleading because Rules 6 and 17 occur
throughout the benchmark. We therefore define **special-rule load** as the
number of active rules after excluding Rules 6 and 17. Paired protocol-score
deltas use 50,000 bootstrap resamples. Full-pass differences use exact
two-sided McNemar tests on the same 90 prompts.

## 4. Main results

| Method | Protocol | Micro | Full pass | BERTScore F1 | BLEU | Quality pass |
|---|---:|---:|---:|---:|---:|---:|
| Prompted baseline | 0.5878 | 0.5773 | 2/90 | 0.7862 | 0.0199 | 0/90 |
| Standard SFT | 0.6365 | 0.6392 | 4/90 | 0.8549 | 0.1009 | 12/90 |
| Ordered curriculum | 0.6083 | 0.6082 | 2/90 | 0.8267 | 0.0570 | 5/90 |
| Metadata SFT | 0.6606 | 0.6615 | 5/90 | 0.8531 | 0.0960 | 13/90 |
| **Protocol-weighted SFT** | **0.7374** | **0.7354** | 8/90 | 0.8574 | 0.1092 | 13/90 |
| **Metadata + weighted** | 0.7303 | 0.7285 | **14/90** | **0.8619** | **0.1351** | **14/90** |

Two models lead for different operational objectives. Protocol-weighted SFT
has the strongest average compliance, improving over standard SFT by 10.09
percentage points. The combined model has the strongest all-or-nothing
reliability, adding 10 exact passes over standard SFT, and the best semantic-overlap
scores.

| Comparison with standard SFT | Score delta | Paired 95% CI | Full-pass delta | Exact McNemar p |
|---|---:|---:|---:|---:|
| Prompted baseline | -0.0487 | [-0.0930, -0.0039] | -2 | 0.6875 |
| Ordered curriculum | -0.0282 | [-0.0640, +0.0071] | -2 | 0.6875 |
| Metadata SFT | +0.0241 | [-0.0119, +0.0610] | +1 | 1.0000 |
| **Protocol-weighted SFT** | **+0.1009** | **[+0.0662, +0.1360]** | +4 | 0.2188 |
| **Metadata + weighted** | **+0.0938** | **[+0.0556, +0.1324]** | **+10** | **0.0020** |

The score improvements from both weighted methods have paired intervals above
zero. The weighted-only exact-pass increase is not statistically resolved on
90 examples. The combined model's 10 additional exact passes are stronger:
10 prompts pass only under the combined model, and none pass only under
standard SFT.

## 5. Where the gains occur

### 5.1 Compositional load

Each cell reports protocol score and exact passes. Sample sizes are fixed by
the test set: 8 examples at load 0, 15 at load 1, 61 at load 2, and 6 at load
3 or above.

| Special-rule load | Standard SFT | Weighted SFT | Metadata + weighted |
|---|---:|---:|---:|
| 0 | 0.6125; 0/8 | 0.7687; 1/8 | **0.7937; 2/8** |
| 1 | 0.7244; 2/15 | 0.8222; 2/15 | **0.8244; 4/15** |
| 2 | 0.6228; 2/61 | **0.7198; 5/61** | 0.7041; **8/61** |
| 3+ | 0.5883; 0/6 | 0.6621; 0/6 | **0.6764; 0/6** |

Weighted training improves mean compliance at every load. The combined model
produces more exact passes at loads 0, 1, and 2, although the load-3+ slice is
too small for a reliable ranking.

### 5.2 Conflict handling

| Method | Non-conflict score; passes (n=63) | Conflict score; passes (n=27) |
|---|---:|---:|
| Prompted baseline | 0.6080; 2 | 0.5407; 0 |
| Standard SFT | 0.6324; 2 | 0.6460; 2 |
| Protocol-weighted SFT | **0.7408; 5** | 0.7294; 3 |
| Metadata + weighted | 0.7152; **9** | **0.7653; 5** |

The combined method gives the strongest conflict result and doubles the exact
conflict passes relative to standard SFT. These are descriptive subgroup results; 27
conflict examples are not enough for broad generalization.

### 5.3 Factorial view

Relative to standard SFT, metadata alone contributes +0.0241 protocol score and
weighting alone contributes +0.1009. With metadata already present, weighting
adds +0.0697. With weighting already present, metadata changes mean score by
−0.0071 but raises exact passes from 8 to 14. The score-scale interaction is
−0.0312, so the interventions are not additive on mean compliance. Metadata
appears to trade a small amount of partial credit for more all-rules-satisfied
outputs when paired with weighting; this is an interpretation, not a causal
mechanism established by the current experiment.

## 6. Error profile

For protocol-weighted SFT, the most frequent failures are Rule 6 (38/90), Rule
2 (19/90), the required conflict note (16/27 applicable cases), and Rule 8
(9 applicable failures). The combined model reduces conflict-note failures to
10 and raises exact pass substantially, but Rule 6 remains the largest error
source with 40 failures. Future improvements should therefore target format
control and explicit conflict realization rather than increasing epochs
without diagnosis.

## 7. Reproducibility and limitations

The maintained pipeline is in `scripts/train.py`, `scripts/generate_test.py`,
and `scripts/evaluate.py`; `scripts/compare_evaluations.py` regenerates the
paired analysis in `results/model-comparison.json`, while `src/eval/ci.py`
produces the extended diagnostics in `results/report-metrics.json`. Six Slurm jobs reproduce
the baseline and five trained variants. The earlier hand-written LoRA loop is
retained as `src/training/custom_sft.py` and is not imported by production
training.

The test set has only 90 examples, exact pass remains low in absolute terms,
and several subgroup cells are small. Wilson intervals for exact pass remain
wide—for example, 14/90 corresponds to approximately [0.095, 0.244]. Automated
checks capture formal compliance but cannot fully establish factual accuracy,
helpfulness, or naturalness. BERTScore and BLEU measure reference overlap, not
protocol correctness. The results therefore support the observed benchmark
effects, not a claim of universal instruction-following reliability.

## 8. Conclusion

Ordinary SFT improves Qwen3-8B over prompting, but it leaves much of the
protocol unsatisfied. Merely ordering examples from easy to hard does not solve
the problem. Protocol-weighted SFT delivers the clearest improvement in average
rule compliance, while metadata plus weighting delivers the strongest exact
completion and semantic overlap. The practical lesson is direct: when a small
set of output tokens determines whether a compositional protocol succeeds,
training should make those tokens matter more.
