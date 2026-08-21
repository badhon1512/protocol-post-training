from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


DEFAULT_RUNS = {
    "baseline": "results/qwen3-8b-prompted-baseline-4071749-evaluation.json",
    "standard_sft": "results/qwen3-8b-hf-normal-10ep-nonthinking-4071751-evaluation.json",
    "curriculum_sft": "results/qwen3-8b-hf-curriculum-10ep-nonthinking-4071750-evaluation.json",
    "metadata_sft": "results/qwen3-8b-hf-metadata-10ep-nonthinking-4073195-evaluation.json",
    "protocol_weighted_sft": "results/qwen3-8b-hf-constraint-weighted-10ep-nonthinking-4073195-evaluation.json",
    "metadata_protocol_weighted_sft": (
        "results/qwen3-8b-hf-metadata-constraint-weighted-10ep-nonthinking-4073195-evaluation.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paired comparisons across SOOFI evaluation files."
    )
    parser.add_argument("--output-path", type=Path, default=Path("results/model-comparison.json"))
    parser.add_argument("--reference", default="standard_sft", choices=DEFAULT_RUNS)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_evaluation(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not payload.get("examples"):
        raise ValueError(f"No evaluated examples in {path}")
    return payload


def index_examples(evaluation: dict) -> dict[str, dict]:
    indexed = {str(item["id"]): item for item in evaluation["examples"]}
    if len(indexed) != len(evaluation["examples"]):
        raise ValueError("Evaluation contains duplicate example IDs")
    return indexed


def paired_bootstrap_ci(
    differences: list[float], samples: int, seed: int
) -> list[float]:
    if not differences:
        return [0.0, 0.0]
    generator = random.Random(seed)
    count = len(differences)
    estimates = [
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    estimates.sort()
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return [lower, upper]


def exact_mcnemar_p_value(model_only_pass: int, reference_only_pass: int) -> float:
    discordant = model_only_pass + reference_only_pass
    if discordant == 0:
        return 1.0
    smaller = min(model_only_pass, reference_only_pass)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def metric_block(evaluation: dict) -> dict:
    summary = evaluation["summary"]
    keys = (
        "count",
        "protocol_score",
        "micro_protocol_score",
        "protocol_pass_count",
        "protocol_pass_rate",
        "protocol_pass_rate_ci_95",
        "reference_similarity",
        "bertscore_f1",
        "bleu",
        "quality_pass_count",
        "quality_pass_rate",
    )
    return {key: summary[key] for key in keys if key in summary}


def compare(model: dict, reference: dict, samples: int, seed: int) -> dict:
    model_examples = index_examples(model)
    reference_examples = index_examples(reference)
    if model_examples.keys() != reference_examples.keys():
        raise ValueError("Evaluation files do not contain the same example IDs")

    ids = sorted(model_examples)
    score_differences = [
        model_examples[item_id]["protocol_score"]
        - reference_examples[item_id]["protocol_score"]
        for item_id in ids
    ]
    model_only_pass = sum(
        bool(model_examples[item_id]["protocol_pass"])
        and not bool(reference_examples[item_id]["protocol_pass"])
        for item_id in ids
    )
    reference_only_pass = sum(
        bool(reference_examples[item_id]["protocol_pass"])
        and not bool(model_examples[item_id]["protocol_pass"])
        for item_id in ids
    )

    subgroup_deltas: dict[str, dict] = {}
    for group_name in ("special_rule_load", "conflict", "language", "case_type"):
        model_groups = model["summary"]["groups"].get(group_name, {})
        reference_groups = reference["summary"]["groups"].get(group_name, {})
        subgroup_deltas[group_name] = {
            value: {
                "count": model_groups[value]["count"],
                "protocol_score_delta": (
                    model_groups[value]["protocol_score"]
                    - reference_groups[value]["protocol_score"]
                ),
                "protocol_pass_count_delta": (
                    model_groups[value]["protocol_pass_count"]
                    - reference_groups[value]["protocol_pass_count"]
                ),
            }
            for value in model_groups.keys() & reference_groups.keys()
        }

    return {
        "protocol_score_delta": sum(score_differences) / len(score_differences),
        "protocol_score_delta_bootstrap_ci_95": paired_bootstrap_ci(
            score_differences, samples, seed
        ),
        "protocol_pass_count_delta": (
            model["summary"]["protocol_pass_count"]
            - reference["summary"]["protocol_pass_count"]
        ),
        "mcnemar": {
            "model_only_pass": model_only_pass,
            "reference_only_pass": reference_only_pass,
            "exact_two_sided_p_value": exact_mcnemar_p_value(
                model_only_pass, reference_only_pass
            ),
        },
        "subgroup_deltas": subgroup_deltas,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be at least 1")

    evaluations = {
        name: load_evaluation(Path(path)) for name, path in DEFAULT_RUNS.items()
    }
    reference = evaluations[args.reference]
    comparisons = {
        name: compare(evaluation, reference, args.bootstrap_samples, args.seed + index)
        for index, (name, evaluation) in enumerate(evaluations.items())
        if name != args.reference
    }

    normal = evaluations["standard_sft"]["summary"]["protocol_score"]
    metadata = evaluations["metadata_sft"]["summary"]["protocol_score"]
    weighted = evaluations["protocol_weighted_sft"]["summary"]["protocol_score"]
    combined = evaluations["metadata_protocol_weighted_sft"]["summary"]["protocol_score"]
    payload = {
        "reference_run": args.reference,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "experiments": {
            name: metric_block(evaluation) for name, evaluation in evaluations.items()
        },
        "paired_comparisons": comparisons,
        "factorial_effects_on_protocol_score": {
            "metadata_without_weighting": metadata - normal,
            "weighting_without_metadata": weighted - normal,
            "metadata_with_weighting": combined - weighted,
            "weighting_with_metadata": combined - metadata,
            "interaction": combined - weighted - metadata + normal,
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"Saved paired comparison to {args.output_path}")


if __name__ == "__main__":
    main()
