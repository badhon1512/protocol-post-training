from __future__ import annotations

import json
import math
import random
from pathlib import Path


RESULTS_DIR = Path(__file__).parents[2] / "results"
OUTPUT_PATH = RESULTS_DIR / "report-metrics.json"
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 42

FILES = {
    "prompted": "qwen3-8b-prompted-baseline-4071749-evaluation.json",
    "standard": "qwen3-8b-hf-normal-10ep-nonthinking-4071751-evaluation.json",
    "ordered": "qwen3-8b-hf-curriculum-10ep-nonthinking-4071750-evaluation.json",
    "metadata": "qwen3-8b-hf-metadata-10ep-nonthinking-4073195-evaluation.json",
    "weighted": "qwen3-8b-hf-constraint-weighted-10ep-nonthinking-4073195-evaluation.json",
    "combined": "qwen3-8b-hf-metadata-constraint-weighted-10ep-nonthinking-4073195-evaluation.json",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_stats(left: dict, right: dict, field: str, seed: int = BOOTSTRAP_SEED) -> dict:
    right_by_id = {item["id"]: item for item in right["examples"]}
    differences = [
        float(item[field]) - float(right_by_id[item["id"]][field])
        for item in left["examples"]
    ]
    rng = random.Random(seed)
    count = len(differences)
    bootstrap_means = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    lower = bootstrap_means[int(0.025 * (BOOTSTRAP_SAMPLES - 1))]
    upper = bootstrap_means[int(0.975 * (BOOTSTRAP_SAMPLES - 1))]
    result = {
        "mean_delta": sum(differences) / count,
        "ci_95": [lower, upper],
    }
    if field == "protocol_pass":
        gains = sum(
            bool(item[field]) and not bool(right_by_id[item["id"]][field])
            for item in left["examples"]
        )
        losses = sum(
            not bool(item[field]) and bool(right_by_id[item["id"]][field])
            for item in left["examples"]
        )
        result.update(
            {
                "gains": gains,
                "losses": losses,
                "mcnemar_exact_p": exact_mcnemar(gains, losses),
            }
        )
    return result


def special_rule_count(item: dict) -> int:
    return len(set(int(rule) for rule in item["active_rules"]) - {6, 17})


def mean(items: list[dict], field: str = "protocol_score") -> float:
    return sum(float(item[field]) for item in items) / len(items) if items else 0.0


def subset(items: list[dict], kind: str) -> list[dict]:
    if kind == "ordinary":
        return [item for item in items if item.get("case_type") == "ordinary_retention"]
    if kind == "one":
        return [item for item in items if special_rule_count(item) == 1]
    if kind == "two_nonconflict":
        return [
            item
            for item in items
            if special_rule_count(item) == 2 and not item.get("has_conflict")
        ]
    if kind == "conflict":
        return [item for item in items if item.get("has_conflict")]
    raise KeyError(kind)


def main() -> None:
    data = {name: read_json(RESULTS_DIR / filename) for name, filename in FILES.items()}
    main_table = {}
    for name, result in data.items():
        summary = result["summary"]
        conflict = summary["groups"]["conflict"]["True"]
        main_table[name] = {
            "protocol_score": summary["protocol_score"],
            "full_passes": round(summary["protocol_pass_rate"] * summary["count"]),
            "conflict_score": conflict["protocol_score"],
            "conflict_passes": round(conflict["protocol_pass_rate"] * conflict["count"]),
            "bertscore_f1": summary["bertscore_f1"],
        }

    comparisons = {
        "standard_minus_prompted_protocol": paired_stats(
            data["standard"], data["prompted"], "protocol_score"
        ),
        "weighted_minus_standard_protocol": paired_stats(
            data["weighted"], data["standard"], "protocol_score", seed=46
        ),
        "combined_minus_standard_protocol": paired_stats(
            data["combined"], data["standard"], "protocol_score", seed=47
        ),
        "combined_minus_weighted_protocol": paired_stats(
            data["combined"], data["weighted"], "protocol_score"
        ),
        "combined_minus_standard_pass": paired_stats(
            data["combined"], data["standard"], "protocol_pass"
        ),
        "combined_minus_weighted_pass": paired_stats(
            data["combined"], data["weighted"], "protocol_pass"
        ),
        "ordered_minus_standard_bertscore": paired_stats(
            data["ordered"], data["standard"], "bertscore_f1"
        ),
    }

    rule_load = {}
    for name, result in data.items():
        bins = {}
        for label, predicate in (
            ("0", lambda value: value == 0),
            ("1", lambda value: value == 1),
            ("2", lambda value: value == 2),
            ("3+", lambda value: value >= 3),
        ):
            values = [
                item
                for item in result["examples"]
                if predicate(special_rule_count(item))
            ]
            bins[label] = {"count": len(values), "protocol_score": mean(values)}
        rule_load[name] = bins

    structural_groups = {
        "overall": None,
        "ordinary": "ordinary",
        "one": "one",
        "two_nonconflict": "two_nonconflict",
        "conflict": "conflict",
    }
    structural_deltas = {}
    standard_items = data["standard"]["examples"]
    for name in ("ordered", "metadata", "weighted", "combined"):
        method_by_id = {item["id"]: item for item in data[name]["examples"]}
        row = {}
        for label, kind in structural_groups.items():
            base_items = standard_items if kind is None else subset(standard_items, kind)
            row[label] = {
                "count": len(base_items),
                "delta": sum(
                    float(method_by_id[item["id"]]["protocol_score"])
                    - float(item["protocol_score"])
                    for item in base_items
                )
                / len(base_items),
            }
        structural_deltas[name] = row

    language_counts = {}
    language_deltas = {}
    languages = sorted({str(item["language"]) for item in standard_items})
    for language in languages:
        base_items = [item for item in standard_items if str(item["language"]) == language]
        language_counts[language] = len(base_items)
    for name in ("ordered", "metadata", "weighted", "combined"):
        method_by_id = {item["id"]: item for item in data[name]["examples"]}
        language_deltas[name] = {}
        for language in languages:
            base_items = [
                item for item in standard_items if str(item["language"]) == language
            ]
            language_deltas[name][language] = sum(
                float(method_by_id[item["id"]]["protocol_score"])
                - float(item["protocol_score"])
                for item in base_items
            ) / len(base_items)

    compromise_note = {}
    for name in ("standard", "ordered"):
        conflict_items = [
            item for item in data[name]["examples"] if item.get("has_conflict")
        ]
        compromise_note[name] = sum(
            bool(item["checks"].get("conflict_note")) for item in conflict_items
        ) / len(conflict_items)

    output = {
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "main_table": main_table,
        "comparisons": comparisons,
        "rule_load": rule_load,
        "structural_deltas": structural_deltas,
        "language_counts": language_counts,
        "language_deltas": language_deltas,
        "compromise_note": compromise_note,
    }
    write_json(OUTPUT_PATH, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
