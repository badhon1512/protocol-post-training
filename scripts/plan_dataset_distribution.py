"""Generate and audit the 1,000-slot multilingual SOOFI distribution."""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

from src.data.distribution import build_manifest
from src.data.protocol_config import (
    GENERAL_MULTILINGUAL, MIN_LANGUAGE_PER_CATEGORY, MIN_LANGUAGE_PER_CELL,
    QUOTAS, SPECIAL,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("data/dolly_soofi_1000_manifest.json"))
    parser.add_argument("--audit", type=Path,
                        default=Path("data/dolly_soofi_1000_distribution.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = arguments()
    manifest = build_manifest(args.seed)
    pair_counts = Counter(
        tuple(sorted(pair))
        for slot in manifest
        for pair in itertools.combinations(slot["selected_special_rules"], 2)
    )
    language_cells = {}
    missing_language_cells = []
    for split in ("train", "validation", "test"):
        for size in QUOTAS:
            cell = [slot for slot in manifest
                    if slot["split"] == split and len(slot["selected_special_rules"]) == size]
            counts = Counter(slot["language"] for slot in cell)
            key = f"{split}/special_count_{size}"
            language_cells[key] = {
                "total": len(cell), "counts": dict(counts),
                "english_share": counts["en"] / len(cell),
            }
            missing = [language for language in GENERAL_MULTILINGUAL
                       if counts[language] < MIN_LANGUAGE_PER_CELL]
            if missing: missing_language_cells.append({"cell": key, "missing": missing})

    rule_language = {}
    for rule in SPECIAL:
        rows = [slot for slot in manifest if rule in slot["selected_special_rules"]]
        counts = Counter(slot["language"] for slot in rows)
        rule_language[str(rule)] = {
            "total": len(rows), "counts": dict(counts),
            "english_share": counts["en"] / len(rows),
            "english_exempt": rule == 2,
        }
    category_language_minimums = {
        str(size): {
            language: sum(slot["language"] == language
                          and len(slot["selected_special_rules"]) == size
                          for slot in manifest)
            for language in GENERAL_MULTILINGUAL
        }
        for size in QUOTAS
    }
    audit = {
        "total": len(manifest),
        "generation_order": "easy_to_hard_by_rule_count_conflict_language_split",
        "first_25_slots": [
            {
                "slot": slot["slot"],
                "split": slot["split"],
                "language": slot["language"],
                "selected_special_rules": slot["selected_special_rules"],
                "conflict_type": slot["conflict_type"],
            }
            for slot in manifest[:25]
        ],
        "splits": dict(Counter(slot["split"] for slot in manifest)),
        "selected_special_rule_counts": dict(Counter(
            str(len(slot["selected_special_rules"]))
            for slot in manifest
        )),
        "conflict_examples": sum(slot["has_conflict"] for slot in manifest),
        "conflict_types": dict(Counter(slot["conflict_type"] for slot in manifest)),
        "resolution_bases": dict(Counter(slot["resolution_basis"] for slot in manifest)),
        "rule_1_override_examples": sum(
            slot["resolution_basis"] == "rule_1_override_then_alphabetical"
            for slot in manifest
        ),
        "minimum_pair_coverage": min(
            pair_counts.get(pair, 0) for pair in itertools.combinations(SPECIAL, 2)
        ),
        "languages": dict(Counter(slot["language"] for slot in manifest)),
        "languages_by_split": {
            split: dict(Counter(slot["language"] for slot in manifest if slot["split"] == split))
            for split in ("train", "validation", "test")
        },
        "language_cells": language_cells,
        "missing_general_language_cells": missing_language_cells,
        "language_by_rule": rule_language,
        "general_language_counts_by_category": category_language_minimums,
        "minimum_general_language_per_category": MIN_LANGUAGE_PER_CATEGORY,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Manifest written to {args.output}")
    print(f"Distribution audit written to {args.audit}")


if __name__ == "__main__":
    main()
