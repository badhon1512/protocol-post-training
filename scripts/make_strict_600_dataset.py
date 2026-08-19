"""Create a 600-row strict dataset with 70/15/15 train/validation/test split."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audit_dataset import check_row, summarize
from src.data.io import load_records
from transformers import AutoTokenizer


TARGETS = {"train": 420, "validation": 90, "test": 90}
LANGUAGES = ("en", "de", "fr", "ar", "hi", "es", "bn")
SPECIAL_RULES = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/dolly_soofi_1000_final.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/dolly_soofi_600_strict.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("results/dataset_600_strict_audit.json"))
    parser.add_argument("--tokenizer", default="google/gemma-3-270m")
    return parser.parse_args()


def selected_rules(row):
    return tuple(int(rule) for rule in row.get("active_rules", []) if int(rule) not in (6, 17))


def row_key(row):
    rules = selected_rules(row)
    return (
        len(rules),
        bool(row.get("has_conflict")),
        row.get("language") != "en",
        row.get("target_word_count", 0),
        row.get("id", ""),
    )


def coverage_score(row, selected, target_split):
    """Prefer rows that add missing language/rule/conflict coverage."""
    split_rows = [item for item in selected if item["split"] == target_split]
    langs = {item["language"] for item in split_rows}
    rules = {rule for item in split_rows for rule in selected_rules(item)}
    has_conflict = any(item.get("has_conflict") for item in split_rows)
    score = 0
    if row["language"] not in langs:
        score += 100
    for rule in selected_rules(row):
        if rule not in rules:
            score += 20
    if row.get("has_conflict") and not has_conflict:
        score += 50
    # English is abundant, so prefer non-English when scores tie.
    if row["language"] != "en":
        score += 5
    return score


def choose_rows(rows):
    # Start from existing validation/test rows because they are already held out.
    selected = []
    remaining = []
    for row in rows:
        if row["split"] in ("validation", "test"):
            selected.append(dict(row))
        else:
            remaining.append(dict(row))

    # Fill validation/test up to 90 each by moving diverse train rows.
    for split in ("validation", "test"):
        while sum(row["split"] == split for row in selected) < TARGETS[split]:
            best = max(
                remaining,
                key=lambda row: (coverage_score(row, selected, split), row_key(row)),
            )
            remaining.remove(best)
            best = dict(best)
            best["split"] = split
            selected.append(best)

    # Fill train to 420, trimming mainly abundant/simple English examples.
    train_needed = TARGETS["train"]
    train_candidates = [row for row in remaining if row["split"] == "train"]

    must_keep = []
    for language in LANGUAGES:
        language_rows = [row for row in train_candidates if row["language"] == language]
        must_keep.extend(sorted(language_rows, key=row_key)[: min(3, len(language_rows))])
    for rule in SPECIAL_RULES:
        rule_rows = [row for row in train_candidates if rule in selected_rules(row)]
        must_keep.extend(sorted(rule_rows, key=row_key)[: min(3, len(rule_rows))])
    conflict_rows = [row for row in train_candidates if row.get("has_conflict")]
    must_keep.extend(sorted(conflict_rows, key=row_key)[: min(20, len(conflict_rows))])

    keep_ids = {row["id"] for row in must_keep}
    train_selected = []
    for row in train_candidates:
        if row["id"] in keep_ids and len(train_selected) < train_needed:
            train_selected.append(row)

    leftovers = [row for row in train_candidates if row["id"] not in {item["id"] for item in train_selected}]
    leftovers = sorted(
        leftovers,
        key=lambda row: (
            row["language"] != "en",          # remove English later by selecting non-English first
            len(selected_rules(row)) >= 3,    # keep harder rows earlier
            bool(row.get("has_conflict")),
            -row.get("target_word_count", 0),
            row.get("id", ""),
        ),
        reverse=True,
    )
    for row in leftovers:
        if len(train_selected) >= train_needed:
            break
        train_selected.append(row)

    selected.extend(train_selected)
    selected = selected[:]
    if len(selected) != 600:
        raise RuntimeError(f"Expected 600 rows, got {len(selected)}")
    return sorted(selected, key=lambda row: (row["split"], row["id"]))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = arguments()
    rows = load_records(args.input)
    chosen = choose_rows(rows)
    write_jsonl(args.output, chosen)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    audits = [check_row(row, tokenizer) for row in chosen]
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "summary": summarize(audits),
        "split_counts": dict(Counter(row["split"] for row in chosen)),
        "language_counts": dict(Counter(row["language"] for row in chosen)),
        "rule_count_distribution": dict(Counter(str(len(selected_rules(row))) for row in chosen)),
        "individual_rule_counts": dict(Counter(str(rule) for row in chosen for rule in selected_rules(row))),
        "conflict_counts": dict(Counter(str(bool(row.get("has_conflict"))) for row in chosen)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(chosen)} rows -> {args.output}")
    print(f"Split counts: {report['split_counts']}")
    print(f"Pass rate: {report['summary']['overall_pass_rate']:.3f}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
