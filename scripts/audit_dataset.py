"""Audit generated SFT samples against prompt triggers and response rules."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from src.data.generation import detected_special_rules, mechanical_signals
from src.data.io import load_records
from src.eval.evaluate import RULE_CHECKS, universal_checks


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/dolly_soofi_1000.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/dataset_audit.json"))
    parser.add_argument("--tokenizer", default="google/gemma-3-270m")
    parser.add_argument("--max-examples", type=int, default=30)
    return parser.parse_args()


def planned_special_rules(row):
    # active_rules includes always-on Rule 6 and derived Rule 17.
    return sorted(int(rule) for rule in row.get("active_rules", []) if int(rule) not in (6, 17))


def check_row(row, tokenizer):
    user_prompt = row.get("user_prompt", "")
    target_response = row.get("target_response", "")
    signals = mechanical_signals(user_prompt, tokenizer)
    planned = planned_special_rules(row)
    detected = sorted(detected_special_rules(user_prompt, row.get("language", "en"), signals))

    checks = {}
    checks["prompt_trigger_match"] = detected == planned
    for name, passed in universal_checks(target_response).items():
        checks[name] = passed

    record = dict(row)
    record["target_response"] = target_response
    skipped_conflict_rules = []
    for rule in [int(rule) for rule in row.get("active_rules", [])]:
        checker = RULE_CHECKS[rule]

        # If this is a conflict row and the reference target itself does not
        # satisfy a lower-priority rule, do not count that rule as failed here.
        if row.get("has_conflict") and not checker(target_response, record):
            skipped_conflict_rules.append(rule)
            continue
        checks[f"rule_{rule}"] = checker(target_response, record)

    if row.get("has_conflict"):
        checks["conflict_note"] = bool(
            re.search(r"\[[^\]]*compromise[^\]]*\]", target_response, re.IGNORECASE)
        )

    failed = [name for name, passed in checks.items() if not passed]
    response_rule_checks = {
        name: passed for name, passed in checks.items() if name.startswith("rule_")
    }
    universal_rule_checks = {
        name: passed for name, passed in checks.items() if name.startswith("universal_")
    }
    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "language": row.get("language"),
        "case_type": row.get("case_type"),
        "active_rules": row.get("active_rules", []),
        "planned_special_rules": planned,
        "detected_special_rules": detected,
        "word_count": signals["user_word_count"],
        "parity": signals["word_parity"],
        "required_response_format": row.get("required_response_format"),
        "checks": checks,
        "failed_checks": failed,
        "pass": not failed,
        "response_rules_pass": all(response_rule_checks.values()) if response_rule_checks else True,
        "universal_rules_pass": all(universal_rule_checks.values()) if universal_rule_checks else True,
        "all_selected_rules_pass": (
            checks.get("prompt_trigger_match", False)
            and (all(response_rule_checks.values()) if response_rule_checks else True)
        ),
        "skipped_conflict_rules": skipped_conflict_rules,
    }


def average(values):
    return sum(values) / len(values) if values else 0.0


def summarize(audits):
    check_totals = defaultdict(list)
    rule_totals = defaultdict(list)
    by_split = defaultdict(list)
    by_language = defaultdict(list)
    by_case_type = defaultdict(list)
    by_rule_count = defaultdict(list)
    by_conflict = defaultdict(list)
    by_rule_combo = defaultdict(list)

    for item in audits:
        by_split[item["split"]].append(item)
        by_language[item["language"]].append(item)
        by_case_type[item["case_type"]].append(item)
        by_rule_count[str(len(item["planned_special_rules"]))].append(item)
        by_conflict["conflict" if item.get("case_type") == "conflict_precedence" else "non_conflict"].append(item)
        combo = ",".join(str(rule) for rule in item["planned_special_rules"]) or "none"
        by_rule_combo[combo].append(item)
        for name, passed in item["checks"].items():
            check_totals[name].append(passed)
            if name.startswith("rule_"):
                rule_totals[name].append(passed)

    def group_summary(items):
        return {
            "count": len(items),
            "pass_rate": average([item["pass"] for item in items]),
            "prompt_trigger_match_rate": average([
                item["checks"].get("prompt_trigger_match", False) for item in items
            ]),
            "response_rules_pass_rate": average([item["response_rules_pass"] for item in items]),
            "universal_rules_pass_rate": average([item["universal_rules_pass"] for item in items]),
            "all_selected_rules_pass_rate": average([item["all_selected_rules_pass"] for item in items]),
            "avg_failed_checks": average([len(item["failed_checks"]) for item in items]),
        }

    return {
        "count": len(audits),
        "overall_pass_rate": average([item["pass"] for item in audits]),
        "prompt_trigger_match_rate": average([
            item["checks"].get("prompt_trigger_match", False) for item in audits
        ]),
        "response_rules_pass_rate": average([item["response_rules_pass"] for item in audits]),
        "universal_rules_pass_rate": average([item["universal_rules_pass"] for item in audits]),
        "all_selected_rules_pass_rate": average([item["all_selected_rules_pass"] for item in audits]),
        "avg_failed_checks": average([len(item["failed_checks"]) for item in audits]),
        "check_pass_rate": {
            name: average(values) for name, values in sorted(check_totals.items())
        },
        "per_rule_pass_rate": {
            name: average(values) for name, values in sorted(rule_totals.items())
        },
        "by_split": {key: group_summary(items) for key, items in sorted(by_split.items())},
        "by_language": {key: group_summary(items) for key, items in sorted(by_language.items())},
        "by_case_type": {key: group_summary(items) for key, items in sorted(by_case_type.items())},
        "by_selected_rule_count": {
            key: group_summary(items) for key, items in sorted(by_rule_count.items())
        },
        "by_conflict_status": {
            key: group_summary(items) for key, items in sorted(by_conflict.items())
        },
        "worst_rule_combinations": {
            key: group_summary(items)
            for key, items in sorted(
                by_rule_combo.items(),
                key=lambda pair: (group_summary(pair[1])["pass_rate"], -len(pair[1]))
            )[:30]
        },
        "top_failed_checks": dict(Counter(
            failed for item in audits for failed in item["failed_checks"]
        ).most_common(30)),
    }


def main():
    args = arguments()
    rows = load_records(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    audits = [check_row(row, tokenizer) for row in rows]
    summary = summarize(audits)

    failed_examples = [
        item for item in audits if item["failed_checks"]
    ][:args.max_examples]
    report = {
        "input": str(args.input),
        "summary": summary,
        "failed_examples": failed_examples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Audited {summary['count']:,} samples from {args.input}")
    print(f"Overall pass rate: {summary['overall_pass_rate']:.3f}")
    print(f"Prompt trigger match rate: {summary['prompt_trigger_match_rate']:.3f}")
    print("Top failed checks:")
    for name, count in summary["top_failed_checks"].items():
        print(f"  {name}: {count}")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
