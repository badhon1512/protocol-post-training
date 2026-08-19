"""Create a cleaned SFT dataset by applying safe deterministic repairs."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from scripts.audit_dataset import check_row, summarize
from src.data.generation import mechanical_signals
from src.data.io import load_records


EM_DASH = "\u2014"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/dolly_soofi_1000.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/dolly_soofi_1000_clean.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("results/dataset_cleaning_report.json"))
    parser.add_argument("--tokenizer", default="google/gemma-3-270m")
    parser.add_argument("--keep-failed", action="store_true",
                        help="Write repaired rows even when they still fail audit.")
    return parser.parse_args()


def content_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def has_em_dash(text):
    return EM_DASH in text or "â€”" in text


def ensure_em_dash(text):
    if has_em_dash(text):
        return text
    return text.rstrip() + f" {EM_DASH} noted."


def ensure_rule_6_format(text, required_format):
    lines = content_lines(text)
    if required_format == "bulleted":
        if not lines:
            return f"- {EM_DASH} noted."
        return "\n".join(
            line if re.match(r"^[\-*\u2022â€¢]", line) else f"- {line}"
            for line in lines
        )
    if required_format == "unbroken_prose":
        return " ".join(lines)
    return text


def ensure_rule_9(text):
    exact = "Was that necessary? It was."
    stripped = text.rstrip()
    if re.search(r"was that necessary\?\s*it was\.?$", stripped, re.IGNORECASE):
        return stripped
    # Remove near-miss rhetorical endings before adding the canonical ending.
    stripped = re.sub(
        r"\s*(?:Pray,|Pause to ask:|Ponder this:|Precisely:|Perfectly necessary,)?\s*"
        r"was that necessary\?\s*(?:plainly|precisely|positively|yes,?)?\s*it was\.?$",
        "",
        stripped,
        flags=re.IGNORECASE,
    ).rstrip()
    return f"{stripped} {exact}"


def ensure_rule_8(text):
    count = len(re.findall(r"\bthank(?:s|\s+you)?\b", text, re.IGNORECASE))
    if count >= 3:
        return text
    extras = ["Thank you"] * (3 - count)
    return text.rstrip() + " " + " ".join(extras) + "."


def ensure_conflict_note(text):
    if re.search(r"\[[^\]]*compromise[^\]]*\]", text, re.IGNORECASE):
        return text
    return text.rstrip() + " [compromise: lower-priority rule is approximated.]"


def refresh_metadata(row, tokenizer):
    signals = mechanical_signals(row["user_prompt"], tokenizer)
    row["word_count"] = signals["user_word_count"]
    row["parity"] = signals["word_parity"]
    row["required_response_format"] = (
        "bulleted" if signals["word_parity"] == "even" else "unbroken_prose"
    )
    active = sorted(set(int(rule) for rule in row.get("active_rules", [])))
    if 6 not in active:
        active.append(6)
    if signals["apostrophe_count"] == 0 and 17 not in active:
        active.append(17)
    if signals["apostrophe_count"] > 0:
        active = [rule for rule in active if rule != 17]
    row["active_rules"] = sorted(active)
    row["target_word_count"] = len(re.findall(r"\b[\wâ€™'-]+\b", row["target_response"], re.UNICODE))
    row["constraint_loss"] = row.get("constraint_loss", {})
    row["constraint_loss"]["mask_source"] = "derive_from_target_after_model_tokenization"
    row["constraint_loss"]["universal_constraints"] = [
        "banned_delve", "required_em_dash", "forbidden_openers"
    ]
    return row


def repair_row(row, audit, tokenizer):
    row = dict(row)
    before = list(audit["failed_checks"])
    text = row.get("target_response", "")

    text = ensure_em_dash(text)
    text = ensure_rule_6_format(text, row.get("required_response_format"))

    active = {int(rule) for rule in row.get("active_rules", [])}
    if 9 in active:
        text = ensure_rule_9(text)
    if 8 in active:
        text = ensure_rule_8(text)
    if row.get("has_conflict"):
        text = ensure_conflict_note(text)

    row["target_response"] = text
    row = refresh_metadata(row, tokenizer)
    after = check_row(row, tokenizer)
    return row, before, after


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = arguments()
    rows = load_records(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    cleaned = []
    rejected = []
    repair_counts = Counter()
    before_audits = []
    after_audits = []

    for row in rows:
        before = check_row(row, tokenizer)
        before_audits.append(before)
        repaired, failed_before, after = repair_row(row, before, tokenizer)
        after_audits.append(after)
        for name in failed_before:
            repair_counts[name] += 1
        if after["pass"] or args.keep_failed:
            cleaned.append(repaired)
        else:
            rejected.append({
                "id": row.get("id"),
                "failed_before": failed_before,
                "failed_after": after["failed_checks"],
            })

    write_jsonl(args.output, cleaned)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "input_count": len(rows),
        "clean_count": len(cleaned),
        "rejected_count": len(rejected),
        "before_summary": summarize(before_audits),
        "after_summary_all_rows": summarize(after_audits),
        "clean_summary": summarize([check_row(row, tokenizer) for row in cleaned]),
        "repair_attempted_for_failed_checks": dict(repair_counts.most_common()),
        "rejected_examples": rejected[:100],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input rows: {len(rows):,}")
    print(f"Clean rows written: {len(cleaned):,} -> {args.output}")
    print(f"Rejected rows: {len(rejected):,}")
    print(f"Before pass rate: {report['before_summary']['overall_pass_rate']:.3f}")
    print(f"Clean pass rate: {report['clean_summary']['overall_pass_rate']:.3f}")
    print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
