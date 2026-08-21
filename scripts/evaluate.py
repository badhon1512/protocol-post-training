import argparse
import json
from json import JSONDecodeError
from pathlib import Path

from src.eval import evaluate_records


def load_generated_records(input_path: Path) -> list[dict]:
    """Load a JSON array, JSONL, or consecutive JSON objects from disk."""
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Generation file is empty: {input_path}")

    try:
        payload = json.loads(text)
    except JSONDecodeError:
        # Recover files produced incrementally as `{...}{...}` or JSONL.
        decoder = json.JSONDecoder()
        records: list[dict] = []
        position = 0

        while position < len(text):
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text):
                break

            try:
                value, position = decoder.raw_decode(text, position)
            except JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON near line {error.lineno}, column {error.colno} "
                    f"in {input_path}"
                ) from error

            if isinstance(value, dict):
                records.append(value)
            elif isinstance(value, list) and all(
                isinstance(record, dict) for record in value
            ):
                records.extend(value)
            else:
                raise ValueError("Each generated record must be a JSON object")

        print("Detected consecutive JSON records; recovered all complete objects.")
        return records

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(record, dict) for record in payload):
        return payload
    raise ValueError("Generation file must contain a JSON object or an array of objects")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated test responses.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("results/qwen3-8b-generations.json"),
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--quality-threshold", type=float, default=0.35)
    parser.add_argument("--bertscore-model", default="xlm-roberta-base")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore when you only need fast protocol/BLEU evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.quality_threshold <= 1:
        raise ValueError("quality-threshold must be between 0 and 1")
    if args.bertscore_batch_size < 1:
        raise ValueError("bertscore-batch-size must be at least 1")
    if not args.input_path.exists():
        raise FileNotFoundError(f"Generation file not found: {args.input_path}")

    print(f"Loading generated responses: {args.input_path}")
    records = load_generated_records(args.input_path)

    print(f"Evaluating {len(records)} responses...")
    results = evaluate_records(
        records,
        quality_threshold=args.quality_threshold,
        bertscore_model=args.bertscore_model,
        bertscore_batch_size=args.bertscore_batch_size,
        compute_bertscore=not args.skip_bertscore,
    )
    output_path = args.output_path or args.input_path.with_name(
        f"{args.input_path.stem}_evaluation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    summary = results["summary"]
    print(f"Protocol score:       {summary['protocol_score']:.3f}")
    print(
        "Protocol pass rate:   "
        f"{summary['protocol_pass_rate']:.3f} "
        f"({summary['protocol_pass_count']}/{summary['count']})"
    )
    print(f"Micro protocol score: {summary['micro_protocol_score']:.3f}")
    print(f"Reference similarity: {summary['reference_similarity']:.3f}")
    if "bertscore_f1" in summary:
        print(f"BERTScore F1:         {summary['bertscore_f1']:.3f}")
    if "bleu" in summary:
        print(f"BLEU:                 {summary['bleu']:.3f}")
    print(
        "Quality pass rate:    "
        f"{summary['quality_pass_rate']:.3f} "
        f"({summary['quality_pass_count']}/{summary['count']})"
    )
    print("Special-rule load (excluding pervasive Rules 6 and 17):")
    for load, group in summary.get("groups", {}).get("special_rule_load", {}).items():
        print(
            f"  - {load}: score={group['protocol_score']:.3f}, "
            f"pass={group['protocol_pass_count']}/{group['count']}"
        )
    if summary.get("metric_warnings"):
        print("Metric warnings:")
        for warning in summary["metric_warnings"]:
            print(f"  - {warning}")
    print("Worst failed checks:")
    for name, count in summary.get("worst_failed_checks", [])[:10]:
        print(f"  - {name}: {count}")
    print(f"Saved detailed evaluation to {output_path}")


if __name__ == "__main__":
    main()
