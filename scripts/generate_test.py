"""Generate responses for every example in the SOOFI test split."""

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from src.generate import InferencePipeline
from src.model.model import Gemma, Qwen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses for the test set.")
    parser.add_argument("--checkpoint", default="outputs/gemma-custom")
    parser.add_argument("--model", choices=("qwen", "gemma"), default="gemma")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/dolly_soofi_600_strict.jsonl"),
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument(
        "--prompt-style",
        choices=("sft_text", "chat"),
        default="chat",
        help="Use chat for newly trained models; use sft_text only for old plain-text checkpoints.",
    )
    return parser.parse_args()


def load_test_records(data_path: Path) -> list[dict]:
    with data_path.open("r", encoding="utf-8") as file:
        if data_path.suffix.lower() == ".jsonl":
            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]
        else:
            records = json.load(file)

    test_records = [record for record in records if record.get("split") == "test"]
    if not test_records:
        raise ValueError(f"No test records found in {data_path}")
    return test_records


def save_results(output_path: Path, results: list[dict]) -> None:
    """Atomically save the current results as one valid JSON array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)


def load_existing_results(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    with output_path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Existing output must be a JSON array: {output_path}")
    return records


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists() and "/" not in args.checkpoint:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.system_prompt_file and not args.system_prompt_file.exists():
        raise FileNotFoundError(f"System prompt not found: {args.system_prompt_file}")

    output_path = args.output_path or (
        Path("results") / f"{checkpoint_path.name}_test_generations.json"
    )
    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8")
        if args.system_prompt_file
        else None
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    model_loader = Qwen() if args.model == "qwen" else Gemma()
    model = model_loader.load_model(str(args.checkpoint))
    processor = model_loader.load_processor(str(args.checkpoint))
    inference = InferencePipeline(model, processor)
    print("Checkpoint loaded.")

    test_records = load_test_records(args.data_path)
    existing_results = load_existing_results(output_path)
    completed_ids = {
        record.get("id")
        for record in existing_results
        if record.get("id") is not None and record.get("generated_response")
    }
    pending_records = [
        record
        for record in test_records
        if record.get("id") not in completed_ids
    ]
    print(
        f"Generating responses for {len(pending_records)} pending test examples "
        f"({len(existing_results)} already saved)."
    )

    results = existing_results
    progress = tqdm(
        range(0, len(pending_records), args.batch_size),
        desc="Test generation",
        unit="batch",
    )
    for start in progress:
        batch = pending_records[start : start + args.batch_size]
        responses = inference.generate(
            [record["user_prompt"] for record in batch],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sample,
            temperature=args.temperature,
            system_prompt=system_prompt,
            prompt_style=args.prompt_style,
        )

        for record, response in zip(batch, responses, strict=True):
            result = dict(record)
            result["generated_response"] = response.strip()
            results.append(result)

        # Save after each batch so an interrupted run keeps all completed responses.
        save_results(output_path, results)

    print(f"Saved {len(results)} responses to {output_path}")


if __name__ == "__main__":
    main()
