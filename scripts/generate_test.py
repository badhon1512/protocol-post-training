import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.data import metadata_system_message
from src.generate import InferencePipeline
from src.model.model import Qwen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses for the test set.")
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/dolly_soofi_600_strict.jsonl"),
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument(
        "--include-mechanical-metadata",
        action="store_true",
        help=(
            "Supply token count, word count, and uppercase ratio for each "
            "original test prompt in a separate system message."
        ),
    )
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
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be at least 1")
    if args.sample and args.temperature <= 0:
        raise ValueError("temperature must be greater than zero when sampling")
    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in the interval (0, 1]")
    if args.top_k < 0:
        raise ValueError("top-k must not be negative")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists() and "/" not in args.checkpoint:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.system_prompt_file and not args.system_prompt_file.exists():
        raise FileNotFoundError(f"System prompt not found: {args.system_prompt_file}")
    if args.system_prompt_file and args.include_mechanical_metadata:
        raise ValueError(
            "Use either --system-prompt-file or --include-mechanical-metadata, not both."
        )

    output_path = args.output_path or (
        Path("results") / f"{checkpoint_path.name}_test_generations.json"
    )
    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8")
        if args.system_prompt_file
        else None
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    model_loader = Qwen()
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
        batch_system_prompts = system_prompt
        batch_metadata = [None] * len(batch)
        if args.include_mechanical_metadata:
            metadata_pairs = [
                metadata_system_message(record["user_prompt"], inference.tokenizer)
                for record in batch
            ]
            batch_system_prompts = [pair[0] for pair in metadata_pairs]
            batch_metadata = [pair[1] for pair in metadata_pairs]
        responses = inference.generate(
            [record["user_prompt"] for record in batch],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            system_prompt=batch_system_prompts,
            prompt_style=args.prompt_style,
        )

        for record, response, metadata in zip(
            batch, responses, batch_metadata, strict=True
        ):
            result = dict(record)
            result["generated_response"] = response.strip()
            if metadata is not None:
                result["mechanical_metadata"] = metadata
            results.append(result)

        # Save after each batch so an interrupted run keeps all completed responses.
        save_results(output_path, results)

    print(f"Saved {len(results)} responses to {output_path}")


if __name__ == "__main__":
    main()
