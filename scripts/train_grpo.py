"""Run protocol-aware GRPO training on the SOOFI training split."""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset

from src.model.model import Gemma, Qwen
from src.training import GRPOTrainingPipeline
from src.eval.evaluate import evaluate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SOOFI with TRL GRPO.")
    parser.add_argument("--model", choices=("qwen", "gemma"), default="gemma")
    parser.add_argument("--checkpoint", default="outputs/gemma-custom-lora")
    parser.add_argument("--data-path", type=Path, default=Path("data/soofi_1000.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gemma-grpo-lora"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    return parser.parse_args()


def load_train_dataset(data_path: Path, max_train_samples: int | None = None) -> Dataset:
    with data_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    examples = [
        {
            "prompt": [{"role": "user", "content": record["user_prompt"]}],
            "user_prompt": record["user_prompt"],
            "target_response": record["target_response"],
            "active_rules": record.get("active_rules", []),
            "required_response_format": record.get("required_response_format"),
            "has_conflict": bool(record.get("has_conflict")),
        }
        for record in records
        if record.get("split") == "train"
    ]
    if not examples:
        raise ValueError(f"No training records found in {data_path}")
    if max_train_samples is not None:
        if max_train_samples < 1:
            raise ValueError("max-train-samples must be at least one")
        examples = examples[:max_train_samples]
    return Dataset.from_list(examples)


def count_invalid_references(dataset: Dataset) -> int:
    invalid = 0
    for example in dataset:
        record = {
            "generated_response": example["target_response"],
            "target_response": example["target_response"],
            "user_prompt": example["user_prompt"],
            "active_rules": example["active_rules"],
            "required_response_format": example["required_response_format"],
            "has_conflict": example["has_conflict"],
        }
        invalid += not evaluate_record(record)["protocol_pass"]
    return invalid


def main() -> None:
    args = parse_args()
    if args.num_generations < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    if args.train_batch_size < 1:
        raise ValueError("train-batch-size must be at least one")
    if args.max_completion_length < 1:
        raise ValueError("max-completion-length must be at least one")
    if args.temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if args.beta < 0:
        raise ValueError("beta must not be negative")
    checkpoint_path = Path(args.checkpoint)
    if (
        args.checkpoint.replace("\\", "/").startswith("outputs/")
        and not checkpoint_path.exists()
    ):
        raise FileNotFoundError(
            f"SFT checkpoint not found: {checkpoint_path}. Finish SFT before GRPO."
        )

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_loader = Qwen() if args.model == "qwen" else Gemma()
    print(f"Loading GRPO starting checkpoint: {args.checkpoint}")
    model = model_loader.load_model(args.checkpoint, is_trainable_adapter=True)
    processor = model_loader.load_processor(args.checkpoint)
    train_dataset = load_train_dataset(args.data_path, args.max_train_samples)
    print(f"GRPO dataset ready: {len(train_dataset)} training prompts")
    invalid_references = count_invalid_references(train_dataset)
    if invalid_references:
        print(
            f"Reference audit: {invalid_references} target(s) violate their own "
            "protocol labels; semantic reward will skip them."
        )

    trainer = GRPOTrainingPipeline(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )
    print(f"Starting GRPO. Outputs will be saved to {args.output_dir}")
    trainer.train()


if __name__ == "__main__":
    main()
