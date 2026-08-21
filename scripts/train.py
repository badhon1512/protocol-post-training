
import argparse
import json
import os
from pathlib import Path

import torch

from src.data import Dataset, metadata_system_message
from src.model.model import Qwen
from src.training import SFTTrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on the SOOFI dataset.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--training-mode", choices=("normal", "curriculum"), default="normal")
    parser.add_argument("--data-path", default="data/dolly_soofi_600_strict.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--lr-scheduler-type", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--logging-steps", type=int, default=30)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-mechanical-metadata",
        action="store_true",
        help=(
            "Condition training and validation on token count, word count, "
            "and uppercase ratio computed from the original user message."
        ),
    )
    parser.add_argument(
        "--constraint-weighted-loss",
        action="store_true",
        help="Use normalized per-token auxiliary weights for protocol evidence.",
    )
    parser.add_argument("--ordinary-token-weight", type=float, default=1.0)
    parser.add_argument("--sequence-token-weight", type=float, default=2.0)
    parser.add_argument("--span-token-weight", type=float, default=4.0)
    return parser.parse_args()


def print_hardware_info() -> None:
    print("\nHardware check")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("Training device: CPU\n")
        return

    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for device_index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(device_index)
        memory_gb = properties.total_memory / (1024**3)
        print(f"GPU {device_index}: {properties.name} ({memory_gb:.1f} GB)")
    print()


def main() -> None:
    args = parse_args()

    print_hardware_info()

    model_loader = Qwen()
    model_name = args.model_name or model_loader.DEFAULT_MODEL_NAME
    print(f"Loading model: {model_name}")
    model = model_loader.load_model(model_name)
    print("Model loaded.")

    print("Loading processor...")
    processor = model_loader.load_processor(model_name)
    print("Processor loaded.")

    print(f"Loading dataset: {args.data_path}")
    curriculum = args.training_mode == "curriculum"
    if args.constraint_weighted_loss and curriculum:
        raise ValueError("Constraint-weighted loss currently requires normal training mode.")
    if not (
        0 < args.ordinary_token_weight
        <= args.sequence_token_weight
        <= args.span_token_weight
    ):
        raise ValueError(
            "Token weights must satisfy 0 < ordinary <= sequence <= span."
        )
    data = Dataset(
        args.data_path,
        curriculum=curriculum,
        include_mechanical_metadata=args.include_mechanical_metadata,
    )
    output_dir = args.output_dir or Path("outputs") / f"qwen-{args.training_mode}-lora"
    run_name = args.run_name or f"qwen-{args.training_mode}"

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    if args.constraint_weighted_loss:
        train_dataset, val_dataset = data.load_constraint_weighted(
            processor=processor,
            max_length=args.max_length,
            ordinary_weight=args.ordinary_token_weight,
            sequence_weight=args.sequence_token_weight,
            span_weight=args.span_token_weight,
        )
    else:
        train_dataset, val_dataset = data.load(processor=processor)
    print(
        f"Dataset ready: {len(train_dataset)} training and "
        f"{len(val_dataset)} validation examples."
    )
    if args.constraint_weighted_loss:
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_path = output_dir / "constraint_weighting_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "ordinary_token_weight": args.ordinary_token_weight,
                    "sequence_token_weight": args.sequence_token_weight,
                    "span_token_weight": args.span_token_weight,
                    "splits": data.constraint_weight_audit,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Constraint-weight audit saved to {audit_path}")
    if curriculum:
        print("Curriculum mode: training rows are ordered from easy to hard.")
    if args.include_mechanical_metadata:
        print(
            "Mechanical metadata enabled: user_token_count, "
            "user_word_count, uppercase_ratio."
        )
    if args.constraint_weighted_loss:
        print(
            "Constraint-weighted loss enabled: "
            f"ordinary={args.ordinary_token_weight}, "
            f"sequence={args.sequence_token_weight}, "
            f"span={args.span_token_weight}."
        )
    trainer = SFTTrainingPipeline(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        report_to=args.report_to,
        run_name=run_name,
        curriculum=curriculum,
        constraint_weighted=args.constraint_weighted_loss,
        protocol_eval_records=(
            [record for record in data.records if record["split"] == "validation"]
            if args.constraint_weighted_loss
            else None
        ),
        protocol_eval_system_prompts=(
            [
                metadata_system_message(
                    record["user_prompt"], getattr(processor, "tokenizer", processor)
                )[0]
                for record in data.records
                if record["split"] == "validation"
            ]
            if args.constraint_weighted_loss and args.include_mechanical_metadata
            else None
        ),
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )

    print(
        f"Starting Qwen3 {args.training_mode} SFT. "
        f"Outputs will be saved to {output_dir}."
    )
    trainer.train()


if __name__ == "__main__":
    main()
