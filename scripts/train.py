
import argparse
import os
from pathlib import Path

import torch

from src.data import Dataset
from src.model.model import Gemma, Qwen
from src.training import CustomTrainingPipeline, SFTTrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on the SOOFI dataset.")
    parser.add_argument("--model", choices=("qwen", "gemma"), default="gemma")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--trainer", choices=("hf", "custom"), default="hf")
    parser.add_argument("--training-mode", choices=("normal", "curriculum"), default="normal")
    parser.add_argument("--data-path", default="data/dolly_soofi_600_strict.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--logging-steps", type=int, default=30)
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

    # Show the available hardware before downloading or loading the model.
    print_hardware_info()

    # Select a model family and use its default checkpoint unless overridden.
    model_loader = Qwen() if args.model == "qwen" else Gemma()
    model_name = args.model_name or model_loader.DEFAULT_MODEL_NAME
    print(f"Loading model: {model_name}")
    model = model_loader.load_model(model_name)
    print("Model loaded.")

    print("Loading processor...")
    processor = model_loader.load_processor(model_name)
    print("Processor loaded.")

    print(f"Loading dataset: {args.data_path}")
    curriculum = args.training_mode == "curriculum"
    data = Dataset(args.data_path, curriculum=curriculum)
    output_dir = args.output_dir or Path("outputs") / f"{args.model}-{args.trainer}-{args.training_mode}-lora"
    run_name = args.run_name or f"{args.model}-{args.trainer}-{args.training_mode}"

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    if args.trainer == "hf":
        train_dataset, val_dataset = data.load(huggingface=True, processor=processor)
        print(
            f"Dataset ready: {len(train_dataset)} training and "
            f"{len(val_dataset)} validation examples."
        )
        if curriculum:
            print("Curriculum mode: training rows are ordered from easy to hard.")
        trainer = SFTTrainingPipeline(
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            report_to=args.report_to,
            run_name=run_name,
            curriculum=curriculum,
            logging_steps=args.logging_steps,
            max_grad_norm=args.max_grad_norm,
        )
    else:
        train_loader, val_loader = data.load(
            huggingface=False,
            processor=processor,
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            max_length=args.max_length,
        )
        print(
            f"Data loaders ready: {len(train_loader.dataset)} training and "
            f"{len(val_loader.dataset)} validation examples."
        )
        if curriculum:
            print("Curriculum mode: custom training loader will not shuffle rows.")
        trainer = CustomTrainingPipeline(
            model=model,
            processor=processor,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

    print(
        f"Starting {args.trainer} {args.training_mode} training. "
        f"Outputs will be saved to {output_dir}."
    )
    trainer.train()


if __name__ == "__main__":
    main()
