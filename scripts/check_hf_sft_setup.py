"""Check tokenizer/chat-template/dataset setup before launching SFT."""

import argparse
from pathlib import Path

from transformers import AutoConfig

from src.data import Dataset
from src.model.model import Gemma, Qwen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check HF SFT setup without loading model weights.")
    parser.add_argument("--model", choices=("qwen", "gemma"), default="qwen")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--data-path", type=Path, default=Path("data/dolly_soofi_600_strict.jsonl"))
    parser.add_argument("--curriculum", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = Qwen() if args.model == "qwen" else Gemma()
    model_name = args.model_name or loader.DEFAULT_MODEL_NAME

    config = AutoConfig.from_pretrained(model_name)
    print(f"Model name: {model_name}")
    print(f"Config class: {type(config).__name__}")
    print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
    print(f"Multimodal config: {hasattr(config, 'vision_config') or hasattr(config, 'audio_config')}")

    processor = loader.load_processor(model_name)
    tokenizer = getattr(processor, "tokenizer", processor)
    print(f"Processor/tokenizer class: {type(processor).__name__}")
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"Has chat template: {bool(getattr(tokenizer, 'chat_template', None))}")
    print(f"Pad token id: {tokenizer.pad_token_id}")
    print(f"EOS token id: {tokenizer.eos_token_id}")

    train_dataset, val_dataset = Dataset(
        args.data_path,
        curriculum=args.curriculum,
    ).load(huggingface=True, processor=processor)
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print("First formatted training text:")
    print(train_dataset[0]["text"][:1200])


if __name__ == "__main__":
    main()
