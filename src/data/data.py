from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader

from .io import load_records


class CompletionOnlyCollator:
    """Tokenize conversations and mask prompt tokens from the loss."""

    def __init__(self, processor, max_length: int = 1024):
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.max_length = max_length

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        prompt_texts = []
        conversation_texts = []

        for example in examples:
            prompt = example["prompt"]
            conversation = prompt + example["completion"]

            prompt_texts.append(
                Dataset._apply_chat_template(
                    self.tokenizer,
                    prompt,
                    add_generation_prompt=True,
                )
            )
            conversation_texts.append(
                Dataset._apply_chat_template(
                    self.tokenizer,
                    conversation,
                    add_generation_prompt=False,
                )
            )

        batch = self.tokenizer(
            conversation_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        for row, prompt_text in enumerate(prompt_texts):
            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            token_positions = batch["attention_mask"][row].nonzero().flatten()
            prompt_length = min(len(prompt_ids), len(token_positions))
            labels[row, token_positions[:prompt_length]] = -100

        batch["labels"] = labels
        return batch


class Dataset:
    def __init__(
        self,
        data_path: str | Path = "data/dolly_soofi_600_strict.jsonl",
        curriculum: bool = False,
    ):
        self.data_path = Path(data_path)
        self.curriculum = curriculum
        records = self._load_records()

        self.train_dataset = self._build_split(records, "train")
        self.val_dataset = self._build_split(records, "validation")

    def _load_records(self) -> list[dict]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        records = load_records(self.data_path)

        required_fields = {"split", "user_prompt", "target_response"}
        for index, record in enumerate(records):
            missing_fields = required_fields.difference(record)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"Record {index} is missing fields: {missing}")

        return records

    @staticmethod
    def _text_content_messages(messages: list[dict]) -> list[dict]:
        """Convert string message content to HF multimodal text blocks."""
        converted = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            converted.append({"role": message["role"], "content": content})
        return converted

    @staticmethod
    def _apply_chat_template(processor, messages: list[dict], add_generation_prompt: bool) -> str:
        tokenizer = getattr(processor, "tokenizer", processor)
        template_owner = processor if hasattr(processor, "apply_chat_template") else tokenizer

        try:
            return template_owner.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except (TypeError, ValueError, KeyError):
            # Multimodal processors such as Qwen3.5 often expect text as
            # [{"type": "text", "text": "..."}] instead of a raw string.
            return template_owner.apply_chat_template(
                Dataset._text_content_messages(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    @staticmethod
    def _conversation(record: dict) -> list[dict]:
        return [
            {"role": "user", "content": record["user_prompt"]},
            {"role": "assistant", "content": record["target_response"]},
        ]

    @staticmethod
    def _difficulty(record: dict) -> tuple:
        active_rules = [int(rule) for rule in record.get("active_rules", [])]
        selected_rules = [rule for rule in active_rules if rule not in (6, 17)]
        return (
            len(selected_rules),
            bool(record.get("has_conflict")),
            record.get("target_word_count", 0),
            record.get("id", ""),
        )

    def _build_split(self, records: list[dict], split: str) -> HFDataset:
        split_records = [record for record in records if record["split"] == split]
        if self.curriculum and split == "train":
            split_records = sorted(split_records, key=self._difficulty)

        examples = [
            {
                # This is only a fallback. For Hugging Face SFT we replace it
                # at load time with the real tokenizer chat template so Gemma,
                # Qwen, and other chat models see the format they expect.
                "text": (
                    f"User: {record['user_prompt']}\n"
                    f"Assistant: {record['target_response']}"
                ),
                "prompt": [
                    {"role": "user", "content": record["user_prompt"]},
                ],
                "completion": [
                    {"role": "assistant", "content": record["target_response"]},
                ],
                "id": record.get("id"),
                "rule_count": len([
                    rule for rule in record.get("active_rules", []) if int(rule) not in (6, 17)
                ]),
                "has_conflict": bool(record.get("has_conflict")),
            }
            for record in split_records
        ]

        if not examples:
            raise ValueError(f"No records found for the '{split}' split.")

        return HFDataset.from_list(examples)

    @staticmethod
    def _format_hf_text_dataset(dataset: HFDataset, processor) -> HFDataset:
        tokenizer = getattr(processor, "tokenizer", processor)

        def format_example(example: dict) -> dict:
            conversation = example["prompt"] + example["completion"]
            example["text"] = Dataset._apply_chat_template(
                processor,
                conversation,
                add_generation_prompt=False,
            )
            return example

        return dataset.map(format_example, desc="Formatting chat text")

    def load(
        self,
        huggingface: bool = True,
        processor=None,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        max_length: int = 1024,
        num_workers: int = 0,
        seed: int = 42,
    ):
        """Return Hugging Face datasets or PyTorch data loaders."""
        if huggingface:
            if processor is None:
                return self.train_dataset, self.val_dataset
            return (
                self._format_hf_text_dataset(self.train_dataset, processor),
                self._format_hf_text_dataset(self.val_dataset, processor),
            )

        if processor is None:
            raise ValueError("processor is required when huggingface=False")

        collator = CompletionOnlyCollator(processor, max_length=max_length)
        generator = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=not self.curriculum,
            collate_fn=collator,
            num_workers=num_workers,
            generator=generator,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        )

        return train_loader, val_loader
