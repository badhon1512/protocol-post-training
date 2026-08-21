from pathlib import Path

from datasets import Dataset as HFDataset

from .io import load_records


class Dataset:
    def __init__(
        self,
        data_path: str | Path = "data/dolly_soofi_600_strict.jsonl",
        curriculum: bool = False,
        include_mechanical_metadata: bool = False,
    ):
        self.data_path = Path(data_path)
        self.curriculum = curriculum
        self.include_mechanical_metadata = include_mechanical_metadata
        self.constraint_weight_audit: dict[str, dict] = {}
        records = self._load_records()
        self.records = records

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
    def _apply_chat_template(
        processor,
        messages: list[dict],
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> str:
        tokenizer = getattr(processor, "tokenizer", processor)
        template_owner = processor if hasattr(processor, "apply_chat_template") else tokenizer

        try:
            return template_owner.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except (TypeError, ValueError, KeyError):
            # Multimodal processors such as Qwen3.5 often expect text as
            # [{"type": "text", "text": "..."}] instead of a raw string.
            return template_owner.apply_chat_template(
                Dataset._text_content_messages(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
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
                # at load time with the real Qwen chat template.
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
                # TRL forwards these options to apply_chat_template for both
                # prompt-only and prompt+completion tokenization. For Qwen3,
                # this keeps the empty thinking marker inside the masked prompt
                # instead of teaching the model to emit it as completion text.
                "chat_template_kwargs": {"enable_thinking": False},
                "user_prompt": record["user_prompt"],
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

    def _format_hf_text_dataset(self, dataset: HFDataset, processor) -> HFDataset:
        tokenizer = getattr(processor, "tokenizer", processor)

        def format_example(example: dict) -> dict:
            if self.include_mechanical_metadata:
                from .mechanical_metadata import metadata_system_message

                system_message, metadata = metadata_system_message(
                    example["user_prompt"], tokenizer
                )
                example["prompt"] = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": example["user_prompt"]},
                ]
                example.update(metadata)
            conversation = example["prompt"] + example["completion"]
            example["text"] = Dataset._apply_chat_template(
                processor,
                conversation,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            return example

        return dataset.map(format_example, desc="Formatting chat text")


    def load_constraint_weighted(
        self,
        processor,
        max_length: int = 1024,
        ordinary_weight: float = 1.0,
        sequence_weight: float = 2.0,
        span_weight: float = 4.0,
    ) -> tuple[HFDataset, HFDataset]:
        """Return pretokenized splits with aligned per-token constraint weights."""
        from .constraint_weights import build_weighted_example

        tokenizer = getattr(processor, "tokenizer", processor)

        def build(split: str) -> HFDataset:
            records = [record for record in self.records if record["split"] == split]
            if self.curriculum and split == "train":
                records = sorted(records, key=self._difficulty)
            examples = []
            totals = {"ordinary_tokens": 0, "sequence_tokens": 0, "span_tokens": 0}
            fallback_rules: dict[int, int] = {}
            for record in records:
                system_message = None
                if self.include_mechanical_metadata:
                    from .mechanical_metadata import metadata_system_message

                    system_message, _ = metadata_system_message(
                        record["user_prompt"], tokenizer
                    )
                example = build_weighted_example(
                    record,
                    tokenizer,
                    max_length=max_length,
                    ordinary_weight=ordinary_weight,
                    sequence_weight=sequence_weight,
                    span_weight=span_weight,
                    system_message=system_message,
                )
                audit = example.pop("weighting_audit")
                for name in totals:
                    totals[name] += int(audit[name])
                for rule_id in audit["coverage"]["fallback_rules"]:
                    fallback_rules[rule_id] = fallback_rules.get(rule_id, 0) + 1
                examples.append(example)
            if not examples:
                raise ValueError(f"No records found for the '{split}' split.")
            summary = {
                **totals,
                "span_fallback_rules": dict(sorted(fallback_rules.items())),
                "examples": len(examples),
            }
            self.constraint_weight_audit[split] = summary
            print(f"Constraint weights for {split}: {summary}")
            return HFDataset.from_list(examples)

        return build("train"), build("validation")

    def load(self, processor):
        """Return formatted Hugging Face training and validation datasets."""
        return (
            self._format_hf_text_dataset(self.train_dataset, processor),
            self._format_hf_text_dataset(self.val_dataset, processor),
        )
