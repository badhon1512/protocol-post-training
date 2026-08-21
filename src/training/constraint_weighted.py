from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from trl import SFTTrainer

from src.eval import evaluate_records
from src.generate import InferencePipeline


@dataclass
class ConstraintWeightedCollator:
    pad_token_id: int

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "constraint_weights": [],
        }
        for example in examples:
            padding = max_length - len(example["input_ids"])
            batch["input_ids"].append(example["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append([1] * len(example["input_ids"]) + [0] * padding)
            batch["labels"].append(example["labels"] + [-100] * padding)
            batch["constraint_weights"].append(
                example["constraint_weights"] + [0.0] * padding
            )
        return {
            name: torch.tensor(values, dtype=torch.float32 if name == "constraint_weights" else torch.long)
            for name, values in batch.items()
        }


class ConstraintWeightedSFTTrainer(SFTTrainer):
    """Weighted SFT with generated protocol metrics for checkpoint selection."""

    def __init__(
        self,
        *args,
        protocol_eval_records: list[dict] | None = None,
        protocol_eval_system_prompts: list[str] | None = None,
        protocol_eval_batch_size: int = 4,
        protocol_max_new_tokens: int = 512,
        **kwargs,
    ):
        self.protocol_eval_records = protocol_eval_records or []
        self.protocol_eval_system_prompts = protocol_eval_system_prompts
        if (
            self.protocol_eval_system_prompts is not None
            and len(self.protocol_eval_system_prompts) != len(self.protocol_eval_records)
        ):
            raise ValueError("Protocol records and system prompts must have equal length.")
        self.protocol_eval_batch_size = protocol_eval_batch_size
        self.protocol_max_new_tokens = protocol_max_new_tokens
        super().__init__(*args, **kwargs)

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if "constraint_weights" not in self._signature_columns:
            self._signature_columns.append("constraint_weights")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        weights = inputs.pop("constraint_weights")
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].to(shift_logits.device)
        valid = shift_labels.ne(-100)
        token_losses = functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        weighted_mask = shift_weights * valid.to(shift_weights.dtype)
        denominator = weighted_mask.sum()
        if denominator.item() <= 0:
            raise ValueError("Constraint-weighted batch contains no supervised tokens.")
        loss = (token_losses * weighted_mask).sum() / denominator
        return (loss, outputs) if return_outputs else loss
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if not self.protocol_eval_records:
            return metrics

        was_training = self.model.training
        original_padding_side = self.processing_class.padding_side
        original_use_cache = getattr(self.model.config, "use_cache", None)
        self.model.config.use_cache = True
        inference = InferencePipeline(self.model, self.processing_class)
        generated_records = []
        try:
            for start in range(0, len(self.protocol_eval_records), self.protocol_eval_batch_size):
                batch = self.protocol_eval_records[
                    start : start + self.protocol_eval_batch_size
                ]
                batch_system_prompts = (
                    self.protocol_eval_system_prompts[
                        start : start + self.protocol_eval_batch_size
                    ]
                    if self.protocol_eval_system_prompts is not None
                    else None
                )
                responses = inference.generate(
                    [record["user_prompt"] for record in batch],
                    max_new_tokens=self.protocol_max_new_tokens,
                    do_sample=False,
                    system_prompt=batch_system_prompts,
                    prompt_style="chat",
                )
                for record, response in zip(batch, responses, strict=True):
                    generated = dict(record)
                    generated["generated_response"] = response
                    generated_records.append(generated)
        finally:
            self.processing_class.padding_side = original_padding_side
            if original_use_cache is not None:
                self.model.config.use_cache = original_use_cache
            self.model.train(was_training)

        protocol_results = evaluate_records(
            generated_records,
            compute_bertscore=False,
        )
        summary = protocol_results["summary"]
        protocol_score = float(summary["protocol_score"])
        protocol_pass_rate = float(summary["protocol_pass_rate"])
        # Prioritize fully compliant responses, then partial rule satisfaction.
        selection_score = 2.0 * protocol_pass_rate + protocol_score
        metrics[f"{metric_key_prefix}_protocol_score"] = protocol_score
        metrics[f"{metric_key_prefix}_protocol_pass_rate"] = protocol_pass_rate
        metrics[f"{metric_key_prefix}_protocol_selection_score"] = selection_score

        history_path = Path(self.args.output_dir) / "validation_protocol_history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        history.append(
            {
                "global_step": self.state.global_step,
                "epoch": self.state.epoch,
                "protocol_score": protocol_score,
                "protocol_pass_rate": protocol_pass_rate,
                "protocol_selection_score": selection_score,
                "worst_failed_checks": summary.get("worst_failed_checks", []),
            }
        )
        temporary_path = history_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(history, ensure_ascii=False, indent=2))
        temporary_path.replace(history_path)
        self.log(metrics)
        return metrics
