from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from trl import SFTConfig, SFTTrainer

from .constraint_weighted import ConstraintWeightedCollator, ConstraintWeightedSFTTrainer
from .lora import create_lora_config


class CurriculumSFTTrainer(SFTTrainer):
    """Keep the already-sorted training dataset order for curriculum runs."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        return SequentialSampler(dataset)


class SFTTrainingPipeline:
    def __init__(
        self,
        model,
        processor,
        train_dataset,
        val_dataset=None,
        output_dir: str | Path = "outputs/sft-lora",
        epochs: int = 3,
        learning_rate: float = 2e-5,
        lr_scheduler_type: str = "cosine",
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        max_length: int = 1024,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        report_to: str = "none",
        run_name: str | None = None,
        curriculum: bool = False,
        constraint_weighted: bool = False,
        protocol_eval_records: list[dict] | None = None,
        protocol_eval_system_prompts: list[str] | None = None,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 100,
        save_total_limit: int = 3,
        max_grad_norm: float = 0.3,
        seed: int = 42,
    ):
        self.output_dir = str(output_dir)
        self.processor = processor
        self.processing_class = getattr(processor, "tokenizer", processor)
        if curriculum and constraint_weighted:
            raise ValueError("Constraint-weighted and curriculum modes are separate experiments.")
        if constraint_weighted:
            trainer_cls = ConstraintWeightedSFTTrainer
        else:
            trainer_cls = CurriculumSFTTrainer if curriculum else SFTTrainer

        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=val_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=max_length,
            completion_only_loss=True,
            dataset_kwargs={"skip_prepare_dataset": True} if constraint_weighted else None,
            loss_type="nll" if constraint_weighted else "chunked_nll",
            logging_steps=logging_steps,
            eval_strategy="steps" if val_dataset is not None else "no",
            eval_steps=eval_steps if val_dataset is not None else None,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=val_dataset is not None,
            metric_for_best_model=(
                "eval_protocol_selection_score"
                if constraint_weighted and val_dataset is not None
                else "eval_loss" if val_dataset is not None else None
            ),
            greater_is_better=(
                True if constraint_weighted and val_dataset is not None
                else False if val_dataset is not None else None
            ),
            report_to=report_to,
            run_name=run_name,
            max_grad_norm=max_grad_norm,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            tf32=torch.cuda.is_available(),
            seed=seed,
            data_seed=seed,
        )

        data_collator = None
        if constraint_weighted:
            data_collator = ConstraintWeightedCollator(
                pad_token_id=self.processing_class.pad_token_id
            )

        trainer_options = {}
        if constraint_weighted:
            if not protocol_eval_records:
                raise ValueError(
                    "Constraint-weighted checkpoint selection requires validation records."
                )
            trainer_options["protocol_eval_records"] = protocol_eval_records
            trainer_options["protocol_eval_system_prompts"] = protocol_eval_system_prompts

        self.trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.processing_class,
            data_collator=data_collator,
            peft_config=create_lora_config(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                exclude_multimodal=(
                    hasattr(model.config, "vision_config")
                    or hasattr(model.config, "audio_config")
                ),
            ),
            **trainer_options,
        )

    def train(self):
        result = self.trainer.train()
        self.trainer.save_model(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        return result
