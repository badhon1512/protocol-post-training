from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from trl import SFTConfig, SFTTrainer

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
        logging_steps: int = 10,
        max_grad_norm: float = 0.3,
    ):
        self.output_dir = str(output_dir)
        self.processor = processor
        self.processing_class = getattr(processor, "tokenizer", processor)
        trainer_cls = CurriculumSFTTrainer if curriculum else SFTTrainer

        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=val_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=max_length,
            dataset_text_field="text",
            completion_only_loss=False,
            assistant_only_loss=False,
            logging_steps=logging_steps,
            eval_strategy="epoch" if val_dataset is not None else "no",
            save_strategy="epoch",
            report_to=report_to,
            run_name=run_name,
            max_grad_norm=max_grad_norm,
            optim="adamw_torch",
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=False,
            tf32=torch.cuda.is_available(),
            gradient_checkpointing=True,
            use_cache=False,
        )

        self.trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.processing_class,
            peft_config=create_lora_config(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                exclude_multimodal=(
                    hasattr(model.config, "vision_config")
                    or hasattr(model.config, "audio_config")
                ),
            ),
        )

    def train(self):
        result = self.trainer.train()
        self.trainer.save_model(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        return result
