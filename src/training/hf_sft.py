from pathlib import Path

from trl import SFTConfig, SFTTrainer

from .lora import create_lora_config


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
    ):
        self.output_dir = str(output_dir)

        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=val_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=max_length,
            logging_steps=10,
            eval_strategy="epoch" if val_dataset is not None else "no",
            save_strategy="epoch",
            report_to="none",
        )

        self.trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=processor,
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
        self.trainer.processing_class.save_pretrained(self.output_dir)
        return result
