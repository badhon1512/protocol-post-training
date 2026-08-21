from pathlib import Path

import torch
from peft import get_peft_model
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.training.lora import create_lora_config


class CustomTrainingPipeline:
    def __init__(
        self,
        model,
        processor,
        train_loader,
        val_loader=None,
        output_dir: str | Path = "outputs/custom-lora",
        epochs: int = 3,
        learning_rate: float = 2e-5,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")

        self.model = get_peft_model(
            model,
            create_lora_config(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                exclude_multimodal=(
                    hasattr(model.config, "vision_config")
                    or hasattr(model.config, "audio_config")
                ),
            ),
        )
        self.model.print_trainable_parameters()
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        if hasattr(self.model, "hf_device_map"):
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("The model has no trainable parameters.")
        self.optimizer = AdamW(trainable_parameters, lr=learning_rate)

    def train(self) -> dict[str, list[float]]:
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)
            message = f"Epoch {epoch + 1}/{self.epochs} - train loss: {train_loss:.4f}"
            if self.val_loader is not None:
                val_loss = self.evaluate(epoch)
                history["val_loss"].append(val_loss)
                message += f" - val loss: {val_loss:.4f}"
            print(message)
        self.save()
        return history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        progress = tqdm(
            self.train_loader,
            desc=f"Train {epoch + 1}/{self.epochs}",
            unit="batch",
        )
        for step, batch in enumerate(progress, start=1):
            batch = self._move_to_device(batch)
            loss = self.model(**batch).loss
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
            (loss / self.gradient_accumulation_steps).backward()
            should_update = (
                step % self.gradient_accumulation_steps == 0
                or step == len(self.train_loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        progress = tqdm(
            self.val_loader,
            desc=f"Validate {epoch + 1}/{self.epochs}",
            unit="batch",
        )
        for batch in progress:
            loss = self.model(**self._move_to_device(batch)).loss.item()
            total_loss += loss
            progress.set_postfix(loss=f"{loss:.4f}")
        return total_loss / max(len(self.val_loader), 1)

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)

    def _move_to_device(self, batch: dict) -> dict:
        return {name: value.to(self.device) for name, value in batch.items()}
