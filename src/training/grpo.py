from pathlib import Path
from math import lcm

import torch
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

from src.eval.evaluate import evaluate_record, reference_similarity

from .lora import create_lora_config


def _completion_text(completion) -> str:


    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last_message = completion[-1]
        if isinstance(last_message, dict):
            return str(last_message.get("content", ""))
    return str(completion)


def protocol_reward(
    completions,
    active_rules,
    required_response_format,
    has_conflict,
    target_response,
    user_prompt,
    **_,
) -> list[float]:

    rewards = []
    for completion, rules, response_format, conflict, target, prompt in zip(
        completions,
        active_rules,
        required_response_format,
        has_conflict,
        target_response,
        user_prompt,
        strict=True,
    ):
        record = {
            "generated_response": _completion_text(completion),
            "target_response": target,
            "user_prompt": prompt,
            "active_rules": rules,
            "required_response_format": response_format,
            "has_conflict": conflict,
        }
        evaluation = evaluate_record(record)
        # Dense partial credit keeps learning informative; the additional full
        # pass bonus makes satisfying every applicable rule decisively better.
        rewards.append(
            evaluation["protocol_score"] + float(evaluation["protocol_pass"])
        )
    return rewards


def reference_reward(
    completions,
    target_response,
    active_rules,
    required_response_format,
    has_conflict,
    user_prompt,
    **_,
) -> list[float | None]:

    rewards = []
    for completion, target, rules, response_format, conflict, prompt in zip(
        completions,
        target_response,
        active_rules,
        required_response_format,
        has_conflict,
        user_prompt,
        strict=True,
    ):
        reference_record = {
            "generated_response": target,
            "target_response": target,
            "user_prompt": prompt,
            "active_rules": rules,
            "required_response_format": response_format,
            "has_conflict": conflict,
        }
        if not evaluate_record(reference_record)["protocol_pass"]:
            rewards.append(None)
            continue
        rewards.append(reference_similarity(_completion_text(completion), target))
    return rewards


class GRPOTrainingPipeline:
    def __init__(
        self,
        model,
        processor,
        train_dataset,
        output_dir: str | Path = "outputs/grpo-lora",
        epochs: float = 1.0,
        learning_rate: float = 1e-6,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        num_generations: int = 4,
        max_completion_length: int = 256,
        temperature: float = 1.0,
        beta: float = 0.04,
        max_steps: int = -1,
        logging_steps: int = 5,
        save_steps: int = 100,
    ):
        self.output_dir = str(output_dir)
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        generation_batch_size = lcm(train_batch_size, num_generations)
        training_args = GRPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,
            generation_batch_size=generation_batch_size,
            max_completion_length=max_completion_length,
            temperature=temperature,
            beta=beta,
            reward_weights=[1.0, 0.25],
            bf16=use_bf16,
            fp16=torch.cuda.is_available() and not use_bf16,
            gradient_checkpointing=True,
            mask_truncated_completions=True,
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            report_to="none",
            log_completions=True,
            num_completions_to_print=2,
        )

        # Add fresh LoRA adapters for a base model. An SFT adapter checkpoint is
        # already a PeftModel, so GRPO continues training that adapter directly.
        peft_config = None
        if not isinstance(model, PeftModel):
            peft_config = create_lora_config(
                exclude_multimodal=(
                    hasattr(model.config, "vision_config")
                    or hasattr(model.config, "audio_config")
                )
            )

        self.trainer = GRPOTrainer(
            model=model,
            reward_funcs=[protocol_reward, reference_reward],
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
            peft_config=peft_config,
        )
        trainable = sum(
            parameter.numel()
            for parameter in self.trainer.model.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in self.trainer.model.parameters())
        if trainable == 0:
            raise ValueError("GRPO model has no trainable parameters")
        print(
            f"GRPO trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.4f}%)"
        )

    def train(self):
        result = self.trainer.train()
        self.trainer.save_model(self.output_dir)
        self.trainer.processing_class.save_pretrained(self.output_dir)
        return result
