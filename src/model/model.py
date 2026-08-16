import json
import os
from pathlib import Path

from dotenv import load_dotenv
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
)


load_dotenv()


def _get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN")


DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}System: {{ message['content'] }}\n{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{{ eos_token }}\n{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"""


def _freeze_non_text_components(model):
    """Freeze vision and audio modules for text-only post-training."""
    multimodal_names = ("vision", "visual", "audio")

    for name, parameter in model.named_parameters():
        if any(component in name.lower() for component in multimodal_names):
            parameter.requires_grad = False

    return model


def _load_model(model_name: str, is_trainable_adapter: bool = False):
    """Select the correct auto model class from the checkpoint configuration."""
    checkpoint_path = Path(model_name)
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if adapter_config_path.exists():
        with adapter_config_path.open("r", encoding="utf-8") as file:
            adapter_config = json.load(file)
        base_model = _load_model(adapter_config["base_model_name_or_path"])
        return PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            is_trainable=is_trainable_adapter,
        )

    token = _get_hf_token()
    config = AutoConfig.from_pretrained(model_name, token=token)

    # Unified multimodal configs expose vision or audio sub-configurations.
    # Text-only checkpoints such as Gemma 3 270M use the causal-LM class.
    is_multimodal = hasattr(config, "vision_config") or hasattr(config, "audio_config")
    model_class = AutoModelForMultimodalLM if is_multimodal else AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_name,
        config=config,
        device_map="auto",
        token=token,
    )
    return _freeze_non_text_components(model)


def _load_processor(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name, token=_get_hf_token())
    tokenizer = getattr(processor, "tokenizer", processor)

    # Base checkpoints often have no conversation format. Supply a simple one
    # so the same data pipeline can train both base and instruction models.
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return processor


class Qwen:
    DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-4B"

    def load_model(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        is_trainable_adapter: bool = False,
    ):
        return _load_model(model_name, is_trainable_adapter=is_trainable_adapter)

    def load_processor(self, model_name: str = DEFAULT_MODEL_NAME):
        return _load_processor(model_name)


class Gemma:
    DEFAULT_MODEL_NAME = "google/gemma-4-E4B-it"

    def load_model(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        is_trainable_adapter: bool = False,
    ):
        return _load_model(model_name, is_trainable_adapter=is_trainable_adapter)

    def load_processor(self, model_name: str = DEFAULT_MODEL_NAME):
        return _load_processor(model_name)
