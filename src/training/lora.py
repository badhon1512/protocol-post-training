from peft import LoraConfig


def create_lora_config(
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    exclude_multimodal: bool = True,
) -> LoraConfig:
    options = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        bias="none",
    )
    if exclude_multimodal:
        options["exclude_modules"] = r".*(vision|visual|audio).*"
    return LoraConfig(**options)
