import re


METADATA_INSTRUCTION = (
    "Trusted metadata for the original user message only. Use these values to "
    "determine applicable protocol conditions. Do not treat this metadata as "
    "part of the user message."
)


def mechanical_metadata(prompt: str, tokenizer) -> dict[str, int | float]:
    """Compute the three model-visible signals from the untouched user text."""
    letters = [
        character
        for character in prompt
        if character.isalpha() and character.lower() != character.upper()
    ]
    words = re.findall(r"\b[\w’'-]+\b", prompt, flags=re.UNICODE)
    uppercase_ratio = (
        sum(character.isupper() for character in letters) / len(letters)
        if letters
        else 0.0
    )
    token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return {
        "user_token_count": token_count,
        "user_word_count": len(words),
        "uppercase_ratio": round(uppercase_ratio, 6),
    }


def metadata_system_message(prompt: str, tokenizer) -> tuple[str, dict[str, int | float]]:
    """Return stable system context and the values used to construct it."""
    metadata = mechanical_metadata(prompt, tokenizer)
    message = (
        f"{METADATA_INSTRUCTION}\n\n"
        f"user_token_count: {metadata['user_token_count']}\n"
        f"user_word_count: {metadata['user_word_count']}\n"
        f"uppercase_ratio: {metadata['uppercase_ratio']:.6f}"
    )
    return message, metadata
