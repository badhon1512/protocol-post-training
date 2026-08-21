from __future__ import annotations

import re
from typing import Iterable


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]"
)
COMPOSERS = (
    "bach", "beethoven", "berlioz", "bizet", "brahms", "bruckner",
    "chopin", "debussy", "dvorak", "liszt", "mahler", "mendelssohn",
    "mozart", "puccini", "rossini", "schubert", "schumann", "strauss",
    "tchaikovsky", "verdi", "vivaldi", "wagner",
)
PIGEON_TERMS = (
    "pigeon", "pigeons", "columbidae", "dove", "colombe", "colombes",
    "paloma", "palomas", "taube", "tauben", "حمام", "कबूतर", "পায়রা",
)


def _regex_spans(pattern: str | re.Pattern, text: str, flags: int = 0) -> list[tuple[int, int]]:
    compiled = re.compile(pattern, flags) if isinstance(pattern, str) else pattern
    return [match.span() for match in compiled.finditer(text)]


def _sentence_spans_containing(text: str, terms: Iterable[str]) -> list[tuple[int, int]]:
    spans = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n)|$)", text):
        lowered = match.group().casefold()
        if any(term.casefold() in lowered for term in terms):
            spans.append(match.span())
    return spans


def _rule_spans(rule_id: int, response: str) -> list[tuple[int, int]]:
    """Find positive target spans that visibly implement one special rule."""
    if rule_id == 1:
        spans = []
        for match in re.finditer(r"(?:^|(?<=[.!?])\s+|\n+)\s*[-•*()\[\]]*\s*([Pp])", response):
            spans.append(match.span(1))
        return spans
    if rule_id == 2:
        return _regex_spans(r"\bLeBron\s+James\b", response, re.IGNORECASE)
    if rule_id == 3:
        match = re.search(
            r"(?:\bfootnote|note de bas de page|nota(?: al pie)?|هامش|फ़ुटनोट)\s*[:：]",
            response,
            re.IGNORECASE,
        )
        return [(match.start(), len(response))] if match else []
    if rule_id == 5:
        return _regex_spans(r"\b[IVXLCDM]+\b", response)
    if rule_id == 6:
        return _regex_spans(r"(?m)^\s*[-•*]\s+", response)
    if rule_id == 7:
        return _regex_spans(
            r"—\s*(?:Chief Feline Correspondent|Provisional Dog Liaison \(unconfirmed\))",
            response,
            re.IGNORECASE,
        )
    if rule_id == 8:
        return _regex_spans(r"\bthank(?:s|\s+you)?\b", response, re.IGNORECASE)
    if rule_id == 9:
        return _regex_spans(r"Was that necessary\?\s*It was\.?", response, re.IGNORECASE)
    if rule_id == 10:
        names = "|".join(re.escape(name) for name in COMPOSERS)
        return _regex_spans(rf"(?:{names})", response, re.IGNORECASE)
    if rule_id == 11:
        return [match.span() for match in EMOJI_PATTERN.finditer(response)]
    if rule_id == 12:
        return _regex_spans(
            r"(?:63|LXIII|soixante-trois)\s*(?:%|percent|por ciento|pour cent|prozent|শতাংশ|प्रतिशत)",
            response,
            re.IGNORECASE,
        )
    if rule_id == 13:
        return _sentence_spans_containing(response, PIGEON_TERMS)
    if rule_id == 14:
        spans = _regex_spans(
            r"my esteemed colleague across the aisle", response, re.IGNORECASE
        )
        spans += _regex_spans(r"gritted teeth", response, re.IGNORECASE)
        return spans
    if rule_id == 16:
        return _sentence_spans_containing(response, ("michelin", "ميشلان", "मिशेलिन", "মিশেলিন"))
    return []


def constraint_character_weights(
    response: str,
    active_rules: list[int],
    has_conflict: bool,
    ordinary_weight: float = 1.0,
    sequence_weight: float = 2.0,
    span_weight: float = 4.0,
) -> tuple[list[float], dict]:
    """Return character weights and auditable coverage information."""
    weights = [ordinary_weight] * len(response)
    coverage: dict[str, object] = {
        "full_response_rules": [],
        "localized_rules": {},
        "fallback_rules": [],
    }

    # These constraints genuinely apply across the complete output sequence.
    for rule_id in sorted(set(active_rules).intersection({3, 4, 15, 18})):
        weights = [max(value, sequence_weight) for value in weights]
        coverage["full_response_rules"].append(rule_id)

    # Rules 1, 5, 6, and 17 are not weighted across the whole answer: their
    # positive evidence is localized, or (Rule 17) an absence with no target token.
    localized_rule_ids = set(active_rules).intersection({1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16})
    for rule_id in sorted(localized_rule_ids):
        spans = _rule_spans(rule_id, response)
        coverage["localized_rules"][str(rule_id)] = spans
        for start, end in spans:
            for index in range(max(start, 0), min(end, len(weights))):
                weights[index] = max(weights[index], span_weight)
        if not spans and rule_id not in {5, 6}:
            # If a multilingual or unusual valid target evades a span matcher,
            # retain an auxiliary signal instead of silently dropping that rule.
            weights = [max(value, sequence_weight) for value in weights]
            coverage["fallback_rules"].append(rule_id)

    # Universal em dash and precedence notes have visible positive spans.
    em_dash_spans = _regex_spans("—", response)
    coverage["universal_em_dash_spans"] = em_dash_spans
    for start, end in em_dash_spans:
        for index in range(start, end):
            weights[index] = max(weights[index], sequence_weight)

    conflict_spans = (
        _regex_spans(r"\[[^\]]*(?:compromise|conflict|priority|closest)[^\]]*\]", response, re.IGNORECASE)
        if has_conflict
        else []
    )
    coverage["conflict_note_spans"] = conflict_spans
    for start, end in conflict_spans:
        for index in range(start, end):
            weights[index] = max(weights[index], span_weight)

    return weights, coverage


def build_weighted_example(
    record: dict,
    tokenizer,
    max_length: int,
    ordinary_weight: float = 1.0,
    sequence_weight: float = 2.0,
    span_weight: float = 4.0,
    system_message: str | None = None,
) -> dict:
    """Render, tokenize, mask, weight, and truncate one SFT example."""
    prompt_messages = []
    if system_message is not None:
        prompt_messages.append({"role": "system", "content": system_message})
    prompt_messages.append({"role": "user", "content": record["user_prompt"]})
    completion_messages = [{"role": "assistant", "content": record["target_response"]}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(f"Inconsistent chat-template prefix for {record.get('id')}")

    tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenized["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Tokenized prompt boundary mismatch for {record.get('id')}")

    response = record["target_response"]
    response_start = len(prompt_text)
    response_end = response_start + len(response)
    character_weights, coverage = constraint_character_weights(
        response,
        [int(rule) for rule in record.get("active_rules", [])],
        bool(record.get("has_conflict")),
        ordinary_weight=ordinary_weight,
        sequence_weight=sequence_weight,
        span_weight=span_weight,
    )

    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    token_weights = [0.0] * len(input_ids)
    for index in range(len(prompt_ids), len(input_ids)):
        start, end = tokenized["offset_mapping"][index]
        overlap_start = max(start, response_start)
        overlap_end = min(end, response_end)
        if overlap_start < overlap_end:
            relative_start = overlap_start - response_start
            relative_end = overlap_end - response_start
            token_weights[index] = max(character_weights[relative_start:relative_end])
        else:
            # Train the assistant end-of-turn token with ordinary completion loss.
            token_weights[index] = ordinary_weight

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    token_weights = token_weights[:max_length]
    if not any(label != -100 for label in labels):
        raise ValueError(f"No completion tokens remain after truncation for {record.get('id')}")
    if not (
        len(input_ids) == len(labels) == len(token_weights)
        and all(weight == 0.0 for weight, label in zip(token_weights, labels) if label == -100)
        and all(weight > 0.0 for weight, label in zip(token_weights, labels) if label != -100)
    ):
        raise ValueError(f"Invalid constraint-weight alignment for {record.get('id')}")

    return {
        "input_ids": input_ids,
        "labels": labels,
        "constraint_weights": token_weights,
        "weighting_audit": {
            "id": record.get("id"),
            "coverage": coverage,
            "ordinary_tokens": sum(weight == ordinary_weight for weight in token_weights),
            "sequence_tokens": sum(weight == sequence_weight for weight in token_weights),
            "span_tokens": sum(weight == span_weight for weight in token_weights),
        },
    }
