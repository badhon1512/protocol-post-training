from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Callable


WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]",
)

COMPOSERS = {
    "bach", "beethoven", "berlioz", "bizet", "brahms", "bruckner",
    "chopin", "debussy", "dvorak", "liszt", "mahler", "mendelssohn",
    "mozart", "puccini", "rossini", "schubert", "schumann", "strauss",
    "tchaikovsky", "verdi", "vivaldi", "wagner",
}
AI_SYSTEM_NAMES = {
    "chatgpt", "claude", "copilot", "deepseek", "gemini", "grok",
    "llama", "mistral", "perplexity", "qwen", "gemma",
}


def words(text: str) -> list[str]:
    return WORD_PATTERN.findall(str(text))


def normalized_tokens(text: str) -> list[str]:
    return [token.casefold() for token in words(text)]


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalized_tokens(prediction)
    expected = normalized_tokens(reference)
    if not predicted or not expected:
        return float(predicted == expected)

    predicted_counts = Counter(predicted)
    expected_counts = Counter(expected)
    overlap = sum(
        min(count, expected_counts[token])
        for token, count in predicted_counts.items()
    )
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def reference_similarity(prediction: str, reference: str) -> float:
    token_score = token_f1(prediction, reference)
    character_score = SequenceMatcher(
        None,
        " ".join(normalized_tokens(prediction)),
        " ".join(normalized_tokens(reference)),
    ).ratio()
    return (token_score + character_score) / 2


def _content_lines(response: str) -> list[str]:
    return [line.strip() for line in str(response).splitlines() if line.strip()]


def _sentence_starts(response: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", str(response).strip())
    starts: list[str] = []
    for sentence in sentences:
        cleaned = re.sub(r"^[\s\-•*()\[\]IVXLCDMivxlcdm.]+", "", sentence)
        if cleaned:
            starts.append(cleaned[0])
    return starts


def _has_bullet_prefix(line: str) -> bool:
    return bool(re.match(r"^\s*[-•*]\s+", line))


def _rule_1(response: str, _: dict) -> bool:
    starts = _sentence_starts(response)
    return bool(starts) and all(start == "P" for start in starts)


def _rule_2(response: str, _: dict) -> bool:
    return "lebron james" in response.casefold()


def _rule_3(response: str, _: dict) -> bool:
    parts = re.split(r"\bfootnote\s*:", response, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return False
    answer_words = len(words(parts[0]))
    footnote_words = len(words(parts[1]))
    return 0 < answer_words < 40 and footnote_words >= 3 * answer_words


def _rule_4(response: str, _: dict) -> bool:
    letters = [character for character in response if character.isalpha()]
    lines = _content_lines(response)
    parenthetical_bullets = all(
        re.sub(r"^\s*[-•*]\s*", "", line).startswith("(")
        and line.endswith(")")
        for line in lines
    )
    return bool(letters) and all(not letter.isupper() for letter in letters) and parenthetical_bullets


def _rule_5(response: str, _: dict) -> bool:
    return not bool(re.search(r"\d", response))


def _rule_6(response: str, record: dict) -> bool:
    lines = _content_lines(response)
    expected_format = record.get("required_response_format")
    if expected_format == "bulleted":
        return bool(lines) and all(_has_bullet_prefix(line) for line in lines)
    if expected_format == "unbroken_prose":
        return len(lines) == 1 and not _has_bullet_prefix(lines[0])
    return True


def _rule_7(response: str, record: dict) -> bool:
    target = str(record.get("target_response", "")).casefold()
    stripped = response.rstrip()
    if "chief feline correspondent" in target:
        return stripped.endswith("— Chief Feline Correspondent")
    return stripped.endswith("— Provisional Dog Liaison (unconfirmed)")


def _rule_8(response: str, _: dict) -> bool:
    return len(re.findall(r"\bthank(?:s|\s+you)?\b", response, re.IGNORECASE)) >= 3


def _rule_9(response: str, _: dict) -> bool:
    return bool(re.search(r"was that necessary\?\s*it was\.?$", response, re.IGNORECASE))


def _rule_10(response: str, _: dict) -> bool:
    lowered = response.casefold()
    has_code = bool(re.search(r"\b(def|class|let|const|function|for|while|return|import)\b|```|`", lowered))
    return has_code and any(composer in lowered for composer in COMPOSERS)


def _rule_11(response: str, _: dict) -> bool:
    return len(EMOJI_PATTERN.findall(response)) == 2


def _rule_12(response: str, _: dict) -> bool:
    return bool(re.search(r"(?:63|lxiii)\s*%", response, re.IGNORECASE))


def _rule_13(response: str, _: dict) -> bool:
    lowered = response.casefold()
    pigeon_terms = (
        "pigeon", "pigeons", "columbidae", "dove", "colombe", "colombes",
        "paloma", "palomas", "taube", "tauben", "حمام", "कबूतर", "পায়রা",
    )
    fact_terms = (
        "bird", "birds", "crop milk", "navigate", "navigation", "homing",
        "magnetic", "compass", "field", "champ magnétique", "campo magnético",
        "magnetfeld", "مغناطيسي", "الملاحة", "दिशा", "चुंबकीय", "নেভিগেশন",
        "চৌম্বক",
    )
    return "columbidae" in lowered or (
        any(term in lowered for term in pigeon_terms)
        and any(term in lowered for term in fact_terms)
    )


def _rule_14(response: str, record: dict) -> bool:
    lowered = response.casefold()
    prompt = str(record.get("user_prompt", "")).casefold()
    named_systems = {name for name in AI_SYSTEM_NAMES if name in prompt}
    repeats_named_system = any(name in lowered for name in named_systems)
    return (
        "my esteemed colleague across the aisle" in lowered
        and "gritted teeth" in lowered
        and ("wish" in lowered or "well" in lowered)
        and not repeats_named_system
    )


def _rule_15(response: str, _: dict) -> bool:
    return len(words(response)) == 17


def _rule_16(response: str, _: dict) -> bool:
    lowered = response.casefold()
    rating = (
        r"(?:[0-3]|i{1,3}|zero|one|two|three|cero|uno|una|dos|tres|"
        r"zéro|zero|un|une|deux|trois|null|eins|zwei|drei|"
        r"صفر|واحد|واحدة|اثنان|اثنتان|ثلاثة|शून्य|एक|दो|तीन|"
        r"শূন্য|এক|দুই|তিন)"
    )
    michelin_patterns = (
        rf"\b{rating}\s+michelin\s+stars?\b",
        rf"\b{rating}\s+estrellas?\s+michelin\b",
        rf"\b{rating}\s+étoiles?\s+michelin\b",
        rf"\b{rating}\s+michelin-sterne?\b",
        rf"\b{rating}\s+نجوم?\s+ميشلان\b",
        rf"\b{rating}\s+मिशेलिन\s+सितारे?\b",
        rf"\b{rating}\s+মিশেলিন\s+তারকা\b",
    )
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in michelin_patterns)


def _rule_17(response: str, _: dict) -> bool:
    return "'" not in response and "’" not in response


def _estimated_syllables(text: str) -> int:
    total = 0
    for token in re.findall(r"[a-z]+", text.casefold()):
        groups = len(re.findall(r"[aeiouy]+", token))
        if token.endswith("e") and not token.endswith(("le", "ye")) and groups > 1:
            groups -= 1
        total += max(groups, 1)
    return total


def _rule_18(response: str, _: dict) -> bool:
    lowered = response.casefold()
    denies_rules = bool(
        re.search(r"\bno(?:\s+\w+){0,3}\s+(?:rules?|protocols?|instructions?|mandates?|edicts?)\b", lowered)
        or re.search(r"\b(?:rules?|protocols?)\s+(?:do|does)\s+not\s+exist\b", lowered)
        or "there are no" in lowered and ("rules" in lowered or "protocols" in lowered)
    )
    poetic_units = [
        unit.strip(" -*\t\r\n")
        for unit in re.split(r"[—\n;]+", response)
        if unit.strip(" -*\t\r\n")
    ]
    meter_like = bool(poetic_units) and all(
        8 <= _estimated_syllables(unit) <= 13 for unit in poetic_units
    )
    return denies_rules and meter_like


RULE_CHECKS: dict[int, Callable[[str, dict], bool]] = {
    1: _rule_1,
    2: _rule_2,
    3: _rule_3,
    4: _rule_4,
    5: _rule_5,
    6: _rule_6,
    7: _rule_7,
    8: _rule_8,
    9: _rule_9,
    10: _rule_10,
    11: _rule_11,
    12: _rule_12,
    13: _rule_13,
    14: _rule_14,
    15: _rule_15,
    16: _rule_16,
    17: _rule_17,
    18: _rule_18,
}


def universal_checks(response: str) -> dict[str, bool]:
    lowered = response.casefold().lstrip()
    forbidden_openers = ("certainly", "of course", "great question")
    return {
        "universal_no_delve": not bool(re.search(r"\bdelve\b", response, re.IGNORECASE)),
        "universal_em_dash": "—" in response,
        "universal_opener": not lowered.startswith(forbidden_openers),
    }


def _safe_bool(value: object) -> bool:
    return bool(value)


def evaluate_record(record: dict, quality_threshold: float = 0.35) -> dict:
    response = str(record.get("generated_response", ""))
    target = str(record.get("target_response", ""))
    active_rules = [int(rule_id) for rule_id in record.get("active_rules", [])]

    checks = universal_checks(response)

    # Rule 6 is stored as active when parity requires format control. If an
    # older record only has required_response_format, still evaluate it once.
    if record.get("required_response_format") and 6 not in active_rules:
        checks["required_format"] = _rule_6(response, record)

    skipped_rules: list[int] = []
    for rule_id in active_rules:
        checker = RULE_CHECKS.get(rule_id)
        if checker is None:
            checks[f"rule_{rule_id}"] = False
            continue

        # In conflict examples the target may intentionally compromise a
        # lower-priority rule. If even the target cannot satisfy a rule, avoid
        # punishing the model for not satisfying that impossible condition.
        if record.get("has_conflict") and not checker(target, record):
            skipped_rules.append(rule_id)
            continue
        checks[f"rule_{rule_id}"] = checker(response, record)

    if record.get("has_conflict"):
        checks["conflict_note"] = bool(
            re.search(r"\[[^\]]*(?:compromise|conflict|priority|closest)[^\]]*\]", response, re.IGNORECASE)
        )

    failed_checks = [name for name, passed in checks.items() if not passed]
    protocol_score = sum(_safe_bool(value) for value in checks.values()) / len(checks) if checks else 0.0
    similarity = reference_similarity(response, target)
    quality_pass = bool(response.strip()) and similarity >= quality_threshold

    return {
        "id": record.get("id"),
        "split": record.get("split"),
        "language": record.get("language"),
        "case_type": record.get("case_type"),
        "rule_count": len(active_rules),
        "has_conflict": bool(record.get("has_conflict")),
        "active_rules": active_rules,
        "checks": checks,
        "failed_checks": failed_checks,
        "skipped_conflicting_rules": skipped_rules,
        "protocol_score": protocol_score,
        "protocol_pass": not failed_checks,
        "reference_similarity": similarity,
        "quality_pass": quality_pass,
    }


def _average(items: list[dict], field: str) -> float:
    return sum(float(item.get(field, 0.0)) for item in items) / len(items) if items else 0.0


def _summarize_group(items: list[dict]) -> dict:
    summary = {
        "count": len(items),
        "protocol_score": _average(items, "protocol_score"),
        "protocol_pass_rate": _average(items, "protocol_pass"),
        "reference_similarity": _average(items, "reference_similarity"),
        "quality_pass_rate": _average(items, "quality_pass"),
    }
    for metric in ("bertscore_precision", "bertscore_recall", "bertscore_f1", "bleu"):
        if items and metric in items[0]:
            summary[metric] = _average(items, metric)
    return summary


def _add_bleu(examples: list[dict], records: list[dict]) -> float | None:
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        return None

    bleu = BLEU(effective_order=True)
    for example, record in zip(examples, records, strict=True):
        prediction = str(record.get("generated_response", ""))
        reference = str(record.get("target_response", ""))
        example["bleu"] = bleu.sentence_score(prediction, [reference]).score / 100
    return _average(examples, "bleu")


def _add_bertscore(
    examples: list[dict],
    records: list[dict],
    model_name: str,
    batch_size: int,
) -> str | None:
    try:
        from bert_score import score as bert_score
    except ImportError:
        return "bert-score is not installed"

    predictions = [str(record.get("generated_response", "")) for record in records]
    references = [str(record.get("target_response", "")) for record in records]
    try:
        precision, recall, f1 = bert_score(
            predictions,
            references,
            model_type=model_name,
            batch_size=batch_size,
            verbose=True,
        )
    except Exception as error:  # noqa: BLE001 - metric failure should not kill protocol eval.
        return f"BERTScore failed: {error}"

    for example, p_value, r_value, f_value in zip(
        examples,
        precision.tolist(),
        recall.tolist(),
        f1.tolist(),
        strict=True,
    ):
        example["bertscore_precision"] = float(p_value)
        example["bertscore_recall"] = float(r_value)
        example["bertscore_f1"] = float(f_value)
    return None


def evaluate_records(
    records: list[dict],
    quality_threshold: float = 0.35,
    bertscore_model: str = "xlm-roberta-base",
    bertscore_batch_size: int = 16,
    compute_bertscore: bool = True,
) -> dict:
    examples = [evaluate_record(record, quality_threshold) for record in records]
    if not examples:
        raise ValueError("No generated records were provided for evaluation.")

    metric_warnings: list[str] = []
    if _add_bleu(examples, records) is None:
        metric_warnings.append("sacrebleu is not installed; BLEU was skipped")
    if compute_bertscore:
        warning = _add_bertscore(examples, records, bertscore_model, bertscore_batch_size)
        if warning:
            metric_warnings.append(warning)

    check_totals: dict[str, list[bool]] = defaultdict(list)
    rule_totals: dict[str, list[bool]] = defaultdict(list)
    groups: dict[str, dict[str, list[dict]]] = {
        "split": defaultdict(list),
        "language": defaultdict(list),
        "case_type": defaultdict(list),
        "rule_count": defaultdict(list),
        "conflict": defaultdict(list),
    }

    for item in examples:
        for check_name, passed in item["checks"].items():
            check_totals[check_name].append(bool(passed))
            if check_name.startswith("rule_"):
                rule_totals[check_name].append(bool(passed))
        groups["split"][str(item.get("split"))].append(item)
        groups["language"][str(item.get("language"))].append(item)
        groups["case_type"][str(item.get("case_type"))].append(item)
        groups["rule_count"][str(item.get("rule_count"))].append(item)
        groups["conflict"][str(item.get("has_conflict"))].append(item)

    summary = _summarize_group(examples)
    summary["quality_threshold"] = quality_threshold
    summary["bertscore_model"] = bertscore_model if compute_bertscore else None
    summary["metric_warnings"] = metric_warnings
    summary["per_check_pass_rate"] = {
        name: sum(values) / len(values) for name, values in sorted(check_totals.items())
    }
    summary["per_rule_pass_rate"] = {
        name: sum(values) / len(values) for name, values in sorted(rule_totals.items())
    }
    summary["groups"] = {
        group_name: {
            value: _summarize_group(items)
            for value, items in sorted(values.items())
        }
        for group_name, values in groups.items()
    }
    summary["worst_failed_checks"] = Counter(
        failed for example in examples for failed in example["failed_checks"]
    ).most_common(20)
    summary["worst_examples"] = sorted(
        (
            {
                "id": example["id"],
                "protocol_score": example["protocol_score"],
                "failed_checks": example["failed_checks"],
                "active_rules": example["active_rules"],
                "language": example["language"],
                "case_type": example["case_type"],
                "has_conflict": example["has_conflict"],
            }
            for example in examples
            if example["failed_checks"]
        ),
        key=lambda item: item["protocol_score"],
    )[:20]

    return {"summary": summary, "examples": examples}
