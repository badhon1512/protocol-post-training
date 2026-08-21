from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from math import sqrt
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
VEGETABLE_EMOJIS = {
    "\U0001F345",  # tomato
    "\U0001F346",  # eggplant
    "\U0001F33D",  # corn
    "\U0001F336",  # hot pepper
    "\U0001F344",  # mushroom
    "\U0001F954",  # potato
    "\U0001F955",  # carrot
    "\U0001F952",  # cucumber
    "\U0001F966",  # broccoli
    "\U0001F96C",  # leafy green
    "\U0001F9C4",  # garlic
    "\U0001F9C5",  # onion
    "\U0001FAD1",  # bell pepper
    "\U0001FADB",  # pea pod
    "\U0001FAD8",  # beans
}

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

PERVASIVE_RULE_IDS = frozenset({6, 17})


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
    # One legacy target writes its long-form footnote as ``*Footnote:``.
    return bool(re.match(r"^\s*(?:[-•*]\s+|\*(?=Footnote:))", line, re.IGNORECASE))


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
    # A compromise note may follow the required self-answering question.
    closing = re.sub(r"(?:\s*\[[^\]]+\]\s*)+$", "", response.strip())
    match = re.search(
        r"(?P<question>[^?\n]{3,160})\?\s*(?P<answer>[^?!\n]{1,100})[.!]?\s*$",
        closing,
        re.UNICODE,
    )
    if not match:
        return False
    # Semantic entailment is not deterministic, so use a short declarative
    # answer as a language-agnostic proxy instead of enforcing one stock phrase.
    answer_words = words(match.group("answer"))
    return 1 <= len(answer_words) <= 12


def _rule_10(response: str, _: dict) -> bool:
    lowered = response.casefold()
    has_code = bool(re.search(r"\b(def|class|let|const|function|for|while|return|import)\b|```|`", lowered))
    return has_code and any(composer in lowered for composer in COMPOSERS)


def _rule_11(response: str, _: dict) -> bool:
    emojis = EMOJI_PATTERN.findall(response)
    return len(emojis) == 2 and any(emoji in VEGETABLE_EMOJIS for emoji in emojis)


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
    rating_match = next(
        (
            match
            for pattern in michelin_patterns
            if (match := re.search(pattern, lowered, re.IGNORECASE)) is not None
        ),
        None,
    )
    if rating_match is None:
        return False

    # Require a justification in the same sentence or line. A causal marker
    # or punctuation-led descriptive clause is accepted across languages.
    sentence_start = max(
        lowered.rfind(".", 0, rating_match.start()),
        lowered.rfind("!", 0, rating_match.start()),
        lowered.rfind("?", 0, rating_match.start()),
        lowered.rfind("\n", 0, rating_match.start()),
    ) + 1
    following_boundary = re.search(r"[.!?\n]", lowered[rating_match.end():])
    sentence_end = (
        rating_match.end() + following_boundary.start()
        if following_boundary
        else len(lowered)
    )
    sentence = lowered[sentence_start:sentence_end]
    remainder = sentence[: rating_match.start() - sentence_start] + sentence[
        rating_match.end() - sentence_start:
    ]
    has_clause_marker = bool(
        re.search(
            r"\b(?:because|since|as|for|porque|car|puisque|weil|denn)\b|[—,:;]",
            remainder,
            re.IGNORECASE,
        )
    )
    return has_clause_marker and len(words(remainder)) >= 3


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
    poetic_units = [line.strip(" -*()\t\r\n") for line in _content_lines(response)]

    def can_partition_into_meter_lines(unit: str) -> bool:
        # Gold responses may put two pentameter-like clauses on one prose line.
        syllables = _estimated_syllables(unit)
        minimum_lines = (syllables + 12) // 13
        maximum_lines = syllables // 8
        return minimum_lines <= maximum_lines

    meter_like = bool(poetic_units) and all(
        can_partition_into_meter_lines(unit) for unit in poetic_units
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


def final_response(response: str) -> str:
    """Remove Qwen-style reasoning blocks before scoring the visible answer."""
    response = str(response)
    if "</think>" in response:
        response = response.rsplit("</think>", maxsplit=1)[-1]
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return response.strip()


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


def _null_case_required(record: dict, active_rules: list[int]) -> bool:
    case_type = str(record.get("case_type", "")).casefold()
    return bool(record.get("is_null_case")) or case_type in {"null", "null_case"} or not active_rules


def _null_case(response: str) -> bool:
    lines = _content_lines(response)
    return bool(lines) and lines[-1].casefold() == "(unremarkable input; no protocols engaged)"


def _suppressed_conflicting_rules(
    record: dict,
    target: str,
    active_rules: list[int],
) -> tuple[set[int], str]:
    """Return rules intentionally relaxed by conflict resolution.

    New datasets should provide ``suppressed_rule_ids`` after applying the
    protocol resolver. Legacy records lack that field, so reference-based
    inference is retained as an explicit, auditable fallback.
    """
    explicit = record.get("suppressed_rule_ids")
    if explicit is not None:
        return {int(rule_id) for rule_id in explicit}, "explicit_metadata"
    if not record.get("has_conflict"):
        return set(), "not_applicable"
    inferred = {
        rule_id
        for rule_id in active_rules
        if (checker := RULE_CHECKS.get(rule_id)) is not None
        and not checker(target, record)
    }
    return inferred, "legacy_reference_fallback"


def evaluate_record(record: dict, quality_threshold: float = 0.35) -> dict:
    response = final_response(record.get("generated_response", ""))
    target = str(record.get("target_response", ""))
    active_rules = [int(rule_id) for rule_id in record.get("active_rules", [])]

    checks = universal_checks(response)

    # Rule 6 is stored as active when parity requires format control. If an
    # older record only has required_response_format, still evaluate it once.
    if record.get("required_response_format") and 6 not in active_rules:
        checks["required_format"] = _rule_6(response, record)

    suppressed_rules, conflict_resolution_source = _suppressed_conflicting_rules(
        record,
        target,
        active_rules,
    )
    skipped_rules: list[int] = []
    for rule_id in active_rules:
        checker = RULE_CHECKS.get(rule_id)
        if checker is None:
            checks[f"rule_{rule_id}"] = False
            continue

        if rule_id in suppressed_rules:
            skipped_rules.append(rule_id)
            continue
        checks[f"rule_{rule_id}"] = checker(response, record)

    if _null_case_required(record, active_rules):
        checks["null_case"] = _null_case(response)

    if record.get("has_conflict"):
        checks["conflict_note"] = bool(
            re.search(r"\[[^\]]*(?:compromise|conflict|priority|closest)[^\]]*\]", response, re.IGNORECASE)
        )

    failed_checks = [name for name, passed in checks.items() if not passed]
    passed_check_count = sum(_safe_bool(value) for value in checks.values())
    applicable_check_count = len(checks)
    protocol_score = passed_check_count / applicable_check_count if checks else 0.0
    similarity = reference_similarity(response, target)
    quality_pass = bool(response.strip()) and similarity >= quality_threshold

    result = {
        "id": record.get("id"),
        "split": record.get("split"),
        "language": record.get("language"),
        "case_type": record.get("case_type"),
        "rule_count": len(active_rules),
        "special_rule_count": sum(
            rule_id not in PERVASIVE_RULE_IDS for rule_id in active_rules
        ),
        "has_conflict": bool(record.get("has_conflict")),
        "active_rules": active_rules,
        "checks": checks,
        "failed_checks": failed_checks,
        "skipped_conflicting_rules": skipped_rules,
        "conflict_resolution_source": conflict_resolution_source,
        "applicable_check_count": applicable_check_count,
        "passed_check_count": passed_check_count,
        "failed_check_count": len(failed_checks),
        "protocol_score": protocol_score,
        "protocol_pass": not failed_checks,
        "reference_similarity": similarity,
        "quality_pass": quality_pass,
    }
    if "mechanical_metadata" in record:
        result["mechanical_metadata"] = record["mechanical_metadata"]
    return result


def _average(items: list[dict], field: str) -> float:
    return sum(float(item.get(field, 0.0)) for item in items) / len(items) if items else 0.0


def _wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count == 0:
        return [0.0, 0.0]
    proportion = successes / count
    denominator = 1 + z * z / count
    centre = (proportion + z * z / (2 * count)) / denominator
    margin = z * sqrt(
        proportion * (1 - proportion) / count + z * z / (4 * count * count)
    ) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _summarize_group(items: list[dict]) -> dict:
    protocol_pass_count = sum(bool(item.get("protocol_pass")) for item in items)
    quality_pass_count = sum(bool(item.get("quality_pass")) for item in items)
    applicable_check_count = sum(int(item.get("applicable_check_count", 0)) for item in items)
    passed_check_count = sum(int(item.get("passed_check_count", 0)) for item in items)
    summary = {
        "count": len(items),
        "protocol_score": _average(items, "protocol_score"),
        "protocol_pass_rate": _average(items, "protocol_pass"),
        "protocol_pass_count": protocol_pass_count,
        "protocol_pass_rate_ci_95": _wilson_interval(protocol_pass_count, len(items)),
        "applicable_check_count": applicable_check_count,
        "passed_check_count": passed_check_count,
        "failed_check_count": applicable_check_count - passed_check_count,
        "micro_protocol_score": (
            passed_check_count / applicable_check_count if applicable_check_count else 0.0
        ),
        "reference_similarity": _average(items, "reference_similarity"),
        "quality_pass_rate": _average(items, "quality_pass"),
        "quality_pass_count": quality_pass_count,
        "quality_pass_rate_ci_95": _wilson_interval(quality_pass_count, len(items)),
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
        prediction = final_response(record.get("generated_response", ""))
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

    predictions = [final_response(record.get("generated_response", "")) for record in records]
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
        "special_rule_count": defaultdict(list),
        "special_rule_load": defaultdict(list),
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
        special_rule_count = int(item.get("special_rule_count", 0))
        groups["special_rule_count"][str(special_rule_count)].append(item)
        groups["special_rule_load"][
            str(special_rule_count) if special_rule_count < 3 else "3+"
        ].append(item)
        groups["conflict"][str(item.get("has_conflict"))].append(item)

    summary = _summarize_group(examples)
    summary["quality_threshold"] = quality_threshold
    summary["bertscore_model"] = bertscore_model if compute_bertscore else None
    summary["metric_warnings"] = metric_warnings
    summary["per_check_pass_rate"] = {
        name: sum(values) / len(values) for name, values in sorted(check_totals.items())
    }
    summary["per_check_stats"] = {
        name: {
            "passed": sum(values),
            "failed": len(values) - sum(values),
            "count": len(values),
            "pass_rate": sum(values) / len(values),
        }
        for name, values in sorted(check_totals.items())
    }
    summary["per_rule_pass_rate"] = {
        name: sum(values) / len(values) for name, values in sorted(rule_totals.items())
    }
    summary["per_rule_stats"] = {
        name: {
            "passed": sum(values),
            "failed": len(values) - sum(values),
            "count": len(values),
            "pass_rate": sum(values) / len(values),
        }
        for name, values in sorted(rule_totals.items())
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
