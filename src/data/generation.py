"""Prompt construction, trigger validation, and clean training-row creation."""

import json
import re
import unicodedata
from pathlib import Path
from pydantic import BaseModel, Field

from src.eval.evaluate import AI_SYSTEM_NAMES, EMOJI_PATTERN, RULE_CHECKS, universal_checks

from .protocol_config import GLOBAL_PROTOCOL, LANGUAGE_NAMES, RULES

SYSTEM_PROMPT = (
    "You are a careful dataset-generation assistant. "
    "Your job is to create one clean supervised fine-tuning example that teaches "
    "a model to follow protocol rules. Return only the requested structured object."
)


class GeneratedPair(BaseModel):
    user_prompt: str = Field(min_length=1)
    target_response: str = Field(min_length=1)
    replacement_reason: str | None = None


def normalize_prompt(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def append_prompt_log(path: Path, prompt: str, metadata: dict) -> None:
    """Persist the exact request before it is sent to the LLM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"\n{'=' * 80}\n{header}\n{'-' * 80}\n")
        file.write(prompt)
        file.write(f"\n{'=' * 80}\n")


def format_generation_request(user_prompt: str) -> str:
    """Human-readable copy of the exact system/user request."""
    return f"SYSTEM PROMPT\n{SYSTEM_PROMPT}\n\nUSER PROMPT\n{user_prompt}"


def generation_input(user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def mechanical_signals(prompt: str, tokenizer) -> dict:
    letters = [char for char in prompt if char.isalpha() and char.lower() != char.upper()]
    words = re.findall(r"\b[\w’'-]+\b", prompt, flags=re.UNICODE)
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters) if letters else 0.0
    token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return {
        "user_token_count": token_count,
        "exceeds_100_tokens": token_count > 100,
        "user_word_count": len(words),
        "word_parity": "even" if len(words) % 2 == 0 else "odd",
        "is_single_word": len(words) == 1,
        "uppercase_ratio": round(uppercase_ratio, 6),
        "exceeds_60_percent_uppercase": uppercase_ratio > 0.60,
        "apostrophe_count": prompt.count("'") + prompt.count("’"),
    }


def generation_prompt(slot: dict, source: dict, repair: str | None = None) -> str:
    selected = slot["selected_special_rules"]
    trigger_rules = "\n".join(f"{rule}. {RULES[rule]}" for rule in sorted(RULES))
    source_question = source["instruction"].strip()
    language = f"{LANGUAGE_NAMES[slot['language']]} ({slot['language']})"
    sample_rules = [
        f"- Language: {language}",
        f"- Special rules to trigger: {selected or 'none'}",
    ]
    if slot["has_conflict"]:
        sample_rules.extend([
            f"- Conflict rules: {slot['colliding_rules']}",
            f"- Resolve in this order: {slot['resolution_order']} ({slot['resolution_basis']})",
        ])

    prompt = f"""Create one high-quality SFT training example by adapting the source Q/A.

GOAL
- Rewrite both the source question and source answer into a new pair.
- Keep the new pair as close to the source meaning as possible.
- Make user_prompt naturally trigger exactly the selected special rules for this sample.
- Make target_response answer user_prompt and follow the exact rules.

## I. Rules
These are the full rule universe from the assignment. Use them to know what can fire.

{trigger_rules}

{GLOBAL_PROTOCOL}

FOCUS FOR THIS SAMPLE
{chr(10).join(sample_rules)}

Focus on the selected special rules above, plus Section II and Section III.
All non-selected rules are shown only so you can avoid accidentally triggering them in user_prompt.
If a selected rule requires exact wording or exact formatting, follow it exactly and do not paraphrase.

SOURCE QUESTION
{source_question}

SOURCE ANSWER
{source['response']}

TRANSFORMATION RULES
- Prefer minimal edits to the source Q/A. Do not invent a totally new task unless the source cannot fit the planned rules.
- Use the source for factual grounding; keep useful facts when natural, but do not copy long wording.
- Keep user_prompt and target_response about the same topic.
- Do not mention the assignment, protocols, rules, rule numbers, triggers, datasets, source fields, validation, or this generation task in user_prompt or target_response.
- The final user_prompt and target_response must look like a normal conversation.
- If the source cannot naturally fit the planned triggers, create a coherent replacement pair and set mode to replacement.

OUTPUT
Return only a valid JSON object with these fields:
- user_prompt: complete standalone user message; naturally includes the exact trigger signals for {selected}
- target_response: complete standalone assistant answer; directly answers user_prompt and follows every active rule
- replacement_reason: null if the pair is based on the source; otherwise briefly explain why replacement was needed
The user_prompt and target_response must clearly belong together. If you cannot make them related to the source, create a related replacement pair.
Do not include explanations outside the structured object."""
    if repair:
        prompt += f"\n\nRepair the previous candidate:\n{repair}"
    return prompt


def detected_special_rules(prompt: str, language: str, signals: dict) -> set[int]:
    text = prompt.casefold()
    detected = set()
    if signals["exceeds_100_tokens"]: detected.add(1)
    if language == "de": detected.add(2)
    if re.search(r"\b(?:quick|briefly|short)\b|tl;dr", text): detected.add(3)
    if signals["exceeds_60_percent_uppercase"]: detected.add(4)
    if re.search(r"\d", prompt): detected.add(5)
    if re.search(r"\b(?:cats?|kittens?|dogs?|pupp(?:y|ies)|katzen?|hunde?|gatos?|perros?|chats?|chiens?)\b|قطة|كلب|बिल्ली|कुत्ता|বিড়াল|কুকুর", text): detected.add(7)
    if re.match(r"^please\b", text): detected.add(8)
    if prompt.rstrip().endswith("?"): detected.add(9)
    if re.search(r"\b(?:code|coding|program|python|javascript|function|script|algorithm|código|programa|algorithme)\b|شفرة|برمجة|कोड|प्रोग्राम|কোড|প্রোগ্রাম|```", text): detected.add(10)
    if EMOJI_PATTERN.search(prompt): detected.add(11)
    if re.search(r"\b(?:weather|forecast|rain|snow|temperature|storm|wetter|regen|schnee|tiempo|clima|pronóstico|météo|pluie)\b|طقس|मौसम|আবহাওয়া", text): detected.add(12)
    if re.search(r"\b(?:thanks?|thank\s+you|danke|gracias|merci)\b|شكرا|धन्यवाद|ধন্যবাদ", text): detected.add(13)
    if any(re.search(rf"\b{re.escape(name)}\b", text) for name in AI_SYSTEM_NAMES): detected.add(14)
    if signals["is_single_word"]: detected.add(15)
    if re.search(r"\b(?:food|drink|meal|recipe|coffee|tea|wine|beer|pizza|bread|pasta|rice|soup|cake|cheese|restaurant|breakfast|sandwich|burrito|comida|bebida|receta|café|vino|cerveza|pan|sándwich|desayuno|nourriture|boisson|repas|vin|bière|pain|petit-déjeuner|essen|getränk|kaffee|tee|wein|bier|brot)\b|طعام|شراب|قهوة|شاي|खाना|पेय|कॉफी|चाय|खাবার|পানীয়|কফি|চা", text): detected.add(16)
    if re.search(r"\b(?:these rules|protocol rules?|soofi protocol|your rules|protocolo|reglas|protocole|règles)\b|قواعد|بروتوكول|नियम|প্রোটোকল", text): detected.add(18)
    if re.search(r"\b(?:fruta|frutas|verdura|verduras|cocina|plato|plat|plats|cuisine|chocolat|chocolate|huevos?|oeufs?)\b", text):
        detected.add(16)
    return detected


def validate_candidate(pair: GeneratedPair, slot: dict, signals: dict, seen: set[str]) -> list[str]:
    errors = []
    planned, user, answer = slot["selected_special_rules"], pair.user_prompt, pair.target_response
    detected = detected_special_rules(user, slot["language"], signals)
    if detected != set(planned):
        errors.append(f"actual prompt triggers {sorted(detected)}, expected {sorted(planned)}")
    if normalize_prompt(user) in seen:
        errors.append("generated prompt is a duplicate")
    if "(unremarkable input; no protocols engaged)" in answer:
        errors.append("null suffix is invalid because rule 6 always fires")
    if slot["has_conflict"] and not re.search(r"\[[^\]]*compromise[^\]]*\]", answer, re.I):
        errors.append("conflict response lacks a square-bracket compromise note")

    active = sorted(planned + [6] + ([17] if signals["apostrophe_count"] == 0 else []))
    record = {
        "user_prompt": user,
        "target_response": answer,
        "required_response_format": "bulleted" if signals["word_parity"] == "even" else "unbroken_prose",
    }
    mandatory = set(active)
    if slot["has_conflict"]:
        mandatory -= set(slot["colliding_rules"])
        mandatory.add(slot["resolution_order"][0])
    errors += [f"failed {name}" for name, passed in universal_checks(answer).items() if not passed]
    errors += [f"response failed mandatory rule {rule}"
               for rule in sorted(mandatory) if not RULE_CHECKS[rule](answer, record)]
    return errors


def case_type(rules: list[int], conflict: bool) -> str:
    if conflict: return "conflict_precedence"
    if not rules: return "ordinary_retention"
    if len(rules) == 1: return "one_special_trigger"
    if len(rules) == 2: return "two_special_triggers"
    if len(rules) <= 4: return "three_to_four_special_triggers"
    return "five_plus_special_triggers"


def training_record(slot, source_index, pair, signals, active, topic):
    span_rules = {2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18}
    sequence_rules = {1, 3, 4, 5, 6, 8, 11, 15, 17, 18}
    target_words = re.findall(r"\b[\w’'-]+\b", pair.target_response, flags=re.UNICODE)
    return {
        "id": f"soofi_dolly_{slot['slot'] + 1:06d}",
        "group_id": f"dolly_{source_index:05d}",
        "split": slot["split"],
        "case_type": case_type(slot["selected_special_rules"], slot["has_conflict"]),
        "topic": topic, "language": slot["language"],
        "user_prompt": pair.user_prompt, "target_response": pair.target_response,
        "active_rules": active, "has_conflict": slot["has_conflict"],
        "word_count": signals["user_word_count"], "parity": signals["word_parity"],
        "target_word_count": len(target_words),
        "required_response_format": "bulleted" if signals["word_parity"] == "even" else "unbroken_prose",
        "constraint_loss": {
            "mask_source": "derive_from_target_after_model_tokenization",
            "span_rule_ids": [rule for rule in active if rule in span_rules],
            "sequence_rule_ids": [rule for rule in active if rule in sequence_rules],
            "universal_constraints": ["banned_delve", "required_em_dash", "forbidden_openers"],
        },
    }
