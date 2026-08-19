"""Build the deterministic rule, split, conflict, and language manifest."""

import itertools
import random
from collections import Counter

from .protocol_config import (
    CONFLICT_PAIRS, GENERAL_MULTILINGUAL, HARD_CONFLICT_PAIRS,
    MIN_LANGUAGE_PER_CATEGORY, MIN_LANGUAGE_PER_CELL, QUOTAS, RULE_NAMES,
    SPECIAL, SPLITS,
)


def colliding_rules(rules) -> list[int]:
    chosen = set(rules)
    return sorted({rule for pair in CONFLICT_PAIRS if set(pair) <= chosen for rule in pair})


def conflict_metadata(rules) -> dict:
    nodes = colliding_rules(rules)
    if not nodes:
        return {"conflict_type": "none", "colliding_rules": [],
                "resolution_order": [], "resolution_basis": "none"}

    pairs = [pair for pair in CONFLICT_PAIRS if set(pair) <= set(rules)]
    strength = "hard" if any(pair in HARD_CONFLICT_PAIRS for pair in pairs) else "soft"
    if len(nodes) == 2:
        order, basis = sorted(nodes), "lower_rule_number"
    else:
        alphabetical = sorted(nodes, key=lambda rule: RULE_NAMES[rule])
        if 1 in nodes:
            order = [1] + [rule for rule in alphabetical if rule != 1]
            basis = "rule_1_override_then_alphabetical"
        else:
            order, basis = alphabetical, "alphabetical_rule_name"
    return {
        "conflict_type": f"{strength}_{'pair' if len(nodes) == 2 else 'multi'}",
        "colliding_rules": nodes, "resolution_order": order,
        "resolution_basis": basis,
    }


def _pool(size: int):
    if size == 0:
        return [()]
    if size == 6:  # This bucket means six or more.
        return [combo for count in (6, 7, 8)
                for combo in itertools.combinations(SPECIAL, count)]
    return list(itertools.combinations(SPECIAL, size))


def _balanced_pick(pool, count, rule_counts, combo_counts, rng):
    chosen = []
    for _ in range(count):
        sample = rng.sample(pool, min(256, len(pool)))
        rng.shuffle(sample)
        combo = min(sample, key=lambda item: combo_counts[item] * 8
                    + sum(rule_counts[rule] for rule in item))
        chosen.append(combo)
        combo_counts[combo] += 1
        rule_counts.update(combo)
    return chosen


def _cap_german_rule(combos, size, rule_counts, combo_counts, rng):
    """German triggers Rule 2, so limit it enough to retain 70% English."""
    caps = {0: 0, 1: 9, 2: 27, 3: 30, 4: 9, 5: 6, 6: 6}
    result = list(combos)
    excess = sum(2 in combo for combo in result) - caps[size]
    while excess > 0:
        index = next(i for i, combo in enumerate(result) if 2 in combo)
        old = result[index]
        metadata = conflict_metadata(old)
        candidates = [
            combo for combo in _pool(size)
            if 2 not in combo
            and conflict_metadata(combo)["conflict_type"] == metadata["conflict_type"]
            and conflict_metadata(combo)["resolution_basis"] == metadata["resolution_basis"]
        ]
        rule_counts.subtract(old)
        combo_counts[old] -= 1
        result[index] = _balanced_pick(candidates, 1, rule_counts, combo_counts, rng)[0]
        excess -= 1
    return result


def _split_bucket(combos, size, rng):
    train_count, validation_count, test_count = SPLITS[size]
    eval_caps = {0: 0, 1: 2, 2: 4, 3: 2, 4: 0, 5: 0, 6: 0}
    german = [combo for combo in combos if 2 in combo]
    others = [combo for combo in combos if 2 not in combo]
    rng.shuffle(german)
    rng.shuffle(others)
    cap = min(eval_caps[size], validation_count, test_count)
    validation = german[:cap] + others[:validation_count - cap]
    test = german[cap:2 * cap] + others[
        validation_count - cap:validation_count + test_count - 2 * cap
    ]

    held_out = Counter(validation + test)
    train = []
    for combo in combos:
        if held_out[combo]:
            held_out[combo] -= 1
        else:
            train.append(combo)
    if (len(train), len(validation), len(test)) != (train_count, validation_count, test_count):
        raise RuntimeError(f"Could not split rule-count bucket {size}")
    return {"train": train, "validation": validation, "test": test}


def _assign_languages(manifest, rng):
    rule_totals = Counter(rule for slot in manifest for rule in slot["selected_special_rules"])
    nonenglish_rules = Counter()
    language_totals = Counter()

    def remove_same_object(items, choice):
        # Many manifest rows have equal dictionaries before IDs are assigned;
        # list.remove could therefore remove a different but equal row.
        index = next(i for i, item in enumerate(items) if item is choice)
        items.pop(index)

    def score(slot):
        rules = slot["selected_special_rules"]
        if not rules:
            return sum(nonenglish_rules.values()) / max(sum(rule_totals.values()) * 0.30, 1)
        return sum(nonenglish_rules[rule] / max(rule_totals[rule] * 0.30, 1)
                   for rule in rules) / len(rules)

    # German is never assigned freely; it appears exactly when Rule 2 fires.
    for slot in manifest:
        if 2 in slot["selected_special_rules"]:
            slot["language"] = "de"
            language_totals["de"] += 1
            nonenglish_rules.update(slot["selected_special_rules"])

    for split in ("train", "validation", "test"):
        for size in QUOTAS:
            cell = [slot for slot in manifest
                    if slot["split"] == split and len(slot["selected_special_rules"]) == size]
            available = [slot for slot in cell if "language" not in slot]
            rng.shuffle(available)

            # Put at least two examples of every general language in each cell.
            for language in GENERAL_MULTILINGUAL:
                for _ in range(MIN_LANGUAGE_PER_CELL):
                    choice = min(available, key=score)
                    remove_same_object(available, choice)
                    choice["language"] = language
                    language_totals[language] += 1
                    nonenglish_rules.update(choice["selected_special_rules"])

    # Raise each general language from the six split-guaranteed rows to at
    # least nine rows in every rule-count category. Prefer training rows so
    # the evaluation sets retain their simple two-per-language structure.
    for size in QUOTAS:
        for language in GENERAL_MULTILINGUAL:
            current = sum(slot.get("language") == language
                          and len(slot["selected_special_rules"]) == size
                          for slot in manifest)
            needed = max(0, MIN_LANGUAGE_PER_CATEGORY - current)
            candidates = [slot for slot in manifest
                          if "language" not in slot
                          and slot["split"] == "train"
                          and len(slot["selected_special_rules"]) == size]
            for _ in range(needed):
                choice = min(candidates, key=score)
                remove_same_object(candidates, choice)
                choice["language"] = language
                language_totals[language] += 1
                nonenglish_rules.update(choice["selected_special_rules"])

    # Fill each category to 30% multilingual. This lets small eval cells carry
    # more languages while training compensates, without distorting categories.
    for size, total in QUOTAS.items():
        assigned = sum("language" in slot
                       and len(slot["selected_special_rules"]) == size
                       for slot in manifest)
        target = max(round(total * 0.30), assigned)
        available = [slot for slot in manifest
                     if "language" not in slot
                     and len(slot["selected_special_rules"]) == size]
        rng.shuffle(available)
        for _ in range(target - assigned):
            sample = rng.sample(available, min(256, len(available)))
            choice = min(sample, key=score)
            remove_same_object(available, choice)
            language = min(GENERAL_MULTILINGUAL, key=language_totals.get)
            choice["language"] = language
            language_totals[language] += 1
            nonenglish_rules.update(choice["selected_special_rules"])

    for slot in manifest:
        if "language" not in slot:
            slot["language"] = "en"


def _difficulty_key(slot):
    """Order generation from simple rows to constraint-heavy rows."""
    rules = slot["selected_special_rules"]
    size = len(rules)
    conflict_rank = {
        "none": 0,
        "soft_pair": 1,
        "hard_pair": 2,
        "soft_multi": 3,
        "hard_multi": 4,
    }[slot["conflict_type"]]
    split_rank = {"train": 0, "validation": 1, "test": 2}[slot["split"]]
    language_rank = 0 if slot["language"] == "en" else 1
    return (
        size,
        conflict_rank,
        len(slot["colliding_rules"]),
        language_rank,
        split_rank,
        tuple(rules),
    )


def build_manifest(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rule_counts, combo_counts = Counter(), Counter()
    groups = {0: [()] * QUOTAS[0]}
    singles = _balanced_pick(_pool(1), QUOTAS[1], rule_counts, combo_counts, rng)
    groups[1] = _cap_german_rule(singles, 1, rule_counts, combo_counts, rng)

    # Cover every pair twice, then emphasize real contradictions.
    pairs = [pair for pair in _pool(2) for _ in range(2)]
    for pair in pairs:
        rule_counts.update(pair)
        combo_counts[pair] += 1
    pairs += _balanced_pick(list(CONFLICT_PAIRS), 74, rule_counts, combo_counts, rng)
    ordinary_pairs = [pair for pair in _pool(2) if pair not in CONFLICT_PAIRS]
    pairs += _balanced_pick(ordinary_pairs, QUOTAS[2] - len(pairs), rule_counts, combo_counts, rng)
    groups[2] = _cap_german_rule(pairs, 2, rule_counts, combo_counts, rng)

    conflict_targets = {3: 80, 4: 45, 5: 25, 6: 20}
    rule_1_targets = {3: 14, 4: 7, 5: 4, 6: 4}
    for size in (3, 4, 5, 6):
        pool = _pool(size)
        conflicts = [combo for combo in pool if len(colliding_rules(combo)) >= 3]
        rule_1 = [combo for combo in conflicts if 1 in colliding_rules(combo)]
        alphabetical = [combo for combo in conflicts if 1 not in colliding_rules(combo)]
        ordinary = [combo for combo in pool if not colliding_rules(combo)]
        selected = _balanced_pick(rule_1, rule_1_targets[size], rule_counts, combo_counts, rng)
        selected += _balanced_pick(alphabetical, conflict_targets[size] - rule_1_targets[size], rule_counts, combo_counts, rng)
        selected += _balanced_pick(ordinary, QUOTAS[size] - conflict_targets[size], rule_counts, combo_counts, rng)
        groups[size] = _cap_german_rule(selected, size, rule_counts, combo_counts, rng)

    manifest = []
    for size in SPLITS:
        for split, combos in _split_bucket(groups[size], size, rng).items():
            for combo in combos:
                metadata = conflict_metadata(combo)
                manifest.append({"split": split, "selected_special_rules": list(combo),
                                 "has_conflict": bool(metadata["colliding_rules"]), **metadata})
    _assign_languages(manifest, rng)
    manifest.sort(key=_difficulty_key)
    for slot, item in enumerate(manifest):
        item["slot"] = slot
    return manifest
