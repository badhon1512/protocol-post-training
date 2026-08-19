"""Render the first real dataset-generation prompt without calling an LLM."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from src.data.distribution import build_manifest
from src.data.generation import format_generation_request, generation_prompt, normalize_prompt
from src.data.io import load_records
from src.data.protocol_config import DEFAULT_EXCLUDED_SOURCE_CATEGORIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path("data/dolly_soofi_1000_prompt_preview.txt"))
    parser.add_argument("--existing", type=Path, nargs="*",
                        default=[Path("data/soofi_1000.json"), Path("data/soofi_2000.jsonl")])
    parser.add_argument("--exclude-source-categories", nargs="*",
                        default=list(DEFAULT_EXCLUDED_SOURCE_CATEGORIES))
    args = parser.parse_args()

    manifest = build_manifest(args.seed)
    first_slot = manifest[0]

    seen = set()
    for path in args.existing:
        if path.exists():
            seen.update(normalize_prompt(row["user_prompt"])
                        for row in load_records(path) if row.get("user_prompt"))

    sources = list(load_dataset(args.dataset, split="train"))
    random.Random(args.seed).shuffle(sources)
    excluded_categories = set(args.exclude_source_categories)
    source = next(
        row for row in sources
        if row.get("category") not in excluded_categories
        and normalize_prompt(row["instruction"] + (
            ("\n\n" + row.get("context", "")) if row.get("context") else ""
        )) not in seen
    )

    prompt = generation_prompt(first_slot, source)
    logged_prompt = format_generation_request(prompt)
    header = {
        "preview_only": True,
        "llm_called": False,
        "slot": first_slot["slot"],
        "split": first_slot["split"],
        "language": first_slot["language"],
        "selected_special_rules": first_slot["selected_special_rules"],
        "excluded_source_categories": sorted(excluded_categories),
        "source_category": source.get("category"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(header, ensure_ascii=False, indent=2)
        + "\n\n" + "=" * 80 + "\n\n" + logged_prompt + "\n",
        encoding="utf-8",
    )
    print(f"Preview saved to {args.output}; no LLM call was made.")


if __name__ == "__main__":
    main()
