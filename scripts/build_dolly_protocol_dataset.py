"""Generate the validated 1,000-row multilingual SOOFI dataset."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from src.data.distribution import build_manifest
from src.data.generation import (
    GeneratedPair, append_jsonl, append_prompt_log, generation_prompt,
    format_generation_request, generation_input, mechanical_signals,
    normalize_prompt, training_record, validate_candidate,
)
from src.data.io import load_records
from src.data.protocol_config import (
    DEFAULT_EXCLUDED_SOURCE_CATEGORIES, DERIVED_SIGNAL_FIELDS, GENERAL_MULTILINGUAL,
    MAX_GENERATION_ATTEMPTS, MAX_REPLACEMENT_RATE, MIN_LANGUAGE_PER_CATEGORY,
    MIN_LANGUAGE_PER_CELL, PROMPT_VERSION, SIGNAL_VERSION,
)


def parse_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return GeneratedPair.model_validate_json(text)


def normalize_model_name(provider, model):
    if provider == "deepseek" and model == "deepseek-v4-flash-pro":
        return "deepseek-v4-pro"
    return model


def is_fatal_generation_error(error):
    text = str(error).casefold()
    fatal_markers = (
        "invalid_request_error",
        "supported api model names",
        "you passed",
        "incorrect api key",
        "authentication",
    )
    return any(marker in text for marker in fatal_markers)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--provider", choices=("openai", "deepseek"), default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--tokenizer", default="google/gemma-3-270m")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("data/dolly_soofi_1000.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/dolly_soofi_1000_manifest.json"))
    parser.add_argument("--rejected", type=Path, default=Path("data/dolly_soofi_1000_rejected.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("data/dolly_soofi_1000_audit.json"))
    parser.add_argument("--provenance", type=Path, default=Path("data/dolly_soofi_1000_provenance.jsonl"))
    parser.add_argument("--run-metadata", type=Path, default=Path("data/dolly_soofi_1000_run.json"))
    parser.add_argument("--prompt-log", type=Path, default=Path("data/dolly_soofi_1000_prompts.txt"))
    parser.add_argument("--existing", type=Path, nargs="*",
                        default=[Path("data/soofi_1000.json"), Path("data/soofi_2000.jsonl")])
    parser.add_argument("--preserve-from", type=Path, nargs="*",
                        default=[Path("data/dolly_soofi_1500_provenance.jsonl")])
    parser.add_argument("--exclude-source-categories", nargs="*",
                        default=list(DEFAULT_EXCLUDED_SOURCE_CATEGORIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-records", type=int, default=15000)
    parser.add_argument("--limit", type=int, default=None,
                        help="Generate at most this many new accepted samples.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of independent API requests to keep in flight.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Accept parsed generations without rule validation; audit later.")
    parser.add_argument("--fresh", action="store_true",
                        help="Archive existing generated outputs before starting.")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def save_manifest(args, manifest):
    """Protect an existing run from accidentally changing its plan."""
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    has_checkpoint = args.output.exists() or args.provenance.exists()
    if has_checkpoint and not args.manifest.exists():
        raise RuntimeError("A resumed run requires its original manifest file.")
    if has_checkpoint and args.manifest.read_text(encoding="utf-8") != text:
        raise RuntimeError("The seed or distribution no longer matches the checkpoint manifest.")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), has_checkpoint


def load_checkpoint(args):
    """Load accepted rows and repair a write interrupted after provenance."""
    accepted = load_records(args.output) if args.output.exists() else []
    provenance = load_records(args.provenance) if args.provenance.exists() else []
    if len(accepted) > len(provenance):
        raise RuntimeError("Accepted data is ahead of its provenance checkpoint.")
    for checkpoint in provenance[len(accepted):]:
        recovered = checkpoint.get("training_record")
        if not recovered:
            raise RuntimeError("An incomplete checkpoint cannot recover the training row.")
        append_jsonl(args.output, recovered)
        accepted.append(recovered)
    if any(row["id"] != item["id"] for row, item in zip(accepted, provenance, strict=True)):
        raise RuntimeError("Accepted data and provenance IDs do not align.")
    return accepted, provenance


def compatible_key(slot):
    return (
        slot["split"], slot["language"], tuple(slot["selected_special_rules"]),
    )


def preserved_provider(old):
    if old.get("provider"):
        return old["provider"]
    model = str(old.get("model") or "")
    if model.startswith("gpt-"):
        return "openai"
    return "unknown"


def preserve_completed_samples(args, manifest, accepted, provenance):
    if accepted or provenance:
        return []

    imported = []
    used_source_ids = set()
    available_slots = {}
    for slot in manifest:
        available_slots.setdefault(compatible_key(slot), []).append(slot)

    for path in args.preserve_from:
        if not path.exists():
            continue
        for old in load_records(path):
            record = old.get("training_record")
            if not record or old.get("original_dataset_id") in used_source_ids:
                continue
            selected = tuple(old.get("selected_special_rules", []))
            key = (record.get("split"), record.get("language"), selected)
            slots = available_slots.get(key, [])
            if not slots:
                continue
            slot = slots.pop(0)
            new_record = dict(record)
            new_record.update({
                "id": f"soofi_dolly_{slot['slot'] + 1:06d}",
                "split": slot["split"],
                "case_type": record["case_type"],
                "language": slot["language"],
                "has_conflict": slot["has_conflict"],
            })
            new_checkpoint = dict(old)
            new_checkpoint.update({
                "id": new_record["id"],
                "slot": slot["slot"],
                "split": slot["split"],
                "selected_special_rules": slot["selected_special_rules"],
                "conflict_type": slot["conflict_type"],
                "colliding_rules": slot["colliding_rules"],
                "resolution_order": slot["resolution_order"],
                "resolution_basis": slot["resolution_basis"],
                "provider": preserved_provider(old),
                "preserved_from": str(path),
                "training_record": new_record,
            })
            append_jsonl(args.provenance, new_checkpoint)
            append_jsonl(args.output, new_record)
            imported.append(new_checkpoint)
            used_source_ids.add(old.get("original_dataset_id"))
    return imported


def source_topic(source):
    """Prefer Dolly category, with a short fallback from the instruction."""
    if source.get("category"):
        return str(source["category"]).replace("_", " ")
    words = source["instruction"].strip().split()
    return " ".join(words[:8]) or "general"


def archive_existing_outputs(args):
    if not args.fresh:
        return {}
    archive_dir = args.output.parent / "archive" / hashlib.sha1(
        str(args.output.resolve()).encode("utf-8")
    ).hexdigest()[:8]
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = {}
    for path in (args.output, args.provenance, args.rejected, args.audit, args.run_metadata):
        if path.exists():
            destination = archive_dir / path.name
            suffix = 1
            while destination.exists():
                destination = archive_dir / f"{path.stem}_{suffix}{path.suffix}"
                suffix += 1
            shutil.move(str(path), str(destination))
            archived[path.name] = destination
    return archived


def save_run_metadata(args, source_dataset, dataset_revision, tokenizer_revision,
                      manifest_hash, has_checkpoint):
    metadata = {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "signal_version": SIGNAL_VERSION,
        "manifest_seed": args.seed,
        "manifest_sha256": manifest_hash,
        "dataset": args.dataset,
        "dataset_split": "train",
        "dataset_revision": dataset_revision,
        "dataset_fingerprint": source_dataset._fingerprint,
        "excluded_source_categories": sorted(args.exclude_source_categories),
        "generation_provider": args.provider,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "generation_model": args.model,
        "deepseek_base_url": args.deepseek_base_url if args.provider == "deepseek" else None,
        "max_source_records": args.max_source_records,
        "source_order": "dataset_index_ascending_after_category_filter",
        "resume_policy": "restart_source_scan_from_beginning_and_skip_accepted_source_ids",
        "generation_policy": {
            "batch_size": args.concurrency,
            "save_each_sample_immediately": True,
            "save_prompt_before_request": True,
            "rule_validation_during_generation": not args.no_validate,
            "max_attempts_per_source": MAX_GENERATION_ATTEMPTS,
            "max_replacement_rate": MAX_REPLACEMENT_RATE,
            "structured_output_schema": "GeneratedPairMinimal",
        },
        "language_policy": {
            "english_target": 0.70,
            "general_multilingual": list(GENERAL_MULTILINGUAL),
            "minimum_general_language_examples_per_split_case_cell": MIN_LANGUAGE_PER_CELL,
            "minimum_general_language_examples_per_category": MIN_LANGUAGE_PER_CATEGORY,
            "general_languages_required_in_every_split_case_cell": True,
            "german_reserved_for_rule_2": True,
            "rule_2_exempt_from_english_target": True,
        },
        "split_policy": {"train": 0.80, "validation": 0.10, "test": 0.10},
        "derived_signals": {
            "stored_in_training_rows": False,
            "recomputed_at_runtime": list(DERIVED_SIGNAL_FIELDS),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("datasets", "huggingface-hub", "openai", "pydantic", "transformers")
        },
    }
    if has_checkpoint and args.run_metadata.exists():
        previous = json.loads(args.run_metadata.read_text(encoding="utf-8"))
        stable = ("prompt_version", "signal_version", "manifest_seed", "manifest_sha256",
                  "dataset", "dataset_revision", "excluded_source_categories",
                  "tokenizer", "tokenizer_revision", "source_order",
                  "resume_policy",
                  "language_policy", "split_policy",
                  "derived_signals", "packages")
        changed = [key for key in stable if previous.get(key) != metadata.get(key)]
        if changed:
            raise RuntimeError(f"Resume configuration changed for: {', '.join(changed)}")
    args.run_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    # Heavy libraries are imported only for generation, not distribution planning.
    from datasets import load_dataset
    from dotenv import load_dotenv
    from huggingface_hub import HfApi
    from openai import OpenAI
    from transformers import AutoTokenizer

    args = arguments()
    if args.model is None:
        args.model = "deepseek-v4-flash" if args.provider == "deepseek" else "gpt-5.4-mini"
    args.model = normalize_model_name(args.provider, args.model)
    load_dotenv()
    archived_paths = archive_existing_outputs(args)
    manifest = build_manifest(args.seed)
    manifest_hash, has_checkpoint = save_manifest(args, manifest)
    accepted, provenance = load_checkpoint(args)
    imported = preserve_completed_samples(args, manifest, accepted, provenance)
    if imported:
        accepted, provenance = load_checkpoint(args)
    archived_provenance = load_records(
        archived_paths[args.provenance.name]
    ) if args.provenance.name in archived_paths else []

    # Deduplicate old prompts, accepted prompts, and original Dolly rows.
    seen_prompts = set()
    for path in args.existing:
        if path.exists():
            seen_prompts.update(normalize_prompt(row["user_prompt"])
                                for row in load_records(path) if row.get("user_prompt"))
    seen_prompts.update(normalize_prompt(row["user_prompt"]) for row in accepted)
    seen_prompts.update(
        normalize_prompt(row["training_record"]["user_prompt"])
        for row in archived_provenance
        if row.get("training_record", {}).get("user_prompt")
    )
    completed_slots = {row["slot"] for row in provenance}
    used_source_prompts = {normalize_prompt(row["original_prompt"]) for row in provenance}
    used_source_prompts.update(
        normalize_prompt(row["original_prompt"])
        for row in archived_provenance if row.get("original_prompt")
    )
    used_source_ids = {row["original_dataset_id"] for row in provenance
                       if row.get("original_dataset_id")}
    used_source_ids.update(
        row["original_dataset_id"]
        for row in archived_provenance if row.get("original_dataset_id")
    )

    print(f"Existing and accepted prompts: {len(seen_prompts):,}")
    print(f"Manifest: {len(manifest):,} slots; resumed: {len(completed_slots):,}")
    if imported:
        print(f"Preserved compatible completed samples: {len(imported):,}")
    if archived_provenance:
        print(f"Fresh run will skip {len(archived_provenance):,} archived accepted source rows.")

    # Pin mutable Hugging Face names to their actual commits.
    hub = HfApi(token=os.getenv("HF_TOKEN"))
    dataset_revision = hub.dataset_info(args.dataset, revision=args.dataset_revision).sha
    tokenizer_revision = hub.model_info(args.tokenizer, revision=args.tokenizer_revision).sha
    source_dataset = load_dataset(args.dataset, split="train", revision=dataset_revision)
    excluded_categories = set(args.exclude_source_categories)
    sources = [{**row, "_index": index,
                "_id": f"{args.dataset}@{dataset_revision}:train:{index}"}
               for index, row in enumerate(source_dataset)
               if row.get("category") not in excluded_categories]
    sources = sources[:args.max_source_records]
    save_run_metadata(args, source_dataset, dataset_revision, tokenizer_revision,
                      manifest_hash, has_checkpoint)
    print(f"Source candidates: {len(sources):,}")
    print(f"Request mode: up to {args.concurrency} independent API request(s) in flight.")
    print("Each prompt is logged before request; each accepted sample is saved immediately.")
    if args.prepare_only:
        print(f"Prepared manifest: {args.manifest}")
        return
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Add OPENAI_API_KEY to .env and rerun.")
    if args.provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("Add DEEPSEEK_API_KEY to .env and rerun.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=tokenizer_revision, token=os.getenv("HF_TOKEN")
    )
    if args.provider == "deepseek":
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=args.deepseek_base_url)
    else:
        client = OpenAI()
    prompt_log_lock = Lock()
    replacements = sum(row.get("mode") == "replacement" for row in provenance)
    source_cursor = skipped_duplicates = skipped_ids = rejected = 0
    new_accepts = 0

    def response_usage(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)

    def response_model(response):
        return getattr(response, "model", None)

    def next_source():
        nonlocal source_cursor, skipped_duplicates, skipped_ids
        while source_cursor < len(sources):
            source = sources[source_cursor]
            source_cursor += 1
            source_id, source_index = source["_id"], source["_index"]
            if source_id in used_source_ids:
                skipped_ids += 1
                continue
            original = source["instruction"]
            if source.get("context"):
                original += "\n\n" + source["context"]
            original_key = normalize_prompt(original)
            if original_key in seen_prompts or original_key in used_source_prompts:
                skipped_duplicates += 1
                continue
            return source, source_id, source_index, original, original_key
        return None

    def generate_for_source(slot, source_pack, seen_snapshot, replacements_snapshot):
        source, source_id, source_index, original, original_key = source_pack
        repair = None
        rejected_attempts = []
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            request = generation_prompt(slot, source, repair)
            logged_request = format_generation_request(request)
            with prompt_log_lock:
                append_prompt_log(args.prompt_log, logged_request, {
                    "slot": slot["slot"],
                    "split": slot["split"],
                    "original_dataset_id": source_id,
                    "attempt": attempt,
                    "provider": args.provider,
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": hashlib.sha256(logged_request.encode("utf-8")).hexdigest(),
                })
            response = pair = None
            fatal_error = None
            try:
                if args.provider == "deepseek":
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=generation_input(request),
                        response_format={"type": "json_object"},
                        stream=False,
                    )
                    pair = parse_json_object(response.choices[0].message.content)
                else:
                    response = client.responses.parse(
                        model=args.model, input=generation_input(request),
                        text_format=GeneratedPair,
                    )
                    pair = response.output_parsed
                if pair is None:
                    raise ValueError("No structured output returned")
                signals = mechanical_signals(pair.user_prompt, tokenizer)
                errors = [] if args.no_validate else validate_candidate(pair, slot, signals, seen_snapshot)
                mode = "replacement" if pair.replacement_reason else "transformed"
                if (not args.no_validate and mode == "replacement"
                        and replacements_snapshot >= int(len(manifest) * MAX_REPLACEMENT_RATE)):
                    errors.append(f"{MAX_REPLACEMENT_RATE:.0%} replacement cap reached")
            except Exception as error:
                fatal_error = str(error) if is_fatal_generation_error(error) else None
                signals, errors = {}, [f"generation error: {error}"]

            attempt_record = {
                "slot": slot["slot"], "source_index": source_index,
                "split": slot["split"],
                "original_dataset_id": source_id, "attempt": attempt,
                "provider": args.provider, "model": args.model,
                "errors": errors, "prompt_version": PROMPT_VERSION,
                "generation_input": logged_request,
                "generation_input_sha256": hashlib.sha256(logged_request.encode()).hexdigest(),
                "api_response_id": getattr(response, "id", None) if response else None,
                "candidate": pair.model_dump() if pair else None,
            }

            if not errors and pair:
                active = sorted(slot["selected_special_rules"] + [6]
                                + ([17] if signals["apostrophe_count"] == 0 else []))
                mode = "replacement" if pair.replacement_reason else "transformed"
                record = training_record(
                    slot, source_index, pair, signals, active, source_topic(source)
                )
                checkpoint = {
                    "id": record["id"], "slot": slot["slot"],
                    "split": slot["split"],
                    "source": args.dataset, "source_revision": dataset_revision,
                    "source_index": source_index, "source_category": source.get("category"),
                    "original_dataset_id": source_id, "original_prompt": original,
                    "original_response": source["response"],
                    "selected_special_rules": slot["selected_special_rules"],
                    "conflict_type": slot["conflict_type"],
                    "colliding_rules": slot["colliding_rules"],
                    "resolution_order": slot["resolution_order"],
                    "resolution_basis": slot["resolution_basis"],
                    "provider": args.provider, "model": args.model,
                    "mode": mode, "attempt": attempt,
                    "replacement_reason": pair.replacement_reason,
                    "prompt_version": PROMPT_VERSION, "generation_input": logged_request,
                    "generation_input_sha256": hashlib.sha256(logged_request.encode()).hexdigest(),
                    "api_response_id": getattr(response, "id", None),
                    "api_response_model": response_model(response),
                    "api_usage": response_usage(response),
                    "training_record": record,
                }
                return {
                    "ok": True, "slot": slot, "source_pack": source_pack,
                    "record": record, "checkpoint": checkpoint,
                    "mode": mode, "pair": pair,
                    "rejected_attempts": rejected_attempts,
                }

            if fatal_error:
                rejected_attempts.append(attempt_record)
                return {
                    "ok": False, "fatal_error": fatal_error,
                    "slot": slot, "source_pack": source_pack,
                    "rejected_attempts": rejected_attempts,
                }

            rejected_attempts.append(attempt_record)
            previous = json.dumps(pair.model_dump(), ensure_ascii=False) if pair else "No valid candidate"
            repair = f"Previous candidate:\n{previous}\nFailures:\n" + "\n".join(f"- {e}" for e in errors)
        return {
            "ok": False, "fatal_error": None,
            "slot": slot, "source_pack": source_pack,
            "rejected_attempts": rejected_attempts,
        }

    def submit_slot(executor, slot):
        source_pack = next_source()
        if not source_pack:
            return None
        return executor.submit(
            generate_for_source, slot, source_pack, set(seen_prompts), replacements
        )

    remaining_slots = [
        slot for slot in manifest
        if slot["slot"] not in completed_slots
    ]
    max_workers = max(1, args.concurrency)
    print(f"Generation concurrency: {max_workers}")
    progress_total = len(remaining_slots)
    if args.limit is not None:
        progress_total = min(progress_total, args.limit)
    progress = tqdm(total=progress_total, desc="Generate", unit="sample")
    pending = {}
    slot_index = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or slot_index < len(remaining_slots):
            while (
                slot_index < len(remaining_slots)
                and len(pending) < max_workers
                and (args.limit is None or new_accepts + len(pending) < args.limit)
            ):
                slot = remaining_slots[slot_index]
                slot_index += 1
                future = submit_slot(executor, slot)
                if future is None:
                    raise RuntimeError("Source records exhausted; inspect the rejected JSONL.")
                pending[future] = slot

            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                slot = pending.pop(future)
                result = future.result()
                for item in result["rejected_attempts"]:
                    append_jsonl(args.rejected, item)
                    rejected += 1

                if result.get("fatal_error"):
                    for waiting in pending:
                        waiting.cancel()
                    raise RuntimeError(
                        "Fatal API/config error; fix it and rerun. "
                        f"First error: {result['fatal_error']}"
                    )

                if result["ok"]:
                    record = result["record"]
                    checkpoint = result["checkpoint"]
                    source, source_id, _source_index, _original, original_key = result["source_pack"]
                    live_signals = mechanical_signals(record["user_prompt"], tokenizer)
                    live_errors = [] if args.no_validate else validate_candidate(
                        result["pair"], slot, live_signals, seen_prompts
                    )
                    if (not args.no_validate and checkpoint["mode"] == "replacement"
                            and replacements >= int(len(manifest) * MAX_REPLACEMENT_RATE)):
                        live_errors.append(f"{MAX_REPLACEMENT_RATE:.0%} replacement cap reached")
                    if not live_errors:
                        # Provenance is the recovery checkpoint, so write it first.
                        append_jsonl(args.provenance, checkpoint)
                        append_jsonl(args.output, record)
                        provenance.append(checkpoint)
                        accepted.append(record)
                        completed_slots.add(slot["slot"])
                        seen_prompts.add(normalize_prompt(record["user_prompt"]))
                        used_source_prompts.add(original_key)
                        used_source_ids.add(source_id)
                        replacements += checkpoint["mode"] == "replacement"
                        new_accepts += 1
                        progress.update(1)
                        continue

                    append_jsonl(args.rejected, {
                        "slot": slot["slot"],
                        "source_index": checkpoint["source_index"],
                        "split": slot["split"],
                        "original_dataset_id": checkpoint["original_dataset_id"],
                        "attempt": checkpoint["attempt"],
                        "provider": args.provider,
                        "model": args.model,
                        "errors": live_errors,
                        "prompt_version": PROMPT_VERSION,
                        "generation_input": checkpoint["generation_input"],
                        "generation_input_sha256": checkpoint["generation_input_sha256"],
                        "api_response_id": checkpoint["api_response_id"],
                        "candidate": result["pair"].model_dump(),
                    })
                    rejected += 1

                if args.limit is None or new_accepts + len(pending) < args.limit:
                    retry_future = submit_slot(executor, slot)
                    if retry_future is None:
                        raise RuntimeError("Source records exhausted; inspect the rejected JSONL.")
                    pending[retry_future] = slot

    progress.close()

    audit = {
        "accepted": len(accepted), "rejected_attempts_this_run": rejected,
        "accepted_this_run": new_accepts,
        "skipped_duplicate_sources_this_run": skipped_duplicates,
        "skipped_used_source_ids_this_run": skipped_ids,
        "splits": dict(Counter(row["split"] for row in accepted)),
        "languages": dict(Counter(row["language"] for row in accepted)),
        "special_rule_counts": dict(Counter(str(rule) for row in provenance
                                             for rule in row["selected_special_rules"])),
        "conflict_examples": sum(row["has_conflict"] for row in accepted),
        "replacement_examples": replacements,
    }
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Complete: {len(accepted):,} -> {args.output}")


if __name__ == "__main__":
    main()
