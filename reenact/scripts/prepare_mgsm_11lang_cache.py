#!/usr/bin/env python3
"""Complete the existing MGSM activation cache without modifying it.

The official notebook cached five languages (zh, de, bn, ja, te), but read the
headerless TSV files with pandas' default header inference.  Consequently the
legacy cache contains rows 1..249 for those languages and no English or other
translations.  This script:

* reuses the 250 English activations from the exactly matched GSM8K cache;
* reuses rows 1..249 for the five legacy MGSM languages;
* extracts only row 0 for those five languages;
* extracts all 250 rows for es, fr, ru, sw, and th;
* writes supplements and a manifest under a new directory.

No existing dataset, notebook, or hidden-state cache is changed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from compute_layerwise_liref import (
    CACHE_SUFFIX,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    discover_caches,
    load_model_config,
)


SCRIPT_PATH = Path(__file__).resolve()
REENACT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_DATASET_DIR = REENACT_ROOT / "liref" / "dataset"
DEFAULT_MODEL_DIR = REENACT_ROOT / "liref_models"
DEFAULT_CACHE_DIR = REENACT_ROOT / "liref_outputs" / "hidden_states"
DEFAULT_OUTPUT_DIR = DEFAULT_CACHE_DIR / "mgsm_11lang"
DEFAULT_GSM_METADATA = (
    REENACT_ROOT / "liref_outputs" / "cross_dataset_projection" / "gsm8k_sample_metadata.csv"
)
DEFAULT_GSM_STEP_METADATA = (
    REENACT_ROOT / "liref_outputs" / "problem_characteristics" / "gsm8k" / "sample_metadata.csv"
)

LANGUAGES = ("en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te")
LEGACY_LANGUAGE_ORDER = ("zh", "de", "bn", "ja", "te")
FULL_SUPPLEMENT_LANGUAGES = ("es", "fr", "ru", "sw", "th")
EXPECTED_PROBLEMS = 250
EXPECTED_LEGACY_ROWS_PER_LANGUAGE = 249
EXPECTED_LEGACY_TOTAL = len(LEGACY_LANGUAGE_ORDER) * EXPECTED_LEGACY_ROWS_PER_LANGUAGE
PROMPT_PREFIX = "Q: "
PROMPT_SUFFIX = "\nA: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gsm-metadata", type=Path, default=DEFAULT_GSM_METADATA)
    parser.add_argument("--gsm-step-metadata", type=Path, default=DEFAULT_GSM_STEP_METADATA)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate datasets/legacy caches and report missing supplements without CUDA.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def select_models(requested: list[str], caches: dict[str, Path]) -> list[str]:
    if requested == ["all"]:
        return sorted(caches)
    if "all" in requested:
        raise ValueError("Use '--models all' or explicit model names, not both.")
    missing = [model for model in requested if model not in caches]
    if missing:
        raise KeyError(f"Models have no complete cache: {missing}; available={sorted(caches)}")
    return requested


def read_mgsm(dataset_dir: Path) -> dict[str, list[tuple[str, str]]]:
    records: dict[str, list[tuple[str, str]]] = {}
    for language in LANGUAGES:
        path = dataset_dir / "mgsm" / f"mgsm_{language}.tsv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        if len(rows) != EXPECTED_PROBLEMS or any(len(row) != 2 for row in rows):
            raise RuntimeError(f"Unexpected MGSM structure for {language}: rows={len(rows)}")
        records[language] = [(str(question), str(answer)) for question, answer in rows]

    aligned = sum(
        len({records[language][index][1] for language in LANGUAGES}) == 1
        for index in range(EXPECTED_PROBLEMS)
    )
    if aligned != EXPECTED_PROBLEMS:
        raise RuntimeError(f"MGSM answer alignment failed: {aligned}/{EXPECTED_PROBLEMS}")
    return records


def build_metadata(
    records: dict[str, list[tuple[str, str]]],
    gsm_metadata_path: Path,
    gsm_step_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gsm_text = pd.read_csv(gsm_metadata_path, dtype={"sample_id": "string"})
    gsm_steps = pd.read_csv(gsm_step_path, dtype={"sample_id": "string"})
    gsm = gsm_text.merge(
        gsm_steps,
        on=["row_index", "sample_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(gsm) != 1319:
        raise RuntimeError(f"Unexpected GSM8K metadata rows: {len(gsm)}")
    question_to_gsm = {question: index for index, question in enumerate(gsm["question"])}
    mapped_gsm_indices = [question_to_gsm.get(records["en"][index][0]) for index in range(250)]
    if any(index is None for index in mapped_gsm_indices):
        raise RuntimeError("MGSM English does not exactly map 250/250 to GSM8K metadata.")
    mapped_gsm_indices = [int(index) for index in mapped_gsm_indices]
    if len(set(mapped_gsm_indices)) != EXPECTED_PROBLEMS:
        raise RuntimeError("MGSM English mapping is not one-to-one.")

    mapping_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for problem_index, gsm_position in enumerate(mapped_gsm_indices):
        gsm_row = gsm.iloc[gsm_position]
        problem_id = f"mgsm_problem_{problem_index:03d}"
        mapping_rows.append(
            {
                "mgsm_problem_id": problem_id,
                "mgsm_en_row_index": problem_index,
                "gsm8k_row_index": int(gsm_row["row_index"]),
                "gsm8k_sample_id": str(gsm_row["sample_id"]),
            }
        )
        for language in LANGUAGES:
            question, answer = records[language][problem_index]
            sample_rows.append(
                {
                    "problem_id": problem_id,
                    "language": language,
                    "row_index_within_language": problem_index,
                    "sample_id": f"{problem_id}_{language}",
                    "question": question,
                    "answer": answer,
                    "question_char_length": len(question),
                    "gsm8k_row_index": int(gsm_row["row_index"]),
                    "gsm8k_sample_id": str(gsm_row["sample_id"]),
                    "solution_calculation_steps": int(gsm_row["solution_calculation_steps"]),
                    "step_group": str(gsm_row["step_group"]),
                }
            )
    sample_metadata = pd.DataFrame(sample_rows)
    mapping = pd.DataFrame(mapping_rows)
    if len(sample_metadata) != 2750 or sample_metadata["sample_id"].nunique() != 2750:
        raise RuntimeError("Canonical MGSM sample metadata is incomplete or non-unique.")
    counts = sample_metadata.groupby("language").size().to_dict()
    if counts != {language: 250 for language in sorted(LANGUAGES)}:
        raise RuntimeError(f"Unexpected language counts: {counts}")
    if sample_metadata.groupby("problem_id")["language"].nunique().ne(11).any():
        raise RuntimeError("A problem does not contain exactly 11 languages.")
    return sample_metadata, mapping


def legacy_paths(cache_dir: Path, model_name: str) -> tuple[Path, Path]:
    base = cache_dir / ".partial" / model_name
    return base / "mgsm.pt", base / "gsm8k.pt"


def validate_tensor_cache(
    path: Path,
    *,
    expected_layers: int,
    expected_rows: int,
    hidden_size: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if sorted(payload) != list(range(expected_layers)):
        raise RuntimeError(f"Cache indices mismatch in {path}")
    shapes = {tuple(payload[index].shape) for index in payload}
    if shapes != {(expected_rows, hidden_size)}:
        raise RuntimeError(f"Unexpected shapes in {path}: {shapes}")
    if any(payload[index].dtype != torch.float32 for index in payload):
        raise RuntimeError(f"Non-float32 source cache: {path}")
    del payload


def supplement_path(output_dir: Path, model_name: str, language: str) -> Path:
    return output_dir / model_name / "supplemental" / f"{language}.pt"


def expected_supplement_rows(language: str) -> int:
    if language in LEGACY_LANGUAGE_ORDER:
        return 1
    if language in FULL_SUPPLEMENT_LANGUAGES:
        return EXPECTED_PROBLEMS
    raise KeyError(language)


def validate_supplement(
    path: Path,
    language: str,
    num_hidden_layers: int,
    hidden_size: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        expected_rows = expected_supplement_rows(language)
        valid = (
            sorted(payload) == list(range(num_hidden_layers))
            and {tuple(payload[index].shape) for index in payload}
            == {(expected_rows, hidden_size)}
            and all(payload[index].dtype == torch.float32 for index in payload)
        )
        del payload
        return bool(valid)
    except Exception:
        return False


def extract_language(
    *,
    model: Any,
    tokenizer: Any,
    questions: list[str],
    device: str,
    batch_size: int,
    num_hidden_layers: int,
) -> dict[int, torch.Tensor]:
    cached: dict[int, list[torch.Tensor]] = {index: [] for index in range(num_hidden_layers)}
    with torch.inference_mode():
        for start in range(0, len(questions), batch_size):
            batch_questions = questions[start : start + batch_size]
            prompts = [f"{PROMPT_PREFIX}{question}{PROMPT_SUFFIX}" for question in batch_questions]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                return_token_type_ids=False,
            ).to(device)
            output = model(**inputs, output_hidden_states=True, use_cache=False)
            if len(output.hidden_states) != num_hidden_layers + 1:
                raise RuntimeError(
                    f"Expected {num_hidden_layers + 1} hidden-state entries, "
                    f"found {len(output.hidden_states)}"
                )
            for cache_index in range(num_hidden_layers):
                cached[cache_index].append(
                    output.hidden_states[cache_index][:, -1, :].detach().cpu().float()
                )
            del output, inputs
            torch.cuda.empty_cache()
            print(
                f"      {min(start + batch_size, len(questions))}/{len(questions)}",
                flush=True,
            )
    return {index: torch.cat(parts, dim=0) for index, parts in cached.items()}


def write_manifest(
    *,
    output_dir: Path,
    model_name: str,
    config: dict[str, Any],
    mgsm_legacy_path: Path,
    gsm8k_path: Path,
    mapping_path: Path,
) -> None:
    language_sources: dict[str, Any] = {
        "en": {
            "type": "gsm8k_exact_mapping",
            "path": str(gsm8k_path.resolve()),
            "row_count": 250,
            "mapping_file": str(mapping_path.resolve()),
        }
    }
    for legacy_position, language in enumerate(LEGACY_LANGUAGE_ORDER):
        language_sources[language] = {
            "type": "row0_supplement_plus_legacy_rows_1_249",
            "supplement_path": str(supplement_path(output_dir, model_name, language).resolve()),
            "legacy_path": str(mgsm_legacy_path.resolve()),
            "legacy_slice_start": legacy_position * 249,
            "legacy_slice_stop": (legacy_position + 1) * 249,
            "row_count": 250,
        }
    for language in FULL_SUPPLEMENT_LANGUAGES:
        language_sources[language] = {
            "type": "full_supplement",
            "supplement_path": str(supplement_path(output_dir, model_name, language).resolve()),
            "row_count": 250,
        }
    manifest = {
        "model": model_name,
        "created_at": datetime.now().astimezone().isoformat(),
        "logical_cache_only": True,
        "original_cache_modified": False,
        "languages": list(LANGUAGES),
        "problems_per_language": 250,
        "total_language_samples": 2750,
        "prompt_template": "Q: {question}\\nA: ",
        "token_position": "last padded input token with left padding",
        "num_hidden_layers": config["num_hidden_layers"],
        "hidden_size": config["hidden_size"],
        "cached_indices": list(range(config["num_hidden_layers"])),
        "analysis_excludes_cache_index": 0,
        "language_sources": language_sources,
    }
    atomic_write_json(output_dir / model_name / "manifest.json", manifest)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    dataset_dir = args.dataset_dir.resolve()
    model_dir = args.model_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_mgsm(dataset_dir)
    sample_metadata, mapping = build_metadata(
        records, args.gsm_metadata.resolve(), args.gsm_step_metadata.resolve()
    )
    sample_metadata_path = output_dir / "mgsm_sample_metadata.csv"
    mapping_path = output_dir / "mgsm_problem_mapping.csv"
    atomic_write_csv(sample_metadata_path, sample_metadata)
    atomic_write_csv(mapping_path, mapping)

    caches = discover_caches(cache_dir)
    selected_models = select_models(args.models, caches)
    print("MGSM 11-language cache preparation", flush=True)
    print(f"  models: {selected_models}", flush=True)
    print("  canonical data: 11 languages × 250 problems = 2750", flush=True)
    print("  reuse: English GSM8K 250 + legacy MGSM 5×249", flush=True)
    print("  extract: 5 missing languages ×250 + 5 missing row-0 = 1255/model", flush=True)

    extraction_plan: dict[str, list[str]] = {}
    for model_name in selected_models:
        config = load_model_config(model_dir, model_name)
        num_hidden_layers = config["num_hidden_layers"]
        hidden_size = config["hidden_size"]
        legacy_mgsm, legacy_gsm8k = legacy_paths(cache_dir, model_name)
        validate_tensor_cache(
            legacy_mgsm,
            expected_layers=num_hidden_layers,
            expected_rows=EXPECTED_LEGACY_TOTAL,
            hidden_size=hidden_size,
        )
        validate_tensor_cache(
            legacy_gsm8k,
            expected_layers=num_hidden_layers,
            expected_rows=1319,
            hidden_size=hidden_size,
        )
        needed = []
        for language in (*LEGACY_LANGUAGE_ORDER, *FULL_SUPPLEMENT_LANGUAGES):
            path = supplement_path(output_dir, model_name, language)
            valid = validate_supplement(path, language, num_hidden_layers, hidden_size)
            if valid and (args.skip_existing or not args.overwrite):
                continue
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Invalid/existing supplement requires --overwrite: {path}")
            needed.append(language)
        extraction_plan[model_name] = needed
        print(f"  {model_name}: missing supplements={needed or 'none'}", flush=True)

    if args.validate_only:
        print("Validation-only complete; no model was loaded.", flush=True)
        return 0

    if any(extraction_plan.values()):
        if not torch.cuda.is_available() or torch.cuda.device_count() <= args.device_id:
            raise RuntimeError(
                f"CUDA device {args.device_id} is unavailable; supplements were not extracted."
            )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{args.device_id}"
    torch.cuda.set_device(args.device_id)
    for model_name in selected_models:
        config = load_model_config(model_dir, model_name)
        num_hidden_layers = config["num_hidden_layers"]
        hidden_size = config["hidden_size"]
        legacy_mgsm, legacy_gsm8k = legacy_paths(cache_dir, model_name)
        needed = extraction_plan[model_name]
        if needed:
            print(f"\n[LOAD] {model_name} on {device}", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir / model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_dir / model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            if any(
                token in str(model.config.model_type).lower()
                for token in ("llama", "mistral", "yi", "gptj")
            ):
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif "qwen" in str(model.config.model_type).lower():
                tokenizer.pad_token = "<|endoftext|>"
            tokenizer.padding_side = "left"
            model.eval().to(device)

            for language in needed:
                if language in LEGACY_LANGUAGE_ORDER:
                    questions = [records[language][0][0]]
                else:
                    questions = [question for question, _ in records[language]]
                print(f"  [EXTRACT] {language}: {len(questions)} samples", flush=True)
                payload = extract_language(
                    model=model,
                    tokenizer=tokenizer,
                    questions=questions,
                    device=device,
                    batch_size=args.batch_size,
                    num_hidden_layers=num_hidden_layers,
                )
                path = supplement_path(output_dir, model_name, language)
                atomic_torch_save(path, payload)
                if not validate_supplement(
                    path, language, num_hidden_layers, hidden_size
                ):
                    raise RuntimeError(f"Written supplement failed validation: {path}")
                print(f"  [SAVED] {path}", flush=True)
                del payload
                gc.collect()

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        for language in (*LEGACY_LANGUAGE_ORDER, *FULL_SUPPLEMENT_LANGUAGES):
            path = supplement_path(output_dir, model_name, language)
            if not validate_supplement(path, language, num_hidden_layers, hidden_size):
                raise RuntimeError(f"Incomplete supplement after processing: {path}")
        write_manifest(
            output_dir=output_dir,
            model_name=model_name,
            config=config,
            mgsm_legacy_path=legacy_mgsm,
            gsm8k_path=legacy_gsm8k,
            mapping_path=mapping_path,
        )
        print(f"[COMPLETE] {model_name}", flush=True)

    print(f"Output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed supplements remain resumable.", file=sys.stderr)
        raise
