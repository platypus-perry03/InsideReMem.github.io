#!/usr/bin/env python3
"""Protocol-authorized AI-only-audited Baseline Calibration v3.1.

This runner evaluates baseline behavior only.  It must not load LiReF,
candidate components, hidden states, hooks, or interventions.  Runtime ML
libraries are imported only after an explicit --execute request and a separate
hash-locked execution authorization have both passed preflight validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
STAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "AI" / "reenact" / "models" / "Meta-Llama-3-8B"
OUTPUT_ROOT = ROOT / "AI" / "reenact" / "liref_outputs" / "rm_decomp" / "v3"

DESIGN_PATH = STAGE_DIR / "calibration_v3_design_frozen.json"
AMENDMENT_PATH = STAGE_DIR / "calibration_v3_1_ai_only_audit_policy_frozen.json"
DATASET_PATH = STAGE_DIR / "calibration_v3_assets" / "calibration_v3_dataset_draft.json"
DATASET_MANIFEST_PATH = STAGE_DIR / "calibration_v3_assets" / "calibration_v3_dataset_manifest.json"
AUTOMATIC_AUDIT_PATH = STAGE_DIR / "calibration_v3_assets" / "calibration_v3_automatic_audit.json"
PRIMARY_AI_AUDIT_PATH = (
    STAGE_DIR
    / "calibration_v3_assets"
    / "ai_audit"
    / "calibration_v3_ai_linguistic_audit_summary.json"
)
ADVERSARIAL_AI_AUDIT_PATH = STAGE_DIR / "calibration_v3_ai_only_additional_review.json"
STATIC_REVIEW_PATH = (
    STAGE_DIR
    / "calibration_v3_assets"
    / "calibration_v3_1_baseline_static_safety_review.json"
)

LOCKED_INPUT_HASHES = {
    "design": "c60a579729376d391582dbc03af9cfd3ba0a1e1743a9e9a884967aacc177adfc",
    "amendment": "15ce7892f12e00360b07ce533188249229dc24409119bd96e4c2c25c1bf8f9de",
    "dataset": "d2187c0623ba9752776cf0251dee3dabf9d80ac04e339cf3eb4bd1d1b42761a1",
    "dataset_manifest": "a157ec7bd463c739ee046f9e3f85a08d2e3ebb7dc6a794a63757653c93fad822",
    "automatic_audit": "e904fcec13b97bf9d09afc707aedd2b37c100c911b560f2b88f0ed270e654e26",
    "primary_ai_audit": "d5e1743e27aa7e78e6879a5d6902cabd7998774844e67d1eb29de9133a70827a",
    "adversarial_ai_audit": "94dcf1ea437ab2a16e30b2475778372e7c78e0e325c7f288d5a6a547e209f712",
}
LOCKED_PATHS = {
    "design": DESIGN_PATH,
    "amendment": AMENDMENT_PATH,
    "dataset": DATASET_PATH,
    "dataset_manifest": DATASET_MANIFEST_PATH,
    "automatic_audit": AUTOMATIC_AUDIT_PATH,
    "primary_ai_audit": PRIMARY_AI_AUDIT_PATH,
    "adversarial_ai_audit": ADVERSARIAL_AI_AUDIT_PATH,
}

MODEL_SMALL_FILE_HASHES = {
    "config.json": "2430cee764b6530ff8673cf9ba8561e1d5a33152d503cd0de909ff5718261441",
    "generation_config.json": "93caf96e269e32b9ee33ad78b0d76d910408d11a63a8b2c49241030836759311",
    "model.safetensors.index.json": "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b",
    "special_tokens_map.json": "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
    "tokenizer.json": "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
    "tokenizer_config.json": "690727b4fed286383df1c7ca5e805124cb70c6eb4529f807c7b2e60ff741da7e",
}
MODEL_PARAMETER_FILE_HASHES = {
    "model-00001-of-00004.safetensors": "f2c144103072514542e327fa8080bd375cb300f2d453fba9ca3aea81d0d4cf33",
    "model-00002-of-00004.safetensors": "d9eee5f23d94405d90b7e9ff88b9443fee42f8528a658f54214c2aba7530d80c",
    "model-00003-of-00004.safetensors": "4b8fbc5e113f69768dd8de84661ea20af8a32b734a9976144b4236c447b40ccc",
    "model-00004-of-00004.safetensors": "5dc34e6bdf2da9e35f0d93b5c333c870f3677dc43dc3a91ea3a8ad28a1fe1acb",
}

EXPECTED_FAMILIES = ("object_count", "points_balance", "temperature")
EXPECTED_PAIR_COUNT = 192
EXPECTED_PROMPT_COUNT = 384
EXPECTED_TEMPLATE_COUNT = 24
EXPECTED_FRAMES_PER_TEMPLATE = 8
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260831
MODEL_DTYPE = "float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.preflight_only == args.execute:
        parser.error("Choose exactly one of --preflight-only or --execute")
    if args.execute and (args.authorization is None or not args.run_id):
        parser.error("--execute requires --authorization and --run-id")
    if args.preflight_only and (args.authorization is not None or args.run_id is not None):
        parser.error("Preflight-only mode does not accept execution authorization or run ID")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_locked_inputs() -> dict[str, Any]:
    actual_hashes = {name: sha256_file(path) for name, path in LOCKED_PATHS.items()}
    mismatches = {
        name: {"expected": LOCKED_INPUT_HASHES[name], "actual": actual_hash}
        for name, actual_hash in actual_hashes.items()
        if actual_hash != LOCKED_INPUT_HASHES[name]
    }
    if mismatches:
        raise RuntimeError(f"Locked input hash mismatch: {mismatches}")

    design = load_json(DESIGN_PATH)
    amendment = load_json(AMENDMENT_PATH)
    dataset = load_json(DATASET_PATH)
    manifest = load_json(DATASET_MANIFEST_PATH)
    automatic = load_json(AUTOMATIC_AUDIT_PATH)
    primary = load_json(PRIMARY_AI_AUDIT_PATH)
    adversarial = load_json(ADVERSARIAL_AI_AUDIT_PATH)
    checks = {
        "dataset_identity": (
            dataset["pair_count"] == EXPECTED_PAIR_COUNT
            and len(dataset["pairs"]) == EXPECTED_PAIR_COUNT
            and dataset["prompt_count"] == EXPECTED_PROMPT_COUNT
            and dataset["independent_template_family_count"] == EXPECTED_TEMPLATE_COUNT
        ),
        "families_exact": tuple(design["dataset_design"]["lexical_families"]) == EXPECTED_FAMILIES,
        "frames_exact": design["dataset_design"]["frames_per_template_family"] == EXPECTED_FRAMES_PER_TEMPLATE,
        "manifest_execution_closed": (
            manifest["baseline_calibration_execution_allowed"] is False
            and manifest["stage_e_pilot_allowed"] is False
        ),
        "automatic_audit_pass": automatic["all_automatic_checks_pass"] is True,
        "primary_ai_audit_pass": primary["dataset_pass"] is True and primary["ai_audit_pass_count"] == 192,
        "adversarial_ai_audit_pass": adversarial["all_checks_pass"] is True and adversarial["pair_count"] == 192,
        "amendment_ai_gate": amendment["ai_only_audit_gate_satisfied"] is True,
        "amendment_implementation_allowed": amendment["baseline_calibration_implementation_allowed"] is True,
        "amendment_execution_closed": (
            amendment["baseline_calibration_execution_allowed"] is False
            and amendment["model_loading_allowed"] is False
            and amendment["stage_e_pilot_allowed"] is False
        ),
        "human_claim_forbidden": amendment["audit_policy"]["human_audited_claim_allowed"] is False,
        "thresholds_frozen": design["acceptance_criteria"]["threshold_change_after_result_allowed"] is False,
        "post_result_selection_forbidden": design["acceptance_criteria"]["result_dependent_template_or_family_exclusion_allowed"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Preflight semantic check failure: {checks}")
    return {
        "status": "preflight_pass_execution_still_requires_separate_authorization",
        "locked_hashes": actual_hashes,
        "checks": checks,
        "pair_count": EXPECTED_PAIR_COUNT,
        "prompt_count": EXPECTED_PROMPT_COUNT,
        "human_audit": "not_performed",
        "ai_only_audited": True,
        "human_audited_evidence": False,
        "model_runtime_imported": False,
        "model_loaded": False,
        "model_forward_performed": False,
        "gpu_used": False,
        "execution_allowed": False,
    }


def validate_execution_authorization(
    path: Path, run_id: str, implementation_sha256: str, static_review_sha256: str
) -> dict[str, Any]:
    authorization = load_json(path.resolve())
    checks = {
        "status": authorization.get("status") == "execution_authorized",
        "execution_allowed": authorization.get("execution_allowed") is True,
        "scope": authorization.get("scope") == "baseline_calibration_v3_1_only",
        "run_id": authorization.get("run_id") == run_id,
        "device": bool(re.fullmatch(r"cuda:\d+", str(authorization.get("device", "")))),
        "batch_size": isinstance(authorization.get("batch_size"), int) and authorization["batch_size"] > 0,
        "dtype": authorization.get("dtype") == MODEL_DTYPE,
        "implementation_hash": authorization.get("implementation_sha256") == implementation_sha256,
        "static_review_hash": authorization.get("static_review_sha256") == static_review_sha256,
        "locked_inputs": authorization.get("locked_input_hashes") == LOCKED_INPUT_HASHES,
        "model_loading": authorization.get("model_loading_allowed") is True,
        "gpu": authorization.get("gpu_execution_allowed") is True,
        "liref_closed": authorization.get("load_liref_direction") is False,
        "hidden_closed": authorization.get("capture_hidden_states") is False,
        "candidate_closed": authorization.get("inspect_candidate_states") is False,
        "hooks_closed": authorization.get("forward_hooks") is False,
        "intervention_closed": authorization.get("activation_or_weight_intervention") is False,
        "pilot_closed": authorization.get("stage_e_pilot_allowed") is False,
        "human_claim_closed": authorization.get("human_audited_evidence") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Execution authorization rejected: {checks}")
    return authorization


def validate_model_file_hashes() -> dict[str, Any]:
    expected = {**MODEL_SMALL_FILE_HASHES, **MODEL_PARAMETER_FILE_HASHES}
    actual = {name: sha256_file(MODEL_DIR / name) for name in expected}
    mismatches = {
        name: {"expected": expected[name], "actual": value}
        for name, value in actual.items()
        if value != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"Model artifact hash mismatch: {mismatches}")
    return {
        "small_artifact_hashes": {name: actual[name] for name in MODEL_SMALL_FILE_HASHES},
        "parameter_file_hashes": {name: actual[name] for name in MODEL_PARAMETER_FILE_HASHES},
    }


def normalize_generated_token(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return normalized if re.fullmatch(r"[0-9]+", normalized) else ""


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def descriptive_bootstrap_ci(values: list[float]) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    means = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def compute_cluster_dz(values: list[float]) -> tuple[float | None, bool, str]:
    if len(values) != 8 or any(not math.isfinite(value) for value in values):
        return None, False, "missing_or_nonfinite_cluster"
    mean_value = statistics.fmean(values)
    if all(value == 0.0 for value in values):
        return 0.0, True, "all_exactly_zero"
    sample_sd = statistics.stdev(values)
    if sample_sd <= sys.float_info.epsilon:
        if mean_value == 0.0:
            return 0.0, True, "near_zero_sd_zero_mean"
        return None, False, "near_zero_sd_nonzero_mean"
    dz = mean_value / sample_sd
    return dz, math.isfinite(dz), "valid"


def summarize_results(
    pair_results: list[dict[str, Any]], design: dict[str, Any]
) -> dict[str, Any]:
    criteria = design["acceptance_criteria"]
    fc = criteria["candidate_forced_choice"]
    generation = criteria["one_token_generation"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in pair_results:
        grouped[result["lexical_family"]].append(result)
    if tuple(sorted(grouped)) != tuple(sorted(EXPECTED_FAMILIES)):
        raise RuntimeError(f"Unexpected result families: {sorted(grouped)}")

    family_summaries: dict[str, Any] = {}
    passed_families: list[str] = []
    failed_families: list[str] = []
    for family in EXPECTED_FAMILIES:
        selected = grouped[family]
        if len(selected) != 64:
            raise RuntimeError(f"Expected 64 pair results for {family}, found {len(selected)}")
        by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in selected:
            by_template[result["template_family_id"]].append(result)
        if len(by_template) != 8 or any(len(items) != 8 for items in by_template.values()):
            raise RuntimeError(f"Invalid template cluster structure for {family}")

        template_contrasts = {
            template_id: statistics.fmean(
                item["conditions"]["arithmetic"]["margin_nats"]
                - item["conditions"]["selector"]["margin_nats"]
                for item in items
            )
            for template_id, items in sorted(by_template.items())
        }
        d_values = list(template_contrasts.values())
        mean_d = statistics.fmean(d_values)
        dz, dz_valid, dz_reason = compute_cluster_dz(d_values)
        positive = sum(value > 0 for value in d_values)
        negative = sum(value < 0 for value in d_values)
        same_sign = max(positive, negative)

        condition_metrics: dict[str, Any] = {}
        for condition in ("arithmetic", "selector"):
            records = [item["conditions"][condition] for item in selected]
            margins = [record["margin_nats"] for record in records]
            fc_count = sum(record["forced_choice_correct"] for record in records)
            generation_count = sum(record["generation_correct"] for record in records)
            valid_generation_count = sum(record["generation_valid_format"] for record in records)
            condition_metrics[condition] = {
                "denominator": len(records),
                "forced_choice_correct_count": fc_count,
                "forced_choice_accuracy": fc_count / len(records),
                "one_token_generation_correct_count": generation_count,
                "one_token_generation_accuracy": generation_count / len(records),
                "one_token_generation_valid_format_count": valid_generation_count,
                "mean_margin_nats": statistics.fmean(margins),
                "margin_10th_percentile": percentile(margins, 0.10),
                "margin_90th_percentile": percentile(margins, 0.90),
                "mean_raw_canonical_log_probability": statistics.fmean(
                    record["correct_log_probability"] for record in records
                ),
                "mean_per_token_geometric_probability": statistics.fmean(
                    record["correct_geometric_probability"] for record in records
                ),
            }

        arithmetic_fc = condition_metrics["arithmetic"]["forced_choice_correct_count"]
        selector_fc = condition_metrics["selector"]["forced_choice_correct_count"]
        arithmetic_gen = condition_metrics["arithmetic"]["one_token_generation_correct_count"]
        selector_gen = condition_metrics["selector"]["one_token_generation_correct_count"]
        margin_overlap = max(
            0.0,
            min(
                condition_metrics["arithmetic"]["margin_90th_percentile"],
                condition_metrics["selector"]["margin_90th_percentile"],
            )
            - max(
                condition_metrics["arithmetic"]["margin_10th_percentile"],
                condition_metrics["selector"]["margin_10th_percentile"],
            ),
        )
        checks = {
            "forced_choice_arithmetic_count_range": fc["minimum_correct_count"] <= arithmetic_fc <= fc["maximum_correct_count"],
            "forced_choice_selector_count_range": fc["minimum_correct_count"] <= selector_fc <= fc["maximum_correct_count"],
            "forced_choice_condition_count_gap": abs(arithmetic_fc - selector_fc) <= fc["maximum_absolute_condition_count_gap"],
            "generation_arithmetic_count_range": generation["minimum_correct_count"] <= arithmetic_gen <= generation["maximum_correct_count"],
            "generation_selector_count_range": generation["minimum_correct_count"] <= selector_gen <= generation["maximum_correct_count"],
            "generation_condition_count_gap": abs(arithmetic_gen - selector_gen) <= generation["maximum_absolute_condition_count_gap"],
            "mean_template_contrast": abs(mean_d) <= criteria["maximum_absolute_mean_template_contrast_nats"],
            "cluster_dz_valid_and_bounded": dz_valid and dz is not None and abs(dz) <= criteria["maximum_absolute_cluster_dz"],
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        (passed_families if status == "PASS" else failed_families).append(family)
        family_summaries[family] = {
            "status": status,
            "checks": checks,
            "condition_metrics": condition_metrics,
            "absolute_forced_choice_correct_count_gap": abs(arithmetic_fc - selector_fc),
            "absolute_generation_correct_count_gap": abs(arithmetic_gen - selector_gen),
            "template_contrasts": template_contrasts,
            "mean_template_contrast_nats": mean_d,
            "cluster_dz": dz,
            "cluster_dz_valid": dz_valid,
            "cluster_dz_reason": dz_reason,
            "positive_template_count": positive,
            "negative_template_count": negative,
            "same_sign_template_count_descriptive": same_sign,
            "condition_margin_10_to_90_percentile_overlap_width": margin_overlap,
            "descriptive_cluster_bootstrap_mean_95ci": descriptive_bootstrap_ci(d_values),
        }
    return {
        "schema_version": "3.1",
        "status": "PASS" if passed_families else "FAIL",
        "result_label": "protocol_authorized_ai_only_audited_baseline_calibration_v3_1",
        "human_audit": "not_performed",
        "human_audited_evidence": False,
        "all_criteria_required": True,
        "pass_fail_unit": "lexical_family",
        "authoritative_accuracy_unit": criteria["authoritative_accuracy_unit"],
        "result_dependent_item_template_or_family_exclusion_performed": False,
        "bootstrap_used_for_pass_fail": False,
        "passed_families": passed_families,
        "failed_families": failed_families,
        "family_summaries": family_summaries,
    }


def load_runtime() -> tuple[Any, Any, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, transformers, (AutoModelForCausalLM, AutoTokenizer)


def answer_token_record(tokenizer: Any, prompt: str, answer: str) -> dict[str, Any]:
    joint = prompt + answer
    encoded = tokenizer(joint, add_special_tokens=True, return_offsets_mapping=True)
    start = len(prompt)
    positions = [
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if right > start and left < len(joint)
    ]
    if len(positions) != 1 or positions[0] == 0:
        raise RuntimeError(f"Expected exactly one answer continuation token for {answer!r}")
    position = positions[0]
    return {
        "input_ids": [int(value) for value in encoded["input_ids"]],
        "target_position": position,
        "target_token_id": int(encoded["input_ids"][position]),
    }


def score_candidates(
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: str,
    batch_size: int,
    records: list[dict[str, Any]],
) -> dict[str, float]:
    prepared = []
    for record in records:
        token_record = answer_token_record(tokenizer, record["prompt"], record["answer"])
        prepared.append({**record, **token_record})
    scores: dict[str, float] = {}
    pad_token_id = int(tokenizer.pad_token_id)
    with torch.inference_mode():
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start:start + batch_size]
            max_length = max(len(item["input_ids"]) for item in batch)
            input_ids = []
            attention_mask = []
            for item in batch:
                padding = max_length - len(item["input_ids"])
                input_ids.append(item["input_ids"] + [pad_token_id] * padding)
                attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
            mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)
            outputs = model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
            for row, item in enumerate(batch):
                position = item["target_position"]
                logits = outputs.logits[row, position - 1].float()
                log_probability = torch.log_softmax(logits, dim=-1)[item["target_token_id"]]
                scores[item["record_id"]] = float(log_probability.item())
            del outputs, input_tensor, mask_tensor
    return scores


def generate_one_token(
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: str,
    batch_size: int,
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    outputs_by_id: dict[str, dict[str, Any]] = {}
    tokenizer.padding_side = "left"
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            encoded = tokenizer(
                [record["prompt"] for record in batch],
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=False,
                output_hidden_states=False,
                output_attentions=False,
            )
            suffix = generated[:, encoded["input_ids"].shape[1]:]
            for row, record in enumerate(batch):
                token_ids = [int(value) for value in suffix[row].tolist()]
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                normalized = normalize_generated_token(decoded)
                outputs_by_id[record["record_id"]] = {
                    "generated_token_ids": token_ids,
                    "generated_text": decoded,
                    "normalized_generation": normalized,
                    "generation_valid_format": bool(normalized),
                    "generation_correct": normalized == record["correct_answer"],
                }
    return outputs_by_id


def registered_hook_count(model: Any) -> int:
    total = 0
    for module in model.modules():
        total += len(getattr(module, "_forward_hooks", {}))
        total += len(getattr(module, "_forward_pre_hooks", {}))
        total += len(getattr(module, "_backward_hooks", {}))
    return total


def execute(args: argparse.Namespace, preflight: dict[str, Any]) -> None:
    implementation_sha256 = sha256_file(Path(__file__).resolve())
    if not STATIC_REVIEW_PATH.exists():
        raise RuntimeError("Static review artifact is missing")
    static_review = load_json(STATIC_REVIEW_PATH)
    if static_review.get("all_checks_pass") is not True:
        raise RuntimeError("Static review is not PASS")
    static_review_sha256 = sha256_file(STATIC_REVIEW_PATH)
    authorization = validate_execution_authorization(
        args.authorization, args.run_id, implementation_sha256, static_review_sha256
    )
    run_dir = args.output_root.resolve() / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "status.json", {"status": "running", "run_id": args.run_id})

    model_hashes = validate_model_file_hashes()
    torch, transformers, runtime_classes = load_runtime()
    AutoModelForCausalLM, AutoTokenizer = runtime_classes
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = authorization["device"]
    batch_size = authorization["batch_size"]
    device_index = int(device.split(":", 1)[1])
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"Authorized CUDA device does not exist: {device}")
    torch.cuda.set_device(device_index)
    random.seed(BOOTSTRAP_SEED)
    torch.manual_seed(BOOTSTRAP_SEED)
    torch.cuda.manual_seed_all(BOOTSTRAP_SEED)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.config.output_hidden_states = False
    model.config.output_attentions = False
    if registered_hook_count(model) != 0:
        raise RuntimeError("Unexpected registered model hooks")
    config_checks = {
        "model_type": model.config.model_type == "llama",
        "hidden_size": model.config.hidden_size == 4096,
        "intermediate_size": model.config.intermediate_size == 14336,
        "num_hidden_layers": model.config.num_hidden_layers == 32,
        "num_attention_heads": model.config.num_attention_heads == 32,
        "num_key_value_heads": model.config.num_key_value_heads == 8,
        "output_hidden_states_false": model.config.output_hidden_states is False,
        "output_attentions_false": model.config.output_attentions is False,
        "registered_hook_count_zero": registered_hook_count(model) == 0,
    }
    if not all(config_checks.values()):
        raise RuntimeError(f"Model config contract mismatch: {config_checks}")

    dataset = load_json(DATASET_PATH)
    design = load_json(DESIGN_PATH)
    candidate_records: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    for pair in dataset["pairs"]:
        for condition in ("arithmetic", "selector"):
            payload = pair["conditions"][condition]
            base_id = f'{pair["pair_id"]}:{condition}'
            correct = payload["canonical_answer"]
            alternative = payload["primary_alternative_answer"]
            candidate_records.extend([
                {"record_id": f"{base_id}:correct", "prompt": payload["full_prompt"], "answer": correct},
                {"record_id": f"{base_id}:alternative", "prompt": payload["full_prompt"], "answer": alternative},
            ])
            if condition == "arithmetic":
                candidate_records.extend([
                    {"record_id": f"{base_id}:start", "prompt": payload["full_prompt"], "answer": str(pair["start"])},
                    {"record_id": f"{base_id}:delta", "prompt": payload["full_prompt"], "answer": str(pair["delta"])},
                ])
            generation_records.append({
                "record_id": base_id,
                "prompt": payload["full_prompt"],
                "correct_answer": correct,
            })

    candidate_scores = score_candidates(
        model, tokenizer, torch, device, batch_size, candidate_records
    )
    generations = generate_one_token(
        model, tokenizer, torch, device, batch_size, generation_records
    )
    pair_results: list[dict[str, Any]] = []
    for pair in dataset["pairs"]:
        conditions: dict[str, Any] = {}
        for condition in ("arithmetic", "selector"):
            base_id = f'{pair["pair_id"]}:{condition}'
            correct_logp = candidate_scores[f"{base_id}:correct"]
            alternative_logp = candidate_scores[f"{base_id}:alternative"]
            margin = correct_logp - alternative_logp
            result = {
                "correct_answer": pair["conditions"][condition]["canonical_answer"],
                "alternative_answer": pair["conditions"][condition]["primary_alternative_answer"],
                "correct_log_probability": correct_logp,
                "alternative_log_probability": alternative_logp,
                "correct_geometric_probability": math.exp(correct_logp),
                "margin_nats": margin,
                "forced_choice_correct": margin > 0.0,
                **generations[base_id],
            }
            if condition == "arithmetic":
                start_logp = candidate_scores[f"{base_id}:start"]
                delta_logp = candidate_scores[f"{base_id}:delta"]
                result["diagnostic_start_log_probability"] = start_logp
                result["diagnostic_delta_log_probability"] = delta_logp
                result["correct_vs_max_operand_diagnostic_log_odds"] = correct_logp - max(start_logp, delta_logp)
            conditions[condition] = result
        pair_results.append({
            "pair_id": pair["pair_id"],
            "lexical_family": pair["lexical_family"],
            "template_family_id": pair["template_family_id"],
            "frame_index": pair["frame_index"],
            "oa_row": pair["oa_row"],
            "operation": pair["operation"],
            "selector_active_entry": pair["selector_active_entry"],
            "conditions": conditions,
        })

    summary = summarize_results(pair_results, design)
    pair_results_path = run_dir / "baseline_pair_results.json"
    summary_path = run_dir / "baseline_calibration_summary.json"
    atomic_json(pair_results_path, {
        "schema_version": "3.1",
        "run_id": args.run_id,
        "pair_count": len(pair_results),
        "prompt_count": EXPECTED_PROMPT_COUNT,
        "results": pair_results,
    })
    atomic_json(summary_path, summary)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "batch_size": batch_size,
        "dtype": MODEL_DTYPE,
        "random_seed": BOOTSTRAP_SEED,
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "model_path": str(MODEL_DIR),
        "model_hashes": model_hashes,
        "model_config_checks": config_checks,
        "preflight": preflight,
        "ai_only_audited": True,
        "human_audit": "not_performed",
        "human_audited_evidence": False,
        "model_forward_performed": True,
        "gpu_used": True,
        "liref_loaded": False,
        "candidate_states_inspected": False,
        "hidden_states_captured": False,
        "hooks_or_interventions_used": False,
    }
    environment_path = run_dir / "environment_and_safety.json"
    atomic_json(environment_path, environment)
    manifest = {
        "schema_version": "3.1",
        "status": "complete",
        "result_label": "protocol_authorized_ai_only_audited_baseline_calibration_v3_1",
        "run_id": args.run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "calibration_status": summary["status"],
        "passed_families": summary["passed_families"],
        "failed_families": summary["failed_families"],
        "ai_only_audited": True,
        "human_audit": "not_performed",
        "human_audited_evidence": False,
        "pilot_allowed_from_this_manifest": False,
        "implementation_sha256": implementation_sha256,
        "static_safety_review_sha256": static_review_sha256,
        "execution_authorization_sha256": sha256_file(args.authorization.resolve()),
        "locked_input_hashes": LOCKED_INPUT_HASHES,
        "results_sha256": sha256_file(pair_results_path),
        "summary_sha256": sha256_file(summary_path),
        "environment_sha256": sha256_file(environment_path),
    }
    atomic_json(run_dir / "run_manifest.json", manifest)
    atomic_json(run_dir / "status.json", {"status": "complete", "run_id": args.run_id})
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    preflight = validate_locked_inputs()
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return
    try:
        execute(args, preflight)
    except Exception as error:
        # A run directory can only be newly created by execute(); preserve a
        # minimal failure record without weakening the fail-closed gate.
        run_dir = args.output_root.resolve() / args.run_id
        if run_dir.is_dir():
            atomic_json(
                run_dir / "status.json",
                {
                    "status": "failed",
                    "run_id": args.run_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "model_result_claim_allowed": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
