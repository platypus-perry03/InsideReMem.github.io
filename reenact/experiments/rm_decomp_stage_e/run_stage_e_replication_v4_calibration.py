#!/usr/bin/env python3
"""Behavior-only Baseline Calibration for Stage E independent replication v4.

The runner reads only the frozen calibration pool.  It cannot load LiReF
directions, candidate-component states, hooks, hidden-state outputs, the
independent replication pool, or interventions.  Runtime ML imports occur only
after a separately frozen, hash-locked execution authorization is validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
STAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = STAGE_DIR / "stage_e_replication_v4_assets"
MODEL_DIR = ROOT / "AI" / "reenact" / "models" / "Meta-Llama-3-8B"
OUTPUT_ROOT = ROOT / "AI" / "reenact" / "liref_outputs" / "rm_decomp" / "v4" / "calibration"

DESIGN_PATH = STAGE_DIR / "stage_e_independent_replication_v4_design_frozen.json"
CATALOG_PATH = STAGE_DIR / "stage_e_replication_v4_template_catalog_frozen.json"
CATALOG_AMENDMENT_PATH = STAGE_DIR / "stage_e_replication_v4_0_1_catalog_amendment_frozen.json"
IMPLEMENTATION_AMENDMENT_PATH = STAGE_DIR / "stage_e_replication_v4_calibration_implementation_amendment_frozen.json"
DATASET_PATH = ASSET_DIR / "calibration_pool_dataset.json"
AUTOMATIC_AUDIT_PATH = ASSET_DIR / "calibration_pool_automatic_audit.json"
PRIMARY_AI_AUDIT_PATH = ASSET_DIR / "calibration_pool_primary_ai_audit.json"
ADVERSARIAL_AI_AUDIT_PATH = ASSET_DIR / "calibration_pool_adversarial_ai_audit.json"
AI_AUDIT_SUMMARY_PATH = ASSET_DIR / "ai_audit_summary.json"
STATIC_REVIEW_PATH = ASSET_DIR / "stage_e_replication_v4_calibration_static_review.json"

LOCKED_INPUT_HASHES = {
    "design": "0382a059f2ac3578446e772939a10dc6911d11b7a90bb4cb0f7bd78ed5ebe106",
    "catalog": "d49fab4ceb9be75ec6d9ec2549b433abcbb108fb57862d12659232a3b7fe186b",
    "catalog_amendment": "0cd26c72db2c3510e4013c4427e25ab3622796b3f6e9f202100c5e7e4e5b68cb",
    "implementation_amendment": "bddc48de28e8861b8a2baad1e32db460e95dd964fb54474266b1657fb92bb5c0",
    "calibration_dataset": "e4b660057b8103533c3303c8defc8a3b03268fac036ff3b8232c9e20662f6ded",
    "calibration_automatic_audit": "aaa777b9e794db807f8a01dc1a9865b4fdcf58bc0ac5a135fddc49752eb6aaa0",
    "calibration_primary_ai_audit": "800f652a876f8971857d7b9dac93d758688291044d588a5e185d132a5cac5b6d",
    "calibration_adversarial_ai_audit": "51edae3f42f8ee6aef1558ec98f8f58c5bdcf280928b8878f2523c484cb31f03",
    "ai_audit_summary": "a98408356b0f8ba40ea01b288bbde333f2dd582136b2601457fb802e86763471",
}
LOCKED_PATHS = {
    "design": DESIGN_PATH,
    "catalog": CATALOG_PATH,
    "catalog_amendment": CATALOG_AMENDMENT_PATH,
    "implementation_amendment": IMPLEMENTATION_AMENDMENT_PATH,
    "calibration_dataset": DATASET_PATH,
    "calibration_automatic_audit": AUTOMATIC_AUDIT_PATH,
    "calibration_primary_ai_audit": PRIMARY_AI_AUDIT_PATH,
    "calibration_adversarial_ai_audit": ADVERSARIAL_AI_AUDIT_PATH,
    "ai_audit_summary": AI_AUDIT_SUMMARY_PATH,
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

EXPECTED_FAMILIES = ("points_balance", "temperature")
EXPECTED_PAIR_COUNT = 128
EXPECTED_PROMPT_COUNT = 256
EXPECTED_TEMPLATE_COUNT = 16
EXPECTED_FRAMES_PER_TEMPLATE = 8
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20260901
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
        parser.error("Preflight mode does not accept authorization or run ID")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Empty percentile input")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def cluster_bootstrap_ci(values: list[float]) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    means = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(BOOTSTRAP_REPETITIONS)]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def cluster_dz(values: list[float]) -> tuple[float | None, bool, str]:
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
    value = mean_value / sample_sd
    return value, math.isfinite(value), "valid"


def validate_locked_inputs() -> dict[str, Any]:
    actual = {name: sha256_file(path) for name, path in LOCKED_PATHS.items()}
    mismatches = {
        name: {"expected": LOCKED_INPUT_HASHES[name], "actual": value}
        for name, value in actual.items()
        if value != LOCKED_INPUT_HASHES[name]
    }
    if mismatches:
        raise RuntimeError(f"Locked input hash mismatch: {mismatches}")
    design = load_json(DESIGN_PATH)
    dataset = load_json(DATASET_PATH)
    automatic = load_json(AUTOMATIC_AUDIT_PATH)
    primary = load_json(PRIMARY_AI_AUDIT_PATH)
    adversarial = load_json(ADVERSARIAL_AI_AUDIT_PATH)
    audit_summary = load_json(AI_AUDIT_SUMMARY_PATH)
    amendment = load_json(IMPLEMENTATION_AMENDMENT_PATH)
    pairs = dataset.get("pairs", [])
    checks = {
        "calibration_pool_only": dataset.get("pool") == "calibration",
        "pair_count": dataset.get("pair_count") == EXPECTED_PAIR_COUNT and len(pairs) == EXPECTED_PAIR_COUNT,
        "prompt_count": dataset.get("prompt_count") == EXPECTED_PROMPT_COUNT,
        "template_count": dataset.get("template_count") == EXPECTED_TEMPLATE_COUNT,
        "families_exact": tuple(sorted({p["lexical_family"] for p in pairs})) == tuple(sorted(EXPECTED_FAMILIES)),
        "frames_exact": all(
            sum(p["template_family_id"] == tid for p in pairs) == EXPECTED_FRAMES_PER_TEMPLATE
            for tid in {p["template_family_id"] for p in pairs}
        ),
        "candidate_contract": all(
            c["correct_choice"] in ("A", "B")
            and c["alternative_choice"] in ("A", "B")
            and c["correct_choice"] != c["alternative_choice"]
            and len(c["correct_choice_token_ids"]) == 1
            and len(c["alternative_choice_token_ids"]) == 1
            for p in pairs for c in p["conditions"].values()
        ),
        "automatic_pass": automatic.get("all_checks_pass") is True,
        "primary_ai_pass": primary.get("status") == "PASS" and primary.get("pass_count") == EXPECTED_PAIR_COUNT,
        "adversarial_ai_pass": adversarial.get("status") == "PASS" and adversarial.get("pass_count") == EXPECTED_TEMPLATE_COUNT,
        "audit_summary_pass": audit_summary.get("status") == "automatic_primary_ai_and_adversarial_ai_audits_all_pass",
        "human_audit_disclosed": audit_summary.get("human_audit") == "not_performed" and audit_summary.get("human_audited_evidence") is False,
        "implementation_allowed": amendment["permissions"]["calibration_runner_implementation_allowed"] is True,
        "execution_still_closed": amendment["permissions"]["model_loading_allowed"] is False and amendment["permissions"]["gpu_forward_allowed"] is False,
        "result_dependent_change_forbidden": design["calibration"]["post_result_item_template_threshold_change_allowed"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Preflight semantic check failure: {checks}")
    return {
        "status": "preflight_pass_execution_requires_separate_authorization",
        "checks": checks,
        "locked_input_hashes": actual,
        "model_runtime_imported": False,
        "model_loaded": False,
        "model_forward_performed": False,
        "gpu_used": False,
        "liref_loaded": False,
        "candidate_components_accessed": False,
        "hidden_states_captured": False,
        "hooks_registered": False,
        "intervention_performed": False,
        "replication_pool_accessed": False,
        "human_audit": "not_performed",
        "human_audited_evidence": False,
        "execution_allowed": False,
    }


def validate_authorization(path: Path, run_id: str, implementation_hash: str, review_hash: str) -> dict[str, Any]:
    authorization = load_json(path.resolve())
    checks = {
        "status": authorization.get("status") == "execution_authorized",
        "scope": authorization.get("scope") == "stage_e_v4_behavioral_calibration_only",
        "execution_allowed": authorization.get("execution_allowed") is True,
        "run_id": authorization.get("run_id") == run_id,
        "device": str(authorization.get("device", "")).startswith("cuda:"),
        "batch_size": isinstance(authorization.get("batch_size"), int) and authorization["batch_size"] > 0,
        "dtype": authorization.get("dtype") == MODEL_DTYPE,
        "implementation_hash": authorization.get("implementation_sha256") == implementation_hash,
        "static_review_hash": authorization.get("static_review_sha256") == review_hash,
        "locked_inputs": authorization.get("locked_input_hashes") == LOCKED_INPUT_HASHES,
        "model_loading": authorization.get("model_loading_allowed") is True,
        "gpu_forward": authorization.get("gpu_forward_allowed") is True,
        "replication_closed": authorization.get("replication_pool_access_allowed") is False,
        "liref_closed": authorization.get("liref_loading_allowed") is False,
        "candidate_closed": authorization.get("candidate_component_access_allowed") is False,
        "hidden_closed": authorization.get("hidden_state_capture_allowed") is False,
        "hook_closed": authorization.get("hooks_allowed") is False,
        "intervention_closed": authorization.get("intervention_allowed") is False,
        "replication_execution_closed": authorization.get("independent_replication_execution_allowed") is False,
        "human_claim_closed": authorization.get("human_audited_evidence") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Execution authorization rejected: {checks}")
    return authorization


def validate_model_hashes() -> dict[str, Any]:
    expected = {**MODEL_SMALL_FILE_HASHES, **MODEL_PARAMETER_FILE_HASHES}
    actual = {name: sha256_file(MODEL_DIR / name) for name in expected}
    mismatches = {name: {"expected": expected[name], "actual": value} for name, value in actual.items() if value != expected[name]}
    if mismatches:
        raise RuntimeError(f"Model artifact hash mismatch: {mismatches}")
    return actual


def load_runtime() -> tuple[Any, Any, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return torch, transformers, (AutoModelForCausalLM, AutoTokenizer)


def normalize_generation(text: str) -> str:
    value = text.strip().upper()
    return value if value in ("A", "B") else ""


def registered_hook_count(model: Any) -> int:
    return sum(
        len(getattr(module, name, {}))
        for module in model.modules()
        for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")
    )


def evaluate_prompts(model: Any, tokenizer: Any, torch: Any, device: str, batch_size: int, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outputs_by_id: dict[str, dict[str, Any]] = {}
    tokenizer.padding_side = "left"
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            batch = records[offset:offset + batch_size]
            encoded = tokenizer([r["prompt"] for r in batch], add_special_tokens=True, padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(
                **encoded,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
            logits = output.logits[:, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            generated_ids = torch.argmax(logits, dim=-1)
            for row, record in enumerate(batch):
                correct_id = record["correct_token_id"]
                alternative_id = record["alternative_token_id"]
                correct_lp = float(log_probs[row, correct_id].item())
                alternative_lp = float(log_probs[row, alternative_id].item())
                generated_id = int(generated_ids[row].item())
                generated_text = tokenizer.decode([generated_id], skip_special_tokens=True)
                normalized = normalize_generation(generated_text)
                outputs_by_id[record["record_id"]] = {
                    "correct_choice": record["correct_choice"],
                    "alternative_choice": record["alternative_choice"],
                    "correct_token_id": correct_id,
                    "alternative_token_id": alternative_id,
                    "correct_log_probability": correct_lp,
                    "alternative_log_probability": alternative_lp,
                    "correct_probability": math.exp(correct_lp),
                    "alternative_probability": math.exp(alternative_lp),
                    "margin_nats": correct_lp - alternative_lp,
                    "forced_choice_correct": correct_lp > alternative_lp,
                    "generated_token_id": generated_id,
                    "generated_text": generated_text,
                    "normalized_generation": normalized,
                    "generation_valid_format": bool(normalized),
                    "generation_correct": normalized == record["correct_choice"],
                }
            del output, logits, log_probs, generated_ids
    return outputs_by_id


def summarize(pair_results: list[dict[str, Any]], design: dict[str, Any]) -> dict[str, Any]:
    criteria = design["calibration"]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_results:
        by_family[row["lexical_family"]].append(row)
    summaries: dict[str, Any] = {}
    passed, failed = [], []
    for family in EXPECTED_FAMILIES:
        rows = by_family[family]
        if len(rows) != 64:
            raise RuntimeError(f"Expected 64 pairs for {family}, found {len(rows)}")
        by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_template[row["template_family_id"]].append(row)
        if len(by_template) != 8 or any(len(v) != 8 for v in by_template.values()):
            raise RuntimeError(f"Invalid cluster structure for {family}")
        d_by_template = {
            template: statistics.fmean(
                item["conditions"]["arithmetic"]["margin_nats"] - item["conditions"]["selector"]["margin_nats"]
                for item in items
            )
            for template, items in sorted(by_template.items())
        }
        d_values = list(d_by_template.values())
        mean_d = statistics.fmean(d_values)
        dz, dz_valid, dz_reason = cluster_dz(d_values)
        condition_metrics: dict[str, Any] = {}
        for condition in ("arithmetic", "selector"):
            condition_rows = [row["conditions"][condition] for row in rows]
            fc_count = sum(bool(r["forced_choice_correct"]) for r in condition_rows)
            gen_count = sum(bool(r["generation_correct"]) for r in condition_rows)
            condition_metrics[condition] = {
                "denominator": 64,
                "forced_choice_correct_count": fc_count,
                "forced_choice_accuracy": fc_count / 64,
                "generation_correct_count": gen_count,
                "generation_accuracy": gen_count / 64,
                "generation_valid_format_count": sum(bool(r["generation_valid_format"]) for r in condition_rows),
                "mean_margin_nats": statistics.fmean(r["margin_nats"] for r in condition_rows),
                "mean_correct_probability": statistics.fmean(r["correct_probability"] for r in condition_rows),
            }
        afc = condition_metrics["arithmetic"]["forced_choice_correct_count"]
        sfc = condition_metrics["selector"]["forced_choice_correct_count"]
        agen = condition_metrics["arithmetic"]["generation_correct_count"]
        sgen = condition_metrics["selector"]["generation_correct_count"]
        fc_low, fc_high = criteria["forced_choice_correct_count_range_inclusive"]
        gen_low, gen_high = criteria["generation_correct_count_range_inclusive"]
        checks = {
            "arithmetic_forced_choice_range": fc_low <= afc <= fc_high,
            "selector_forced_choice_range": fc_low <= sfc <= fc_high,
            "forced_choice_condition_gap": abs(afc - sfc) <= criteria["forced_choice_max_condition_count_gap"],
            "arithmetic_generation_range": gen_low <= agen <= gen_high,
            "selector_generation_range": gen_low <= sgen <= gen_high,
            "generation_condition_gap": abs(agen - sgen) <= criteria["generation_max_condition_count_gap"],
            "mean_template_contrast": abs(mean_d) <= criteria["maximum_absolute_mean_template_contrast_nats"],
            "cluster_dz": dz_valid and dz is not None and abs(dz) <= criteria["maximum_absolute_cluster_dz"],
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        (passed if status == "PASS" else failed).append(family)
        summaries[family] = {
            "status": status,
            "checks": checks,
            "condition_metrics": condition_metrics,
            "forced_choice_count_gap": abs(afc - sfc),
            "generation_count_gap": abs(agen - sgen),
            "template_contrasts_nats": d_by_template,
            "mean_template_contrast_nats": mean_d,
            "cluster_dz": dz,
            "cluster_dz_valid": dz_valid,
            "cluster_dz_reason": dz_reason,
            "descriptive_cluster_bootstrap_mean_95ci": cluster_bootstrap_ci(d_values),
        }
    return {
        "schema_version": "4.0",
        "result_label": "stage_e_v4_behavioral_calibration",
        "status": "PASS" if passed else "FAIL",
        "passed_families": passed,
        "failed_families": failed,
        "points_balance_pass_required_for_primary_replication": True,
        "primary_replication_gate_open": "points_balance" in passed,
        "interaction_replication_gate_open": all(f in passed for f in EXPECTED_FAMILIES),
        "all_frozen_criteria_required": True,
        "result_dependent_exclusion_or_threshold_change_performed": False,
        "human_audit": "not_performed",
        "human_audited_evidence": False,
        "family_summaries": summaries,
    }


def execute(args: argparse.Namespace, preflight: dict[str, Any]) -> None:
    implementation_hash = sha256_file(Path(__file__).resolve())
    if not STATIC_REVIEW_PATH.exists():
        raise RuntimeError("Static review artifact is missing")
    static_review = load_json(STATIC_REVIEW_PATH)
    if static_review.get("all_checks_pass") is not True:
        raise RuntimeError("Static review is not PASS")
    review_hash = sha256_file(STATIC_REVIEW_PATH)
    authorization = validate_authorization(args.authorization, args.run_id, implementation_hash, review_hash)
    run_dir = args.output_root.resolve() / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "status.json", {"status": "running", "run_id": args.run_id})
    try:
        model_hashes = validate_model_hashes()
        torch, transformers, runtime_classes = load_runtime()
        AutoModelForCausalLM, AutoTokenizer = runtime_classes
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = authorization["device"]
        device_index = int(device.split(":", 1)[1])
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(f"Authorized device unavailable: {device}")
        torch.cuda.set_device(device_index)
        random.seed(BOOTSTRAP_SEED)
        torch.manual_seed(BOOTSTRAP_SEED)
        torch.cuda.manual_seed_all(BOOTSTRAP_SEED)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, trust_remote_code=False, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        model.config.output_hidden_states = False
        model.config.output_attentions = False
        config_checks = {
            "model_type": model.config.model_type == "llama",
            "hidden_size": model.config.hidden_size == 4096,
            "intermediate_size": model.config.intermediate_size == 14336,
            "num_hidden_layers": model.config.num_hidden_layers == 32,
            "num_attention_heads": model.config.num_attention_heads == 32,
            "num_key_value_heads": model.config.num_key_value_heads == 8,
            "registered_hook_count_zero": registered_hook_count(model) == 0,
            "output_hidden_states_false": model.config.output_hidden_states is False,
            "output_attentions_false": model.config.output_attentions is False,
        }
        if not all(config_checks.values()):
            raise RuntimeError(f"Model contract failure: {config_checks}")
        dataset = load_json(DATASET_PATH)
        records: list[dict[str, Any]] = []
        for pair in dataset["pairs"]:
            for condition in ("arithmetic", "selector"):
                payload = pair["conditions"][condition]
                records.append({
                    "record_id": f'{pair["pair_id"]}:{condition}',
                    "prompt": payload["full_prompt"],
                    "correct_choice": payload["correct_choice"],
                    "alternative_choice": payload["alternative_choice"],
                    "correct_token_id": int(payload["correct_choice_token_ids"][0]),
                    "alternative_token_id": int(payload["alternative_choice_token_ids"][0]),
                })
        scored = evaluate_prompts(model, tokenizer, torch, device, authorization["batch_size"], records)
        if registered_hook_count(model) != 0:
            raise RuntimeError("Unexpected hooks after evaluation")
        pair_results = []
        for pair in dataset["pairs"]:
            pair_results.append({
                "pair_id": pair["pair_id"],
                "lexical_family": pair["lexical_family"],
                "template_family_id": pair["template_family_id"],
                "frame_index": pair["frame_index"],
                "factors": pair["factors"],
                "conditions": {
                    condition: scored[f'{pair["pair_id"]}:{condition}']
                    for condition in ("arithmetic", "selector")
                },
            })
        summary = summarize(pair_results, load_json(DESIGN_PATH))
        environment = {
            "run_id": args.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
            "device_name": torch.cuda.get_device_name(device_index),
            "batch_size": authorization["batch_size"],
            "dtype": MODEL_DTYPE,
            "tf32": False,
            "model_hashes": model_hashes,
            "model_config_checks": config_checks,
            "registered_hook_count_final": registered_hook_count(model),
            "liref_loaded": False,
            "candidate_components_accessed": False,
            "hidden_states_captured": False,
            "hooks_registered": False,
            "intervention_performed": False,
            "replication_pool_accessed": False,
            "human_audit": "not_performed",
            "human_audited_evidence": False,
        }
        atomic_json(run_dir / "pair_results.json", {"pair_count": len(pair_results), "pairs": pair_results})
        atomic_json(run_dir / "summary.json", summary)
        atomic_json(run_dir / "environment.json", environment)
        output_hashes = {name: sha256_file(run_dir / name) for name in ("pair_results.json", "summary.json", "environment.json")}
        manifest = {
            "schema_version": "4.0",
            "run_id": args.run_id,
            "status": "complete",
            "scope": "stage_e_v4_behavioral_calibration_only",
            "implementation_sha256": implementation_hash,
            "static_review_sha256": review_hash,
            "authorization_sha256": sha256_file(args.authorization.resolve()),
            "locked_input_hashes": LOCKED_INPUT_HASHES,
            "output_hashes": output_hashes,
            "passed_families": summary["passed_families"],
            "failed_families": summary["failed_families"],
            "primary_replication_gate_open": summary["primary_replication_gate_open"],
            "interaction_replication_gate_open": summary["interaction_replication_gate_open"],
            "independent_replication_automatically_executed": False,
            "human_audited_evidence": False,
        }
        atomic_json(run_dir / "run_manifest.json", manifest)
        atomic_json(run_dir / "status.json", {"status": "complete", "run_id": args.run_id})
        print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))
    except Exception as error:
        atomic_json(run_dir / "status.json", {"status": "failed", "run_id": args.run_id, "error": repr(error)})
        raise


def main() -> None:
    args = parse_args()
    preflight = validate_locked_inputs()
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    execute(args, preflight)


if __name__ == "__main__":
    main()
