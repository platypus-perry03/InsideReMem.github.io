#!/usr/bin/env python3
"""Limited same-sample Stage E Pilot v3.2.

The runner stores scalar readouts only. Runtime ML libraries, the frozen LiReF
direction, the model, and read-only capture hooks are unavailable until an
explicit --execute request passes a separate hash-locked authorization.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
STAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "AI" / "reenact" / "models" / "Meta-Llama-3-8B"
OUTPUT_ROOT = ROOT / "AI" / "reenact" / "liref_outputs" / "rm_decomp" / "v3_2"

DESIGN_PATH = STAGE_DIR / "stage_e_limited_pilot_v3_2_design_frozen.json"
AMENDMENT_PATH = STAGE_DIR / "stage_e_pilot_continuation_amendment_v3_2_frozen.json"
CANDIDATE_MANIFEST_PATH = STAGE_DIR / "stage_e_pilot_v3_2_candidate_manifest.json"
DATASET_PATH = STAGE_DIR / "calibration_v3_assets" / "calibration_v3_dataset_draft.json"
BASELINE_RESULTS_PATH = (
    ROOT
    / "AI"
    / "reenact"
    / "liref_outputs"
    / "rm_decomp"
    / "v3"
    / "calv3_1_baseline_20260830_02"
    / "baseline_pair_results.json"
)
DIRECTION_PATH = (
    ROOT
    / "AI"
    / "reenact"
    / "liref_outputs"
    / "rm_decomp"
    / "v2"
    / "a_core"
    / "checkpoints"
    / "discovery_liref_directions.pt"
)
STAGE_A_CANDIDATE_PATH = (
    ROOT
    / "AI"
    / "reenact"
    / "liref_outputs"
    / "rm_decomp"
    / "v2"
    / "a_core"
    / "manifests"
    / "candidate_manifest.json"
)
STAGE_C_DESIGN_PATH = (
    ROOT
    / "AI"
    / "reenact"
    / "liref_outputs"
    / "rm_decomp"
    / "v2"
    / "c_causal_v2_c01"
    / "manifests"
    / "causal_design.json"
)
STATIC_REVIEW_PATH = (
    STAGE_DIR / "stage_e_limited_pilot_v3_2_static_safety_review.json"
)

LOCKED_INPUT_HASHES = {
    "design": "b8b74d59975230589adae9ec91eeb2e98cefcb80e818f41f15bad45e3f58e3b3",
    "amendment": "3e9d14c676c96ee6d950761d39faa176c0e9267d613856d1bd57ff826d41e5fe",
    "candidate_manifest": "e9d20904967512fa73581d528eb3026a3142f1a5d2c8b0f878e847dbc7eeb233",
    "dataset": "d2187c0623ba9752776cf0251dee3dabf9d80ac04e339cf3eb4bd1d1b42761a1",
    "baseline_pair_results": "c4562fcfb109083b8c501ac90af64ae5a8f6e6f7f33485619a86366aaec78e6a",
    "discovery_liref_directions": "55647779ecf44a33143f66800af9ae3b2767d34b99b8877abd3711b6bba6adf6",
    "stage_a_candidate_manifest": "244b7397790fc71224ed77aafb4b4a1f267cd2dbebe67336b8560d67acbb52b9",
    "stage_c_causal_design": "96db9e6dae8d1a6ec75ac7933b19ac657b41a1306260948dd8d4e4ce7e6cd697",
}
LOCKED_PATHS = {
    "design": DESIGN_PATH,
    "amendment": AMENDMENT_PATH,
    "candidate_manifest": CANDIDATE_MANIFEST_PATH,
    "dataset": DATASET_PATH,
    "baseline_pair_results": BASELINE_RESULTS_PATH,
    "discovery_liref_directions": DIRECTION_PATH,
    "stage_a_candidate_manifest": STAGE_A_CANDIDATE_PATH,
    "stage_c_causal_design": STAGE_C_DESIGN_PATH,
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
CONDITIONS = ("arithmetic", "selector")
PRIMARY_ENDPOINTS = (
    "layer31_liref_projection",
    "L31N13336_contribution",
    "L29H00030_contribution",
    "L30H00006_contribution",
    "L29H00031_contribution",
)
SECONDARY_SCALARS = (
    "L31N13336_signed_z",
    "L31N13336_absolute_z",
    "L29H00030_pre_o_l2_norm",
    "L30H00006_pre_o_l2_norm",
    "L29H00031_pre_o_l2_norm",
)
EXPECTED_PAIR_COUNT = 192
EXPECTED_PROMPT_COUNT = 384
EXPECTED_TEMPLATE_COUNT = 24
EXPECTED_FRAMES_PER_TEMPLATE = 8
BOOTSTRAP_REPETITIONS = 10000
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
        parser.error("Preflight-only mode does not accept authorization or run ID")
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
    if not values:
        raise ValueError("Cannot compute percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def sample_dz(values: list[float]) -> tuple[float | None, str]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        return None, "insufficient_or_nonfinite"
    mean_value = statistics.fmean(values)
    if all(value == 0.0 for value in values):
        return 0.0, "all_exactly_zero"
    sample_sd = statistics.stdev(values)
    if sample_sd <= 2.220446049250313e-16:
        return (0.0, "near_zero_sd_zero_mean") if mean_value == 0.0 else (None, "near_zero_sd_nonzero_mean")
    return mean_value / sample_sd, "valid"


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 3:
        return None
    ranks_a = average_ranks(values_a)
    ranks_b = average_ranks(values_b)
    mean_a = statistics.fmean(ranks_a)
    mean_b = statistics.fmean(ranks_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    denominator = math.sqrt(
        sum((a - mean_a) ** 2 for a in ranks_a)
        * sum((b - mean_b) ** 2 for b in ranks_b)
    )
    return None if denominator == 0.0 else numerator / denominator


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
    candidates = load_json(CANDIDATE_MANIFEST_PATH)
    dataset = load_json(DATASET_PATH)
    baseline = load_json(BASELINE_RESULTS_PATH)
    dataset_ids = {pair["pair_id"] for pair in dataset["pairs"]}
    baseline_ids = {pair["pair_id"] for pair in baseline["results"]}
    checks = {
        "design_frozen": design["status"] == "design_frozen_implementation_allowed_model_execution_not_authorized",
        "amendment_frozen": amendment["status"] == "limited_same_sample_pilot_continuation_frozen_model_execution_not_authorized",
        "candidate_set_exact": [row["component_id"] for row in candidates["candidates"]]
        == ["L31N13336", "L29H00030", "L30H00006", "L29H00031"],
        "candidate_reselection_closed": candidates["candidate_addition_or_reselection_allowed"] is False,
        "dataset_counts": dataset["pair_count"] == EXPECTED_PAIR_COUNT
        and dataset["prompt_count"] == EXPECTED_PROMPT_COUNT
        and len(dataset["pairs"]) == EXPECTED_PAIR_COUNT,
        "baseline_counts": baseline["pair_count"] == EXPECTED_PAIR_COUNT
        and len(baseline["results"]) == EXPECTED_PAIR_COUNT,
        "pair_ids_match": dataset_ids == baseline_ids and len(dataset_ids) == EXPECTED_PAIR_COUNT,
        "all_pairs_primary": design["population"]["include_all_pairs"] is True,
        "correct_only_closed": design["population"]["correct_only_primary_filter_allowed"] is False,
        "execution_closed": design["current_permissions"]["pilot_model_loading_allowed"] is False
        and design["current_permissions"]["pilot_gpu_forward_allowed"] is False,
        "intervention_closed": design["current_permissions"]["intervention_patching_suppression_allowed"] is False,
        "scalar_only": design["output_contract"]["scalar_outputs_only"] is True
        and design["output_contract"]["save_raw_hidden_state_tensors"] is False,
        "statistics_exact": design["statistics"]["bootstrap_repetitions"] == BOOTSTRAP_REPETITIONS
        and design["statistics"]["bootstrap_seed"] == BOOTSTRAP_SEED,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Preflight semantic check failure: {checks}")
    return {
        "status": "preflight_pass_execution_still_requires_separate_authorization",
        "checks": checks,
        "locked_input_hashes": actual_hashes,
        "pair_count": EXPECTED_PAIR_COUNT,
        "prompt_count": EXPECTED_PROMPT_COUNT,
        "model_runtime_imported": False,
        "direction_runtime_loaded": False,
        "model_loaded": False,
        "model_forward_performed": False,
        "gpu_used": False,
        "pilot_execution_allowed": False,
        "intervention_performed": False,
    }


def validate_execution_authorization(
    path: Path, run_id: str, implementation_sha256: str, static_review_sha256: str
) -> dict[str, Any]:
    authorization = load_json(path.resolve())
    checks = {
        "status": authorization.get("status") == "execution_authorized",
        "scope": authorization.get("scope") == "limited_same_sample_pilot_v3_2_only",
        "execution": authorization.get("execution_allowed") is True,
        "run_id": authorization.get("run_id") == run_id,
        "device": isinstance(authorization.get("device"), str)
        and authorization["device"].startswith("cuda:"),
        "batch_size": authorization.get("batch_size") == 8,
        "dtype": authorization.get("dtype") == MODEL_DTYPE,
        "implementation": authorization.get("implementation_sha256") == implementation_sha256,
        "static_review": authorization.get("static_review_sha256") == static_review_sha256,
        "inputs": authorization.get("locked_input_hashes") == LOCKED_INPUT_HASHES,
        "model_loading": authorization.get("model_loading_allowed") is True,
        "gpu_forward": authorization.get("gpu_forward_allowed") is True,
        "direction": authorization.get("load_frozen_liref_direction_allowed") is True,
        "scalar_capture": authorization.get("read_only_scalar_capture_allowed") is True,
        "raw_tensor_closed": authorization.get("save_raw_state_tensors") is False,
        "candidate_search_closed": authorization.get("candidate_addition_or_reselection") is False,
        "intervention_closed": authorization.get("intervention_patching_or_suppression") is False,
        "confirmatory_closed": authorization.get("confirmatory_claim_allowed") is False,
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
    return actual


def pair_prompt_rows(
    prompt_scalars: list[dict[str, Any]],
    dataset: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_prompt_keys = {
        (pair["pair_id"], condition)
        for pair in dataset["pairs"]
        for condition in CONDITIONS
    }
    actual_prompt_keys = {
        (row["pair_id"], row["condition"]) for row in prompt_scalars
    }
    if (
        len(prompt_scalars) != EXPECTED_PROMPT_COUNT
        or len(actual_prompt_keys) != EXPECTED_PROMPT_COUNT
        or actual_prompt_keys != expected_prompt_keys
    ):
        raise RuntimeError("Prompt scalar keys are missing, duplicated, or unexpected")
    prompt_lookup = {
        (row["pair_id"], row["condition"]): row for row in prompt_scalars
    }
    if len(baseline["results"]) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("Baseline pair result count mismatch")
    baseline_lookup = {row["pair_id"]: row for row in baseline["results"]}
    if set(baseline_lookup) != {pair["pair_id"] for pair in dataset["pairs"]}:
        raise RuntimeError("Baseline pair keys are missing, duplicated, or unexpected")
    results = []
    for pair in dataset["pairs"]:
        pair_id = pair["pair_id"]
        condition_rows = {
            condition: prompt_lookup[(pair_id, condition)] for condition in CONDITIONS
        }
        row = {
            "pair_id": pair_id,
            "lexical_family": pair["lexical_family"],
            "template_family_id": pair["template_family_id"],
            "frame_index": pair["frame_index"],
            "operation": pair["operation"],
            "arithmetic_generation_correct": baseline_lookup[pair_id]["conditions"]["arithmetic"]["generation_correct"],
            "arithmetic_forced_choice_correct": baseline_lookup[pair_id]["conditions"]["arithmetic"]["forced_choice_correct"],
            "arithmetic_margin_nats": baseline_lookup[pair_id]["conditions"]["arithmetic"]["margin_nats"],
        }
        for endpoint in PRIMARY_ENDPOINTS + SECONDARY_SCALARS:
            row[endpoint] = (
                condition_rows["arithmetic"][endpoint]
                - condition_rows["selector"][endpoint]
            )
        results.append(row)
    if len(results) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("Pair difference count mismatch")
    return results


def template_cluster_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(pair_rows) != EXPECTED_PAIR_COUNT or len({row["pair_id"] for row in pair_rows}) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("Primary pair population must contain all 192 unique pairs")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["lexical_family"], row["template_family_id"])].append(row)
    output = []
    for (family, template_id), rows in sorted(grouped.items()):
        if len(rows) != EXPECTED_FRAMES_PER_TEMPLATE:
            raise RuntimeError(f"Template {template_id} does not have 8 frames")
        record = {
            "lexical_family": family,
            "template_family_id": template_id,
            "pair_count": len(rows),
            "mean_arithmetic_margin_nats": statistics.fmean(
                row["arithmetic_margin_nats"] for row in rows
            ),
        }
        for endpoint in PRIMARY_ENDPOINTS + SECONDARY_SCALARS:
            record[endpoint] = statistics.fmean(row[endpoint] for row in rows)
        output.append(record)
    if len(output) != EXPECTED_TEMPLATE_COUNT:
        raise RuntimeError("Template cluster count mismatch")
    return output


def bootstrap_indices(family_sizes: dict[str, int]) -> list[dict[str, list[int]]]:
    rng = random.Random(BOOTSTRAP_SEED)
    return [
        {
            family: [rng.randrange(size) for _ in range(size)]
            for family, size in family_sizes.items()
        }
        for _ in range(BOOTSTRAP_REPETITIONS)
    ]


def scalar_effect_summary(values: list[float], bootstrap_means: list[float]) -> dict[str, Any]:
    mean_value = statistics.fmean(values)
    dz, dz_reason = sample_dz(values)
    ci = [percentile(bootstrap_means, 0.025), percentile(bootstrap_means, 0.975)]
    return {
        "cluster_count": len(values),
        "mean_effect": mean_value,
        "cluster_dz": dz,
        "cluster_dz_reason": dz_reason,
        "positive_cluster_count": sum(value > 0 for value in values),
        "negative_cluster_count": sum(value < 0 for value in values),
        "zero_cluster_count": sum(value == 0 for value in values),
        "cluster_bootstrap_95ci": ci,
        "pilot_signal_ci_excludes_zero": ci[0] > 0.0 or ci[1] < 0.0,
        "confirmatory_significance_claim_allowed": False,
    }


def aggregate_primary(template_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in template_rows:
        by_family[row["lexical_family"]].append(row)
    if tuple(sorted(by_family)) != tuple(sorted(EXPECTED_FAMILIES)):
        raise RuntimeError("Unexpected families in template rows")
    if any(len(by_family[family]) != 8 for family in EXPECTED_FAMILIES):
        raise RuntimeError("Each family must contain exactly 8 template clusters")
    samples = bootstrap_indices({family: 8 for family in EXPECTED_FAMILIES})
    family_output: dict[str, Any] = {family: {} for family in EXPECTED_FAMILIES}
    overall_output: dict[str, Any] = {}
    interaction_output: dict[str, Any] = {}
    family_pairs = (
        ("object_count", "points_balance"),
        ("object_count", "temperature"),
        ("points_balance", "temperature"),
    )
    for endpoint in PRIMARY_ENDPOINTS:
        bootstrap_family_means: dict[str, list[float]] = {
            family: [] for family in EXPECTED_FAMILIES
        }
        bootstrap_overall = []
        bootstrap_interactions = {
            f"{left}_minus_{right}": [] for left, right in family_pairs
        }
        for replicate in samples:
            replicate_means = {}
            for family in EXPECTED_FAMILIES:
                rows = by_family[family]
                mean_value = statistics.fmean(
                    rows[index][endpoint] for index in replicate[family]
                )
                bootstrap_family_means[family].append(mean_value)
                replicate_means[family] = mean_value
            bootstrap_overall.append(
                statistics.fmean(replicate_means[family] for family in EXPECTED_FAMILIES)
            )
            for left, right in family_pairs:
                bootstrap_interactions[f"{left}_minus_{right}"].append(
                    replicate_means[left] - replicate_means[right]
                )
        family_means = {}
        for family in EXPECTED_FAMILIES:
            values = [row[endpoint] for row in by_family[family]]
            family_output[family][endpoint] = scalar_effect_summary(
                values, bootstrap_family_means[family]
            )
            family_means[family] = statistics.fmean(values)
        all_values = [row[endpoint] for row in template_rows]
        equal_family_mean = statistics.fmean(family_means.values())
        simple_cluster_mean = statistics.fmean(all_values)
        if not math.isclose(equal_family_mean, simple_cluster_mean, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("Overall aggregation equivalence check failed")
        overall_output[endpoint] = {
            **scalar_effect_summary(all_values, bootstrap_overall),
            "equal_weight_family_mean": equal_family_mean,
            "simple_24_cluster_mean": simple_cluster_mean,
            "aggregation_equivalence_check": True,
        }
        interaction_output[endpoint] = {}
        for left, right in family_pairs:
            label = f"{left}_minus_{right}"
            distribution = bootstrap_interactions[label]
            ci = [percentile(distribution, 0.025), percentile(distribution, 0.975)]
            interaction_output[endpoint][label] = {
                "effect_difference": family_means[left] - family_means[right],
                "cluster_bootstrap_95ci": ci,
                "pilot_signal_ci_excludes_zero": ci[0] > 0.0 or ci[1] < 0.0,
                "exploratory_derived_directional_hypothesis": (
                    endpoint == "L31N13336_contribution"
                    and label in ("object_count_minus_temperature", "points_balance_minus_temperature")
                ),
                "confirmatory_interaction_claim_allowed": False,
            }
    return {
        "family_effects": family_output,
        "overall_effects": overall_output,
        "interaction_effects": interaction_output,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "hard_pass_fail_applied": False,
    }


def grouped_secondary_means(
    pair_rows: list[dict[str, Any]], grouping: str
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    scopes = {"overall": pair_rows}
    scopes.update(
        {
            family: [row for row in pair_rows if row["lexical_family"] == family]
            for family in EXPECTED_FAMILIES
        }
    )
    for scope, rows in scopes.items():
        output[scope] = {}
        levels = sorted({str(row[grouping]) for row in rows})
        for level in levels:
            selected = [row for row in rows if str(row[grouping]) == level]
            output[scope][level] = {
                "pair_count": len(selected),
                "endpoint_means": {
                    endpoint: statistics.fmean(row[endpoint] for row in selected)
                    for endpoint in PRIMARY_ENDPOINTS
                },
                "p_values_computed": False,
            }
    return output


def secondary_diagnostics(
    pair_rows: list[dict[str, Any]], template_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    correlation_output: dict[str, Any] = {}
    scopes = {"overall": template_rows}
    scopes.update(
        {
            family: [row for row in template_rows if row["lexical_family"] == family]
            for family in EXPECTED_FAMILIES
        }
    )
    for scope, rows in scopes.items():
        margins = [row["mean_arithmetic_margin_nats"] for row in rows]
        correlation_output[scope] = {
            endpoint: spearman(margins, [row[endpoint] for row in rows])
            for endpoint in PRIMARY_ENDPOINTS
        }
    return {
        "operation": grouped_secondary_means(pair_rows, "operation"),
        "arithmetic_generation_correct": grouped_secondary_means(
            pair_rows, "arithmetic_generation_correct"
        ),
        "arithmetic_forced_choice_correct": grouped_secondary_means(
            pair_rows, "arithmetic_forced_choice_correct"
        ),
        "template_cluster_spearman_arithmetic_margin_vs_internal_effect": correlation_output,
        "secondary_results_may_replace_primary": False,
        "secondary_p_values_computed": False,
        "candidate_selection_from_secondary_allowed": False,
    }


def load_runtime() -> tuple[Any, Any, Any, Any]:
    import numpy
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return numpy, torch, transformers, (AutoModelForCausalLM, AutoTokenizer)


def registered_hook_count(model: Any) -> int:
    total = 0
    for module in model.modules():
        total += len(getattr(module, "_forward_hooks", {}))
        total += len(getattr(module, "_forward_pre_hooks", {}))
        total += len(getattr(module, "_backward_hooks", {}))
    return total


class ScalarCapture:
    """Read-only hooks that retain scalar Python lists, never raw state tensors."""

    def __init__(self, model: Any, torch: Any, directions: Any) -> None:
        self.model = model
        self.torch = torch
        self.handles: list[Any] = []
        self.values: dict[str, list[float]] = {}
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        self.directions = {
            layer: torch.as_tensor(directions[layer], dtype=dtype, device=device)
            for layer in (29, 30, 31)
        }
        layer31 = model.model.layers[31]
        self.neuron_projection = torch.dot(
            layer31.mlp.down_proj.weight[:, 13336], self.directions[31]
        )
        self.head_projections = {}
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        for component_id, layer_index, head_index in (
            ("L29H00030", 29, 30),
            ("L30H00006", 30, 6),
            ("L29H00031", 29, 31),
        ):
            weight = model.model.layers[layer_index].self_attn.o_proj.weight
            block = weight[:, head_index * head_dim : (head_index + 1) * head_dim]
            self.head_projections[component_id] = torch.mv(
                block.T, self.directions[layer_index]
            )

    @staticmethod
    def _python_scalars(tensor: Any) -> list[float]:
        return [float(value) for value in tensor.detach().float().cpu().tolist()]

    def reset(self) -> None:
        self.values = {}

    def install(self) -> None:
        if self.handles:
            raise RuntimeError("Scalar capture hooks already installed")
        self.handles.append(
            self.model.model.layers[31].register_forward_hook(self._layer31_hook())
        )
        self.handles.append(
            self.model.model.layers[29].self_attn.o_proj.register_forward_pre_hook(
                self._attention_hook(29, (("L29H00030", 30), ("L29H00031", 31)))
            )
        )
        self.handles.append(
            self.model.model.layers[30].self_attn.o_proj.register_forward_pre_hook(
                self._attention_hook(30, (("L30H00006", 6),))
            )
        )
        self.handles.append(
            self.model.model.layers[31].mlp.down_proj.register_forward_pre_hook(
                self._neuron_hook()
            )
        )

    def _layer31_hook(self):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            scalar = self.torch.mv(tensor[:, -1, :], self.directions[31])
            self.values["layer31_liref_projection"] = self._python_scalars(scalar)

        return hook

    def _attention_hook(self, layer_index: int, candidates: tuple[tuple[str, int], ...]):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            tensor = args[0][:, -1, :]
            head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
            heads = tensor.reshape(-1, self.model.config.num_attention_heads, head_dim)
            for component_id, head_index in candidates:
                state = heads[:, head_index, :]
                contribution = self.torch.mv(state, self.head_projections[component_id])
                norm = self.torch.linalg.vector_norm(state.float(), dim=-1)
                self.values[f"{component_id}_contribution"] = self._python_scalars(contribution)
                self.values[f"{component_id}_pre_o_l2_norm"] = self._python_scalars(norm)

        return hook

    def _neuron_hook(self):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            z = args[0][:, -1, 13336]
            contribution = z * self.neuron_projection
            self.values["L31N13336_contribution"] = self._python_scalars(contribution)
            self.values["L31N13336_signed_z"] = self._python_scalars(z)
            self.values["L31N13336_absolute_z"] = self._python_scalars(z.abs())

        return hook

    def batch_rows(self, expected_batch_size: int) -> dict[str, list[float]]:
        expected = set(PRIMARY_ENDPOINTS + SECONDARY_SCALARS)
        if set(self.values) != expected:
            raise RuntimeError(f"Incomplete scalar capture: {sorted(self.values)}")
        if any(len(values) != expected_batch_size for values in self.values.values()):
            raise RuntimeError("Scalar capture batch-size mismatch")
        return {name: list(values) for name, values in self.values.items()}

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


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
    numpy, torch, transformers, runtime_classes = load_runtime()
    AutoModelForCausalLM, AutoTokenizer = runtime_classes
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = authorization["device"]
    device_index = int(device.split(":", 1)[1])
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"Authorized CUDA device does not exist: {device}")
    torch.cuda.set_device(device_index)
    random.seed(BOOTSTRAP_SEED)
    numpy.random.seed(BOOTSTRAP_SEED)
    torch.manual_seed(BOOTSTRAP_SEED)
    torch.cuda.manual_seed_all(BOOTSTRAP_SEED)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    direction_payload = torch.load(
        DIRECTION_PATH, map_location="cpu", weights_only=False
    )
    directions = numpy.asarray(
        direction_payload["result"]["unit_directions"], dtype=numpy.float64
    )
    if directions.shape != (32, 4096) or not numpy.all(numpy.isfinite(directions)):
        raise RuntimeError(f"Unexpected frozen direction array: {directions.shape}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
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
    baseline_hook_count = registered_hook_count(model)
    if baseline_hook_count != 0:
        raise RuntimeError("Unexpected pre-existing model hooks")
    config_checks = {
        "model_type": model.config.model_type == "llama",
        "layers": model.config.num_hidden_layers == 32,
        "hidden": model.config.hidden_size == 4096,
        "intermediate": model.config.intermediate_size == 14336,
        "heads": model.config.num_attention_heads == 32,
        "kv_heads": model.config.num_key_value_heads == 8,
        "left_padding": tokenizer.padding_side == "left",
        "output_hidden_states_false": model.config.output_hidden_states is False,
        "output_attentions_false": model.config.output_attentions is False,
    }
    if not all(config_checks.values()):
        raise RuntimeError(f"Model contract mismatch: {config_checks}")

    dataset = load_json(DATASET_PATH)
    baseline = load_json(BASELINE_RESULTS_PATH)
    prompt_specs = []
    for pair in dataset["pairs"]:
        for condition in CONDITIONS:
            prompt_specs.append(
                {
                    "pair_id": pair["pair_id"],
                    "condition": condition,
                    "lexical_family": pair["lexical_family"],
                    "template_family_id": pair["template_family_id"],
                    "frame_index": pair["frame_index"],
                    "operation": pair["operation"],
                    "prompt": pair["conditions"][condition]["full_prompt"],
                }
            )
    capture = ScalarCapture(model, torch, directions)
    prompt_scalars = []
    capture.install()
    try:
        with torch.inference_mode():
            for start in range(0, len(prompt_specs), authorization["batch_size"]):
                batch = prompt_specs[start : start + authorization["batch_size"]]
                encoded = tokenizer(
                    [row["prompt"] for row in batch],
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
                capture.reset()
                model(
                    **encoded,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
                scalar_batch = capture.batch_rows(len(batch))
                for index, spec in enumerate(batch):
                    row = {name: value for name, value in spec.items() if name != "prompt"}
                    for endpoint in PRIMARY_ENDPOINTS + SECONDARY_SCALARS:
                        row[endpoint] = scalar_batch[endpoint][index]
                    prompt_scalars.append(row)
                del encoded, scalar_batch
    finally:
        capture.remove()
    if registered_hook_count(model) != baseline_hook_count:
        raise RuntimeError("Read-only capture hooks were not fully removed")
    if len(prompt_scalars) != EXPECTED_PROMPT_COUNT:
        raise RuntimeError("Prompt scalar row count mismatch")
    if any(
        not math.isfinite(row[endpoint])
        for row in prompt_scalars
        for endpoint in PRIMARY_ENDPOINTS + SECONDARY_SCALARS
    ):
        raise RuntimeError("Non-finite scalar output")

    pair_rows = pair_prompt_rows(prompt_scalars, dataset, baseline)
    template_rows = template_cluster_rows(pair_rows)
    primary = aggregate_primary(template_rows)
    secondary = secondary_diagnostics(pair_rows, template_rows)
    output_paths = {
        "prompt_scalars": run_dir / "prompt_scalars.json",
        "pair_differences": run_dir / "pair_differences.json",
        "template_cluster_effects": run_dir / "template_cluster_effects.json",
        "family_effects": run_dir / "family_effects.json",
        "overall_effects": run_dir / "overall_effects.json",
        "interaction_effects": run_dir / "interaction_effects.json",
        "secondary_diagnostics": run_dir / "secondary_diagnostics.json",
    }
    atomic_json(output_paths["prompt_scalars"], {"row_count": len(prompt_scalars), "rows": prompt_scalars})
    atomic_json(output_paths["pair_differences"], {"row_count": len(pair_rows), "rows": pair_rows})
    atomic_json(output_paths["template_cluster_effects"], {"row_count": len(template_rows), "rows": template_rows})
    atomic_json(output_paths["family_effects"], primary["family_effects"])
    atomic_json(output_paths["overall_effects"], primary["overall_effects"])
    atomic_json(output_paths["interaction_effects"], primary["interaction_effects"])
    atomic_json(output_paths["secondary_diagnostics"], secondary)

    summary = {
        "schema_version": "3.2",
        "status": "complete_limited_same_sample_pilot_no_pass_fail_gate",
        "evidence_class": "limited_same_sample_pilot",
        "independent_or_confirmatory": False,
        "hard_pass_fail_applied": False,
        "primary_population_pair_count": len(pair_rows),
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "family_effects": primary["family_effects"],
        "overall_effects": primary["overall_effects"],
        "interaction_effects": primary["interaction_effects"],
        "behavioral_calibration_outcome": {
            "passed_families": [],
            "failed_families": list(EXPECTED_FAMILIES),
            "behavioral_equivalence_achieved": False,
        },
        "intervention_performed": False,
        "claim_limits": load_json(DESIGN_PATH)["claim_limits"],
    }
    summary_path = run_dir / "pilot_summary.json"
    atomic_json(summary_path, summary)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "batch_size": authorization["batch_size"],
        "dtype": MODEL_DTYPE,
        "random_seed": BOOTSTRAP_SEED,
        "model_hashes": model_hashes,
        "model_config_checks": config_checks,
        "preflight": preflight,
        "frozen_direction_sha256": sha256_file(DIRECTION_PATH),
        "direction_reestimated": False,
        "read_only_capture_hook_count": 4,
        "hooks_removed_after_capture": registered_hook_count(model) == baseline_hook_count,
        "raw_state_tensors_saved": False,
        "candidate_reselection_performed": False,
        "intervention_patching_or_suppression_performed": False,
        "gpu_used": True,
    }
    environment_path = run_dir / "environment_and_safety.json"
    atomic_json(environment_path, environment)
    manifest = {
        "schema_version": "3.2",
        "status": "complete",
        "run_id": args.run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "limited_same_sample_pilot",
        "independent_or_confirmatory": False,
        "implementation_sha256": implementation_sha256,
        "static_review_sha256": static_review_sha256,
        "execution_authorization_sha256": sha256_file(args.authorization.resolve()),
        "locked_input_hashes": LOCKED_INPUT_HASHES,
        "output_sha256": {name: sha256_file(path) for name, path in output_paths.items()},
        "summary_sha256": sha256_file(summary_path),
        "environment_sha256": sha256_file(environment_path),
        "intervention_allowed_from_this_manifest": False,
        "confirmatory_claim_allowed": False,
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
        run_dir = args.output_root.resolve() / args.run_id
        if run_dir.is_dir():
            atomic_json(
                run_dir / "status.json",
                {
                    "status": "failed",
                    "run_id": args.run_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "scientific_result_claim_allowed": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
