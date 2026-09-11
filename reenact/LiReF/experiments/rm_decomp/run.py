#!/usr/bin/env python3
"""Stage A: internal decomposition and localization of the LiReF R/M gap."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from core import (  # noqa: E402
    GROUP_MEMORY,
    GROUP_REASONING,
    PROMPT_TEMPLATE_ID,
    atomic_csv,
    atomic_json,
    atomic_torch,
    build_identity,
    canonical_hash,
    choose_sanity_indices,
    component_pass,
    direction_pass,
    environment_payload,
    finalize_errors,
    freeze_run_inputs,
    load_config,
    load_dataset_and_split,
    load_model_and_tokenizer,
    release_model,
    seed_everything,
    sha256_file,
    validate_model_contract,
)
from stats import (  # noqa: E402
    GroupMoments,
    assign_signed_ranks,
    benjamini_hochberg,
    safe_spearman,
    summarize_moments,
)


CONFOUND_LIMITATION = (
    "This stage localizes components associated with the R/M representation gap in the fixed "
    "MMLU-Pro split. It does not establish that the components implement reasoning or memorization "
    "rather than dataset, topic, lexical, length, numeric-content, or prompt-format differences. "
    "Controlled-input and causal intervention experiments are required for that stronger claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a"], default="a")
    parser.add_argument("--phase", choices=["sanity", "full"], required=True)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu_id is not None:
        config["gpu_id"] = args.gpu_id
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    unhashed = {key: value for key, value in config.items() if key not in {"config_path", "config_hash"}}
    config["config_hash"] = canonical_hash(unhashed)
    return config


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def ensure_environment(config: dict[str, Any], output_root: Path, identity: dict[str, Any], device: torch.device) -> None:
    payload = environment_payload(device)
    payload.update(
        {
            "identity_hash": identity["identity_hash"],
            "config_hash": config["config_hash"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    )
    path = output_root / "a_core" / "checkpoints" / "environment.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("Frozen environment.json differs from the current environment")
    else:
        atomic_json(path, payload)


def checkpoint_load(path: Path, identity: dict[str, Any], direction_sha: str | None = None) -> Any | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("identity_hash") != identity["identity_hash"]:
        raise RuntimeError(f"Stale checkpoint identity: {path}")
    if direction_sha is not None and payload.get("direction_artifact_sha256") != direction_sha:
        raise RuntimeError(f"Stale checkpoint direction hash: {path}")
    return payload["result"]


def checkpoint_save(
    path: Path,
    result: Any,
    identity: dict[str, Any],
    direction_sha: str | None = None,
) -> None:
    atomic_torch(
        path,
        {
            "identity_hash": identity["identity_hash"],
            "config_hash": identity["config_hash"],
            "direction_artifact_sha256": direction_sha,
            "result": result,
        },
    )


def compare_existing_cache(
    cache_path: Path,
    samples: dict[int, np.ndarray],
) -> dict[str, Any]:
    if not cache_path.exists():
        return {
            "available": False,
            "pass": False,
            "reason": f"Existing LiReF cache not found: {cache_path}",
        }
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        raise RuntimeError("Existing LiReF cache is not a layer dictionary")
    rows = sorted(samples)
    per_module = []
    all_cosines: list[float] = []
    all_max_abs: list[float] = []
    # Existing notebook cache k contains embedding at k=0 and module k-1 output at k>=1.
    for module_index in range(31):
        cache_index = module_index + 1
        if cache_index not in cache:
            raise KeyError(f"Existing cache missing layer key {cache_index}")
        cached = cache[cache_index][rows].float()
        fresh = torch.as_tensor(np.stack([samples[row][module_index] for row in rows])).float()
        cosine = F.cosine_similarity(cached, fresh, dim=-1)
        max_abs_by_sample = (cached - fresh).abs().amax(dim=-1)
        all_cosines.extend(cosine.tolist())
        all_max_abs.extend(max_abs_by_sample.tolist())
        per_module.append(
            {
                "module_index": module_index,
                "existing_cache_index": cache_index,
                "mean_cosine": float(cosine.mean()),
                "min_cosine": float(cosine.min()),
                "max_abs_error": float(max_abs_by_sample.max()),
            }
        )
        del cached, fresh
    return {
        "available": True,
        "n_samples": len(rows),
        "compared_module_indices": list(range(31)),
        "unavailable_module_index": 31,
        "reason_module_31_unavailable": "The old output_hidden_states cache stops at cache index 31, corresponding to decoder module 30.",
        "mean_cosine": float(np.mean(all_cosines)),
        "min_cosine": float(np.min(all_cosines)),
        "max_abs_error": float(np.max(all_max_abs)),
        "per_module": per_module,
    }


def run_sanity(
    config: dict[str, Any],
    data: dict[str, Any],
    identity: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> None:
    sanity_dir = output_root / "a_core" / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    indices = choose_sanity_indices(data, int(config["sanity_per_group"]))
    model = None
    status_path = sanity_dir / "stage_status.json"
    try:
        model, tokenizer = load_model_and_tokenizer(config, device)
        contract = validate_model_contract(model, tokenizer, config)
        directions = direction_pass(
            model,
            tokenizer,
            device,
            data,
            indices,
            int(config["batch_size"]),
            float(config["direction_epsilon"]),
            capture_samples=False,
        )
        components = component_pass(
            model,
            tokenizer,
            device,
            data,
            indices,
            int(config["batch_size"]),
            directions["unit_directions"],
            sanity_manual_z=True,
            capture_samples=True,
        )
        errors = finalize_errors(components["reconstruction"])
        sanity_head_stats = summarize_moments(
            GroupMoments.from_state_dict(components["head_moments"]),
            float(config["statistics_epsilon"]),
        )
        sanity_neuron_stats = summarize_moments(
            GroupMoments.from_state_dict(components["neuron_moments"]),
            float(config["statistics_epsilon"]),
        )
        finite_stat_fractions = {
            "head_welch_p": float(np.isfinite(sanity_head_stats["welch_p"]).mean()),
            "neuron_welch_p": float(np.isfinite(sanity_neuron_stats["welch_p"]).mean()),
        }
        cache_alignment = compare_existing_cache(Path(config["existing_cache_path"]), components["sample_h_out"])
        cache_alignment["pass"] = bool(
            cache_alignment.get("available")
            and cache_alignment.get("min_cosine", -math.inf) >= float(config["tolerances"]["cache_cosine_min"])
        )
        checks = {
            "layer_vector": errors["layer_vector"]["max_abs_error"] <= config["tolerances"]["layer_vector_max_abs"],
            "layer_scalar": errors["layer_scalar"]["max_abs_error"] <= config["tolerances"]["layer_scalar_max_abs"],
            "head_scalar": errors["head_scalar"]["max_abs_error"] <= config["tolerances"]["head_scalar_max_abs"],
            "ffn_scalar": errors["ffn_scalar"]["max_abs_error"] <= config["tolerances"]["ffn_scalar_max_abs"],
            "ffn_vector": errors["ffn_vector"]["max_abs_error"] <= config["tolerances"]["ffn_vector_max_abs"],
            "gated_activation": errors["gated_activation"]["max_abs_error"] <= config["tolerances"]["gated_activation_max_abs"],
            "cache_alignment": cache_alignment["pass"],
            "dtype_float32": next(model.parameters()).dtype == torch.float32,
            "all_directions_valid": bool(np.all(directions["valid_direction"])),
            "balanced_sanity_subset": bool(
                int((data["labels"][indices] == GROUP_MEMORY).sum()) == int(config["sanity_per_group"])
                and int((data["labels"][indices] == GROUP_REASONING).sum()) == int(config["sanity_per_group"])
            ),
            "statistics_finite_fraction": bool(
                finite_stat_fractions["head_welch_p"] >= 0.99
                and finite_stat_fractions["neuron_welch_p"] >= 0.99
            ),
        }
        passed = all(bool(value) for value in checks.values())
        atomic_json(sanity_dir / "model_contract.json", jsonable(contract))
        atomic_json(
            sanity_dir / "reconstruction_checks.json",
            jsonable({"errors": errors, "finite_stat_fractions": finite_stat_fractions, "checks": checks}),
        )
        atomic_json(sanity_dir / "cache_alignment.json", jsonable(cache_alignment))
        atomic_json(
            sanity_dir / "sanity_samples.json",
            {
                "row_indices": indices.tolist(),
                "question_ids": [data["question_ids"][int(index)] for index in indices],
                "n_memory": int((data["labels"][indices] == GROUP_MEMORY).sum()),
                "n_reasoning": int((data["labels"][indices] == GROUP_REASONING).sum()),
            },
        )
        status = {
            "phase": "sanity",
            "status": "PASS" if passed else "FAIL",
            "identity_hash": identity["identity_hash"],
            "config_hash": config["config_hash"],
            "checks": checks,
            "confound_limitation": CONFOUND_LIMITATION,
        }
        atomic_json(status_path, status)
        if not passed:
            raise RuntimeError(f"Sanity gate failed: {checks}")
        print(f"SANITY PASS: {status_path}")
    except Exception as exc:
        if not status_path.exists():
            atomic_json(
                status_path,
                {
                    "phase": "sanity",
                    "status": "ERROR",
                    "identity_hash": identity["identity_hash"],
                    "config_hash": config["config_hash"],
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    finally:
        release_model(model)


def enforce_sanity_gate(output_root: Path, identity: dict[str, Any]) -> None:
    status_path = output_root / "a_core" / "sanity" / "stage_status.json"
    if not status_path.exists():
        raise RuntimeError("Full run blocked: sanity/stage_status.json does not exist")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "PASS":
        raise RuntimeError(f"Full run blocked: sanity status is {status.get('status')}")
    if status.get("identity_hash") != identity["identity_hash"]:
        raise RuntimeError("Full run blocked: sanity identity hash does not match current inputs/code")
    if status.get("config_hash") != identity["config_hash"]:
        raise RuntimeError("Full run blocked: sanity config hash does not match")


def component_frame(state: dict[str, np.ndarray], component_type: str, epsilon: float) -> pd.DataFrame:
    moments = GroupMoments.from_state_dict(state)
    summary = summarize_moments(moments, epsilon)
    shape = summary["Delta"].shape
    layer_grid, index_grid = np.indices(shape)
    flat: dict[str, Any] = {
        "component_type": np.repeat(component_type, np.prod(shape)),
        "module_index": layer_grid.ravel(),
        "component_index": index_grid.ravel(),
    }
    prefix = "H" if component_type == "head" else "N"
    flat["component_id"] = [
        f"L{layer:02d}{prefix}{index:05d}" for layer, index in zip(layer_grid.ravel(), index_grid.ravel())
    ]
    for key, values in summary.items():
        flat[key] = np.asarray(values).ravel()
    return pd.DataFrame(flat)


def attach_neuron_activations(frame: pd.DataFrame, result: dict[str, Any]) -> pd.DataFrame:
    counts = result["activation_counts"]
    means = result["activation_sums"] / counts[:, None, None]
    out = frame.copy()
    out["memory_activation_mean"] = means[GROUP_MEMORY].ravel()
    out["reasoning_activation_mean"] = means[GROUP_REASONING].ravel()
    out["activation_Delta"] = out["reasoning_activation_mean"] - out["memory_activation_mean"]
    return out


def layer_tables(result: dict[str, Any], directions: dict[str, Any], split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit = directions["unit_directions"]
    raw = directions["raw_directions"]
    eps = 1e-12
    vector_rows = []
    scalar_rows = []
    for layer in range(unit.shape[0]):
        vectors = {name: result["vector_deltas"][name][layer] for name in ("input", "attention", "mlp", "output")}
        reconstruction = vectors["input"] + vectors["attention"] + vectors["mlp"]
        output = vectors["output"]
        input_norm = float(np.linalg.norm(vectors["input"]))
        output_norm = float(np.linalg.norm(output))
        cosine = math.nan
        if input_norm > eps and output_norm > eps:
            cosine = float(np.dot(vectors["input"], output) / (input_norm * output_norm))
            cosine = float(np.clip(cosine, -1.0, 1.0))
        vector_rows.append(
            {
                "split": split_name,
                "module_index": layer,
                "input_gap_norm": input_norm,
                "attention_gap_norm": float(np.linalg.norm(vectors["attention"])),
                "mlp_gap_norm": float(np.linalg.norm(vectors["mlp"])),
                "output_gap_norm": output_norm,
                "discovery_direction_norm": float(np.linalg.norm(raw[layer])),
                "input_output_cosine": cosine,
                "input_output_rotation_degrees": math.degrees(math.acos(cosine)) if math.isfinite(cosine) else math.nan,
                "attention_parallel_to_liref": float(np.dot(vectors["attention"], unit[layer])),
                "attention_orthogonal_norm": float(np.linalg.norm(vectors["attention"] - np.dot(vectors["attention"], unit[layer]) * unit[layer])),
                "mlp_parallel_to_liref": float(np.dot(vectors["mlp"], unit[layer])),
                "mlp_orthogonal_norm": float(np.linalg.norm(vectors["mlp"] - np.dot(vectors["mlp"], unit[layer]) * unit[layer])),
                "vector_reconstruction_max_abs": float(np.max(np.abs(output - reconstruction))),
                "vector_reconstruction_l2": float(np.linalg.norm(output - reconstruction)),
            }
        )
        projections = {name: float(np.dot(value, unit[layer])) for name, value in vectors.items()}
        scalar_rows.append(
            {
                "split": split_name,
                "module_index": layer,
                "input_projection_Delta": projections["input"],
                "attention_projection_Delta": projections["attention"],
                "mlp_projection_Delta": projections["mlp"],
                "output_projection_Delta": projections["output"],
                "scalar_reconstruction_error": projections["output"] - projections["input"] - projections["attention"] - projections["mlp"],
                "valid_direction": bool(directions["valid_direction"][layer]),
            }
        )
    return pd.DataFrame(vector_rows), pd.DataFrame(scalar_rows)


def select_candidates(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selected: dict[str, set[str]] = {}
    for _, row in frame.iterrows():
        sources = set()
        if pd.notna(row["rank_global"]) and int(row["rank_global"]) <= int(config["topk_global_per_sign"]):
            sources.add("global")
        if pd.notna(row["rank_layer"]) and int(row["rank_layer"]) <= int(config["topk_per_layer_per_sign"]):
            sources.add("per_layer")
        if sources:
            selected[row["component_id"]] = sources
    result = frame[frame["component_id"].isin(selected)].copy()
    result["selection_source"] = result["component_id"].map(lambda key: "+".join(sorted(selected[key])))
    result["detailed_candidate"] = (
        result["rank_global"].notna()
        & (result["rank_global"].astype("Int64") <= int(config["topk_detailed_per_sign"]))
    )
    return result


def validate_candidates(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    candidates = select_candidates(discovery, config)
    dcols = {column: f"{column}_discovery" for column in candidates.columns if column not in {"component_type", "module_index", "component_index", "component_id", "selection_source", "detailed_candidate"}}
    vcols = {column: f"{column}_validation" for column in validation.columns if column not in {"component_type", "module_index", "component_index", "component_id"}}
    left = candidates.rename(columns=dcols)
    right = validation.rename(columns=vcols)
    joined = left.merge(right, on=["component_type", "module_index", "component_index", "component_id"], how="left", validate="one_to_one")
    joined["bh_q_validation"] = benjamini_hochberg(joined["welch_p_validation"].to_numpy())
    joined["fdr_family_size"] = len(joined)
    joined["fdr_valid_p_count"] = int(np.isfinite(joined["welch_p_validation"]).sum())
    joined["same_sign"] = np.sign(joined["Delta_discovery"]) == np.sign(joined["Delta_validation"])
    joined["reproduced"] = joined["same_sign"] & (joined["bh_q_validation"] < float(config["fdr_alpha"]))
    return joined


def ranking_stability(discovery: pd.DataFrame, validation: pd.DataFrame, component_type: str, config: dict[str, Any]) -> pd.DataFrame:
    joined = discovery[["component_id", "Delta"]].merge(
        validation[["component_id", "Delta"]], on="component_id", suffixes=("_discovery", "_validation"), validate="one_to_one"
    )
    rho, p_value, reason = safe_spearman(joined["Delta_discovery"].to_numpy(), joined["Delta_validation"].to_numpy())
    rows = []
    for sign in ("positive", "negative"):
        ascending = sign == "negative"
        d = discovery[discovery["sign_group"] == sign].sort_values("Delta", ascending=ascending, kind="mergesort")
        v = validation[validation["sign_group"] == sign].sort_values("Delta", ascending=ascending, kind="mergesort")
        for k in sorted({int(config["topk_detailed_per_sign"]), int(config["topk_global_per_sign"]) } ):
            ds = set(d.head(k)["component_id"])
            vs = set(v.head(k)["component_id"])
            rows.append(
                {
                    "component_type": component_type,
                    "sign_group": sign,
                    "top_k": k,
                    "overlap_count": len(ds & vs),
                    "overlap_fraction": len(ds & vs) / max(k, 1),
                    "spearman_rho_all_components": rho,
                    "spearman_p_all_components": p_value,
                    "spearman_na_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def save_figures(
    figures_dir: Path,
    layer_vector: pd.DataFrame,
    layer_scalar: pd.DataFrame,
    head_candidates: pd.DataFrame,
    neuron_candidates: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figures_dir / f"{name}.png", dpi=180)
        plt.savefig(figures_dir / f"{name}.pdf")
        plt.close()

    plt.figure(figsize=(8, 4.5))
    for split, group in layer_vector.groupby("split"):
        plt.plot(group["module_index"], group["output_gap_norm"], marker="o", ms=3, label=split)
    plt.xlabel("Decoder module index")
    plt.ylabel("||mean(R) - mean(M)||")
    plt.title("Layerwise R/M output-gap norm")
    plt.legend()
    save("layer_vector_geometry")

    discovery = layer_scalar[layer_scalar["split"] == "discovery"]
    plt.figure(figsize=(9, 4.5))
    plt.plot(discovery["module_index"], discovery["input_projection_Delta"], label="residual carry")
    plt.plot(discovery["module_index"], discovery["attention_projection_Delta"], label="attention addition")
    plt.plot(discovery["module_index"], discovery["mlp_projection_Delta"], label="MLP addition")
    plt.plot(discovery["module_index"], discovery["output_projection_Delta"], label="layer output", lw=2)
    plt.axhline(0, color="black", lw=0.7)
    plt.xlabel("Decoder module index")
    plt.ylabel("R-M projection difference")
    plt.title("Scalar LiReF decomposition (Discovery)")
    plt.legend(ncol=2)
    save("layer_scalar_decomposition")

    for name, candidates, title in (
        ("head_candidate_validation", head_candidates, "Attention-head candidate validation"),
        ("neuron_candidate_validation", neuron_candidates, "FFN-neuron candidate validation"),
    ):
        display = candidates.sort_values("abs_Delta_discovery", ascending=False).head(30)
        x = np.arange(len(display))
        plt.figure(figsize=(12, 5))
        plt.bar(x - 0.2, display["Delta_discovery"], width=0.4, label="Discovery")
        plt.bar(x + 0.2, display["Delta_validation"], width=0.4, label="Validation")
        plt.xticks(x, display["component_id"], rotation=75, ha="right", fontsize=7)
        plt.axhline(0, color="black", lw=0.7)
        plt.ylabel("R-M contribution difference")
        plt.title(title)
        plt.legend()
        save(name)

    plt.figure(figsize=(7, 4.5))
    for component_type, group in stability.groupby("component_type"):
        plt.plot(group["top_k"].astype(str) + "-" + group["sign_group"], group["overlap_fraction"], marker="o", label=component_type)
    plt.ylim(0, 1)
    plt.ylabel("Discovery/Validation top-K overlap")
    plt.xlabel("K-sign")
    plt.title("Candidate ranking stability")
    plt.legend()
    save("ranking_stability")


def run_full(
    config: dict[str, Any],
    data: dict[str, Any],
    identity: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> None:
    enforce_sanity_gate(output_root, identity)
    root = output_root / "a_core"
    checkpoints = root / "checkpoints"
    tables = root / "tables"
    figures = root / "figures"
    manifests = root / "manifests"
    for path in (checkpoints, tables, figures, manifests):
        path.mkdir(parents=True, exist_ok=True)
    status_path = root / "stage_status.json"
    model = None
    try:
        model, tokenizer = load_model_and_tokenizer(config, device)
        contract = validate_model_contract(model, tokenizer, config)

        direction_path = checkpoints / "discovery_liref_directions.pt"
        directions = checkpoint_load(direction_path, identity)
        if directions is None:
            directions = direction_pass(
                model, tokenizer, device, data, data["indices"]["discovery"], int(config["batch_size"]),
                float(config["direction_epsilon"]), capture_samples=False,
            )
            directions["sample_h_out"] = {}
            checkpoint_save(direction_path, directions, identity)
        direction_sha = sha256_file(direction_path)

        split_results: dict[str, Any] = {}
        for split_name in ("discovery", "validation"):
            checkpoint_path = checkpoints / f"{split_name}_component_decomposition.pt"
            result = checkpoint_load(checkpoint_path, identity, direction_sha)
            if result is None:
                result = component_pass(
                    model,
                    tokenizer,
                    device,
                    data,
                    data["indices"][split_name],
                    int(config["batch_size"]),
                    directions["unit_directions"],
                    sanity_manual_z=False,
                    capture_samples=False,
                )
                result["sample_h_out"] = {}
                checkpoint_save(checkpoint_path, result, identity, direction_sha)
            split_results[split_name] = result

        vector_artifact = {
            "identity_hash": identity["identity_hash"],
            "config_hash": config["config_hash"],
            "direction_artifact_sha256": direction_sha,
            "definition": "Each tensor is mean(Reasoning) - mean(Memory), indexed [decoder_module, hidden_dimension].",
            "discovery_liref_raw_direction": torch.from_numpy(directions["raw_directions"]),
            "discovery_liref_unit_direction": torch.from_numpy(directions["unit_directions"]),
            "splits": {
                split: {name: torch.from_numpy(values) for name, values in result["vector_deltas"].items()}
                for split, result in split_results.items()
            },
        }
        vector_path = root / "layer_mean_difference_vectors.pt"
        atomic_torch(vector_path, vector_artifact)

        vector_frames, scalar_frames = [], []
        component_frames: dict[str, dict[str, pd.DataFrame]] = {"head": {}, "neuron": {}}
        for split, result in split_results.items():
            vector_frame, scalar_frame = layer_tables(result, directions, split)
            vector_frames.append(vector_frame)
            scalar_frames.append(scalar_frame)
            head = assign_signed_ranks(component_frame(result["head_moments"], "head", float(config["statistics_epsilon"])))
            neuron = attach_neuron_activations(
                assign_signed_ranks(component_frame(result["neuron_moments"], "neuron", float(config["statistics_epsilon"]))),
                result,
            )
            component_frames["head"][split] = head
            component_frames["neuron"][split] = neuron
            atomic_csv(tables / f"{split}_head_statistics.csv.gz", head)
            atomic_csv(tables / f"{split}_neuron_statistics.csv.gz", neuron)

        layer_vector = pd.concat(vector_frames, ignore_index=True)
        layer_scalar = pd.concat(scalar_frames, ignore_index=True)
        atomic_csv(tables / "layer_vector_geometry.csv", layer_vector)
        atomic_csv(tables / "layer_scalar_decomposition.csv", layer_scalar)

        candidate_tables: dict[str, pd.DataFrame] = {}
        stability_frames = []
        for component_type in ("head", "neuron"):
            discovery = component_frames[component_type]["discovery"]
            validation = component_frames[component_type]["validation"]
            candidate_tables[component_type] = validate_candidates(discovery, validation, config)
            atomic_csv(tables / f"{component_type}_candidate_validation.csv", candidate_tables[component_type])
            stability_frames.append(ranking_stability(discovery, validation, component_type, config))
        stability = pd.concat(stability_frames, ignore_index=True)
        atomic_csv(tables / "ranking_stability.csv", stability)

        candidate_manifest = {
            "run_id": config["run_id"],
            "identity_hash": identity["identity_hash"],
            "config_hash": config["config_hash"],
            "dataset_sha256": identity["dataset_sha256"],
            "split_sha256": identity["split_sha256"],
            "split_seed": identity["split_seed"],
            "prompt_template_id": PROMPT_TEMPLATE_ID,
            "prompt_template_sha256": identity["prompt_template_sha256"],
            "direction_artifact": str(direction_path),
            "direction_artifact_sha256": direction_sha,
            "layer_vector_artifact": str(vector_path),
            "layer_vector_artifact_sha256": sha256_file(vector_path),
            "selection_rule": {
                "discovery_only": True,
                "topk_global_per_sign": config["topk_global_per_sign"],
                "topk_per_layer_per_sign": config["topk_per_layer_per_sign"],
                "topk_detailed_per_sign": config["topk_detailed_per_sign"],
                "validation_candidates_reselected": False,
                "validation_bh_fdr_alpha": config["fdr_alpha"],
                "bh_applied_only_to_finite_p_values": True,
            },
            "candidate_counts": {
                key: {
                    "total": len(value),
                    "finite_validation_p": int(np.isfinite(value["welch_p_validation"]).sum()),
                    "reproduced": int(value["reproduced"].sum()),
                }
                for key, value in candidate_tables.items()
            },
            "candidates": {
                key: value[["component_id", "module_index", "component_index", "sign_group_discovery", "selection_source", "detailed_candidate"]].to_dict("records")
                for key, value in candidate_tables.items()
            },
            "confound_limitation": CONFOUND_LIMITATION,
        }
        atomic_json(manifests / "candidate_manifest.json", jsonable(candidate_manifest))
        atomic_json(manifests / "model_contract.json", jsonable(contract))

        save_figures(figures, layer_vector, layer_scalar, candidate_tables["head"], candidate_tables["neuron"], stability)

        reconstruction = {split: finalize_errors(result["reconstruction"]) for split, result in split_results.items()}
        summary = {
            "stage": "A — Internal Decomposition & Localization",
            "status": "PASS",
            "identity_hash": identity["identity_hash"],
            "config_hash": config["config_hash"],
            "n_discovery": len(data["indices"]["discovery"]),
            "n_validation": len(data["indices"]["validation"]),
            "group_counts": {
                split: {
                    "memory": int((data["labels"][indices] == GROUP_MEMORY).sum()),
                    "reasoning": int((data["labels"][indices] == GROUP_REASONING).sum()),
                }
                for split, indices in data["indices"].items()
            },
            "all_discovery_directions_valid": bool(np.all(directions["valid_direction"])),
            "reconstruction": reconstruction,
            "candidate_counts": candidate_manifest["candidate_counts"],
            "artifacts": {
                "layer_vectors": str(vector_path),
                "tables": str(tables),
                "figures": str(figures),
                "candidate_manifest": str(manifests / "candidate_manifest.json"),
            },
            "interpretation_scope": (
                "Stage A identifies where and through which residual, attention, MLP, head, and neuron outputs "
                "the fixed-dataset R/M mean-representation gap is associated. It is localization/correlation, not causal proof."
            ),
            "confound_limitation": CONFOUND_LIMITATION,
            "next_stage": "B characterizes the fixed Stage-A candidates; causal validation and controlled-input tests remain separate requirements.",
        }
        atomic_json(root / "summary.json", jsonable(summary))
        atomic_json(status_path, {"status": "PASS", "identity_hash": identity["identity_hash"], "config_hash": config["config_hash"]})
        print(f"FULL PASS: {root / 'summary.json'}")
    except Exception as exc:
        atomic_json(
            status_path,
            {
                "status": "ERROR",
                "identity_hash": identity["identity_hash"],
                "config_hash": config["config_hash"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        release_model(model)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config.resolve()), args)
    output_root = Path(config["output_root"])
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / "a_core" / ".matplotlib"))
    seed_everything(int(config["split_seed"]))
    data = load_dataset_and_split(config)
    identity = build_identity(config, data, SCRIPT_DIR)
    freeze_run_inputs(output_root, config, identity)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Stage A run")
    device = torch.device(f"cuda:{int(config['gpu_id'])}")
    ensure_environment(config, output_root, identity, device)
    if args.phase == "sanity":
        run_sanity(config, data, identity, output_root, device)
    else:
        run_full(config, data, identity, output_root, device)


if __name__ == "__main__":
    main()
