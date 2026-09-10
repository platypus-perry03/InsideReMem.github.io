#!/usr/bin/env python3
"""LiReF Stage B: frozen-candidate feature sensitivity characterization."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from torch.nn import functional as F
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage_b_core import (  # noqa: E402
    AtomicCsvSink,
    CONTROLLED_COLUMNS,
    FeatureExtractor,
    StageBCapture,
    atomic_csv,
    build_controls,
    code_checksum,
    component_lookup,
    frozen_candidates,
    frozen_provenance,
    jsonable,
    load_dataset_and_split,
    load_directions,
    load_model_and_tokenizer,
    load_stage_a_assets,
    model_parameter_checksum,
    projection_metadata,
    projections_for_components,
    read_json,
    release_model,
    require_status,
    sha256_file,
    token_semantic_role,
    unique_components,
    validate_backend_tolerances,
    validate_confirmatory_power,
    validate_controlled_manifest,
    validate_feature_schema,
    write_json,
    write_status,
)
from stage_b_stats import benjamini_hochberg, paired_summary, specificity_summary  # noqa: E402


INTERPRETATION_BOUNDARY = (
    "These results characterize feature sensitivity in frozen Meta-Llama-3-8B Base, "
    "the frozen MMLU-Pro/controlled data, the question-only prompt, and the last prompt token. "
    "They do not establish a reasoning/memorization component or a causal mechanism."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "prepare",
            "sanity",
            "natural",
            "freeze_hypotheses",
            "generate_pilot",
            "approve_pilot",
            "pilot",
            "freeze_confirmatory",
            "confirmatory",
            "report",
        ],
    )
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(path)
    config["config_path"] = str(path.resolve())
    if args.gpu_id is not None:
        config["gpu_id"] = args.gpu_id
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    config["config_hash"] = __import__("hashlib").sha256(
        json.dumps(
            {key: value for key, value in config.items() if key not in {"config_path", "config_hash"}},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return config


def paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["output_root"])
    return {
        "root": root,
        "manifests": root / "manifests",
        "tables": root / "tables",
        "sanity": root / "sanity",
        "cards": root / "candidate_cards",
        "logs": root / "logs",
        "candidates": root / "manifests" / "frozen_stage_b_candidates.json",
        "controls": root / "manifests" / "frozen_control_components.json",
        "feature_schema": root / "manifests" / "feature_schema.json",
        "hypotheses": root / "manifests" / "feature_hypothesis_manifest.json",
        "pilot_pairs": root / "manifests" / "controlled_pair_manifest_pilot.csv",
        "confirmatory_pairs": root / "manifests" / "controlled_pair_manifest.csv",
        "confirmatory_design_draft": root / "manifests" / "confirmatory_design_draft.json",
        "confirmatory_design": root / "manifests" / "confirmatory_design.json",
        "provenance": root / "manifests" / "provenance.json",
    }


def ensure_dirs(all_paths: dict[str, Path]) -> None:
    for name in ("root", "manifests", "tables", "sanity", "cards", "logs"):
        all_paths[name].mkdir(parents=True, exist_ok=True)


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def load_frozen_components(all_paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    candidates = read_json(all_paths["candidates"])
    controls = read_json(all_paths["controls"])
    lookup = component_lookup(candidates, controls)
    return candidates, controls, lookup


def run_prepare(config: dict[str, Any], all_paths: dict[str, Path]) -> None:
    validate_backend_tolerances(config)
    validate_confirmatory_power(config, require_count=False)
    schema_source = Path(config["feature_schema_path"])
    schema = read_json(schema_source)
    validate_feature_schema(schema)
    assets = load_stage_a_assets(config)
    candidates = frozen_candidates(config, assets)
    frozen_config = all_paths["manifests"] / "frozen_config.json"
    if frozen_config.exists() and read_json(frozen_config) != config:
        raise RuntimeError("Existing frozen Stage B config differs; use a new stage_b_run_id/output_root")
    write_json(frozen_config, config)
    if all_paths["feature_schema"].exists() and sha256_file(all_paths["feature_schema"]) != sha256_file(schema_source):
        raise RuntimeError("Existing frozen feature schema differs; use a new Stage B experiment")
    shutil.copyfile(schema_source, all_paths["feature_schema"])

    candidate_payload = {
        "stage_b_run_id": config["stage_b_run_id"],
        "candidate_source": str(assets["manifest_path"]),
        "selection_rule": "detailed_candidate == true",
        "validation_metadata_used_for_selection": False,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    write_json(all_paths["candidates"], candidate_payload)

    device = torch.device(f"cuda:{int(config['gpu_id'])}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage B requires a CUDA GPU")
    model = None
    try:
        model, _ = load_model_and_tokenizer(config, device)
        if model.training:
            raise RuntimeError("Model must remain in eval mode")
        directions = load_directions(assets["direction_path"])
        head_norms, neuron_abs = projection_metadata(model, directions)
        controls = build_controls(config, assets, candidates, head_norms, neuron_abs)
    finally:
        release_model(model)
    controls.update(
        {
            "stage_b_run_id": config["stage_b_run_id"],
            "candidate_manifest_sha256": sha256_file(all_paths["candidates"]),
            "candidate_ranking_modified": False,
        }
    )
    write_json(all_paths["controls"], controls)

    draft = {**config["confirmatory_power"], "approved": False, "reviewer_id": "", "freeze_timestamp": ""}
    if not all_paths["confirmatory_design_draft"].exists():
        write_json(all_paths["confirmatory_design_draft"], draft)
    provenance = frozen_provenance(config, assets, all_paths["feature_schema"], controls_path=all_paths["controls"])
    write_json(all_paths["provenance"], provenance)
    write_status(
        all_paths["root"],
        "prepare",
        "PASS",
        stage_b_run_id=config["stage_b_run_id"],
        config_hash=config["config_hash"],
        candidate_count=len(candidates),
        control_count=len(controls["controls"]),
        feature_schema_sha256=sha256_file(all_paths["feature_schema"]),
        frozen_candidates_sha256=sha256_file(all_paths["candidates"]),
        frozen_controls_sha256=sha256_file(all_paths["controls"]),
        model_parameter_checksum=provenance["model_parameter_checksum"],
        interpretation_boundary=INTERPRETATION_BOUNDARY,
    )
    print(f"prepare PASS: {len(candidates)} candidates, {len(controls['controls'])} control associations")


def encode(tokenizer: Any, prompts: list[str], device: torch.device, offsets: bool) -> tuple[dict[str, torch.Tensor], Any]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        truncation=False,
        return_token_type_ids=False,
        return_offsets_mapping=offsets,
    )
    offset_mapping = encoded.pop("offset_mapping", None)
    if not bool(torch.all(encoded["attention_mask"][:, -1] == 1)):
        raise RuntimeError("Index -1 must be the final prompt token")
    return {key: value.to(device) for key, value in encoded.items()}, offset_mapping


@torch.inference_mode()
def run_sanity(config: dict[str, Any], all_paths: dict[str, Path]) -> None:
    require_status(all_paths["root"], "prepare")
    assets = load_stage_a_assets(config)
    candidates_payload, _, lookup = load_frozen_components(all_paths)
    candidate_components = {
        row["component_id"]: {
            "component_id": row["component_id"],
            "component_type": row["component_type"],
            "module_index": row["module_index"],
            "component_index": row["component_index"],
        }
        for row in candidates_payload["candidates"]
    }
    data = load_dataset_and_split(config)
    discovery = data["indices"]["discovery"]
    labels = data["labels"]
    memory = discovery[labels[discovery] == 0][:2]
    reasoning = discovery[labels[discovery] == 1][:2]
    indices = np.sort(np.concatenate([memory, reasoning]))
    prompts = [data["prompts"][int(index)] for index in indices]
    device = torch.device(f"cuda:{int(config['gpu_id'])}")
    directions = load_directions(assets["direction_path"])
    parameter_before, _ = model_parameter_checksum(Path(config["model_path"]))
    model = None
    capture = None
    try:
        model, tokenizer = load_model_and_tokenizer(config, device)
        projections = projections_for_components(model, directions, candidate_components)
        capture = StageBCapture(model, candidate_components, capture_sources=True)
        capture.install()
        encoded, _ = encode(tokenizer, prompts, device, offsets=False)

        capture.reset()
        baseline = model(**encoded, use_cache=False, output_attentions=False, return_dict=True)
        capture.validate()
        baseline_logits = baseline.logits[:, -1, :].detach().float().clone()
        baseline_hidden = {layer: value.float().clone() for layer, value in capture.h_out.items()}
        baseline_heads = {layer: value.float().clone() for layer, value in capture.pre_o.items()}

        capture.reset()
        eager = model(**encoded, use_cache=False, output_attentions=True, return_dict=True)
        capture.validate()
        eager_logits = eager.logits[:, -1, :].detach().float()

        logit_error = float((baseline_logits - eager_logits).abs().max())
        hidden_errors = []
        hidden_cosines = []
        for layer, baseline_value in baseline_hidden.items():
            eager_value = capture.h_out[layer].float()
            hidden_errors.append(float((baseline_value - eager_value).abs().max()))
            hidden_cosines.extend(F.cosine_similarity(baseline_value, eager_value, dim=-1).tolist())

        head_errors = []
        source_errors = []
        source_rows = []
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        groups = model.config.num_attention_heads // model.config.num_key_value_heads
        for component_id, row in candidate_components.items():
            if row["component_type"] != "head":
                continue
            layer = int(row["module_index"])
            head = int(row["component_index"])
            q = projections[component_id].float()
            baseline_pre = baseline_heads[layer].reshape(len(indices), model.config.num_attention_heads, head_dim)[:, head]
            eager_pre = capture.pre_o[layer].float().reshape(len(indices), model.config.num_attention_heads, head_dim)[:, head]
            baseline_total = (baseline_pre * q).sum(dim=-1)
            eager_total = (eager_pre * q).sum(dim=-1)
            head_error = (eager_total - baseline_total).abs()
            head_errors.extend(head_error.tolist())
            values = capture.values[layer].float().reshape(len(indices), -1, model.config.num_key_value_heads, head_dim)
            values = values[:, :, head // groups, :]
            attention = eager.attentions[layer][:, head, -1, :].float()
            source = attention * torch.einsum("bsd,d->bs", values, q)
            source_error = (source.sum(dim=-1) - eager_total).abs()
            source_errors.extend(source_error.tolist())
            source_rows.append(
                {
                    "component_id": component_id,
                    "head_backend_max_abs_error": float(head_error.max()),
                    "source_reconstruction_max_abs_error": float(source_error.max()),
                }
            )
        tolerance = config["backend_tolerances"]
        metrics = {
            "logit_max_abs_error": logit_error,
            "hidden_state_max_abs_error": max(hidden_errors),
            "hidden_state_min_cosine": min(hidden_cosines),
            "head_reconstruction_mean_abs_error": float(np.mean(head_errors)),
            "head_reconstruction_max_abs_error": float(np.max(head_errors)),
            "source_reconstruction_mean_abs_error": float(np.mean(source_errors)),
            "source_reconstruction_max_abs_error": float(np.max(source_errors)),
        }
        checks = {
            "logits": metrics["logit_max_abs_error"] <= tolerance["logit_max_abs_tolerance"],
            "hidden_max_abs": metrics["hidden_state_max_abs_error"] <= tolerance["hidden_state_max_abs_tolerance"],
            "hidden_cosine": metrics["hidden_state_min_cosine"] >= tolerance["hidden_state_cosine_tolerance"],
            "head_mean": metrics["head_reconstruction_mean_abs_error"] <= tolerance["head_reconstruction_mean_tolerance"],
            "head_max": metrics["head_reconstruction_max_abs_error"] <= tolerance["head_reconstruction_max_tolerance"],
            "source_mean": metrics["source_reconstruction_mean_abs_error"] <= tolerance["source_reconstruction_mean_tolerance"],
            "source_max": metrics["source_reconstruction_max_abs_error"] <= tolerance["source_reconstruction_max_tolerance"],
            "gqa_mapping": groups == 4,
            "left_padding_last_token": bool(torch.all(encoded["attention_mask"][:, -1] == 1)),
            "model_eval": not model.training,
        }
        parameter_after, _ = model_parameter_checksum(Path(config["model_path"]))
        checks["parameter_checksum_unchanged"] = parameter_before == parameter_after
        payload = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "indices": indices.tolist(),
            "attention_backend_normal": model.config._attn_implementation,
            "attention_backend_source": "eager via output_attentions=True",
            "tolerances": tolerance,
            "metrics": metrics,
            "checks": checks,
            "per_head": source_rows,
            "model_parameter_checksum_before": parameter_before,
            "model_parameter_checksum_after": parameter_after,
        }
        write_json(all_paths["sanity"] / "backend_equivalence.json", payload)
        write_status(all_paths["root"], "sanity", payload["status"], **{key: value for key, value in payload.items() if key != "status"})
        if payload["status"] != "PASS":
            raise RuntimeError(f"Backend/source decomposition sanity failed: {checks}")
        print("sanity PASS", json.dumps(metrics, indent=2))
    finally:
        if capture is not None:
            capture.remove()
        release_model(model)


def response_fieldnames() -> list[str]:
    return [
        "analysis_split", "row_index", "question_id", "label", "category", "token_length",
        "numeric_span_count", "has_numeric", "relation_span_count", "has_relation",
        "operator_span_count", "has_operator", "candidate_id", "component_id", "component_type",
        "component_role", "control_kind", "module_index", "component_index", "activation",
        "projection", "total_contribution",
    ]


def source_fieldnames() -> list[str]:
    return [
        "analysis_split", "row_index", "question_id", "label", "category", "candidate_id",
        "module_index", "component_index", "source_token_index", "token_id", "token_text",
        "char_start", "char_end", "source_span_id", "semantic_role", "feature_family",
        "attention_weight", "source_contribution", "span_contribution", "selected_reason",
    ]


@torch.inference_mode()
def run_natural(config: dict[str, Any], all_paths: dict[str, Path]) -> None:
    require_status(all_paths["root"], "prepare")
    require_status(all_paths["root"], "sanity")
    assets = load_stage_a_assets(config)
    candidates_payload, _, lookup = load_frozen_components(all_paths)
    components = unique_components(lookup)
    schema = read_json(all_paths["feature_schema"])
    validate_feature_schema(schema)
    extractor = FeatureExtractor(schema)
    data = load_dataset_and_split(config)
    indices = data["indices"]["discovery"]
    device = torch.device(f"cuda:{int(config['gpu_id'])}")
    directions = load_directions(assets["direction_path"])
    response_path = all_paths["tables"] / "natural_responses.csv.gz"
    source_path = all_paths["tables"] / "head_source_contributions.csv.gz"
    response_sink = AtomicCsvSink(response_path, response_fieldnames())
    source_sink = AtomicCsvSink(source_path, source_fieldnames())
    parameter_before, _ = model_parameter_checksum(Path(config["model_path"]))
    model = None
    capture = None
    reconstruction_errors: list[float] = []
    try:
        model, tokenizer = load_model_and_tokenizer(config, device)
        projections = projections_for_components(model, directions, components)
        capture = StageBCapture(model, components, capture_sources=True)
        capture.install()
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        groups = model.config.num_attention_heads // model.config.num_key_value_heads
        batch_size = int(config["batch_size"])
        candidate_heads = [row for row in candidates_payload["candidates"] if row["component_type"] == "head"]
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            prompts = [data["prompts"][int(index)] for index in batch_indices]
            capture.reset()
            encoded, offsets = encode(tokenizer, prompts, device, offsets=True)
            result = model(**encoded, use_cache=False, output_attentions=True, return_dict=True)
            capture.validate()
            token_lengths = encoded["attention_mask"].sum(dim=-1).tolist()
            feature_rows = [extractor.summarize(data["records"][int(index)]["question"]) for index in batch_indices]
            scores: dict[str, torch.Tensor] = {}
            for component_id, row in components.items():
                if row["component_type"] == "neuron":
                    activation = capture.z[row["module_index"]][:, row["component_index"]].float()
                    scores[component_id] = activation
                else:
                    pre = capture.pre_o[row["module_index"]].reshape(len(batch_indices), model.config.num_attention_heads, head_dim)
                    scores[component_id] = (pre[:, row["component_index"]].float() * projections[component_id].float()).sum(dim=-1)
            for association in lookup.values():
                component_id = association["component_id"]
                projection = projections[component_id]
                for offset, row_index in enumerate(batch_indices):
                    feature = feature_rows[offset]
                    activation = float(scores[component_id][offset]) if association["component_type"] == "neuron" else None
                    total = float(scores[component_id][offset] * projection.float()) if association["component_type"] == "neuron" else float(scores[component_id][offset])
                    response_sink.writerow(
                        {
                            "analysis_split": "discovery",
                            "row_index": int(row_index),
                            "question_id": data["question_ids"][int(row_index)],
                            "label": "reasoning" if int(data["labels"][int(row_index)]) == 1 else "memory",
                            "category": data["records"][int(row_index)]["category"],
                            "token_length": int(token_lengths[offset]),
                            **{key: feature[key] for key in ("numeric_span_count", "has_numeric", "relation_span_count", "has_relation", "operator_span_count", "has_operator")},
                            "candidate_id": association["candidate_id"],
                            "component_id": component_id,
                            "component_type": association["component_type"],
                            "component_role": association["role"],
                            "control_kind": association["control_kind"],
                            "module_index": association["module_index"],
                            "component_index": association["component_index"],
                            "activation": activation,
                            "projection": float(projection.norm()) if projection.ndim else float(projection),
                            "total_contribution": total,
                        }
                    )
            for head_row in candidate_heads:
                component_id = head_row["component_id"]
                layer = int(head_row["module_index"])
                head = int(head_row["component_index"])
                q = projections[component_id].float()
                values = capture.values[layer].float().reshape(len(batch_indices), -1, model.config.num_key_value_heads, head_dim)
                values = values[:, :, head // groups, :]
                attention = result.attentions[layer][:, head, -1, :].float()
                source_scores = attention * torch.einsum("bsd,d->bs", values, q)
                direct = scores[component_id]
                errors = (source_scores.sum(dim=-1) - direct).abs()
                reconstruction_errors.extend(errors.tolist())
                topk = int(config["natural"]["source_contribution_topk"])
                for offset, row_index in enumerate(batch_indices):
                    mask = encoded["attention_mask"][offset].bool().cpu().numpy()
                    ids = encoded["input_ids"][offset].detach().cpu().tolist()
                    prompt = prompts[offset]
                    question = data["records"][int(row_index)]["question"]
                    prefix_end = prompt.index(question)
                    suffix_start = prefix_end + len(question)
                    spans = feature_rows[offset]["feature_spans"]
                    source_np = source_scores[offset].detach().cpu().numpy()
                    attention_np = attention[offset].detach().cpu().numpy()
                    valid_indices = [i for i, value in enumerate(mask.tolist()) if value]
                    ranked = sorted(valid_indices, key=lambda i: (-abs(float(source_np[i])), i))[:topk]
                    annotated = []
                    token_rows = {}
                    for token_index in valid_indices:
                        char_start, char_end = map(int, offsets[offset, token_index].tolist())
                        if char_start == char_end == 0:
                            role, family = "prompt_prefix", "special_token"
                        else:
                            role, family = token_semantic_role(char_start, char_end, spans, prefix_end, suffix_start)
                        if role in {"numeric", "relation", "operator"}:
                            annotated.append(token_index)
                        token_rows[token_index] = (char_start, char_end, role, family)
                    selected = sorted(set(ranked) | set(annotated))
                    span_totals = {}
                    for token_index in valid_indices:
                        _, _, role, family = token_rows[token_index]
                        key = f"{role}:{family}"
                        span_totals[key] = span_totals.get(key, 0.0) + float(source_np[token_index])
                    for token_index in selected:
                        char_start, char_end, role, family = token_rows[token_index]
                        key = f"{role}:{family}"
                        reason = "top_abs_contribution" if token_index in ranked else "feature_annotated"
                        if token_index in ranked and token_index in annotated:
                            reason = "top_abs_contribution+feature_annotated"
                        source_sink.writerow(
                            {
                                "analysis_split": "discovery",
                                "row_index": int(row_index),
                                "question_id": data["question_ids"][int(row_index)],
                                "label": "reasoning" if int(data["labels"][int(row_index)]) == 1 else "memory",
                                "category": data["records"][int(row_index)]["category"],
                                "candidate_id": component_id,
                                "module_index": layer,
                                "component_index": head,
                                "source_token_index": token_index,
                                "token_id": ids[token_index],
                                "token_text": tokenizer.convert_ids_to_tokens(ids[token_index]),
                                "char_start": char_start,
                                "char_end": char_end,
                                "source_span_id": key,
                                "semantic_role": role,
                                "feature_family": family,
                                "attention_weight": float(attention_np[token_index]),
                                "source_contribution": float(source_np[token_index]),
                                "span_contribution": span_totals[key],
                                "selected_reason": reason,
                            }
                        )
            if start == 0 or (start // batch_size + 1) % 25 == 0 or start + batch_size >= len(indices):
                print(f"natural: {min(start + batch_size, len(indices))}/{len(indices)}", flush=True)
            del encoded, result
        tolerance = config["backend_tolerances"]
        source_mean = float(np.mean(reconstruction_errors))
        source_max = float(np.max(reconstruction_errors))
        if source_mean > tolerance["source_reconstruction_mean_tolerance"] or source_max > tolerance["source_reconstruction_max_tolerance"]:
            raise RuntimeError(f"Natural source reconstruction exceeded frozen tolerances: mean={source_mean}, max={source_max}")
        response_sink.close(True)
        source_sink.close(True)
        response_sink = source_sink = None
        parameter_after, _ = model_parameter_checksum(Path(config["model_path"]))
        if parameter_before != parameter_after:
            raise RuntimeError("Model parameter checksum changed during natural inference")
    finally:
        if response_sink is not None:
            response_sink.close(False)
        if source_sink is not None:
            source_sink.close(False)
        if capture is not None:
            capture.remove()
        release_model(model)
    summarize_natural(config, all_paths, response_path, candidates_payload)
    write_status(
        all_paths["root"], "natural", "PASS",
        response_sha256=sha256_file(response_path),
        source_contribution_sha256=sha256_file(source_path),
        source_reconstruction_mean_abs_error=source_mean,
        source_reconstruction_max_abs_error=source_max,
        model_parameter_checksum_before=parameter_before,
        model_parameter_checksum_after=parameter_after,
        interpretation_boundary=INTERPRETATION_BOUNDARY,
    )


def summarize_natural(config: dict[str, Any], all_paths: dict[str, Path], response_path: Path, candidates_payload: dict[str, Any]) -> None:
    frame = pd.read_csv(response_path, low_memory=False)
    frame = frame[frame["component_role"] == "candidate"].copy()
    rows = []
    for component_id, group in frame.groupby("component_id", sort=True):
        endpoint = "activation" if group["component_type"].iloc[0] == "neuron" else "total_contribution"
        values = group[endpoint].astype(float)
        memory = values[group["label"] == "memory"]
        reasoning = values[group["label"] == "reasoning"]
        rows.append({
            "component_id": component_id, "component_type": group["component_type"].iloc[0],
            "endpoint": endpoint, "analysis": "rm_descriptive", "level": "reasoning-minus-memory",
            "n": len(group), "mean": float(values.mean()), "difference": float(reasoning.mean() - memory.mean()),
            "association": None, "p_value_exploratory": None,
        })
        for feature in ("has_numeric", "has_relation", "has_operator"):
            present = bool_series(group[feature])
            if present.any() and (~present).any():
                difference = float(values[present].mean() - values[~present].mean())
            else:
                difference = math.nan
            rows.append({
                "component_id": component_id, "component_type": group["component_type"].iloc[0],
                "endpoint": endpoint, "analysis": "binary_feature_descriptive", "level": feature,
                "n": len(group), "mean": float(values.mean()), "difference": difference,
                "association": None, "p_value_exploratory": None,
            })
        for feature in ("token_length", "numeric_span_count", "relation_span_count", "operator_span_count"):
            x = group[feature].astype(float).to_numpy()
            y = values.to_numpy()
            if np.ptp(x) == 0:
                rho, p = math.nan, math.nan
            else:
                result = scipy_stats.spearmanr(x, y)
                rho, p = float(result.statistic), float(result.pvalue)
            rows.append({
                "component_id": component_id, "component_type": group["component_type"].iloc[0],
                "endpoint": endpoint, "analysis": "spearman_exploratory", "level": feature,
                "n": len(group), "mean": float(values.mean()), "difference": None,
                "association": rho, "p_value_exploratory": p,
            })
        for category, category_group in group.assign(_endpoint=values).groupby("category", sort=True):
            rows.append({
                "component_id": component_id, "component_type": group["component_type"].iloc[0],
                "endpoint": endpoint, "analysis": "category_descriptive", "level": category,
                "n": len(category_group), "mean": float(category_group["_endpoint"].mean()),
                "difference": None, "association": None, "p_value_exploratory": None,
            })
    summary = pd.DataFrame(rows)
    atomic_csv(all_paths["tables"] / "natural_characterization.csv", summary)
    atomic_csv(
        all_paths["tables"] / "neuron_responses.csv.gz",
        frame[frame["component_type"] == "neuron"],
    )
    atomic_csv(
        all_paths["tables"] / "head_responses.csv.gz",
        frame[frame["component_type"] == "head"],
    )
    draft_candidates = []
    for candidate in candidates_payload["candidates"]:
        cid = candidate["component_id"]
        sub = summary[(summary["component_id"] == cid) & summary["difference"].notna()].copy()
        sub["abs_difference"] = sub["difference"].abs()
        associations = sub.sort_values(["abs_difference", "level"], ascending=[False, True]).head(8)
        draft_candidates.append(
            {
                "candidate_id": cid,
                "component_type": candidate["component_type"],
                "natural_associations": jsonable(associations.to_dict(orient="records")),
                "hypotheses": [],
            }
        )
    manifest = {
        "stage_b_run_id": config["stage_b_run_id"],
        "approved": False,
        "reviewer_id": "",
        "freeze_timestamp": "",
        "maximum_confirmatory_hypotheses_per_candidate": int(config["natural"]["hypotheses_max_per_candidate"]),
        "taxonomy": ["numeric_value", "numeric_representation", "numeric_count", "numeric_relevance", "relation_semantics", "lexical_paraphrase", "domain", "token_length"],
        "required_hypothesis_fields": ["hypothesis_id", "feature_family", "description", "natural_association_metric", "selection_reason", "competing_hypotheses", "analysis_label", "approved"],
        "candidates": draft_candidates,
        "warning": "Natural associations generate hypotheses only and are not confirmatory evidence.",
    }
    write_json(all_paths["hypotheses"], manifest)


def validate_hypotheses(config: dict[str, Any], all_paths: dict[str, Path], require_approved: bool) -> dict[str, Any]:
    manifest = read_json(all_paths["hypotheses"])
    candidates = read_json(all_paths["candidates"])
    expected = {row["component_id"] for row in candidates["candidates"]}
    actual = {row["candidate_id"] for row in manifest.get("candidates", [])}
    if actual != expected:
        raise RuntimeError("Hypothesis manifest candidate set differs from frozen candidates")
    maximum = int(config["natural"]["hypotheses_max_per_candidate"])
    hypothesis_ids = []
    required = {"hypothesis_id", "feature_family", "description", "natural_association_metric", "selection_reason", "competing_hypotheses", "analysis_label", "approved"}
    for candidate in manifest["candidates"]:
        hypotheses = candidate.get("hypotheses", [])
        confirmatory = [row for row in hypotheses if row.get("analysis_label") == "confirmatory"]
        if len(confirmatory) > maximum:
            raise RuntimeError(f"Too many confirmatory hypotheses for {candidate['candidate_id']}")
        for hypothesis in hypotheses:
            if required.difference(hypothesis):
                raise RuntimeError(f"Hypothesis fields missing for {candidate['candidate_id']}")
            hypothesis_ids.append(hypothesis["hypothesis_id"])
            if require_approved and hypothesis.get("approved") is not True:
                raise RuntimeError(f"Unapproved hypothesis: {hypothesis['hypothesis_id']}")
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        raise RuntimeError("hypothesis_id must be globally unique")
    if require_approved:
        if manifest.get("approved") is not True or not str(manifest.get("reviewer_id", "")).strip() or not str(manifest.get("freeze_timestamp", "")).strip():
            raise RuntimeError("Hypothesis manifest requires top-level approval, reviewer, and freeze timestamp")
        if not hypothesis_ids:
            raise RuntimeError("No hypotheses were frozen")
    return manifest


def run_freeze_hypotheses(config: dict[str, Any], all_paths: dict[str, Path]) -> None:
    require_status(all_paths["root"], "natural")
    manifest = validate_hypotheses(config, all_paths, require_approved=True)
    write_status(
        all_paths["root"], "freeze_hypotheses", "PASS",
        hypothesis_manifest_sha256=sha256_file(all_paths["hypotheses"]),
        n_hypotheses=sum(len(row["hypotheses"]) for row in manifest["candidates"]),
        reviewer_id=manifest["reviewer_id"],
        freeze_timestamp=manifest["freeze_timestamp"],
    )


def rules_to_manifest(config: dict[str, Any], all_paths: dict[str, Path], split: str, destination: Path) -> None:
    hypothesis_status = require_status(all_paths["root"], "freeze_hypotheses")
    if hypothesis_status["hypothesis_manifest_sha256"] != sha256_file(all_paths["hypotheses"]):
        raise RuntimeError("Frozen hypothesis manifest changed")
    rules_path = Path(config["template_rules_path"])
    payload = read_json(rules_path)
    if payload.get("approved") is not True:
        raise RuntimeError("template_rules.json requires human approved=true")
    hypotheses = validate_hypotheses(config, all_paths, require_approved=True)
    allowed = {
        hypothesis["hypothesis_id"]
        for candidate in hypotheses["candidates"]
        for hypothesis in candidate["hypotheses"]
        if hypothesis["analysis_label"] == "confirmatory"
    }
    rows = [dict(row) for row in payload.get("rules", []) if row.get("split") == split]
    if not rows:
        raise RuntimeError(f"No template rules found for split={split}")
    schema = read_json(all_paths["feature_schema"])
    extractor = FeatureExtractor(schema)
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    for row in rows:
        if row.get("hypothesis_id") not in allowed:
            raise RuntimeError(f"Rule references a non-frozen confirmatory hypothesis: {row.get('hypothesis_id')}")
        for column in CONTROLLED_COLUMNS:
            row.setdefault(column, "")
        for variant in ("original", "modified"):
            text = str(row[f"{variant}_text"])
            prompt = config["prompt_template"].format(question=text)
            token_length = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            numeric_spans = [item for item in extractor.spans(text) if item["semantic_role"] == "numeric"]
            length_key = f"token_length_{variant}"
            spans_key = f"number_spans_{variant}"
            if str(row[length_key]).strip() and int(row[length_key]) != token_length:
                raise RuntimeError(f"Frozen rule token length mismatch for {row['pair_id']} {variant}")
            row[length_key] = token_length
            serialized_spans = json.dumps(numeric_spans, ensure_ascii=False, sort_keys=True)
            if str(row[spans_key]).strip() and str(row[spans_key]) != serialized_spans:
                raise RuntimeError(f"Frozen rule numeric spans mismatch for {row['pair_id']} {variant}")
            row[spans_key] = serialized_spans
    frame = pd.DataFrame(rows, columns=CONTROLLED_COLUMNS)
    validate_controlled_manifest(frame, split=split, require_approved=False)
    atomic_csv(destination, frame)


def run_generate_pilot(config: dict[str, Any], all_paths: dict[str, Path]) -> None:
    rules_to_manifest(config, all_paths, "pilot", all_paths["pilot_pairs"])
    write_status(all_paths["root"], "generate_pilot", "PASS", pilot_manifest_sha256=sha256_file(all_paths["pilot_pairs"]), approved=False)


def run_approve_pilot(config: dict[str, Any], all_paths: dict[str, Path]) -> None:
    require_status(all_paths["root"], "generate_pilot")
    frame = pd.read_csv(all_paths["pilot_pairs"], keep_default_na=False)
    validate_controlled_manifest(frame, split="pilot", require_approved=True)
    write_status(
        all_paths["root"], "approve_pilot", "PASS",
        pilot_manifest_sha256=sha256_file(all_paths["pilot_pairs"]),
        n_pair=len(frame), n_template_family=frame["template_family"].nunique(),
        pilot_use_restriction="variance and runtime estimation only; excluded from confirmatory claims",
    )


def controlled_response_fields() -> list[str]:
    return [
        "split", "pair_id", "hypothesis_id", "feature_family", "template_id", "template_family", "variant",
        "candidate_id", "component_id", "component_type", "component_role", "control_kind",
        "module_index", "component_index", "activation", "projection", "total_contribution",
    ]


def controlled_source_fields() -> list[str]:
    return [
        "split", "pair_id", "hypothesis_id", "feature_family", "template_id", "template_family", "variant",
        "candidate_id", "module_index", "component_index", "source_token_index", "token_id", "token_text",
        "char_start", "char_end", "source_span_id", "semantic_role", "source_feature_family",
        "attention_weight", "source_contribution", "span_contribution", "selected_reason",
    ]


def controlled_span_fields() -> list[str]:
    return [
        "split", "pair_id", "hypothesis_id", "feature_family", "template_id", "template_family", "variant",
        "candidate_id", "module_index", "component_index", "source_span_id", "semantic_role",
        "source_feature_family", "span_contribution",
    ]


@torch.inference_mode()
def extract_controlled_responses(config: dict[str, Any], all_paths: dict[str, Path], manifest_path: Path, split: str, output_path: Path) -> dict[str, Any]:
    assets = load_stage_a_assets(config)
    _, _, lookup = load_frozen_components(all_paths)
    components = unique_components(lookup)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    validate_controlled_manifest(manifest, split=split, require_approved=True)
    hypothesis_manifest = validate_hypotheses(config, all_paths, require_approved=True)
    candidate_by_hypothesis = {
        hypothesis["hypothesis_id"]: candidate["candidate_id"]
        for candidate in hypothesis_manifest["candidates"]
        for hypothesis in candidate["hypotheses"]
        if hypothesis["analysis_label"] == "confirmatory"
    }
    device = torch.device(f"cuda:{int(config['gpu_id'])}")
    directions = load_directions(assets["direction_path"])
    parameter_before, _ = model_parameter_checksum(Path(config["model_path"]))
    sink = AtomicCsvSink(output_path, controlled_response_fields())
    capture_sources = split == "confirmatory"
    source_path = all_paths["tables"] / "confirmatory_head_source_contributions.csv.gz"
    span_path = all_paths["tables"] / "confirmatory_head_span_contributions.csv.gz"
    source_sink = AtomicCsvSink(source_path, controlled_source_fields()) if capture_sources else None
    span_sink = AtomicCsvSink(span_path, controlled_span_fields()) if capture_sources else None
    extractor = FeatureExtractor(read_json(all_paths["feature_schema"])) if capture_sources else None
    source_reconstruction_errors: list[float] = []
    model = None
    capture = None
    started = time.time()
    try:
        model, tokenizer = load_model_and_tokenizer(config, device)
        projections = projections_for_components(model, directions, components)
        capture = StageBCapture(model, components, capture_sources=capture_sources)
        capture.install()
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        flattened = []
        for _, row in manifest.iterrows():
            for variant in ("original", "modified"):
                flattened.append((row, variant, config["prompt_template"].format(question=row[f"{variant}_text"])))
        batch_size = int(config["batch_size"])
        for start in range(0, len(flattened), batch_size):
            batch = flattened[start : start + batch_size]
            capture.reset()
            encoded, offsets = encode(tokenizer, [item[2] for item in batch], device, offsets=capture_sources)
            model_output = model(**encoded, use_cache=False, output_attentions=capture_sources, return_dict=True)
            capture.validate()
            scores = {}
            for component_id, component in components.items():
                if component["component_type"] == "neuron":
                    scores[component_id] = capture.z[component["module_index"]][:, component["component_index"]].float()
                else:
                    pre = capture.pre_o[component["module_index"]].reshape(len(batch), model.config.num_attention_heads, head_dim)
                    scores[component_id] = (pre[:, component["component_index"]].float() * projections[component_id].float()).sum(dim=-1)
            for offset, (pair, variant, _) in enumerate(batch):
                target_candidate = candidate_by_hypothesis[pair["hypothesis_id"]]
                relevant = [association for association in lookup.values() if association["candidate_id"] == target_candidate]
                for association in relevant:
                    component_id = association["component_id"]
                    projection = projections[component_id]
                    activation = float(scores[component_id][offset]) if association["component_type"] == "neuron" else None
                    total = float(scores[component_id][offset] * projection.float()) if association["component_type"] == "neuron" else float(scores[component_id][offset])
                    sink.writerow({
                        "split": split, "pair_id": pair["pair_id"], "hypothesis_id": pair["hypothesis_id"],
                        "feature_family": pair["feature_family"], "template_id": pair["template_id"],
                        "template_family": pair["template_family"], "variant": variant,
                        "candidate_id": target_candidate, "component_id": component_id,
                        "component_type": association["component_type"], "component_role": association["role"],
                        "control_kind": association["control_kind"], "module_index": association["module_index"],
                        "component_index": association["component_index"], "activation": activation,
                        "projection": float(projection.norm()) if projection.ndim else float(projection),
                        "total_contribution": total,
                    })
            if capture_sources:
                groups = model.config.num_attention_heads // model.config.num_key_value_heads
                for offset, (pair, variant, prompt) in enumerate(batch):
                    target_candidate = candidate_by_hypothesis[pair["hypothesis_id"]]
                    component = components[target_candidate]
                    if component["component_type"] != "head":
                        continue
                    layer = int(component["module_index"])
                    head = int(component["component_index"])
                    q = projections[target_candidate].float()
                    values = capture.values[layer].float().reshape(len(batch), -1, model.config.num_key_value_heads, head_dim)
                    values = values[offset, :, head // groups, :]
                    attention = model_output.attentions[layer][offset, head, -1, :].float()
                    source_scores = attention * torch.einsum("sd,d->s", values, q)
                    source_reconstruction_errors.append(float(abs(source_scores.sum() - scores[target_candidate][offset])))
                    text = str(pair[f"{variant}_text"])
                    features = extractor.summarize(text)
                    prefix_end = prompt.index(text)
                    suffix_start = prefix_end + len(text)
                    mask = encoded["attention_mask"][offset].bool().cpu().numpy()
                    ids = encoded["input_ids"][offset].detach().cpu().tolist()
                    source_np = source_scores.detach().cpu().numpy()
                    attention_np = attention.detach().cpu().numpy()
                    valid_indices = [i for i, value in enumerate(mask.tolist()) if value]
                    ranked = sorted(valid_indices, key=lambda i: (-abs(float(source_np[i])), i))[: int(config["natural"]["source_contribution_topk"])]
                    annotated: list[int] = []
                    token_rows: dict[int, tuple[int, int, str, str]] = {}
                    for token_index in valid_indices:
                        char_start, char_end = map(int, offsets[offset, token_index].tolist())
                        if char_start == char_end == 0:
                            role, source_family = "prompt_prefix", "special_token"
                        else:
                            role, source_family = token_semantic_role(char_start, char_end, features["feature_spans"], prefix_end, suffix_start)
                        if role in {"numeric", "relation", "operator"}:
                            annotated.append(token_index)
                        token_rows[token_index] = (char_start, char_end, role, source_family)
                    span_totals: dict[str, float] = {}
                    span_meta: dict[str, tuple[str, str]] = {}
                    for token_index in valid_indices:
                        _, _, role, source_family = token_rows[token_index]
                        span_id = f"{role}:{source_family}"
                        span_totals[span_id] = span_totals.get(span_id, 0.0) + float(source_np[token_index])
                        span_meta[span_id] = (role, source_family)
                    base = {
                        "split": split, "pair_id": pair["pair_id"], "hypothesis_id": pair["hypothesis_id"],
                        "feature_family": pair["feature_family"], "template_id": pair["template_id"],
                        "template_family": pair["template_family"], "variant": variant,
                        "candidate_id": target_candidate, "module_index": layer, "component_index": head,
                    }
                    for span_id, span_total in sorted(span_totals.items()):
                        role, source_family = span_meta[span_id]
                        span_sink.writerow({
                            **base, "source_span_id": span_id, "semantic_role": role,
                            "source_feature_family": source_family, "span_contribution": span_total,
                        })
                    for token_index in sorted(set(ranked) | set(annotated)):
                        char_start, char_end, role, source_family = token_rows[token_index]
                        span_id = f"{role}:{source_family}"
                        reason = "top_abs_contribution" if token_index in ranked else "feature_annotated"
                        if token_index in ranked and token_index in annotated:
                            reason = "top_abs_contribution+feature_annotated"
                        source_sink.writerow({
                            **base, "source_token_index": token_index, "token_id": ids[token_index],
                            "token_text": tokenizer.convert_ids_to_tokens(ids[token_index]), "char_start": char_start,
                            "char_end": char_end, "source_span_id": span_id, "semantic_role": role,
                            "source_feature_family": source_family, "attention_weight": float(attention_np[token_index]),
                            "source_contribution": float(source_np[token_index]), "span_contribution": span_totals[span_id],
                            "selected_reason": reason,
                        })
            if start == 0 or (start // batch_size + 1) % 25 == 0 or start + batch_size >= len(flattened):
                print(f"{split}: {min(start + batch_size, len(flattened))}/{len(flattened)} variants", flush=True)
        sink.close(True)
        sink = None
        if source_sink is not None:
            source_sink.close(True)
            source_sink = None
        if span_sink is not None:
            span_sink.close(True)
            span_sink = None
        if capture_sources:
            tolerance = config["backend_tolerances"]
            source_mean = float(np.mean(source_reconstruction_errors)) if source_reconstruction_errors else 0.0
            source_max = float(np.max(source_reconstruction_errors)) if source_reconstruction_errors else 0.0
            if source_mean > tolerance["source_reconstruction_mean_tolerance"] or source_max > tolerance["source_reconstruction_max_tolerance"]:
                raise RuntimeError(f"Confirmatory source reconstruction exceeded tolerances: mean={source_mean}, max={source_max}")
        parameter_after, _ = model_parameter_checksum(Path(config["model_path"]))
        if parameter_before != parameter_after:
            raise RuntimeError("Model parameter checksum changed during controlled inference")
    finally:
        if sink is not None:
            sink.close(False)
        if source_sink is not None:
            source_sink.close(False)
        if span_sink is not None:
            span_sink.close(False)
        if capture is not None:
            capture.remove()
        release_model(model)
    metadata = {
        "response_sha256": sha256_file(output_path),
        "model_parameter_checksum_before": parameter_before,
        "model_parameter_checksum_after": parameter_after,
        "runtime_seconds": time.time() - started,
        "n_pair": len(manifest),
        "n_template_family": manifest["template_family"].nunique(),
    }
    if capture_sources:
        metadata.update({
            "head_source_contributions_sha256": sha256_file(source_path),
            "head_span_contributions_sha256": sha256_file(span_path),
            "source_reconstruction_mean_abs_error": source_mean,
            "source_reconstruction_max_abs_error": source_max,
        })
    return metadata


def summarize_controlled(
    config: dict[str, Any],
    response_path: Path,
    split: str,
    output_path: Path,
    specificity_path: Path | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(response_path, low_memory=False)
    keys = ["pair_id", "hypothesis_id", "feature_family", "template_family", "candidate_id", "component_id", "component_type", "component_role", "control_kind"]
    endpoints = []
    for endpoint, component_type in (("activation", "neuron"), ("total_contribution", "head")):
        subset = frame[frame["component_type"] == component_type]
        if subset.empty:
            continue
        pivot = subset.pivot(index=keys, columns="variant", values=endpoint).reset_index()
        pivot["difference"] = pivot["modified"] - pivot["original"]
        pivot["endpoint"] = endpoint
        endpoints.append(pivot)
    if not endpoints:
        raise RuntimeError("No analyzable controlled responses")
    paired = pd.concat(endpoints, ignore_index=True)
    settings = config["statistics"]
    summaries = []
    for group_key, group in paired.groupby(["hypothesis_id", "feature_family", "candidate_id", "component_id", "component_type", "component_role", "control_kind", "endpoint"], sort=True):
        summary = paired_summary(
            group["template_family"], group["difference"],
            int(settings["bootstrap_iterations"]), int(settings["permutation_iterations"]), int(settings["random_seed"]),
        )
        summaries.append(dict(zip(["hypothesis_id", "feature_family", "candidate_id", "component_id", "component_type", "component_role", "control_kind", "endpoint"], group_key)) | summary)
    result = pd.DataFrame(summaries)
    result["analysis_family"] = np.where(result["component_role"] == "candidate", "primary_candidate_effect", result["control_kind"] + "_control_effect")
    result["bh_q"] = np.nan
    for _, family in result.groupby(["analysis_family", "component_type", "feature_family"], sort=True):
        result.loc[family.index, "bh_q"] = benjamini_hochberg(family["sign_flip_p"].astype(float))
    atomic_csv(output_path, result)
    if specificity_path is not None:
        candidate = paired[paired["component_role"] == "candidate"][
            ["pair_id", "hypothesis_id", "feature_family", "template_family", "candidate_id", "component_type", "endpoint", "difference"]
        ].rename(columns={"difference": "candidate_difference"})
        controls = paired[paired["component_role"] == "control"].copy()
        merged = controls.merge(
            candidate,
            on=["pair_id", "hypothesis_id", "feature_family", "template_family", "candidate_id", "component_type", "endpoint"],
            validate="many_to_one",
        )
        specificity_rows = []
        specificity_keys = [
            "hypothesis_id", "feature_family", "candidate_id", "component_id",
            "component_type", "control_kind", "endpoint",
        ]
        for group_key, group in merged.groupby(specificity_keys, sort=True):
            summary = specificity_summary(
                group["template_family"], group["candidate_difference"], group["difference"],
                int(settings["bootstrap_iterations"]), int(settings["permutation_iterations"]), int(settings["random_seed"]),
            )
            specificity_rows.append(dict(zip(specificity_keys, group_key)) | summary)
        specificity = pd.DataFrame(specificity_rows)
        specificity["analysis_family"] = specificity["control_kind"] + "_control_specificity"
        specificity["bh_q"] = np.nan
        for _, family in specificity.groupby(["analysis_family", "component_type", "feature_family"], sort=True):
            specificity.loc[family.index, "bh_q"] = benjamini_hochberg(family["sign_flip_p"].astype(float))
        atomic_csv(specificity_path, specificity)
    return result


def summarize_controlled_source_spans(config: dict[str, Any], span_path: Path, output_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(span_path, low_memory=False)
    columns = [
        "hypothesis_id", "feature_family", "candidate_id", "source_span_id", "semantic_role",
        "source_feature_family", "n_template", "mean_template_effect", "template_sd", "cohen_dz",
        "ci95_low", "ci95_high", "sign_flip_p", "stat_na_reason", "analysis_family", "bh_q",
    ]
    if frame.empty:
        result = pd.DataFrame(columns=columns)
        atomic_csv(output_path, result)
        return result
    keys = [
        "pair_id", "hypothesis_id", "feature_family", "template_family", "candidate_id",
        "source_span_id", "semantic_role", "source_feature_family",
    ]
    pivot = frame.pivot(index=keys, columns="variant", values="span_contribution").reset_index()
    for variant in ("original", "modified"):
        if variant not in pivot:
            pivot[variant] = 0.0
    pivot[["original", "modified"]] = pivot[["original", "modified"]].fillna(0.0)
    pivot["difference"] = pivot["modified"] - pivot["original"]
    settings = config["statistics"]
    rows = []
    group_keys = ["hypothesis_id", "feature_family", "candidate_id", "source_span_id", "semantic_role", "source_feature_family"]
    for group_key, group in pivot.groupby(group_keys, sort=True):
        summary = paired_summary(
            group["template_family"], group["difference"],
            int(settings["bootstrap_iterations"]), int(settings["permutation_iterations"]), int(settings["random_seed"]),
        )
        rows.append(dict(zip(group_keys, group_key)) | summary)
    result = pd.DataFrame(rows)
    result["analysis_family"] = "head_source_span_exploratory_effect"
    result["bh_q"] = np.nan
    for _, family in result.groupby(["analysis_family", "feature_family"], sort=True):
        result.loc[family.index, "bh_q"] = benjamini_hochberg(family["sign_flip_p"].astype(float))
    atomic_csv(output_path, result)
    return result


def run_pilot(config: dict[str, Any], all_paths: dict[str, Path]) -> None:
    approval = require_status(all_paths["root"], "approve_pilot")
    if approval["pilot_manifest_sha256"] != sha256_file(all_paths["pilot_pairs"]):
        raise RuntimeError("Approved pilot pair manifest changed")
    require_status(all_paths["root"], "sanity")
    output = all_paths["tables"] / "pilot_responses.csv.gz"
    metadata = extract_controlled_responses(config, all_paths, all_paths["pilot_pairs"], "pilot", output)
    summary_path = all_paths["tables"] / "pilot_variance_estimates.csv"
    summarize_controlled(config, output, "pilot", summary_path)
    write_status(
        all_paths["root"], "pilot", "PASS", **metadata,
        variance_table_sha256=sha256_file(summary_path),
        use_restriction="Pilot estimates are excluded from final CI, p-values, FDR, and claims.",
    )


def run_freeze_confirmatory(config: dict[str, Any], all_paths: dict[str, Path]) -> None:
    require_status(all_paths["root"], "pilot")
    design = read_json(all_paths["confirmatory_design_draft"])
    combined = {"confirmatory_power": {key: design.get(key) for key in config["confirmatory_power"]}}
    validate_confirmatory_power(combined, require_count=True)
    if design.get("approved") is not True or not str(design.get("reviewer_id", "")).strip() or not str(design.get("freeze_timestamp", "")).strip():
        raise RuntimeError("Confirmatory design requires approved=true, reviewer_id, and freeze_timestamp")
    rules_to_manifest(config, all_paths, "confirmatory", all_paths["confirmatory_pairs"])
    confirmatory = pd.read_csv(all_paths["confirmatory_pairs"], keep_default_na=False)
    validate_controlled_manifest(confirmatory, "confirmatory", require_approved=True)
    pilot = pd.read_csv(all_paths["pilot_pairs"], keep_default_na=False)
    overlap = set(pilot["template_family"].astype(str)) & set(confirmatory["template_family"].astype(str))
    if overlap:
        raise RuntimeError(f"Pilot and confirmatory template families overlap: {sorted(overlap)}")
    hypotheses = validate_hypotheses(config, all_paths, require_approved=True)
    planned = int(design["planned_confirmatory_template_count"])
    actual = int(confirmatory["template_family"].nunique())
    per_hypothesis_counts = confirmatory.groupby("hypothesis_id")["template_family"].nunique().astype(int)
    frozen_hypothesis_ids = {
        hypothesis["hypothesis_id"]
        for candidate in hypotheses["candidates"]
        for hypothesis in candidate["hypotheses"]
        if hypothesis["analysis_label"] == "confirmatory"
    }
    if set(per_hypothesis_counts.index.astype(str)) != frozen_hypothesis_ids:
        raise RuntimeError("Confirmatory manifest must cover every and only frozen confirmatory hypothesis")
    n_primary = sum(
        hypothesis["analysis_label"] == "confirmatory"
        for candidate in hypotheses["candidates"]
        for hypothesis in candidate["hypotheses"]
    )
    alpha_adjusted = float(design["two_sided_alpha"]) / max(n_primary, 1)
    z_alpha = float(scipy_stats.norm.ppf(1.0 - alpha_adjusted / 2.0))
    z_power = float(scipy_stats.norm.ppf(float(design["target_power"])))
    standardized_effect = float(design["minimum_effect_of_interest_dz"])
    exclusion_rate = float(design["expected_template_exclusion_rate"])
    required_before_exclusion = math.ceil(((z_alpha + z_power) / standardized_effect) ** 2)
    recommended_count = math.ceil(required_before_exclusion / (1.0 - exclusion_rate))
    if planned < recommended_count:
        raise RuntimeError(
            f"planned_confirmatory_template_count={planned} is below the frozen power recommendation {recommended_count}"
        )
    if not bool((per_hypothesis_counts == planned).all()):
        raise RuntimeError(
            f"Every hypothesis requires {planned} confirmatory template families; got {per_hypothesis_counts.to_dict()}"
        )
    if planned > int(design["maximum_template_count"]):
        raise RuntimeError("Confirmatory per-hypothesis template count exceeds the frozen maximum")
    frozen = {
        **design,
        "frozen_config_sha256": sha256_file(all_paths["manifests"] / "frozen_config.json"),
        "frozen_candidates_sha256": sha256_file(all_paths["candidates"]),
        "frozen_controls_sha256": sha256_file(all_paths["controls"]),
        "model_parameter_checksum": model_parameter_checksum(Path(config["model_path"]))[0],
        "pilot_manifest_sha256": sha256_file(all_paths["pilot_pairs"]),
        "confirmatory_manifest_sha256": sha256_file(all_paths["confirmatory_pairs"]),
        "hypothesis_manifest_sha256": sha256_file(all_paths["hypotheses"]),
        "feature_schema_sha256": sha256_file(all_paths["feature_schema"]),
        "statistical_config": config["statistics"],
        "statistical_config_sha256": __import__("hashlib").sha256(json.dumps(config["statistics"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "pilot_confirmatory_template_overlap": [],
        "n_frozen_primary_hypotheses": n_primary,
        "planning_alpha_after_bonferroni": alpha_adjusted,
        "recommended_template_count": recommended_count,
        "confirmatory_template_count_per_hypothesis": per_hypothesis_counts.to_dict(),
        "power_formula": "ceil(((z_(1-alpha_adjusted/2)+z_power)/minimum_effect_dz)^2/(1-exclusion_rate))",
    }
    write_json(all_paths["confirmatory_design"], frozen)
    write_status(
        all_paths["root"], "freeze_confirmatory", "PASS",
        confirmatory_design_sha256=sha256_file(all_paths["confirmatory_design"]),
        controlled_pair_manifest_sha256=sha256_file(all_paths["confirmatory_pairs"]),
        n_pair=len(confirmatory), n_template_family=actual,
    )


def revalidate_confirmatory_freeze(config: dict[str, Any], all_paths: dict[str, Path]) -> None:
    status = require_status(all_paths["root"], "freeze_confirmatory")
    if status["confirmatory_design_sha256"] != sha256_file(all_paths["confirmatory_design"]):
        raise RuntimeError("Frozen confirmatory design changed")
    if status["controlled_pair_manifest_sha256"] != sha256_file(all_paths["confirmatory_pairs"]):
        raise RuntimeError("Frozen confirmatory pair manifest changed")
    design = read_json(all_paths["confirmatory_design"])
    for name, path in (
        ("frozen_config_sha256", all_paths["manifests"] / "frozen_config.json"),
        ("frozen_candidates_sha256", all_paths["candidates"]),
        ("frozen_controls_sha256", all_paths["controls"]),
        ("hypothesis_manifest_sha256", all_paths["hypotheses"]),
        ("feature_schema_sha256", all_paths["feature_schema"]),
    ):
        if design[name] != sha256_file(path):
            raise RuntimeError(f"Frozen artifact changed: {name}")
    if design["model_parameter_checksum"] != model_parameter_checksum(Path(config["model_path"]))[0]:
        raise RuntimeError("Model parameter checksum differs from the frozen confirmatory design")


def run_confirmatory(config: dict[str, Any], all_paths: dict[str, Path]) -> None:
    require_status(all_paths["root"], "sanity")
    revalidate_confirmatory_freeze(config, all_paths)
    output = all_paths["tables"] / "confirmatory_responses.csv.gz"
    metadata = extract_controlled_responses(config, all_paths, all_paths["confirmatory_pairs"], "confirmatory", output)
    effects = all_paths["tables"] / "controlled_effects.csv"
    specificity = all_paths["tables"] / "matched_control_results.csv"
    summarize_controlled(config, output, "confirmatory", effects, specificity_path=specificity)
    source_effects = all_paths["tables"] / "head_source_span_effects.csv"
    summarize_controlled_source_spans(
        config,
        all_paths["tables"] / "confirmatory_head_span_contributions.csv.gz",
        source_effects,
    )
    write_status(
        all_paths["root"], "confirmatory", "PASS", **metadata,
        controlled_effects_sha256=sha256_file(effects),
        matched_control_results_sha256=sha256_file(specificity),
        head_source_span_effects_sha256=sha256_file(source_effects),
        confirmatory_design_sha256=sha256_file(all_paths["confirmatory_design"]),
        interpretation_boundary=INTERPRETATION_BOUNDARY,
    )


def run_report(config: dict[str, Any], all_paths: dict[str, Path]) -> None:
    require_status(all_paths["root"], "confirmatory")
    revalidate_confirmatory_freeze(config, all_paths)
    candidates = read_json(all_paths["candidates"])
    hypotheses = validate_hypotheses(config, all_paths, require_approved=True)
    natural = pd.read_csv(all_paths["tables"] / "natural_characterization.csv")
    effects = pd.read_csv(all_paths["tables"] / "controlled_effects.csv")
    hypothesis_by_candidate = {row["candidate_id"]: row["hypotheses"] for row in hypotheses["candidates"]}
    for candidate in candidates["candidates"]:
        cid = candidate["component_id"]
        card = {
            "candidate_id": cid,
            "component_type": candidate["component_type"],
            "module_index_zero_based": candidate["module_index"],
            "layer_one_based": int(candidate["module_index"]) + 1,
            "component_index": candidate["component_index"],
            "stage_a": candidate["stage_a_metadata"],
            "natural": jsonable(natural[natural["component_id"] == cid].to_dict(orient="records")),
            "frozen_hypotheses": hypothesis_by_candidate[cid],
            "controlled": jsonable(effects[effects["candidate_id"] == cid].to_dict(orient="records")),
            "sanity_status": read_json(all_paths["root"] / "status" / "sanity.json")["status"],
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        }
        write_json(all_paths["cards"] / f"{cid}.json", card)
        lines = [
            f"# {cid}", "", f"- Type: {candidate['component_type']}",
            f"- Layer: {int(candidate['module_index']) + 1} (module index {candidate['module_index']})",
            f"- Component index: {candidate['component_index']}", "", "## Interpretation boundary", "", INTERPRETATION_BOUNDARY, "",
            "## Frozen hypotheses", "", "```json", json.dumps(hypothesis_by_candidate[cid], ensure_ascii=False, indent=2), "```", "",
            "## Confirmatory effects", "", "```json", json.dumps(card["controlled"], ensure_ascii=False, indent=2), "```", "",
        ]
        (all_paths["cards"] / f"{cid}.md").write_text("\n".join(lines), encoding="utf-8")
    assets = load_stage_a_assets(config)
    provenance = frozen_provenance(
        config, assets, all_paths["feature_schema"], all_paths["controls"], all_paths["hypotheses"],
        all_paths["confirmatory_pairs"], all_paths["confirmatory_design"],
    )
    write_json(all_paths["provenance"], provenance)
    summary = {
        "stage_b_run_id": config["stage_b_run_id"],
        "status": "PASS",
        "candidate_count": len(candidates["candidates"]),
        "candidate_card_count": len(list(all_paths["cards"].glob("*.json"))),
        "provenance": provenance,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    write_json(all_paths["root"] / "stage_b_summary.json", summary)
    write_status(all_paths["root"], "report", "PASS", summary_sha256=sha256_file(all_paths["root"] / "stage_b_summary.json"))


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args)
    all_paths = paths(config)
    ensure_dirs(all_paths)
    phase_functions = {
        "prepare": run_prepare,
        "sanity": run_sanity,
        "natural": run_natural,
        "freeze_hypotheses": run_freeze_hypotheses,
        "generate_pilot": run_generate_pilot,
        "approve_pilot": run_approve_pilot,
        "pilot": run_pilot,
        "freeze_confirmatory": run_freeze_confirmatory,
        "confirmatory": run_confirmatory,
        "report": run_report,
    }
    try:
        phase_functions[args.phase](config, all_paths)
        return 0
    except Exception as exc:
        write_status(
            all_paths["root"], args.phase, "FAIL",
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
