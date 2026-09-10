#!/usr/bin/env python3
"""AI-audited exploratory activation patching for frozen Stage D items."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_A_DIR = SCRIPT_DIR.parent / "rm_decomp"
STAGE_B_DIR = SCRIPT_DIR.parent / "rm_decomp_b"
for source in (STAGE_A_DIR, STAGE_B_DIR):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from core import load_model_and_tokenizer, release_model, sha256_file  # noqa: E402
from stage_b_core import load_directions, model_parameter_checksum  # noqa: E402


LABELS = ("A", "B", "C", "D")
CONDITIONS = ("A", "B", "C", "D")
COMPONENT_PATTERN = re.compile(r"^L(\d{2})([HN])(\d{5})$")
PATCH_OPERATIONS = {
    "A_to_B": ("A", "B"),
    "C_to_D": ("C", "D"),
    "B_to_A": ("B", "A"),
    "D_to_C": ("D", "C"),
    "A_to_C": ("A", "C"),
    "B_to_D": ("B", "D"),
    "C_to_A": ("C", "A"),
    "D_to_B": ("D", "B"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temporary, path)


def write_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(path)
    if args.gpu_id is not None:
        config["gpu_id"] = args.gpu_id
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    config["config_path"] = str(path.resolve())
    config["config_hash"] = canonical_hash(
        {key: value for key, value in config.items() if key not in {"config_path", "config_hash"}}
    )
    return config


def paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["output_root"])
    return {
        "root": root,
        "manifests": root / "manifests",
        "tables": root / "tables",
        "status": root / "status",
        "logs": root / "logs",
        "design": root / "manifests" / "exploratory_design.json",
        "groups": root / "manifests" / "patch_groups.json",
        "states": root / "checkpoints" / "baseline_component_states.pt",
        "logits": root / "checkpoints" / "baseline_logits.pt",
        "baseline": root / "tables" / "baseline_responses.csv.gz",
        "patch": root / "tables" / "patch_responses.csv.gz",
    }


def ensure_dirs(p: dict[str, Path]) -> None:
    for path in (p["root"], p["manifests"], p["tables"], p["status"], p["logs"], p["states"].parent):
        path.mkdir(parents=True, exist_ok=True)


def status(p: dict[str, Path], phase: str, **payload: Any) -> None:
    write_json(
        p["status"] / f"{phase}.json",
        {"phase": phase, "status": "PASS", "timestamp": utc_now(), **payload},
    )


def require(p: dict[str, Path], phase: str) -> None:
    path = p["status"] / f"{phase}.json"
    if not path.exists() or read_json(path).get("status") != "PASS":
        raise RuntimeError(f"Required phase is not PASS: {phase}")


def component_from_id(component_id: str) -> dict[str, Any]:
    match = COMPONENT_PATTERN.fullmatch(component_id)
    if match is None:
        raise ValueError(f"Invalid component ID: {component_id}")
    return {
        "component_id": component_id,
        "component_type": "head" if match.group(2) == "H" else "neuron",
        "module_index": int(match.group(1)),
        "component_index": int(match.group(3)),
    }


def build_patch_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = read_json(Path(config["stage_c_conditions"]))["conditions"]
    candidates = list(config["primary_candidates"])
    groups = []
    for candidate_id in candidates:
        groups.append(
            {
                "group_id": f"candidate::{candidate_id}",
                "owner_candidate_id": candidate_id,
                "group_role": "candidate",
                "control_kind": "candidate",
                "components": [component_from_id(candidate_id)],
            }
        )
        controls = [
            row for row in conditions
            if row["owner_candidate_id"] == candidate_id and row["component_role"] == "control"
        ]
        if len(controls) != 4 or sum(row["control_kind"] == "matched" for row in controls) != 1:
            raise RuntimeError(f"Expected one matched and three random controls for {candidate_id}")
        for row in controls:
            component = row["components"][0]
            groups.append(
                {
                    "group_id": f"control::{candidate_id}::{row['control_kind']}::{component['component_id']}",
                    "owner_candidate_id": candidate_id,
                    "group_role": "control",
                    "control_kind": row["control_kind"],
                    "components": [component],
                }
            )
    groups.append(
        {
            "group_id": "joint4",
            "owner_candidate_id": "joint4",
            "group_role": "joint",
            "control_kind": "joint",
            "components": [component_from_id(value) for value in candidates],
        }
    )
    return groups


def prepare(config: dict[str, Any], p: dict[str, Path]) -> None:
    ensure_dirs(p)
    if config.get("allow_ai_audited_exploratory") is not True or config.get("human_audit_passed") is not False:
        raise RuntimeError("Exploratory override must be explicit and must not claim human audit PASS")
    source = Path(config["source_root"])
    source_status = read_json(source / "status" / "audit_package.json")
    source_audit = read_json(source / "human_audit" / "ai_full_audit_summary.json")
    linguistic_audit = read_json(source / "human_audit" / "ai_linguistic_audit_summary.json")
    if source_status.get("confirmatory_executed") is not False:
        raise RuntimeError("Source run already executed confirmatory inference")
    if source_audit.get("overall_status") != "PASS_TO_BLIND_HUMAN_AUDIT":
        raise RuntimeError("Source-grounded AI audit did not pass")
    if linguistic_audit.get("overall_status") != "PASS_TO_BLIND_HUMAN_AUDIT":
        raise RuntimeError("AI linguistic audit did not pass")
    items = pd.read_csv(source / "manifests" / "confirmatory_items.csv")
    if items["chain_id"].nunique() != 160 or len(items) != 640:
        raise RuntimeError("Frozen exploratory source must contain 160 complete 2x2 chains")
    groups = build_patch_groups(config)
    unique_components = {
        component["component_id"]: component
        for group in groups for component in group["components"]
    }
    model_checksum, model_files = model_parameter_checksum(Path(config["model_path"]))
    design = {
        "run_id": config["run_id"],
        "execution_label": config["execution_label"],
        "confirmatory_claim_allowed": False,
        "reason": "User explicitly chose continuation after disclosure that two blind human reviews were absent.",
        "source_run": str(source.resolve()),
        "source_items_hash": sha256_file(source / "manifests" / "confirmatory_items.csv"),
        "source_ai_audits": {
            "source_grounded": source_audit,
            "linguistic": linguistic_audit,
        },
        "feature_effects": {
            "E_R": "0.5*((S_B-S_A)+(S_D-S_C))",
            "E_M": "0.5*((S_A-S_C)+(S_B-S_D))",
        },
        "patch_operations": PATCH_OPERATIONS,
        "patch_position": "component input at last prompt token only",
        "head_state": "selected head slice of o_proj input",
        "neuron_state": "selected gated activation at down_proj input",
        "groups": groups,
        "unique_component_count": len(unique_components),
        "model_parameter_checksum": model_checksum,
        "model_parameter_files": model_files,
        "config_hash": config["config_hash"],
    }
    write_json(p["design"], design)
    write_json(p["groups"], {"groups": groups, "unique_components": list(unique_components.values())})
    write_json(p["manifests"] / "frozen_config.json", config)
    status(
        p,
        "prepare",
        execution_label=config["execution_label"],
        confirmatory_claim_allowed=False,
        chain_count=160,
        patch_group_count=len(groups),
        unique_component_count=len(unique_components),
    )


class StateCapture:
    def __init__(self, model: Any, components: list[dict[str, Any]]) -> None:
        self.model = model
        self.components = components
        self.final: torch.Tensor | None = None
        self.values: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def install(self) -> None:
        def final_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.final = tensor.detach()[:, -1, :].clone()

        self.handles.append(self.model.model.layers[31].register_forward_hook(final_hook))
        for component in self.components:
            layer = self.model.model.layers[int(component["module_index"])]
            module = layer.self_attn.o_proj if component["component_type"] == "head" else layer.mlp.down_proj
            self.handles.append(module.register_forward_pre_hook(self._component_hook(component)))

    def _component_hook(self, component: dict[str, Any]):
        component_id = component["component_id"]
        index = int(component["component_index"])

        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            values = args[0].detach()[:, -1, :]
            if component["component_type"] == "head":
                head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
                captured = values[:, index * head_dim:(index + 1) * head_dim]
            else:
                captured = values[:, index:index + 1]
            self.values[component_id] = captured.to("cpu", dtype=torch.float16).clone()

        return hook

    def reset(self) -> None:
        self.final = None
        self.values.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class PatchIntervention:
    def __init__(self, model: Any, components: list[dict[str, Any]]) -> None:
        self.model = model
        self.components = components
        self.sources: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def install(self) -> None:
        for component in self.components:
            layer = self.model.model.layers[int(component["module_index"])]
            module = layer.self_attn.o_proj if component["component_type"] == "head" else layer.mlp.down_proj
            self.handles.append(module.register_forward_pre_hook(self._hook(component)))

    def _hook(self, component: dict[str, Any]):
        component_id = component["component_id"]
        index = int(component["component_index"])

        def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            values = args[0].clone()
            source = self.sources[component_id].to(device=values.device, dtype=values.dtype)
            if component["component_type"] == "head":
                head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
                values[:, -1, index * head_dim:(index + 1) * head_dim] = source
            else:
                values[:, -1, index] = source[:, 0]
            return (values, *args[1:])

        return hook

    def set_sources(self, sources: dict[str, torch.Tensor]) -> None:
        self.sources = sources

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def encode(tokenizer: Any, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    output = tokenizer(prompts, return_tensors="pt", padding="longest", return_token_type_ids=False)
    if not bool(torch.all(output["attention_mask"][:, -1] == 1)):
        raise RuntimeError("Last index is not the final prompt token")
    return {key: value.to(device) for key, value in output.items()}


def label_token_ids(tokenizer: Any) -> list[int]:
    result = []
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Label is not one token: {label} -> {ids}")
        result.append(ids[0])
    return result


@torch.inference_mode()
def baseline(config: dict[str, Any], p: dict[str, Path]) -> None:
    require(p, "prepare")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    source = Path(config["source_root"])
    items = pd.read_csv(source / "manifests" / "confirmatory_items.csv")
    items = items.sort_values(["condition", "chain_id"], kind="stable").reset_index(drop=True)
    groups_payload = read_json(p["groups"])
    components = groups_payload["unique_components"]
    directions = load_directions(Path(config["stage_a_root"]) / "checkpoints" / "discovery_liref_directions.pt")
    device = torch.device(f"cuda:{config['gpu_id']}")
    model = None
    rows: list[dict[str, Any]] = []
    state_parts: dict[str, list[torch.Tensor]] = {row["component_id"]: [] for row in components}
    logits_parts: list[torch.Tensor] = []
    try:
        model, tokenizer = load_model_and_tokenizer(config, device)
        direction = torch.as_tensor(directions[31], device=device, dtype=torch.float32)
        label_ids = label_token_ids(tokenizer)
        capture = StateCapture(model, components)
        capture.install()
        for start in range(0, len(items), int(config["batch_size"])):
            batch = items.iloc[start:start + int(config["batch_size"])]
            capture.reset()
            encoded = encode(tokenizer, batch["prompt"].tolist(), device)
            output = model(**encoded, use_cache=False, return_dict=True)
            if capture.final is None or set(capture.values) != set(state_parts):
                raise RuntimeError("Incomplete baseline capture")
            logits = output.logits[:, -1, :]
            option_probs = torch.softmax(logits[:, label_ids].float(), dim=-1)
            scores = torch.mv(capture.final.float(), direction).cpu().numpy()
            predictions = option_probs.argmax(dim=-1).cpu().numpy()
            for offset, (_, item) in enumerate(batch.iterrows()):
                correct_index = LABELS.index(str(item["correct_label"]))
                rows.append(
                    {
                        "row_id": start + offset,
                        "chain_id": item["chain_id"],
                        "condition": item["condition"],
                        "template_family": item["template_family"],
                        "reasoning_exact_length_matched": bool(item["reasoning_exact_length_matched"]),
                        "memory_exact_length_matched": bool(item["memory_exact_length_matched"]),
                        "score": float(scores[offset]),
                        "correct_probability": float(option_probs[offset, correct_index].item()),
                        "predicted_label": LABELS[int(predictions[offset])],
                        "correct": int(predictions[offset]) == correct_index,
                    }
                )
            for component_id, value in capture.values.items():
                state_parts[component_id].append(value)
            logits_parts.append(logits.detach().to("cpu", dtype=torch.float16))
            print(f"baseline {min(start + int(config['batch_size']), len(items))}/{len(items)}", flush=True)
            del output, logits, encoded
        capture.remove()
    finally:
        release_model(model)
    response = pd.DataFrame(rows)
    states = {key: torch.cat(value, dim=0) for key, value in state_parts.items()}
    logits = torch.cat(logits_parts, dim=0)
    write_csv(p["baseline"], response)
    write_torch(
        p["states"],
        {"states": states, "row_keys": response[["row_id", "chain_id", "condition"]].to_dict("records")},
    )
    write_torch(p["logits"], {"logits": logits, "label": config["execution_label"]})
    status(
        p,
        "baseline",
        rows=len(response),
        chains=int(response["chain_id"].nunique()),
        captured_components=len(states),
        state_hash=sha256_file(p["states"]),
        logits_hash=sha256_file(p["logits"]),
    )


def output_diagnostics(logits: torch.Tensor, baseline_logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = baseline_logits.to(logits.device, dtype=logits.dtype)
    log_q = F.log_softmax(logits.float(), dim=-1)
    p = F.softmax(baseline.float(), dim=-1)
    kl = (p * (F.log_softmax(baseline.float(), dim=-1) - log_q)).sum(dim=-1)
    changed = logits.argmax(dim=-1) != baseline.argmax(dim=-1)
    rms = (logits.float() - baseline.float()).square().mean(dim=-1).sqrt()
    return kl.cpu().numpy(), changed.to(torch.int8).cpu().numpy(), rms.cpu().numpy()


@torch.inference_mode()
def patch(config: dict[str, Any], p: dict[str, Path]) -> None:
    require(p, "baseline")
    source_root = Path(config["source_root"])
    items = pd.read_csv(source_root / "manifests" / "confirmatory_items.csv")
    items = items.sort_values(["condition", "chain_id"], kind="stable").reset_index(drop=True)
    baseline_frame = pd.read_csv(p["baseline"])
    state_payload = torch.load(p["states"], map_location="cpu", weights_only=False)
    states: dict[str, torch.Tensor] = state_payload["states"]
    baseline_logits = torch.load(p["logits"], map_location="cpu", weights_only=False)["logits"]
    groups = read_json(p["groups"])["groups"]
    completed: set[tuple[str, str]] = set()
    response_parts: list[pd.DataFrame] = []
    if p["patch"].exists():
        prior = pd.read_csv(p["patch"])
        response_parts.append(prior)
        completed = set(zip(prior["group_id"], prior["operation"]))
    condition_indices = {
        condition: np.flatnonzero(items["condition"].to_numpy() == condition)
        for condition in CONDITIONS
    }
    directions = load_directions(Path(config["stage_a_root"]) / "checkpoints" / "discovery_liref_directions.pt")
    device = torch.device(f"cuda:{config['gpu_id']}")
    model = None
    try:
        model, tokenizer = load_model_and_tokenizer(config, device)
        direction = torch.as_tensor(directions[31], device=device, dtype=torch.float32)
        label_ids = label_token_ids(tokenizer)
        for group_number, group in enumerate(groups, 1):
            intervention = PatchIntervention(model, group["components"])
            intervention.install()
            try:
                for operation, (source_condition, target_condition) in PATCH_OPERATIONS.items():
                    if (group["group_id"], operation) in completed:
                        continue
                    print(f"[patch {group_number}/{len(groups)}] {group['group_id']} {operation}", flush=True)
                    source_indices = condition_indices[source_condition]
                    target_indices = condition_indices[target_condition]
                    target_items = items.iloc[target_indices].reset_index(drop=True)
                    operation_rows = []
                    for start in range(0, len(target_items), int(config["batch_size"])):
                        stop = min(start + int(config["batch_size"]), len(target_items))
                        batch = target_items.iloc[start:stop]
                        intervention.set_sources(
                            {
                                component["component_id"]: states[component["component_id"]][source_indices[start:stop]]
                                for component in group["components"]
                            }
                        )
                        capture = StateCapture(model, [])
                        capture.install()
                        encoded = encode(tokenizer, batch["prompt"].tolist(), device)
                        output = model(**encoded, use_cache=False, return_dict=True)
                        if capture.final is None:
                            raise RuntimeError("Patched final state was not captured")
                        logits = output.logits[:, -1, :]
                        option_probs = torch.softmax(logits[:, label_ids].float(), dim=-1)
                        scores = torch.mv(capture.final.float(), direction).cpu().numpy()
                        base_slice = baseline_logits[target_indices[start:stop]]
                        kl, top1, rms = output_diagnostics(logits, base_slice)
                        predictions = option_probs.argmax(dim=-1).cpu().numpy()
                        for offset, (_, item) in enumerate(batch.iterrows()):
                            correct_index = LABELS.index(str(item["correct_label"]))
                            baseline_row = baseline_frame.iloc[target_indices[start + offset]]
                            operation_rows.append(
                                {
                                    "group_id": group["group_id"],
                                    "owner_candidate_id": group["owner_candidate_id"],
                                    "group_role": group["group_role"],
                                    "control_kind": group["control_kind"],
                                    "operation": operation,
                                    "source_condition": source_condition,
                                    "target_condition": target_condition,
                                    "chain_id": item["chain_id"],
                                    "template_family": item["template_family"],
                                    "score": float(scores[offset]),
                                    "baseline_target_score": float(baseline_row["score"]),
                                    "correct_probability": float(option_probs[offset, correct_index].item()),
                                    "baseline_correct_probability": float(baseline_row["correct_probability"]),
                                    "predicted_label": LABELS[int(predictions[offset])],
                                    "correct": int(predictions[offset]) == correct_index,
                                    "next_token_kl": float(kl[offset]),
                                    "top1_changed": int(top1[offset]),
                                    "logit_rms_change": float(rms[offset]),
                                }
                            )
                        capture.remove()
                        del output, logits, encoded
                    response_parts.append(pd.DataFrame(operation_rows))
                    combined = pd.concat(response_parts, ignore_index=True)
                    write_csv(p["patch"], combined)
                    completed.add((group["group_id"], operation))
            finally:
                intervention.remove()
    finally:
        release_model(model)
    response = pd.concat(response_parts, ignore_index=True)
    status(
        p,
        "patch",
        rows=len(response),
        patch_groups=len(groups),
        operations_per_group=len(PATCH_OPERATIONS),
        response_hash=sha256_file(p["patch"]),
    )


def bh_adjust(values: pd.Series) -> np.ndarray:
    p_values = values.to_numpy(dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def vector_stats(values: np.ndarray, reps: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    bootstrap = values[draws].mean(axis=1)
    test = stats.ttest_1samp(values, 0.0, alternative="two-sided")
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "paired_dz": mean / sd if sd else math.nan,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "p_value": float(test.pvalue),
    }


def feature_vectors(baseline_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    wide = baseline_frame.pivot(index="chain_id", columns="condition", values="score").sort_index()
    vectors = {
        "E_R": 0.5 * ((wide["B"] - wide["A"]) + (wide["D"] - wide["C"])),
        "E_M": 0.5 * ((wide["A"] - wide["C"]) + (wide["B"] - wide["D"])),
    }
    return wide, {key: value.to_numpy(float) for key, value in vectors.items()}


def patch_vectors(
    baseline_wide: pd.DataFrame,
    group: pd.DataFrame,
    feature: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    patched = group.pivot(index="chain_id", columns="operation", values="score").reindex(baseline_wide.index)
    n_r = 0.5 * ((patched["A_to_B"] - baseline_wide["A"]) + (patched["C_to_D"] - baseline_wide["C"]))
    t_r = 0.5 * ((patched["B_to_A"] - baseline_wide["A"]) + (patched["D_to_C"] - baseline_wide["C"]))
    n_m = 0.5 * ((baseline_wide["A"] - patched["A_to_C"]) + (baseline_wide["B"] - patched["B_to_D"]))
    t_m = 0.5 * ((baseline_wide["A"] - patched["C_to_A"]) + (baseline_wide["B"] - patched["D_to_B"]))
    return {
        "R_attenuation": feature["E_R"] - n_r.to_numpy(float),
        "R_transfer": t_r.to_numpy(float),
        "M_attenuation": feature["E_M"] - n_m.to_numpy(float),
        "M_transfer": t_m.to_numpy(float),
    }


def report(config: dict[str, Any], p: dict[str, Path]) -> None:
    require(p, "patch")
    baseline_frame = pd.read_csv(p["baseline"])
    patch_frame = pd.read_csv(p["patch"])
    baseline_wide, feature = feature_vectors(baseline_frame)
    chain_meta = baseline_frame.drop_duplicates("chain_id").set_index("chain_id").reindex(baseline_wide.index)
    feature_rows = []
    for number, effect in enumerate(("E_R", "E_M")):
        metrics = vector_stats(feature[effect], int(config["bootstrap_replicates"]), int(config["seed"]) + number)
        family_values = pd.Series(feature[effect], index=baseline_wide.index).groupby(chain_meta["template_family"]).mean()
        exact_column = "reasoning_exact_length_matched" if effect == "E_R" else "memory_exact_length_matched"
        exact_mask = chain_meta[exact_column].astype(bool).to_numpy()
        exact_mean = float(feature[effect][exact_mask].mean()) if exact_mask.any() else math.nan
        feature_rows.append(
            {
                "effect": effect,
                **metrics,
                "template_family_sign_consistency": float((family_values > 0).mean()),
                "exact_length_n": int(exact_mask.sum()),
                "exact_length_mean": exact_mean,
                "exact_length_expected_direction": bool(exact_mean > 0) if exact_mask.any() else False,
            }
        )
    feature_table = pd.DataFrame(feature_rows)
    feature_table["q_value"] = bh_adjust(feature_table["p_value"])
    feature_table["criterion_pass"] = (
        (feature_table["ci_low"] > 0)
        & (feature_table["q_value"] < float(config["fdr_alpha"]))
        & (feature_table["paired_dz"].abs() >= float(config["primary_abs_paired_dz_min"]))
        & (feature_table["template_family_sign_consistency"] >= float(config["template_sign_consistency_min"]))
        & feature_table["exact_length_expected_direction"]
    )
    write_csv(p["tables"] / "feature_effects.csv", feature_table)

    vectors_by_group: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for group_number, (group_id, group) in enumerate(patch_frame.groupby("group_id", sort=False)):
        vectors = patch_vectors(baseline_wide, group, feature)
        vectors_by_group[group_id] = vectors
        first = group.iloc[0]
        for metric_number, (metric, values) in enumerate(vectors.items()):
            factor = metric[0]
            feature_name = f"E_{factor}"
            stats_row = vector_stats(
                values,
                int(config["bootstrap_replicates"]),
                int(config["seed"]) + 1000 + group_number * 10 + metric_number,
            )
            feature_mean = float(feature[feature_name].mean())
            rows.append(
                {
                    "group_id": group_id,
                    "owner_candidate_id": first["owner_candidate_id"],
                    "group_role": first["group_role"],
                    "control_kind": first["control_kind"],
                    "metric": metric,
                    "factor": factor,
                    "mode": metric.split("_", 1)[1],
                    **stats_row,
                    "fraction_of_feature_effect": stats_row["mean"] / feature_mean if feature_mean else math.nan,
                    "mean_correct_probability_change": float((group["correct_probability"] - group["baseline_correct_probability"]).mean()),
                    "mean_next_token_kl": float(group["next_token_kl"].mean()),
                    "top1_change_rate": float(group["top1_changed"].mean()),
                    "mean_logit_rms_change": float(group["logit_rms_change"].mean()),
                }
            )
    patch_table = pd.DataFrame(rows)
    patch_table["q_value"] = np.nan
    candidate_mask = patch_table["group_role"] == "candidate"
    for metric in patch_table.loc[candidate_mask, "metric"].unique():
        mask = candidate_mask & (patch_table["metric"] == metric)
        patch_table.loc[mask, "q_value"] = bh_adjust(patch_table.loc[mask, "p_value"])
    feature_pass = feature_table.set_index("effect")["criterion_pass"].to_dict()
    patch_table["criterion_pass"] = False
    for index, row in patch_table[candidate_mask].iterrows():
        patch_table.loc[index, "criterion_pass"] = bool(
            feature_pass[f"E_{row['factor']}"]
            and row["ci_low"] > 0
            and row["q_value"] < float(config["fdr_alpha"])
            and abs(row["paired_dz"]) >= float(config["individual_patch_abs_dz_min"])
            and row["fraction_of_feature_effect"] >= float(config["individual_patch_fraction_min"])
        )
    write_csv(p["tables"] / "patch_effects.csv", patch_table)

    comparisons = []
    for candidate_number, candidate_id in enumerate(config["primary_candidates"]):
        candidate_group = f"candidate::{candidate_id}"
        controls = patch_table[
            (patch_table["owner_candidate_id"] == candidate_id)
            & (patch_table["group_role"] == "control")
        ]["group_id"].unique()
        for metric_number, metric in enumerate(vectors_by_group[candidate_group]):
            for control_number, control_group in enumerate(controls):
                difference = vectors_by_group[candidate_group][metric] - vectors_by_group[control_group][metric]
                metrics = vector_stats(
                    difference,
                    int(config["bootstrap_replicates"]),
                    int(config["seed"]) + 50000 + candidate_number * 100 + metric_number * 10 + control_number,
                )
                control_row = patch_table[
                    (patch_table["group_id"] == control_group) & (patch_table["metric"] == metric)
                ].iloc[0]
                comparisons.append(
                    {
                        "candidate_id": candidate_id,
                        "metric": metric,
                        "control_group_id": control_group,
                        "control_kind": control_row["control_kind"],
                        **metrics,
                        "candidate_greater_than_control": bool(metrics["ci_low"] > 0),
                    }
                )
    comparison_table = pd.DataFrame(comparisons)
    write_csv(p["tables"] / "patch_control_comparisons.csv", comparison_table)

    joint_rows = []
    all_control_groups = patch_table[patch_table["group_role"] == "control"]["group_id"].unique()
    for metric in vectors_by_group["joint4"]:
        joint = patch_table[(patch_table["group_id"] == "joint4") & (patch_table["metric"] == metric)].iloc[0]
        differences = []
        for number, control_group in enumerate(all_control_groups):
            values = vectors_by_group["joint4"][metric] - vectors_by_group[control_group][metric]
            differences.append(
                vector_stats(values, int(config["bootstrap_replicates"]), int(config["seed"]) + 80000 + number)["ci_low"]
            )
        joint_rows.append(
            {
                "metric": metric,
                "mean": joint["mean"],
                "ci_low": joint["ci_low"],
                "ci_high": joint["ci_high"],
                "fraction_of_feature_effect": joint["fraction_of_feature_effect"],
                "minimum_joint_minus_control_ci_low": min(differences),
                "criterion_pass": bool(
                    feature_pass[f"E_{joint['factor']}"]
                    and joint["ci_low"] > 0
                    and joint["fraction_of_feature_effect"] >= float(config["joint_patch_fraction_min"])
                    and min(differences) > 0
                ),
            }
        )
    joint_table = pd.DataFrame(joint_rows)
    write_csv(p["tables"] / "joint_patch_effects.csv", joint_table)

    summary = {
        "run_id": config["run_id"],
        "execution_label": config["execution_label"],
        "confirmatory_claim_allowed": False,
        "feature_results": feature_table.to_dict("records"),
        "individual_patch_pass_count": int(patch_table["criterion_pass"].sum()),
        "joint_patch_pass_count": int(joint_table["criterion_pass"].sum()),
        "weight_rescue_eligible": bool(patch_table["criterion_pass"].any()),
        "interpretation_boundary": (
            "These AI-audited exploratory results cannot be labeled protocol-confirmatory because two independent "
            "blind human reviews were not completed."
        ),
    }
    write_json(p["root"] / "exploratory_summary.json", summary)
    lines = [
        "# Stage D v2_d07 — AI 검수 기반 탐색적 결과",
        "",
        "> 두 명의 독립 인간 검수를 생략했으므로 정식 confirmatory 결과가 아니다.",
        "",
        "## Feature effect",
        "",
    ]
    for row in feature_table.to_dict("records"):
        lines.append(
            f"- `{row['effect']}`: mean={row['mean']:.4f}, 95% CI=[{row['ci_low']:.4f}, "
            f"{row['ci_high']:.4f}], dz={row['paired_dz']:.3f}, PASS={bool(row['criterion_pass'])}"
        )
    lines.extend(
        [
            "",
            "## Activation patching",
            "",
            f"- Individual criterion PASS: {int(patch_table['criterion_pass'].sum())}개",
            f"- Joint criterion PASS: {int(joint_table['criterion_pass'].sum())}개",
            f"- Weight rescue 진행 조건: {bool(patch_table['criterion_pass'].any())}",
            "",
            "세부 결과는 `tables/feature_effects.csv`, `patch_effects.csv`, "
            "`patch_control_comparisons.csv`, `joint_patch_effects.csv`에 저장했다.",
        ]
    )
    (p["root"] / "RESULTS_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status(
        p,
        "report",
        feature_pass_count=int(feature_table["criterion_pass"].sum()),
        individual_patch_pass_count=int(patch_table["criterion_pass"].sum()),
        joint_patch_pass_count=int(joint_table["criterion_pass"].sum()),
        weight_rescue_eligible=bool(patch_table["criterion_pass"].any()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["prepare", "baseline", "patch", "report"])
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "exploratory_config.json")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)
    p = paths(config)
    ensure_dirs(p)
    {"prepare": prepare, "baseline": baseline, "patch": patch, "report": report}[args.phase](config, p)


if __name__ == "__main__":
    main()
