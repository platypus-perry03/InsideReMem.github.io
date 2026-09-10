#!/usr/bin/env python3
"""Frozen last-token component interventions for LiReF R/M causal validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_B_DIR = SCRIPT_DIR.parent / "rm_decomp_b"
STAGE_A_CODE_DIR = SCRIPT_DIR.parent / "rm_decomp"
for source in (BASE_B_DIR, STAGE_A_CODE_DIR):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from core import load_dataset_and_split, release_model, sha256_file, sha256_text  # noqa: E402
from stage_b_core import load_directions, load_model_and_tokenizer, model_parameter_checksum  # noqa: E402


GROUP_MEMORY = 0
GROUP_REASONING = 1
COMPONENT_PATTERN = re.compile(r"^L(\d{2})([HN])(\d{5})$")

GAP_COLUMNS = [
    "condition_id", "owner_candidate_id", "component_role", "control_kind", "alpha",
    "row_index", "question_id", "label", "category", "score", "next_token_kl",
    "top1_changed", "logit_rms_change",
]
MEDIATION_COLUMNS = [
    "condition_id", "owner_candidate_id", "component_role", "control_kind", "alpha",
    "pair_id", "base_id", "relation_family", "relevance", "variant", "score",
    "correct_answer", "foil_answer", "correct_logprob", "foil_logprob", "answer_log_odds",
    "next_token_kl", "top1_changed", "logit_rms_change",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
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
    temporary.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["prepare", "sanity", "gap", "gap-report", "mediation", "report"])
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def validate_execution_authorization(args: argparse.Namespace, config: dict[str, Any], p: dict[str, Path]) -> None:
    if args.authorization is None:
        raise RuntimeError("--authorization is required for GPU execution")
    authorization = read_json(args.authorization)
    expected = {
        "config_sha256": sha256_file(Path(config["config_path"])),
        "implementation_sha256": sha256_file(Path(__file__)),
        "static_review_sha256": sha256_file(SCRIPT_DIR / "STATIC_REVIEW_FULL20.md"),
        "design_sha256": sha256_file(p["design"]),
        "conditions_sha256": sha256_file(p["conditions"]),
        "means_sha256": sha256_file(p["means"]),
        "dataset_sha256": sha256_file(Path(config["dataset_path"])),
        "split_sha256": sha256_file(Path(config["split_path"])),
        "direction_sha256": sha256_file(Path(config["stage_a_root"]) / "checkpoints" / "discovery_liref_directions.pt"),
    }
    if authorization.get("status") != "FROZEN_EXECUTION_AUTHORIZED":
        raise RuntimeError("Execution authorization is not frozen/authorized")
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise RuntimeError(f"Authorization hash mismatch for {key}")
    current_model_hash, _ = model_parameter_checksum(Path(config["model_path"]))
    if authorization.get("model_parameters_sha256") != current_model_hash:
        raise RuntimeError("Authorization model parameter hash mismatch")
    if authorization.get("candidate_count") != len(config["causal_candidates"]):
        raise RuntimeError("Authorization candidate count mismatch")
    if authorization.get("gpu_id") != config["gpu_id"] or authorization.get("batch_size") != config["batch_size"]:
        raise RuntimeError("Authorization runtime setting mismatch")


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(path)
    if args.gpu_id is not None:
        config["gpu_id"] = args.gpu_id
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    config["config_path"] = str(path.resolve())
    config["config_hash"] = canonical_hash({k: v for k, v in config.items() if k not in {"config_path", "config_hash"}})
    return config


def paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["output_root"])
    return {
        "root": root,
        "manifests": root / "manifests",
        "tables": root / "tables",
        "status": root / "status",
        "logs": root / "logs",
        "figures": root / "figures",
        "cards": root / "candidate_cards",
        "design": root / "manifests" / "causal_design.json",
        "conditions": root / "manifests" / "intervention_conditions.json",
        "pairs": root / "manifests" / "mediation_pairs.csv",
        "means": root / "manifests" / "discovery_neuron_reference_means.json",
    }


def ensure_dirs(p: dict[str, Path]) -> None:
    for key in ("root", "manifests", "tables", "status", "logs", "figures", "cards"):
        p[key].mkdir(parents=True, exist_ok=True)


def write_status(p: dict[str, Path], phase: str, **payload: Any) -> None:
    write_json(p["status"] / f"{phase}.json", {"phase": phase, "status": "PASS", "timestamp": utc_now(), **payload})


def require_status(p: dict[str, Path], phase: str) -> None:
    target = p["status"] / f"{phase}.json"
    if not target.exists() or read_json(target).get("status") != "PASS":
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


def apply_component_edit(array: np.ndarray, component_type: str, component_index: int, alpha: float, head_dim: int, mean: float) -> np.ndarray:
    result = array.copy()
    if component_type == "head":
        start = component_index * head_dim
        result[:, -1, start : start + head_dim] *= 1.0 - alpha
    else:
        result[:, -1, component_index] = (1.0 - alpha) * result[:, -1, component_index] + alpha * mean
    return result


@dataclass(frozen=True)
class Condition:
    condition_id: str
    owner_candidate_id: str
    component_role: str
    control_kind: str
    alpha: float
    components: tuple[dict[str, Any], ...]

    def payload(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "owner_candidate_id": self.owner_candidate_id,
            "component_role": self.component_role,
            "control_kind": self.control_kind,
            "alpha": self.alpha,
            "components": list(self.components),
        }


class Intervention:
    def __init__(self, model: Any, condition: Condition, neuron_means: dict[str, float]) -> None:
        self.model = model
        self.condition = condition
        self.neuron_means = neuron_means
        self.handles: list[Any] = []

    def install(self) -> None:
        for component in self.condition.components:
            layer = self.model.model.layers[component["module_index"]]
            module = layer.self_attn.o_proj if component["component_type"] == "head" else layer.mlp.down_proj
            self.handles.append(module.register_forward_pre_hook(self._hook(component)))

    def _hook(self, component: dict[str, Any]):
        alpha = float(self.condition.alpha)
        index = int(component["component_index"])
        component_id = component["component_id"]
        component_type = component["component_type"]
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads

        def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if alpha == 0.0:
                return None
            values = args[0].clone()
            if component_type == "head":
                start = index * head_dim
                values[:, -1, start : start + head_dim] *= 1.0 - alpha
            else:
                mean = torch.as_tensor(self.neuron_means[component_id], device=values.device, dtype=values.dtype)
                values[:, -1, index] = (1.0 - alpha) * values[:, -1, index] + alpha * mean
            return (values, *args[1:])

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class FinalCapture:
    def __init__(self, model: Any, layer_index: int) -> None:
        self.model = model
        self.layer_index = layer_index
        self.handle: Any | None = None
        self.value: torch.Tensor | None = None

    def install(self) -> None:
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.value = tensor.detach()[:, -1, :].clone()
        self.handle = self.model.model.layers[self.layer_index].register_forward_hook(hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def build_conditions(config: dict[str, Any], controls_payload: dict[str, Any]) -> list[Condition]:
    candidates = list(config["causal_candidates"])
    output = [Condition("baseline", "baseline", "baseline", "baseline", 0.0, tuple())]
    for candidate_id in candidates:
        component = component_from_id(candidate_id)
        for alpha in config["candidate_alphas"]:
            output.append(Condition(f"candidate::{candidate_id}::a{alpha:g}", candidate_id, "candidate", "candidate", float(alpha), (component,)))
        for row in controls_payload["controls"]:
            if row["candidate_id"] != candidate_id:
                continue
            control = component_from_id(row["control_id"])
            output.append(Condition(
                f"control::{candidate_id}::{row['control_kind']}::{row['control_id']}::a1",
                candidate_id, "control", row["control_kind"], float(config["control_alpha"]), (control,),
            ))
    if bool(config.get("include_joint_intervention", True)):
        joint_name = f"joint{len(candidates)}"
        joint = tuple(component_from_id(value) for value in candidates)
        for alpha in (0.5, 1.0):
            output.append(Condition(f"{joint_name}::a{alpha:g}", joint_name, "joint", "joint", alpha, joint))
    if len({row.condition_id for row in output}) != len(output):
        raise RuntimeError("Duplicate intervention condition ID")
    return output


SUBJECTS = ["Mira", "Noah", "Lena", "Owen", "Asha", "Evan", "Iris", "Theo", "Nora", "Liam", "Sofia", "Mason"]
ITEMS = ["tokens", "cards", "shells", "beads", "tickets", "stamps", "coins", "tiles", "markers", "folders", "keys", "blocks"]
RELATIONS = [
    ("credited_debited", "was credited with", "was debited", "credited with", "debited"),
    ("deposited_withdrew", "deposited", "withdrew", "deposited", "withdrew"),
    ("awarded_forfeited", "was awarded", "forfeited", "awarded", "forfeited"),
    ("earned_spent", "earned", "spent", "earned", "spent"),
]


def make_mediation_pairs(context_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(context_count):
        agent = SUBJECTS[i % len(SUBJECTS)]
        distractor = SUBJECTS[(i + 5) % len(SUBJECTS)]
        item = ITEMS[(i * 5 + 1) % len(ITEMS)]
        other_item = ITEMS[(i * 7 + 3) % len(ITEMS)]
        start = 12 + (i % 9)
        delta = 2 + (i % 6)
        other_start = 10 + (i % 7)
        other_delta = 2 + ((i + 2) % 5)
        family, positive, negative, positive_span, negative_span = RELATIONS[i % len(RELATIONS)]
        if family == "credited_debited":
            pos_clause = f"{agent}'s account {positive} {delta} {item}."
            neg_clause = f"{agent}'s account {negative} {delta} {item}."
        elif family == "deposited_withdrew":
            pos_clause = f"{agent} {positive} {delta} {item}."
            neg_clause = f"{agent} {negative} {delta} {item}."
        elif family == "awarded_forfeited":
            pos_clause = f"{agent} {positive} {delta} {item}."
            neg_clause = f"{agent} {negative} {delta} {item}."
        else:
            pos_clause = f"{agent} {positive} {delta} {item}."
            neg_clause = f"{agent} {negative} {delta} {item}."
        base_id = f"C{i:03d}"
        relevant_original = f"{agent} started with {start} {item}. {pos_clause} How many {item} does {agent} have now?"
        relevant_modified = f"{agent} started with {start} {item}. {neg_clause} How many {item} does {agent} have now?"
        irrelevant_original = (
            f"{agent} started with {start} {item}. {pos_clause} "
            f"{distractor} started with {other_start} {other_item} and received {other_delta} more. "
            f"How many {other_item} does {distractor} have now?"
        )
        irrelevant_modified = irrelevant_original.replace(pos_clause, neg_clause)
        for relevance, original, modified, answer_original, answer_modified, foil_original, foil_modified in (
            ("relevant", relevant_original, relevant_modified, start + delta, start - delta, start - delta, start + delta),
            ("irrelevant", irrelevant_original, irrelevant_modified, other_start + other_delta, other_start + other_delta, other_start - other_delta, other_start - other_delta),
        ):
            rows.append({
                "pair_id": f"{base_id}::{relevance}", "base_id": base_id, "relation_family": family,
                "relevance": relevance, "original_text": original, "modified_text": modified,
                "changed_span_original": positive_span, "changed_span_modified": negative_span,
                "expected_answer_original": str(answer_original), "expected_answer_modified": str(answer_modified),
                "foil_answer_original": str(foil_original), "foil_answer_modified": str(foil_modified),
                "approved": True, "approval_basis": "frozen_deterministic_generator_and_protocol_review",
                "individual_human_review": False,
            })
    return rows


def iter_batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def encode_prompts(tokenizer: Any, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=False, return_token_type_ids=False)
    if not bool(torch.all(encoded["attention_mask"][:, -1] == 1)):
        raise RuntimeError("Last index is not the final prompt token")
    return {key: value.to(device) for key, value in encoded.items()}


def output_diagnostics(logits: torch.Tensor, baseline_logits: torch.Tensor | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = logits.shape[0]
    if baseline_logits is None:
        zeros = np.zeros(batch, dtype=np.float64)
        return zeros, zeros.astype(np.int8), zeros
    base = baseline_logits.to(logits.device, dtype=logits.dtype)
    log_q = F.log_softmax(logits.float(), dim=-1)
    p = F.softmax(base.float(), dim=-1)
    kl = (p * (F.log_softmax(base.float(), dim=-1) - log_q)).sum(dim=-1)
    changed = logits.argmax(dim=-1) != base.argmax(dim=-1)
    rms = (logits.float() - base.float()).square().mean(dim=-1).sqrt()
    return kl.cpu().numpy(), changed.to(torch.int8).cpu().numpy(), rms.cpu().numpy()


@torch.inference_mode()
def infer_prompts(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    condition: Condition,
    direction: np.ndarray,
    neuron_means: dict[str, float],
    batch_size: int,
    baseline_logits: torch.Tensor | None = None,
    answer_pairs: list[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, torch.Tensor]:
    capture = FinalCapture(model, 31)
    capture.install()
    intervention = Intervention(model, condition, neuron_means)
    intervention.install()
    direction_tensor = torch.as_tensor(direction, device=device, dtype=next(model.parameters()).dtype)
    rows: list[dict[str, Any]] = []
    logits_parts: list[torch.Tensor] = []
    offset = 0
    try:
        for batch_prompts in iter_batches(prompts, batch_size):
            encoded = encode_prompts(tokenizer, batch_prompts, device)
            output = model(**encoded, use_cache=False, return_dict=True)
            if capture.value is None:
                raise RuntimeError("Final hidden state was not captured")
            logits = output.logits[:, -1, :]
            scores = torch.mv(capture.value, direction_tensor).float().cpu().numpy()
            base_slice = None if baseline_logits is None else baseline_logits[offset : offset + len(batch_prompts)]
            kl, top1, rms = output_diagnostics(logits, base_slice)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            for local in range(len(batch_prompts)):
                row = {"score": float(scores[local]), "next_token_kl": float(kl[local]), "top1_changed": int(top1[local]), "logit_rms_change": float(rms[local])}
                if answer_pairs is not None:
                    correct, foil = answer_pairs[offset + local]
                    row.update({
                        "correct_logprob": float(log_probs[local, correct].item()),
                        "foil_logprob": float(log_probs[local, foil].item()),
                        "answer_log_odds": float((log_probs[local, correct] - log_probs[local, foil]).item()),
                    })
                rows.append(row)
            logits_parts.append(logits.detach().to("cpu", dtype=torch.float16))
            offset += len(batch_prompts)
            capture.value = None
            del output, logits, encoded
    finally:
        intervention.remove()
        capture.remove()
    return pd.DataFrame(rows), torch.cat(logits_parts, dim=0)


def gap_metrics(base: np.ndarray, intervention: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    base_g = float(intervention[labels == GROUP_REASONING].mean() * 0 + base[labels == GROUP_REASONING].mean() - base[labels == GROUP_MEMORY].mean())
    int_g = float(intervention[labels == GROUP_REASONING].mean() - intervention[labels == GROUP_MEMORY].mean())
    return {
        "G_base": base_g, "G_intervention": int_g, "delta_G": int_g - base_g,
        "abs_G_base": abs(base_g), "abs_G_intervention": abs(int_g),
        "delta_abs_G": abs(int_g) - abs(base_g), "gap_reduction": abs(base_g) - abs(int_g),
    }


def bootstrap_gap(base: np.ndarray, intervention: np.ndarray, labels: np.ndarray, reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    memory = np.flatnonzero(labels == GROUP_MEMORY)
    reasoning = np.flatnonzero(labels == GROUP_REASONING)
    memory_draws = rng.choice(memory, (reps, len(memory)), replace=True)
    reasoning_draws = rng.choice(reasoning, (reps, len(reasoning)), replace=True)
    base_g = base[reasoning_draws].mean(axis=1) - base[memory_draws].mean(axis=1)
    intervention_g = intervention[reasoning_draws].mean(axis=1) - intervention[memory_draws].mean(axis=1)
    values = np.column_stack([intervention_g - base_g, np.abs(base_g) - np.abs(intervention_g)])
    return {
        "delta_G_ci_low": float(np.quantile(values[:, 0], 0.025)),
        "delta_G_ci_high": float(np.quantile(values[:, 0], 0.975)),
        "gap_reduction_ci_low": float(np.quantile(values[:, 1], 0.025)),
        "gap_reduction_ci_high": float(np.quantile(values[:, 1], 0.975)),
    }


def permutation_group_difference(changes: np.ndarray, labels: np.ndarray, reps: int, seed: int) -> float:
    observed = abs(float(changes[labels == GROUP_REASONING].mean() - changes[labels == GROUP_MEMORY].mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(reps):
        shuffled = rng.permutation(labels)
        statistic = abs(float(changes[shuffled == GROUP_REASONING].mean() - changes[shuffled == GROUP_MEMORY].mean()))
        exceed += statistic >= observed
    return float((exceed + 1) / (reps + 1))


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def prepare(config: dict[str, Any], p: dict[str, Path]) -> None:
    ensure_dirs(p)
    stage_a = Path(config["stage_a_root"])
    stage_b = Path(config["stage_b_root"])
    extension = Path(config["stage_b_extension_root"])
    a_identity = read_json(stage_a / "checkpoints" / "input_manifest.json")
    a_config = read_json(stage_a / "checkpoints" / "frozen_config.json")
    if a_config["prompt_template"] != config["prompt_template"]:
        raise RuntimeError("Prompt template differs from Stage A")
    candidate_payload = read_json(extension / "manifests" / "frozen_stage_b_candidates.json")
    controls_payload = read_json(extension / "manifests" / "frozen_control_components.json")
    frozen = set(config["causal_candidates"])
    known = {row["component_id"] for row in candidate_payload["candidates"]}
    expected_count = int(config.get("expected_candidate_count", 5))
    if not frozen.issubset(known) or len(frozen) != expected_count or len(config["causal_candidates"]) != expected_count:
        raise RuntimeError(f"Causal candidate set is not the frozen {expected_count}-candidate set")
    conditions = build_conditions(config, controls_payload)

    natural = pd.read_csv(stage_b / "tables" / "natural_responses.csv.gz")
    needed_neurons = {
        component["component_id"] for condition in conditions for component in condition.components
        if component["component_type"] == "neuron"
    }
    discovery = natural[(natural["analysis_split"] == "discovery") & natural["component_id"].isin(needed_neurons)]
    means = discovery.groupby("component_id", sort=True)["activation"].mean().to_dict()
    if set(means) != needed_neurons or not all(math.isfinite(float(v)) for v in means.values()):
        raise RuntimeError(f"Missing Discovery neuron means: {sorted(needed_neurons.difference(means))}")
    write_json(p["means"], {"source": str((stage_b / "tables" / "natural_responses.csv.gz").resolve()), "analysis_split": "discovery", "means": means})

    pairs = pd.DataFrame(make_mediation_pairs(int(config["mediation_context_count"])))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    for variant in ("original", "modified"):
        pairs[f"token_length_{variant}"] = [len(tokenizer(config["prompt_template"].format(question=text), add_special_tokens=True)["input_ids"]) for text in pairs[f"{variant}_text"]]
        for answer_type in ("expected_answer", "foil_answer"):
            # The frozen prompt already ends in a standalone whitespace token ("A: ").
            # Therefore the answer continuation itself is tokenized without another prefix space.
            token_ids = [tokenizer(str(value), add_special_tokens=False)["input_ids"] for value in pairs[f"{answer_type}_{variant}"]]
            if any(len(value) != 1 for value in token_ids):
                raise RuntimeError("Controlled answers must each be exactly one token with a leading space")
            pairs[f"{answer_type}_{variant}_token_id"] = [value[0] for value in token_ids]
    atomic_csv(p["pairs"], pairs)
    write_json(p["conditions"], {"conditions": [row.payload() for row in conditions]})

    parameter_checksum, parameter_files = model_parameter_checksum(Path(config["model_path"]))
    code_files = {path.name: sha256_file(path) for path in sorted(SCRIPT_DIR.iterdir()) if path.is_file()}
    design = {
        "causal_run_id": config["causal_run_id"], "config_hash": config["config_hash"],
        "candidate_selection_source": config.get("candidate_selection_source", "Stage B b04 preregistered robust lexical/relevance intersection"),
        "candidate_set": config["causal_candidates"], "candidate_set_hash": canonical_hash(config["causal_candidates"]),
        "primary_population": "Stage A validation/heldout split (600 natural MMLU-Pro questions)",
        "primary_endpoint": "Layer 31 last-prompt-token h_out projected on frozen Discovery LiReF direction",
        "signed_gap": "G=mean(score|R)-mean(score|M)", "primary_effect": "delta_abs_G=abs(G_int)-abs(G_base)",
        "desired_direction": "delta_abs_G < 0", "intervention_position": config["intervention_position"],
        "head_formula": config["head_suppression_formula"], "neuron_formula": config["neuron_clamp_formula"],
        "neuron_reference": "pooled Discovery activation mean frozen before causal inference",
        "alphas": config["candidate_alphas"], "controls": "one Stage B matched plus three frozen random controls per candidate",
        "joint_intervention": (
            f"all {len(config['causal_candidates'])} candidates at alpha 0.5 and 1.0"
            if bool(config.get("include_joint_intervention", True)) else "not_performed"
        ),
        "mediation_endpoint": "attenuation of relevant-minus-irrelevant relation manipulation interaction on new frozen templates",
        "behavior_endpoint": "correct-versus-foil one-token answer log odds on new controlled prompts",
        "global_disruption_diagnostics": ["next-token KL(base||intervention)", "top-1 token change rate", "logit RMS change"],
        "statistics": {"bootstrap_replicates": config["bootstrap_replicates"], "permutation_replicates": config["permutation_replicates"], "BH_FDR": config["fdr_alpha"]},
        "interpretation_boundary": config["interpretation_boundary"],
        "hashes": {
            "stage_a_identity_hash": a_identity["identity_hash"],
            "stage_a_direction": sha256_file(stage_a / "checkpoints" / "discovery_liref_directions.pt"),
            "stage_b_candidates": sha256_file(extension / "manifests" / "frozen_stage_b_candidates.json"),
            "stage_b_controls": sha256_file(extension / "manifests" / "frozen_control_components.json"),
            "stage_b_extension_summary": sha256_file(extension / "stage_b_extension_summary.json"),
            "dataset": sha256_file(Path(config["dataset_path"])), "split": sha256_file(Path(config["split_path"])),
            "prompt": sha256_text(config["prompt_template"]), "model_parameters": parameter_checksum,
            "model_parameter_files": parameter_files, "code": canonical_hash(code_files), "code_files": code_files,
        },
        "individual_controlled_pair_human_review": False,
        "controlled_pair_review_note": "Pairs are deterministically generated from a frozen, user-approved protocol and automatically validated; individual human pair review was not claimed.",
    }
    write_json(p["design"], design)
    write_json(p["manifests"] / "frozen_config.json", config)
    write_status(p, "prepare", condition_count=len(conditions), mediation_pair_count=len(pairs), hashes={"design": sha256_file(p["design"]), "conditions": sha256_file(p["conditions"]), "pairs": sha256_file(p["pairs"]), "means": sha256_file(p["means"])})


def load_runtime(config: dict[str, Any], p: dict[str, Path]) -> tuple[Any, Any, torch.device, np.ndarray, dict[str, float], list[Condition]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    model, tokenizer = load_model_and_tokenizer(config, device)
    directions = load_directions(Path(config["stage_a_root"]) / "checkpoints" / "discovery_liref_directions.pt")
    means = {k: float(v) for k, v in read_json(p["means"])["means"].items()}
    conditions = [Condition(
        row["condition_id"], row["owner_candidate_id"], row["component_role"], row["control_kind"],
        float(row["alpha"]), tuple(row["components"]),
    ) for row in read_json(p["conditions"])["conditions"]]
    return model, tokenizer, device, directions, means, conditions


def sanity(config: dict[str, Any], p: dict[str, Path]) -> None:
    require_status(p, "prepare")
    before, _ = model_parameter_checksum(Path(config["model_path"]))
    model = None
    try:
        model, tokenizer, device, directions, means, conditions = load_runtime(config, p)
        data = load_dataset_and_split(config)
        indices = data["indices"]["validation"][:4]
        prompts = [data["prompts"][int(i)] for i in indices]
        baseline = next(row for row in conditions if row.condition_id == "baseline")
        sham = next(row for row in conditions if row.condition_id == f"candidate::{config['causal_candidates'][0]}::a0")
        full_head = next(row for row in conditions if row.condition_id == f"candidate::{config['causal_candidates'][0]}::a1")
        base_frame, base_logits = infer_prompts(model, tokenizer, device, prompts, baseline, directions[31], means, len(prompts))
        sham_frame, sham_logits = infer_prompts(model, tokenizer, device, prompts, sham, directions[31], means, len(prompts), base_logits)
        head_frame, _ = infer_prompts(model, tokenizer, device, prompts, full_head, directions[31], means, len(prompts), base_logits)
        score_error = float(np.max(np.abs(base_frame["score"].to_numpy() - sham_frame["score"].to_numpy())))
        logit_error = float((base_logits.float() - sham_logits.float()).abs().max().item())
        if score_error != 0.0 or logit_error != 0.0:
            raise RuntimeError(f"alpha=0 sham is not exact: score={score_error}, logits={logit_error}")
        if not bool((head_frame["next_token_kl"] >= 0).all()):
            raise RuntimeError("Negative KL divergence")
    finally:
        release_model(model)
    after, _ = model_parameter_checksum(Path(config["model_path"]))
    if before != after:
        raise RuntimeError("Model parameter checksum changed")
    write_status(p, "sanity", alpha0_score_max_abs=score_error, alpha0_logit_max_abs=logit_error, full_head_mean_kl=float(head_frame["next_token_kl"].mean()), model_checksum_before=before, model_checksum_after=after)


def gap_phase(config: dict[str, Any], p: dict[str, Path]) -> None:
    require_status(p, "sanity")
    model = None
    try:
        model, tokenizer, device, directions, means, conditions = load_runtime(config, p)
        data = load_dataset_and_split(config)
        indices = [int(value) for value in data["indices"]["validation"]]
        # Length bucketing changes only execution order. row_index restores the frozen
        # validation order during analysis and avoids excessive left-padding compute.
        indices.sort(key=lambda value: len(tokenizer(data["prompts"][value], add_special_tokens=True)["input_ids"]))
        prompts = [data["prompts"][value] for value in indices]
        metadata = pd.DataFrame({
            "row_index": indices,
            "question_id": [data["question_ids"][value] for value in indices],
            "label": [int(data["labels"][value]) for value in indices],
            "category": [data["records"][value]["category"] for value in indices],
        })
        all_frames: list[pd.DataFrame] = []
        baseline_logits: torch.Tensor | None = None
        baseline_inference: pd.DataFrame | None = None
        for number, condition in enumerate(conditions, 1):
            print(f"[gap {number}/{len(conditions)}] {condition.condition_id}", flush=True)
            if condition.component_role == "candidate" and condition.alpha == 0.0:
                if baseline_inference is None or baseline_logits is None:
                    raise RuntimeError("Baseline must precede sham conditions")
                frame = baseline_inference.copy()
                logits = baseline_logits
            else:
                frame, logits = infer_prompts(model, tokenizer, device, prompts, condition, directions[31], means, int(config["batch_size"]), baseline_logits)
            if condition.condition_id == "baseline":
                baseline_logits = logits
                baseline_inference = frame.copy()
            frame = pd.concat([metadata.reset_index(drop=True), frame], axis=1)
            for key in ("condition_id", "owner_candidate_id", "component_role", "control_kind", "alpha"):
                frame[key] = getattr(condition, key)
            all_frames.append(frame[GAP_COLUMNS])
            if condition.condition_id != "baseline":
                del logits
        atomic_csv(p["tables"] / "gap_responses.csv.gz", pd.concat(all_frames, ignore_index=True))
    finally:
        release_model(model)
    write_status(p, "gap", response_rows=sum(len(frame) for frame in all_frames), validation_samples=len(metadata), condition_count=len(conditions))


def mediation_phase(config: dict[str, Any], p: dict[str, Path]) -> None:
    require_status(p, "gap")
    pairs = pd.read_csv(p["pairs"])
    prompt_rows: list[dict[str, Any]] = []
    for row in pairs.to_dict("records"):
        for variant in ("original", "modified"):
            prompt_rows.append({
                "pair_id": row["pair_id"], "base_id": row["base_id"], "relation_family": row["relation_family"],
                "relevance": row["relevance"], "variant": variant, "question": row[f"{variant}_text"],
                "correct_answer": str(row[f"expected_answer_{variant}"]), "foil_answer": str(row[f"foil_answer_{variant}"]),
                "correct_token_id": int(row[f"expected_answer_{variant}_token_id"]), "foil_token_id": int(row[f"foil_answer_{variant}_token_id"]),
            })
    metadata = pd.DataFrame(prompt_rows)
    prompts = [config["prompt_template"].format(question=row["question"]) for row in prompt_rows]
    answer_pairs = [(row["correct_token_id"], row["foil_token_id"]) for row in prompt_rows]
    model = None
    try:
        model, tokenizer, device, directions, means, conditions = load_runtime(config, p)
        order = np.argsort([len(tokenizer(prompt, add_special_tokens=True)["input_ids"]) for prompt in prompts], kind="stable")
        prompts = [prompts[int(i)] for i in order]
        answer_pairs = [answer_pairs[int(i)] for i in order]
        metadata = metadata.iloc[order].reset_index(drop=True)
        all_frames: list[pd.DataFrame] = []
        baseline_logits: torch.Tensor | None = None
        baseline_inference: pd.DataFrame | None = None
        for number, condition in enumerate(conditions, 1):
            print(f"[mediation {number}/{len(conditions)}] {condition.condition_id}", flush=True)
            if condition.component_role == "candidate" and condition.alpha == 0.0:
                if baseline_inference is None or baseline_logits is None:
                    raise RuntimeError("Baseline must precede sham conditions")
                frame = baseline_inference.copy()
                logits = baseline_logits
            else:
                frame, logits = infer_prompts(model, tokenizer, device, prompts, condition, directions[31], means, int(config["batch_size"]), baseline_logits, answer_pairs)
            if condition.condition_id == "baseline":
                baseline_logits = logits
                baseline_inference = frame.copy()
            frame = pd.concat([metadata.reset_index(drop=True), frame], axis=1)
            for key in ("condition_id", "owner_candidate_id", "component_role", "control_kind", "alpha"):
                frame[key] = getattr(condition, key)
            all_frames.append(frame[MEDIATION_COLUMNS])
            if condition.condition_id != "baseline":
                del logits
        atomic_csv(p["tables"] / "mediation_responses.csv.gz", pd.concat(all_frames, ignore_index=True))
    finally:
        release_model(model)
    write_status(p, "mediation", response_rows=sum(len(frame) for frame in all_frames), controlled_prompts=len(metadata), condition_count=len(conditions))


def analyze_gap(config: dict[str, Any], p: dict[str, Path]) -> pd.DataFrame:
    frame = pd.read_csv(p["tables"] / "gap_responses.csv.gz")
    base = frame[frame["condition_id"] == "baseline"].sort_values("row_index")
    base_scores = base["score"].to_numpy()
    labels = base["label"].to_numpy(dtype=np.int8)
    rows = []
    for number, (condition_id, group) in enumerate(frame.groupby("condition_id", sort=False)):
        group = group.sort_values("row_index")
        values = group["score"].to_numpy()
        metrics = gap_metrics(base_scores, values, labels)
        metrics.update(bootstrap_gap(base_scores, values, labels, int(config["bootstrap_replicates"]), int(config["seed"]) + number))
        metrics["p_permutation_delta_G"] = permutation_group_difference(values - base_scores, labels, int(config["permutation_replicates"]), int(config["seed"]) + 10000 + number)
        first = group.iloc[0]
        rows.append({
            "condition_id": condition_id, "owner_candidate_id": first["owner_candidate_id"],
            "component_role": first["component_role"], "control_kind": first["control_kind"], "alpha": first["alpha"],
            **metrics, "mean_next_token_kl": float(group["next_token_kl"].mean()),
            "top1_change_rate": float(group["top1_changed"].mean()), "mean_logit_rms_change": float(group["logit_rms_change"].mean()),
        })
    output = pd.DataFrame(rows)
    if bool(config.get("permutation_fdr_full_alpha_only", False)):
        candidate_mask = (output["component_role"] == "candidate") & (output["alpha"] == 1.0)
    else:
        candidate_mask = (output["component_role"] == "candidate") & (output["alpha"] > 0)
    output["q_delta_G_candidates"] = np.nan
    output.loc[candidate_mask, "q_delta_G_candidates"] = bh_adjust(output.loc[candidate_mask, "p_permutation_delta_G"])
    atomic_csv(p["tables"] / "gap_effects.csv", output)

    comparisons = []
    score_by_condition = {
        condition_id: group.sort_values("row_index")["score"].to_numpy()
        for condition_id, group in frame.groupby("condition_id", sort=False)
    }
    memory = np.flatnonzero(labels == GROUP_MEMORY)
    reasoning = np.flatnonzero(labels == GROUP_REASONING)
    for candidate_number, candidate_id in enumerate(config["causal_candidates"]):
        candidate = output[(output["condition_id"] == f"candidate::{candidate_id}::a1")].iloc[0]
        controls = output[(output["owner_candidate_id"] == candidate_id) & (output["component_role"] == "control")]
        candidate_condition = f"candidate::{candidate_id}::a1"
        matched_condition = controls[controls["control_kind"] == "matched"]["condition_id"].iloc[0]
        random_conditions = controls[controls["control_kind"] == "random"]["condition_id"].tolist()
        rng = np.random.default_rng(int(config["seed"]) + 70000 + candidate_number)
        boot = np.empty((int(config["bootstrap_replicates"]), 2), dtype=np.float64)
        for i in range(len(boot)):
            sampled = np.concatenate([
                rng.choice(memory, len(memory), replace=True),
                rng.choice(reasoning, len(reasoning), replace=True),
            ])
            base_i = base_scores[sampled]
            labels_i = labels[sampled]
            candidate_reduction = gap_metrics(base_i, score_by_condition[candidate_condition][sampled], labels_i)["gap_reduction"]
            matched_reduction = gap_metrics(base_i, score_by_condition[matched_condition][sampled], labels_i)["gap_reduction"]
            random_reduction = np.mean([
                gap_metrics(base_i, score_by_condition[value][sampled], labels_i)["gap_reduction"]
                for value in random_conditions
            ])
            boot[i] = candidate_reduction - matched_reduction, candidate_reduction - random_reduction
        comparisons.append({
            "candidate_id": candidate_id, "candidate_gap_reduction": candidate["gap_reduction"],
            "matched_gap_reduction": float(controls[controls["control_kind"] == "matched"]["gap_reduction"].mean()),
            "random_mean_gap_reduction": float(controls[controls["control_kind"] == "random"]["gap_reduction"].mean()),
            "candidate_minus_matched": float(candidate["gap_reduction"] - controls[controls["control_kind"] == "matched"]["gap_reduction"].mean()),
            "candidate_minus_random_mean": float(candidate["gap_reduction"] - controls[controls["control_kind"] == "random"]["gap_reduction"].mean()),
            "candidate_minus_matched_ci_low": float(np.quantile(boot[:, 0], 0.025)),
            "candidate_minus_matched_ci_high": float(np.quantile(boot[:, 0], 0.975)),
            "candidate_minus_random_mean_ci_low": float(np.quantile(boot[:, 1], 0.025)),
            "candidate_minus_random_mean_ci_high": float(np.quantile(boot[:, 1], 0.975)),
            "dose_monotonic_abs_gap": bool(
                output[output["condition_id"] == f"candidate::{candidate_id}::a0"]["abs_G_intervention"].iloc[0]
                >= output[output["condition_id"] == f"candidate::{candidate_id}::a0.5"]["abs_G_intervention"].iloc[0]
                >= candidate["abs_G_intervention"]
            ),
        })
    atomic_csv(p["tables"] / "gap_control_comparisons.csv", pd.DataFrame(comparisons))
    return output


def _interaction(group: pd.DataFrame, value: str) -> tuple[float, pd.Series]:
    pivot = group.pivot_table(index=["base_id", "relevance"], columns="variant", values=value, aggfunc="first")
    pivot["pair_effect"] = pivot["modified"] - pivot["original"]
    wide = pivot["pair_effect"].unstack("relevance")
    per_base = wide["relevant"] - wide["irrelevant"]
    return float(per_base.mean()), per_base


def analyze_mediation(config: dict[str, Any], p: dict[str, Path]) -> pd.DataFrame:
    frame = pd.read_csv(p["tables"] / "mediation_responses.csv.gz")
    baseline = frame[frame["condition_id"] == "baseline"]
    base_score_interaction, base_score_vector = _interaction(baseline, "score")
    base_behavior_relevant = baseline[baseline["relevance"] == "relevant"].groupby("base_id")["answer_log_odds"].mean()
    base_behavior_irrelevant = baseline[baseline["relevance"] == "irrelevant"].groupby("base_id")["answer_log_odds"].mean()
    rows = []
    rng = np.random.default_rng(int(config["seed"]) + 50000)
    for condition_id, group in frame.groupby("condition_id", sort=False):
        score_interaction, score_vector = _interaction(group, "score")
        relevant = group[group["relevance"] == "relevant"].groupby("base_id")["answer_log_odds"].mean()
        irrelevant = group[group["relevance"] == "irrelevant"].groupby("base_id")["answer_log_odds"].mean()
        behavior_selectivity = float((relevant - base_behavior_relevant).mean() - (irrelevant - base_behavior_irrelevant).mean())
        attenuation = abs(base_score_interaction) - abs(score_interaction)
        base_ids = base_score_vector.index
        base_values = base_score_vector.to_numpy()
        score_values = score_vector.reindex(base_ids).to_numpy()
        relevant_change = (relevant - base_behavior_relevant).reindex(base_ids).to_numpy()
        irrelevant_change = (irrelevant - base_behavior_irrelevant).reindex(base_ids).to_numpy()
        draws = rng.integers(0, len(base_ids), size=(int(config["bootstrap_replicates"]), len(base_ids)))
        boot = np.column_stack([
            np.abs(base_values[draws].mean(axis=1)) - np.abs(score_values[draws].mean(axis=1)),
            relevant_change[draws].mean(axis=1) - irrelevant_change[draws].mean(axis=1),
        ])
        first = group.iloc[0]
        rows.append({
            "condition_id": condition_id, "owner_candidate_id": first["owner_candidate_id"],
            "component_role": first["component_role"], "control_kind": first["control_kind"], "alpha": first["alpha"],
            "baseline_score_interaction": base_score_interaction, "intervention_score_interaction": score_interaction,
            "score_interaction_attenuation": attenuation,
            "attenuation_ci_low": float(np.quantile(boot[:, 0], 0.025)), "attenuation_ci_high": float(np.quantile(boot[:, 0], 0.975)),
            "behavior_relevant_minus_irrelevant_change": behavior_selectivity,
            "behavior_selectivity_ci_low": float(np.quantile(boot[:, 1], 0.025)), "behavior_selectivity_ci_high": float(np.quantile(boot[:, 1], 0.975)),
            "mean_next_token_kl": float(group["next_token_kl"].mean()), "top1_change_rate": float(group["top1_changed"].mean()),
        })
    output = pd.DataFrame(rows)
    atomic_csv(p["tables"] / "mediation_effects.csv", output)
    return output


def gap_only_report(config: dict[str, Any], p: dict[str, Path]) -> None:
    """Report the frozen natural held-out gap test without synthetic mediation."""
    require_status(p, "gap")
    gaps = analyze_gap(config, p)
    comparisons = pd.read_csv(p["tables"] / "gap_control_comparisons.csv")
    candidates = gaps[(gaps["component_role"] == "candidate") & (gaps["alpha"] == 1.0)].copy()
    baseline = gaps[gaps["condition_id"] == "baseline"].iloc[0]
    frozen_path = Path(config["stage_b_extension_root"]) / "manifests" / "frozen_stage_b_candidates.json"
    frozen_payload = read_json(frozen_path)
    stage_a = {
        row["component_id"]: row["stage_a_metadata"]
        for row in frozen_payload["candidates"]
        if row["component_id"] in set(config["causal_candidates"])
    }
    if set(stage_a) != set(config["causal_candidates"]):
        raise RuntimeError("Frozen Stage A metadata does not cover every full-set candidate")

    cards: list[dict[str, Any]] = []
    for candidate_id in config["causal_candidates"]:
        gap_row = candidates[candidates["owner_candidate_id"] == candidate_id].iloc[0]
        control_row = comparisons[comparisons["candidate_id"] == candidate_id].iloc[0]
        metadata = stage_a[candidate_id]
        heldout_same_sign = bool(metadata.get("same_sign", False))
        heldout_q = float(metadata["bh_q_validation"])
        checks = {
            "heldout_same_sign": heldout_same_sign,
            "heldout_bh_q": bool(heldout_q < float(config["fdr_alpha"])),
            "gap_reduction_ci": bool(gap_row["gap_reduction_ci_low"] > 0),
            "permutation_bh_q": bool(gap_row["q_delta_G_candidates"] < float(config["fdr_alpha"])),
            "dose_monotonic": bool(control_row["dose_monotonic_abs_gap"]),
            "candidate_minus_matched_ci": bool(control_row["candidate_minus_matched_ci_low"] > 0),
            "candidate_minus_random_ci": bool(control_row["candidate_minus_random_mean_ci_low"] > 0),
        }
        card = {
            "candidate_id": candidate_id,
            "component_type": component_from_id(candidate_id)["component_type"],
            "Delta_discovery": float(metadata["Delta_discovery"]),
            "Delta_validation": float(metadata["Delta_validation"]),
            "heldout_bh_q": heldout_q,
            "G_base": float(gap_row["G_base"]),
            "G_intervention": float(gap_row["G_intervention"]),
            "gap_reduction": float(gap_row["gap_reduction"]),
            "gap_reduction_95ci": [float(gap_row["gap_reduction_ci_low"]), float(gap_row["gap_reduction_ci_high"])],
            "permutation_bh_q": float(gap_row["q_delta_G_candidates"]),
            "candidate_minus_matched": float(control_row["candidate_minus_matched"]),
            "candidate_minus_matched_95ci": [float(control_row["candidate_minus_matched_ci_low"]), float(control_row["candidate_minus_matched_ci_high"])],
            "candidate_minus_random_mean": float(control_row["candidate_minus_random_mean"]),
            "candidate_minus_random_mean_95ci": [float(control_row["candidate_minus_random_mean_ci_low"]), float(control_row["candidate_minus_random_mean_ci_high"])],
            "abs_G_alpha_0": float(gap_row["abs_G_base"]),
            "abs_G_alpha_1": float(gap_row["abs_G_intervention"]),
            "mean_next_token_kl": float(gap_row["mean_next_token_kl"]),
            "top1_change_rate": float(gap_row["top1_change_rate"]),
            "checks": checks,
            "strict_pass": bool(all(checks.values())),
        }
        write_json(p["cards"] / f"{candidate_id}.json", card)
        cards.append(card)

    pass_cards = [row for row in cards if row["strict_pass"]]
    summary = {
        "causal_run_id": config["causal_run_id"],
        "status": "COMPLETE",
        "scope": "frozen Meta-Llama Stage A detailed candidates; all candidates tested",
        "candidate_count": len(cards),
        "strict_pass_count": len(pass_cards),
        "strict_pass_heads": sum(row["component_type"] == "head" for row in pass_cards),
        "strict_pass_neurons": sum(row["component_type"] == "neuron" for row in pass_cards),
        "passing_components": [row["candidate_id"] for row in pass_cards],
        "primary_baseline_G": float(baseline["G_base"]),
        "primary_baseline_abs_G": float(baseline["abs_G_base"]),
        "candidate_cards": cards,
        "direct_count_comparison_warning": "This exhausts the frozen Meta-Llama 20-candidate list, but candidate nomination still differs from cross-model global top-5-per-type screening.",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(p["root"] / "full20_summary.json", summary)

    lines = [
        "# Meta-Llama-3-8B frozen 20-candidate 전수 인과 검증", "",
        "상태: **COMPLETE**", "",
        "## 한 줄 결론", "",
        f"Stage A에서 동결된 후보 20개를 모두 최종 시험에 올렸고, **{len(pass_cards)}/20개**가 heldout·gap 감소·dose·FDR·matched/random control의 엄격 기준을 모두 통과했다.", "",
        "| 후보 | 종류 | Discovery Δ | Heldout Δ | Gap 감소 | 95% CI | strict |", "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for card in cards:
        lines.append(
            f"| `{card['candidate_id']}` | {card['component_type']} | {card['Delta_discovery']:+.6f} | {card['Delta_validation']:+.6f} | "
            f"{card['gap_reduction']:+.6f} | [{card['gap_reduction_95ci'][0]:+.6f}, {card['gap_reduction_95ci'][1]:+.6f}] | "
            f"{'PASS' if card['strict_pass'] else 'FAIL'} |"
        )
    lines.extend([
        "", "## 해석", "",
        "- 특정 후보를 마지막 prompt token에서 약화했을 때 최종 R/M LiReF gap이 줄고, 그 감소가 matched/random control보다 크며 dose response와 통계 기준을 모두 만족해야 PASS다.",
        "- 이 run은 기존 Stage B의 5개 필터를 제거하고 Stage A의 frozen detailed 후보 20개 전부를 같은 최종 시험에 올렸다.",
        "- 다만 cross-model은 각 모델의 전역 positive-Delta top Head 5개+Neuron 5개를 지명했으므로, 이 결과와 cross-model PASS 개수는 여전히 component 총량의 직접 비교가 아니다.",
        "- synthetic mediation이나 새로운 후보 탐색은 수행하지 않았다.",
    ])
    (p["root"] / "RESULTS_FULL20_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_status(
        p, "gap-report", strict_pass_count=len(pass_cards),
        summary_sha256=sha256_file(p["root"] / "full20_summary.json"),
        report_sha256=sha256_file(p["root"] / "RESULTS_FULL20_KO.md"),
    )


def report(config: dict[str, Any], p: dict[str, Path]) -> None:
    require_status(p, "mediation")
    gaps = analyze_gap(config, p)
    mediation = analyze_mediation(config, p)
    comparisons = pd.read_csv(p["tables"] / "gap_control_comparisons.csv")
    candidates = gaps[(gaps["component_role"] == "candidate") & (gaps["alpha"] == 1.0)].copy()
    baseline = gaps[gaps["condition_id"] == "baseline"].iloc[0]
    joint_name = f"joint{len(config['causal_candidates'])}"
    joint = gaps[gaps["condition_id"] == f"{joint_name}::a1"].iloc[0]
    med_candidates = mediation[(mediation["component_role"] == "candidate") & (mediation["alpha"] == 1.0)]
    cards = []
    for candidate_id in config["causal_candidates"]:
        gap_row = candidates[candidates["owner_candidate_id"] == candidate_id].iloc[0]
        med_row = med_candidates[med_candidates["owner_candidate_id"] == candidate_id].iloc[0]
        control_row = comparisons[comparisons["candidate_id"] == candidate_id].iloc[0]
        verdict = (
            bool(gap_row["gap_reduction_ci_low"] > 0)
            and bool(gap_row["q_delta_G_candidates"] < config["fdr_alpha"])
            and bool(control_row["candidate_minus_matched_ci_low"] > 0)
            and bool(control_row["candidate_minus_random_mean_ci_low"] > 0)
            and bool(control_row["dose_monotonic_abs_gap"])
        )
        card = {
            "candidate_id": candidate_id, "causal_gap_criterion_pass": verdict,
            "gap_reduction": gap_row["gap_reduction"], "gap_reduction_95ci": [gap_row["gap_reduction_ci_low"], gap_row["gap_reduction_ci_high"]],
            "q_delta_G": gap_row["q_delta_G_candidates"], "dose_monotonic": control_row["dose_monotonic_abs_gap"],
            "candidate_minus_matched": control_row["candidate_minus_matched"], "candidate_minus_random_mean": control_row["candidate_minus_random_mean"],
            "candidate_minus_matched_95ci": [control_row["candidate_minus_matched_ci_low"], control_row["candidate_minus_matched_ci_high"]],
            "candidate_minus_random_mean_95ci": [control_row["candidate_minus_random_mean_ci_low"], control_row["candidate_minus_random_mean_ci_high"]],
            "relation_interaction_attenuation": med_row["score_interaction_attenuation"],
            "relation_attenuation_95ci": [med_row["attenuation_ci_low"], med_row["attenuation_ci_high"]],
            "relation_attenuation_supported": bool(med_row["attenuation_ci_low"] > 0),
            "behavior_relevant_minus_irrelevant_change": med_row["behavior_relevant_minus_irrelevant_change"],
            "behavior_selective_impairment_supported": bool(med_row["behavior_selectivity_ci_high"] < 0),
            "mean_next_token_kl": gap_row["mean_next_token_kl"], "top1_change_rate": gap_row["top1_change_rate"],
            "interpretation_boundary": config["interpretation_boundary"],
        }
        write_json(p["cards"] / f"{candidate_id}.json", card)
        cards.append(card)
    pass_count = sum(bool(row["causal_gap_criterion_pass"]) for row in cards)
    convergent_count = sum(
        bool(row["causal_gap_criterion_pass"] and row["behavior_selective_impairment_supported"])
        for row in cards
    )
    relation_supported_count = sum(bool(row["relation_attenuation_supported"]) for row in cards)
    summary = {
        "causal_run_id": config["causal_run_id"], "status": "PASS", "candidate_count": len(config["causal_candidates"]),
        "primary_baseline_G": baseline["G_base"], "primary_baseline_abs_G": baseline["abs_G_base"],
        "joint_full_G": joint["G_intervention"], "joint_full_abs_G": joint["abs_G_intervention"],
        "joint_full_gap_reduction": joint["gap_reduction"], "candidate_primary_criterion_pass_count": pass_count,
        "gap_and_behavior_convergent_count": convergent_count,
        "relation_attenuation_supported_count": relation_supported_count,
        "candidate_cards": cards, "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(p["root"] / "causal_summary.json", summary)

    lines = [
        "# LiReF R/M Causal Validation 결과", "", "## 한 줄 결론", "",
        f"동결된 {len(config['causal_candidates'])}개 후보 중 **{pass_count}/{len(config['causal_candidates'])}개**가 사전 정의한 단일-candidate 인과 gap 기준을 통과했습니다.", "",
        f"그중 **{convergent_count}개**는 관계가 답에 필요한 조건에서 정답-vs-foil 로그오즈도 선택적으로 감소해 표현·행동 증거가 함께 나타났습니다.", "",
        "이 결과는 아래 표와 통제 결과를 함께 읽어야 하며, 인과 기준을 통과하지 못한 경우에도 파일을 삭제하거나 후보를 교체하지 않았습니다.", "",
        "## 주 분석", "",
        f"- Baseline signed G: `{baseline['G_base']:.6f}`", f"- Baseline |G|: `{baseline['abs_G_base']:.6f}`",
        f"- 5개 동시 100% 조작 후 |G|: `{joint['abs_G_intervention']:.6f}`",
        f"- 5개 동시 조작 gap 감소량: `{joint['gap_reduction']:.6f}`", "",
        "| 후보 | |G| 감소 | 95% CI | matched 대비 | random 평균 대비 | dose 단조 | 관계효과 감쇠 | 행동 선택성 변화 | KL | 기준 통과 |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|",
    ]
    for card in cards:
        lines.append(
            f"| {card['candidate_id']} | {card['gap_reduction']:.6f} | [{card['gap_reduction_95ci'][0]:.6f}, {card['gap_reduction_95ci'][1]:.6f}] | "
            f"{card['candidate_minus_matched']:.6f} | {card['candidate_minus_random_mean']:.6f} | {'예' if card['dose_monotonic'] else '아니오'} | "
            f"{card['relation_interaction_attenuation']:.6f} | {card['behavior_relevant_minus_irrelevant_change']:.6f} | {card['mean_next_token_kl']:.6g} | {'PASS' if card['causal_gap_criterion_pass'] else 'FAIL'} |"
        )
    lines.extend([
        "", "## 정확한 해석", "",
        "- PASS는 해당 component를 마지막 prompt token에서 억제/clamp했을 때 validation R/M LiReF gap이 신뢰구간 기준으로 감소하고, 감소량이 frozen matched/random control보다 크며, 0→50→100% dose 순서가 단조였다는 뜻입니다.",
        "- 추가로 candidate ΔG permutation test가 후보군 BH-FDR 0.05를 통과해야 PASS입니다.",
        "- 관계효과 감쇠와 행동 선택성은 별도의 새 deterministic held-out 관계 문항에서 측정한 보조 인과 지표입니다.",
        f"- 관계 hidden-score interaction 감쇠는 95% CI 기준 **{relation_supported_count}/5개 후보**에서 지지됐습니다. 따라서 이 run만으로 특정 후보가 Stage B 관계 민감성을 매개한다고 확정할 수 없습니다.",
        f"- gap PASS와 관계-relevant 정답 로그오즈의 선택적 저하가 동시에 나온 후보는 **{convergent_count}개**이며, 이는 더 강한 수렴 증거이지만 행동 과제 범위는 synthetic arithmetic에 한정됩니다.",
        "- KL과 top-1 변화율은 단순한 전체 출력 붕괴 여부를 판단하는 진단값입니다.",
        "- 이 실험은 '이 모델·데이터·question-only prompt·마지막 token' 범위에서 component의 인과적 기여를 평가합니다. reasoning 전반의 단일 원인 또는 필요충분조건을 증명하지 않습니다.",
        "", "## 설계상 제한", "",
        "- controlled pair는 동결된 규칙 생성기와 자동 검증을 사용했으며, 개별 문항을 사람이 전수 검수했다고 주장하지 않습니다.",
        "- 행동 지표는 한 token 숫자 정답과 foil의 로그오즈이므로 자연 MMLU-Pro 정답 정확도와 동일하지 않습니다.",
        "- 첫 실험은 Stage A/B와 직접 대응하도록 모든 token이 아닌 마지막 prompt token만 조작했습니다.",
    ])
    (p["root"] / "RESULTS_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_code_files = {
        path.name: sha256_file(path)
        for path in sorted(SCRIPT_DIR.iterdir())
        if path.is_file() and path.suffix in {".py", ".json", ".sh", ".md"}
    }
    write_json(p["manifests"] / "final_analysis_code_provenance.json", {
        "timestamp": utc_now(), "code_hash": canonical_hash(final_code_files), "files": final_code_files,
        "note": "This is the exact report-time analysis code provenance. The inference design hash remains frozen in causal_design.json."
    })
    checksums = {
        str(path.relative_to(p["root"])): sha256_file(path)
        for path in sorted(p["root"].rglob("*"))
        if path.is_file()
        and path.name != "artifact_checksums.json"
        and path != p["status"] / "report.json"
    }
    write_json(p["root"] / "artifact_checksums.json", checksums)
    write_status(p, "report", candidate_primary_criterion_pass_count=pass_count, artifact_count=len(checksums), summary_sha256=sha256_file(p["root"] / "causal_summary.json"), report_sha256=sha256_file(p["root"] / "RESULTS_KO.md"))


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)
    p = paths(config)
    ensure_dirs(p)
    if args.phase in {"sanity", "gap"}:
        validate_execution_authorization(args, config, p)
    {"prepare": prepare, "sanity": sanity, "gap": gap_phase, "gap-report": gap_only_report, "mediation": mediation_phase, "report": report}[args.phase](config, p)


if __name__ == "__main__":
    main()
