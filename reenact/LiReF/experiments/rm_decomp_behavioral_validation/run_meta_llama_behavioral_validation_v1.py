#!/usr/bin/env python3
"""Same-sample exploratory behavioral validation for frozen Meta-Llama R/M components."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[3]
DESIGN_PATH = SCRIPT_DIR / "design_v1_frozen.json"
STATIC_REVIEW_PATH = SCRIPT_DIR / "STATIC_REVIEW_V1.md"
IMPLEMENTATION_PATH = Path(__file__).resolve()
COMPONENT_RE = re.compile(r"^L(\d{2})([HN])(\d{5})$")
GROUP_NAMES = {0: "M", 1: "R"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("execute", "report"), required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, key: str) -> int:
    payload = hashlib.sha256(f"{seed}::{key}".encode("utf-8")).digest()
    return int.from_bytes(payload[:8], "big") % (2**32)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def load_design() -> dict[str, Any]:
    design = read_json(DESIGN_PATH)
    if design.get("study_id") != "meta_llama_behavioral_validation_v1":
        raise RuntimeError("Unexpected study_id")
    if design.get("automatic_pdf_update_allowed") is not False:
        raise RuntimeError("Automatic PDF updates must remain prohibited")
    if design.get("evaluation_split") != "heldout_600_same_sample_exploratory":
        raise RuntimeError("Unexpected evaluation split")
    if design.get("intervention_position") != "last_prompt_token_only":
        raise RuntimeError("Only last-token intervention is allowed")
    if design.get("candidate_alphas") != [0.5, 1.0]:
        raise RuntimeError("Candidate alpha contract changed")
    return design


def parse_component(component_id: str) -> tuple[str, int, int]:
    match = COMPONENT_RE.fullmatch(component_id)
    if not match:
        raise ValueError(f"Invalid component id: {component_id}")
    return ("head" if match.group(2) == "H" else "neuron", int(match.group(1)), int(match.group(3)))


def validate_candidate_manifest(design: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(resolve(design["candidate_manifest_path"]))
    rows = manifest.get("candidates", [])
    if manifest.get("status") != "FROZEN_BEFORE_BEHAVIORAL_RESULTS" or len(rows) != 13:
        raise RuntimeError("Candidate manifest is not the frozen 13-candidate union")
    ids = [row["component_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate candidate ids")
    means: dict[str, float] = {}
    for row in rows:
        candidate_kind = parse_component(row["component_id"])[0]
        if parse_component(row["matched_control"])[0] != candidate_kind:
            raise RuntimeError("Matched control component type mismatch")
        if parse_component(row["random_control"])[0] != candidate_kind:
            raise RuntimeError("Random control component type mismatch")
        for component_id, value in row["neuron_means"].items():
            if parse_component(component_id)[0] != "neuron":
                raise RuntimeError("Head unexpectedly has a neuron mean")
            value = float(value)
            if component_id in means and not math.isclose(means[component_id], value, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"Conflicting neuron reference means: {component_id}")
            means[component_id] = value
        for component_id in (row["component_id"], row["matched_control"], row["random_control"]):
            if parse_component(component_id)[0] == "neuron" and component_id not in row["neuron_means"]:
                raise RuntimeError(f"Missing frozen neuron reference mean: {component_id}")
    manifest["neuron_means"] = means
    return manifest


def model_shard_manifest(model_path: Path) -> dict[str, str]:
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("No safetensor model shards")
    return {path.name: sha256_file(path) for path in shards}


def locked_inputs(design: dict[str, Any]) -> dict[str, Any]:
    model_path = resolve(design["model_path"])
    return {
        "design_sha256": sha256_file(DESIGN_PATH),
        "candidate_manifest_sha256": sha256_file(resolve(design["candidate_manifest_path"])),
        "dataset_sha256": sha256_file(resolve(design["dataset_path"])),
        "split_sha256": sha256_file(resolve(design["split_path"])),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "model_index_sha256": sha256_file(model_path / "model.safetensors.index.json"),
        "model_shards": model_shard_manifest(model_path),
    }


def validate_authorization(path: Path, design: dict[str, Any]) -> dict[str, Any]:
    authorization = read_json(path)
    if authorization.get("status") != "FROZEN_EXECUTION_AUTHORIZED":
        raise RuntimeError("Execution is not authorized")
    if authorization.get("run_id") != design["run_id"]:
        raise RuntimeError("Run id mismatch")
    if authorization.get("implementation_sha256") != sha256_file(IMPLEMENTATION_PATH):
        raise RuntimeError("Implementation hash mismatch")
    if authorization.get("static_review_sha256") != sha256_file(STATIC_REVIEW_PATH):
        raise RuntimeError("Static review hash mismatch")
    if authorization.get("locked_inputs") != locked_inputs(design):
        raise RuntimeError("Locked input hash mismatch")
    for key in ("logical_device", "dtype", "batch_size", "physical_gpu"):
        if authorization.get(key) != design.get(key):
            raise RuntimeError(f"Authorization setting mismatch: {key}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(design["physical_gpu"]):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match frozen physical GPU")
    return authorization


def build_prompt(record: dict[str, Any], design: dict[str, Any]) -> str:
    labels = design["answer_labels"]
    choices = "\n".join(f"{labels[index]}. {choice}" for index, choice in enumerate(record["options"]))
    return design["prompt_template"].format(question=record["question"], choices=choices)


def load_records(design: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = read_json(resolve(design["dataset_path"]))
    split = read_json(resolve(design["split_path"]))
    if split.get("dataset_sha256") != sha256_file(resolve(design["dataset_path"])):
        raise RuntimeError("Dataset does not match frozen split")
    indices = [int(value) for value in split["heldout"]["row_indices"]]
    if len(indices) != 600:
        raise RuntimeError("Expected 600 heldout rows")
    records = []
    for row_index in indices:
        source = dataset[row_index]
        answer_index = int(source["answer_index"])
        if not 0 <= answer_index < len(source["options"]) <= len(design["answer_labels"]):
            raise RuntimeError(f"Invalid answer/options at row {row_index}")
        records.append({
            "row_index": row_index,
            "question_id": str(source["question_id"]),
            "category": source["category"],
            "group": int(float(source["memory_reason_score"]) > float(design["score_threshold"])),
            "answer_index": answer_index,
            "option_count": len(source["options"]),
            "prompt": build_prompt(source, design),
        })
    if sum(row["group"] == 0 for row in records) != 324 or sum(row["group"] == 1 for row in records) != 276:
        raise RuntimeError("Frozen M/R heldout counts changed")
    return records


def answer_token_ids(tokenizer: Any, design: dict[str, Any]) -> list[int]:
    result = []
    for label in design["answer_labels"]:
        tokens = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(tokens) != 1:
            raise RuntimeError(f"Answer label is not one token: {label} -> {tokens}")
        result.append(int(tokens[0]))
    if len(set(result)) != len(result):
        raise RuntimeError("Answer label token ids are not unique")
    return result


def iter_batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class Intervention:
    def __init__(self, model: Any, component_id: str, alpha: float, neuron_mean: float | None) -> None:
        self.model = model
        self.component_id = component_id
        self.alpha = float(alpha)
        self.neuron_mean = neuron_mean
        self.handle: Any | None = None

    def install(self) -> None:
        kind, layer_index, component_index = parse_component(self.component_id)
        layer = self.model.model.layers[layer_index]
        module = layer.self_attn.o_proj if kind == "head" else layer.mlp.down_proj
        if kind == "neuron" and self.neuron_mean is None:
            raise RuntimeError(f"Neuron mean missing: {self.component_id}")

        def hook(_module: Any, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            values = args[0].clone()
            if kind == "head":
                head_dim = int(getattr(self.model.config, "head_dim", self.model.config.hidden_size // self.model.config.num_attention_heads))
                start = component_index * head_dim
                values[:, -1, start : start + head_dim] *= 1.0 - self.alpha
            else:
                reference = torch.as_tensor(self.neuron_mean, device=values.device, dtype=values.dtype)
                values[:, -1, component_index] = (1.0 - self.alpha) * values[:, -1, component_index] + self.alpha * reference
            return (values, *args[1:])

        self.handle = module.register_forward_pre_hook(hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def registered_hook_count(model: Any) -> int:
    return sum(
        len(getattr(module, "_forward_hooks", {}))
        + len(getattr(module, "_forward_pre_hooks", {}))
        + len(getattr(module, "_backward_hooks", {}))
        for module in model.modules()
    )


@torch.inference_mode()
def evaluate_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    records: list[dict[str, Any]],
    label_token_ids: list[int],
    batch_size: int,
    condition_id: str,
    component_id: str | None = None,
    alpha: float = 0.0,
    neuron_mean: float | None = None,
) -> pd.DataFrame:
    intervention = None
    if component_id is not None:
        intervention = Intervention(model, component_id, alpha, neuron_mean)
        intervention.install()
    rows = []
    label_tensor = torch.as_tensor(label_token_ids, device=device, dtype=torch.long)
    try:
        for batch in iter_batches(records, batch_size):
            encoded = tokenizer(
                [row["prompt"] for row in batch],
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if not bool(torch.all(encoded["attention_mask"][:, -1] == 1)):
                raise RuntimeError("Last tensor index is not the final prompt token")
            output = model(
                **encoded,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
            logits = output.logits[:, -1, :].float()
            option_logits = logits.index_select(1, label_tensor)
            vocab_top = logits.argmax(dim=-1)
            for local, record in enumerate(batch):
                count = int(record["option_count"])
                available = option_logits[local, :count]
                log_probs = F.log_softmax(available, dim=-1)
                probabilities = log_probs.exp()
                answer = int(record["answer_index"])
                predicted = int(available.argmax().item())
                foil = torch.cat((available[:answer], available[answer + 1 :])).max()
                rows.append({
                    "condition_id": condition_id,
                    "component_id": component_id or "baseline",
                    "alpha": float(alpha),
                    "row_index": record["row_index"],
                    "question_id": record["question_id"],
                    "category": record["category"],
                    "group": record["group"],
                    "option_count": count,
                    "answer_index": answer,
                    "predicted_index": predicted,
                    "forced_choice_correct": int(predicted == answer),
                    "correct_probability": float(probabilities[answer].item()),
                    "correct_log_probability": float(log_probs[answer].item()),
                    "correct_vs_best_foil_margin": float((available[answer] - foil).item()),
                    "vocab_top1_is_valid_label": int(int(vocab_top[local].item()) in label_token_ids[:count]),
                    "vocab_top1_is_correct_label": int(int(vocab_top[local].item()) == label_token_ids[answer]),
                })
            del output, logits, option_logits, encoded
    finally:
        if intervention is not None:
            intervention.remove()
    if registered_hook_count(model) != 0:
        raise RuntimeError("Intervention hook leaked after condition")
    return pd.DataFrame(rows)


def baseline_summary(frame: pd.DataFrame, design: dict[str, Any]) -> dict[str, Any]:
    groups = {}
    checks = []
    gate = design["baseline_solvability_gate"]
    for group_value, group_name in GROUP_NAMES.items():
        subset = frame[frame["group"] == group_value]
        accuracy = float(subset["forced_choice_correct"].mean())
        chance = float((1.0 / subset["option_count"]).mean())
        row_checks = {
            "accuracy_at_least_absolute_minimum": accuracy >= float(gate["minimum_forced_choice_accuracy_each_group"]),
            "accuracy_at_least_chance_plus_minimum": accuracy >= chance + float(gate["minimum_accuracy_above_mean_uniform_chance_each_group"]),
        }
        checks.extend(row_checks.values())
        groups[group_name] = {
            "n": int(len(subset)),
            "forced_choice_correct": int(subset["forced_choice_correct"].sum()),
            "forced_choice_accuracy": accuracy,
            "mean_uniform_chance": chance,
            "mean_correct_probability": float(subset["correct_probability"].mean()),
            "mean_correct_log_probability": float(subset["correct_log_probability"].mean()),
            "mean_margin": float(subset["correct_vs_best_foil_margin"].mean()),
            "valid_label_vocab_top1_rate": float(subset["vocab_top1_is_valid_label"].mean()),
            "correct_label_vocab_top1_rate": float(subset["vocab_top1_is_correct_label"].mean()),
            "checks": row_checks,
        }
    return {"status": "PASS" if all(checks) else "FAIL", "groups": groups}


def bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def bootstrap_mean_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(values), size=(reps, len(values)), replace=True)
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def signflip_p(values: np.ndarray, reps: int, seed: int) -> float:
    observed = float(values.mean())
    if observed <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(reps):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(values), replace=True)
        exceed += float((values * signs).mean()) >= observed
    return (exceed + 1) / (reps + 1)


def mcnemar_damage_p(base_correct: np.ndarray, changed_correct: np.ndarray) -> float:
    damaged = int(np.sum((base_correct == 1) & (changed_correct == 0)))
    improved = int(np.sum((base_correct == 0) & (changed_correct == 1)))
    discordant = damaged + improved
    if discordant == 0:
        return 1.0
    return float(stats.binomtest(damaged, discordant, p=0.5, alternative="greater").pvalue)


def aligned(frame: pd.DataFrame, condition_id: str, rows: list[int]) -> pd.DataFrame:
    subset = frame[frame["condition_id"] == condition_id].set_index("row_index").loc[rows]
    if subset.index.tolist() != rows:
        raise RuntimeError(f"Row alignment failed: {condition_id}")
    return subset


def analyze_candidates(
    responses: pd.DataFrame,
    manifest: dict[str, Any],
    design: dict[str, Any],
) -> pd.DataFrame:
    baseline = responses[responses["condition_id"] == "baseline"].set_index("row_index")
    rows = []
    reps = int(design["bootstrap_replicates"])
    permutations = int(design["permutation_replicates"])
    seed = int(design["seed"])
    for candidate in manifest["candidates"]:
        component = candidate["component_id"]
        for group_value, group_name in GROUP_NAMES.items():
            indices = baseline[baseline["group"] == group_value].index.tolist()
            base = baseline.loc[indices]
            half = aligned(responses, f"candidate::{component}::a0.5", indices)
            full = aligned(responses, f"candidate::{component}::a1", indices)
            matched = aligned(responses, f"control::{candidate['matched_control']}::a1", indices)
            random_control = aligned(responses, f"control::{candidate['random_control']}::a1", indices)
            accuracy_drop = base["forced_choice_correct"].to_numpy(float) - full["forced_choice_correct"].to_numpy(float)
            half_accuracy_drop = base["forced_choice_correct"].to_numpy(float) - half["forced_choice_correct"].to_numpy(float)
            probability_drop = base["correct_probability"].to_numpy(float) - full["correct_probability"].to_numpy(float)
            half_probability_drop = base["correct_probability"].to_numpy(float) - half["correct_probability"].to_numpy(float)
            logp_drop = base["correct_log_probability"].to_numpy(float) - full["correct_log_probability"].to_numpy(float)
            margin_drop = base["correct_vs_best_foil_margin"].to_numpy(float) - full["correct_vs_best_foil_margin"].to_numpy(float)
            matched_probability_drop = base["correct_probability"].to_numpy(float) - matched["correct_probability"].to_numpy(float)
            random_probability_drop = base["correct_probability"].to_numpy(float) - random_control["correct_probability"].to_numpy(float)
            candidate_minus_matched = probability_drop - matched_probability_drop
            candidate_minus_random = probability_drop - random_probability_drop
            key = f"{component}::{group_name}"
            row = {
                "component_id": component,
                "source": candidate["source"],
                "group": group_name,
                "n": len(indices),
                "baseline_correct": int(base["forced_choice_correct"].sum()),
                "alpha_0_5_correct": int(half["forced_choice_correct"].sum()),
                "alpha_1_correct": int(full["forced_choice_correct"].sum()),
                "accuracy_drop_alpha_0_5": float(half_accuracy_drop.mean()),
                "accuracy_drop": float(accuracy_drop.mean()),
                "accuracy_drop_ci": bootstrap_mean_ci(accuracy_drop, reps, stable_seed(seed, key + "::acc")),
                "accuracy_mcnemar_p": mcnemar_damage_p(base["forced_choice_correct"].to_numpy(int), full["forced_choice_correct"].to_numpy(int)),
                "probability_drop_alpha_0_5": float(half_probability_drop.mean()),
                "probability_drop": float(probability_drop.mean()),
                "probability_drop_ci": bootstrap_mean_ci(probability_drop, reps, stable_seed(seed, key + "::prob")),
                "probability_signflip_p": signflip_p(probability_drop, permutations, stable_seed(seed, key + "::sign")),
                "correct_log_probability_drop": float(logp_drop.mean()),
                "margin_drop": float(margin_drop.mean()),
                "matched_probability_drop": float(matched_probability_drop.mean()),
                "random_probability_drop": float(random_probability_drop.mean()),
                "candidate_minus_matched_probability_drop": float(candidate_minus_matched.mean()),
                "candidate_minus_matched_ci": bootstrap_mean_ci(candidate_minus_matched, reps, stable_seed(seed, key + "::matched")),
                "candidate_minus_random_probability_drop": float(candidate_minus_random.mean()),
                "candidate_minus_random_ci": bootstrap_mean_ci(candidate_minus_random, reps, stable_seed(seed, key + "::random")),
                "probability_dose_monotonic": bool(0.0 <= half_probability_drop.mean() <= probability_drop.mean()),
            }
            rows.append(row)
    result = pd.DataFrame(rows)
    result["accuracy_mcnemar_bh_q"] = bh(result["accuracy_mcnemar_p"].to_numpy())
    result["probability_signflip_bh_q"] = bh(result["probability_signflip_p"].to_numpy())
    threshold = float(design["fdr_alpha"])
    result["accuracy_signal"] = (
        result["accuracy_drop_ci"].map(lambda value: value[0] > 0)
        & (result["accuracy_mcnemar_bh_q"] < threshold)
    )
    result["probability_signal"] = (
        result["probability_drop_ci"].map(lambda value: value[0] > 0)
        & (result["probability_signflip_bh_q"] < threshold)
        & result["probability_dose_monotonic"]
        & result["candidate_minus_matched_ci"].map(lambda value: value[0] > 0)
        & result["candidate_minus_random_ci"].map(lambda value: value[0] > 0)
    )
    result["strict_behavioral_signal"] = result["accuracy_signal"] & result["probability_signal"]
    result["probability_only_signal"] = (~result["accuracy_signal"]) & result["probability_signal"]
    return result


def condition_plan(manifest: dict[str, Any], design: dict[str, Any]) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    means = manifest["neuron_means"]
    seen: set[tuple[str, float]] = set()
    for candidate in manifest["candidates"]:
        for alpha in design["candidate_alphas"]:
            key = (candidate["component_id"], float(alpha))
            if key not in seen:
                conditions.append({
                    "condition_id": f"candidate::{key[0]}::a{alpha:g}",
                    "component_id": key[0], "alpha": float(alpha),
                    "neuron_mean": means.get(key[0]),
                })
                seen.add(key)
        for control_key in ("matched_control", "random_control"):
            component = candidate[control_key]
            key = (component, float(design["control_alpha"]))
            if key not in seen:
                conditions.append({
                    "condition_id": f"control::{component}::a1",
                    "component_id": component, "alpha": float(design["control_alpha"]),
                    "neuron_mean": means.get(component),
                })
                seen.add(key)
    return conditions


def load_model(design: dict[str, Any], device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    model_path = resolve(design["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    model.config.output_hidden_states = False
    model.config.output_attentions = False
    checks = {
        "model_type": model.config.model_type == "llama",
        "layers_32": model.config.num_hidden_layers == 32,
        "hidden_size_4096": model.config.hidden_size == 4096,
        "intermediate_size_14336": model.config.intermediate_size == 14336,
        "attention_heads_32": model.config.num_attention_heads == 32,
        "key_value_heads_8": model.config.num_key_value_heads == 8,
        "float32": next(model.parameters()).dtype == torch.float32,
        "eval": not model.training,
        "no_initial_hooks": registered_hook_count(model) == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Model contract failed: {checks}")
    return model, tokenizer, checks


def release_model(model: Any | None) -> None:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def execute(args: argparse.Namespace, design: dict[str, Any], authorization: dict[str, Any]) -> None:
    if args.device != design["logical_device"]:
        raise RuntimeError("Logical CUDA device differs from frozen design")
    output = resolve(design["output_root"]) / design["run_id"]
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite run directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "RUNNING_BASELINE"})
    records = load_records(design)
    manifest = validate_candidate_manifest(design)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    torch.manual_seed(int(design["seed"]))
    torch.cuda.manual_seed_all(int(design["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = None
    try:
        model, tokenizer, model_checks = load_model(design, device)
        labels = answer_token_ids(tokenizer, design)
        records.sort(key=lambda row: len(tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]))
        print("[baseline] evaluating 600 heldout questions", flush=True)
        baseline = evaluate_condition(
            model, tokenizer, device, records, labels, int(design["batch_size"]), "baseline",
        )
        baseline_gate = baseline_summary(baseline, design)
        atomic_csv(output / "baseline_item_results.csv.gz", baseline)
        write_json(output / "baseline_summary.json", baseline_gate)
        if baseline_gate["status"] != "PASS":
            summary = {
                "study_id": design["study_id"], "run_id": design["run_id"],
                "status": "BASELINE_SOLVABILITY_FAIL_INTERVENTION_NOT_RUN",
                "baseline": baseline_gate, "candidate_intervention_run": False,
                "claim_boundary": design["interpretation"],
            }
            write_json(output / "summary.json", summary)
            write_json(output / "status.json", {"status": summary["status"]})
            print("[baseline] FAIL — candidate intervention blocked", flush=True)
            return
        print("[baseline] PASS — candidate intervention authorized by frozen gate", flush=True)
        write_json(output / "status.json", {"status": "RUNNING_INTERVENTIONS"})
        frames = [baseline]
        conditions = condition_plan(manifest, design)
        for number, condition in enumerate(conditions, 1):
            print(f"[intervention {number}/{len(conditions)}] {condition['condition_id']}", flush=True)
            frames.append(evaluate_condition(
                model, tokenizer, device, records, labels, int(design["batch_size"]),
                condition["condition_id"], condition["component_id"], condition["alpha"], condition["neuron_mean"],
            ))
        responses = pd.concat(frames, ignore_index=True)
        atomic_csv(output / "behavioral_item_responses.csv.gz", responses)
        candidate_results = analyze_candidates(responses, manifest, design)
        serialized = candidate_results.copy()
        for column in [name for name in serialized.columns if name.endswith("_ci")]:
            serialized[column] = serialized[column].map(json.dumps)
        atomic_csv(output / "candidate_behavioral_results.csv", serialized)
        strict = candidate_results[candidate_results["strict_behavioral_signal"]]
        probability_only = candidate_results[candidate_results["probability_only_signal"]]
        summary = {
            "study_id": design["study_id"], "run_id": design["run_id"], "status": "COMPLETE",
            "same_sample_exploratory": True,
            "baseline": baseline_gate,
            "candidate_count": len(manifest["candidates"]),
            "physical_intervention_condition_count": len(conditions),
            "strict_behavioral_signal_rows": strict[["component_id", "group"]].to_dict("records"),
            "strict_behavioral_signal_components": sorted(strict["component_id"].unique().tolist()),
            "probability_only_signal_rows": probability_only[["component_id", "group"]].to_dict("records"),
            "probability_only_signal_components": sorted(probability_only["component_id"].unique().tolist()),
            "no_strict_behavioral_signal_components": sorted(
                set(row["component_id"] for row in manifest["candidates"]) - set(strict["component_id"])
            ),
            "model_checks": model_checks,
            "claim_boundary": design["interpretation"],
            "hashes": {
                **locked_inputs(design),
                "implementation_sha256": sha256_file(IMPLEMENTATION_PATH),
                "static_review_sha256": sha256_file(STATIC_REVIEW_PATH),
                "authorization_sha256": sha256_file(args.authorization.resolve()),
                "candidate_results_sha256": sha256_file(output / "candidate_behavioral_results.csv"),
            },
            "environment": {
                "python": platform.python_version(), "torch": torch.__version__,
                "transformers": __import__("transformers").__version__, "device": str(device),
                "gpu": torch.cuda.get_device_name(device), "dtype": design["dtype"], "batch_size": design["batch_size"],
            },
        }
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"status": "COMPLETE", "summary_sha256": sha256_file(output / "summary.json")})
        print(
            f"[complete] strict={len(summary['strict_behavioral_signal_components'])}, "
            f"probability_only={len(summary['probability_only_signal_components'])}", flush=True,
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "ERROR", "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        release_model(model)


def report(design: dict[str, Any]) -> None:
    output = resolve(design["output_root"]) / design["run_id"]
    summary = read_json(output / "summary.json")
    lines = [
        "# Meta-Llama Behavioral Validation v1 결과", "",
        f"상태: **{summary['status']}**", "",
        "## Baseline", "",
    ]
    for group in ("M", "R"):
        row = summary["baseline"]["groups"][group]
        lines.append(
            f"- {group}: {row['forced_choice_correct']}/{row['n']} "
            f"({row['forced_choice_accuracy']:.3f}), mean chance={row['mean_uniform_chance']:.3f}, "
            f"mean correct probability={row['mean_correct_probability']:.3f}"
        )
    if summary["status"] == "COMPLETE":
        lines.extend([
            "", "## Candidate 결과", "",
            f"- strict behavioral signal: {', '.join(summary['strict_behavioral_signal_components']) or 'none'}",
            f"- probability-only signal: {', '.join(summary['probability_only_signal_components']) or 'none'}",
            "", "상세 R/M별 수치는 `candidate_behavioral_results.csv`에 저장했다.",
        ])
    lines.extend([
        "", "## 해석 제한", "",
        "- 기존 heldout 600을 다시 사용한 same-sample exploratory behavioral validation이다.",
        "- 성공해도 reasoning neuron, memorization store, 완전한 circuit 또는 independent replication을 뜻하지 않는다.",
        "- 이 실행은 `result.pdf`를 자동 수정하지 않는다.",
    ])
    (output / "RESULTS_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    design = load_design()
    manifest = validate_candidate_manifest(design)
    preflight = {
        "design": design["study_id"], "candidate_count": len(manifest["candidates"]),
        "locked_inputs": locked_inputs(design),
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    if args.phase == "execute":
        if args.authorization is None:
            raise RuntimeError("--authorization is required")
        authorization = validate_authorization(args.authorization.resolve(), design)
        execute(args, design, authorization)
    else:
        report(design)


if __name__ == "__main__":
    main()
