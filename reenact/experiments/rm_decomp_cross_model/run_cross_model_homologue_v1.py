#!/usr/bin/env python3
"""Cross-model late-component discovery and causal R/M-gap validation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
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
STAGE_A_DIR = SCRIPT_DIR.parent / "rm_decomp"
if str(STAGE_A_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_A_DIR))

from core import load_dataset_and_split  # noqa: E402


GROUP_MEMORY = 0
GROUP_REASONING = 1
COMPONENT_RE = re.compile(r"^L(\d{2})([HN])(\d{5})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("model", "report"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--design", type=Path, default=SCRIPT_DIR / "design_v1_1_frozen.json")
    parser.add_argument("--authorization", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_design(path: Path) -> dict[str, Any]:
    design = read_json(path)
    if design.get("study_id") != "cross_model_functional_homologue_v1_1":
        raise RuntimeError("Unexpected design study_id")
    if design.get("result_pdf_update_allowed") is not False:
        raise RuntimeError("The frozen design must prohibit automatic result.pdf updates")
    if design.get("layer_scope") != "all_transformer_blocks":
        raise RuntimeError("v1.1 must search every Transformer block")
    if int(design["discovery_max_candidates_per_component_type"]) != 5:
        raise RuntimeError("Frozen candidate cap must be five per component type")
    return design


def model_entry(design: dict[str, Any], name: str) -> dict[str, Any]:
    hits = [row for row in design["models"] if row["name"] == name]
    if len(hits) != 1:
        raise RuntimeError(f"Model is not uniquely listed in frozen design: {name}")
    return hits[0]


def model_shard_manifest(model_path: Path) -> dict[str, str]:
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"No safetensor shards found: {model_path}")
    return {shard.name: sha256_file(shard) for shard in shards}


def validate_authorization(
    path: Path,
    design: dict[str, Any],
    design_path: Path,
    implementation_path: Path,
    static_review_path: Path,
    entry: dict[str, Any],
) -> dict[str, Any]:
    auth = read_json(path)
    if auth.get("status") != "FROZEN_EXECUTION_AUTHORIZED":
        raise RuntimeError("Execution authorization is not frozen/authorized")
    if auth.get("design_sha256") != sha256_file(design_path):
        raise RuntimeError("Authorization design hash mismatch")
    if auth.get("implementation_sha256") != sha256_file(implementation_path):
        raise RuntimeError("Authorization implementation hash mismatch")
    if auth.get("static_review_sha256") != sha256_file(static_review_path):
        raise RuntimeError("Authorization static-review hash mismatch")
    model = entry["name"]
    if model not in auth.get("authorized_models", []):
        raise RuntimeError(f"Model is not authorized: {model}")
    if auth.get("dtype") != design["dtype"]:
        raise RuntimeError("Authorization dtype mismatch")
    if int(auth.get("batch_size", {}).get(model, -1)) != int(entry["batch_size"]):
        raise RuntimeError(f"Authorization batch-size mismatch: {model}")
    expected_inputs = auth.get("inputs", {})
    actual_inputs = {
        "dataset_sha256": sha256_file(resolve(WORKSPACE, design["dataset_path"])),
        "split_sha256": sha256_file(resolve(WORKSPACE, design["split_path"])),
        "layerwise_liref_sha256": {
            name: sha256_file(resolve(WORKSPACE, design["layerwise_root"]) / name / "liref_vectors_heldout.pt")
            for name in auth.get("authorized_models", [])
        },
    }
    if expected_inputs != actual_inputs:
        raise RuntimeError("Authorization dataset/split/LiReF input lock mismatch")
    expected_model = auth.get("model_locks", {}).get(model)
    if not expected_model:
        raise RuntimeError(f"Missing model lock: {model}")
    model_path = resolve(WORKSPACE, entry["model_path"])
    actual_model = {
        "config_sha256": sha256_file(model_path / "config.json"),
        "index_sha256": sha256_file(model_path / "model.safetensors.index.json"),
        "shards": model_shard_manifest(model_path),
    }
    if expected_model != actual_model:
        raise RuntimeError(f"Model weight/config lock mismatch: {model}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(entry["gpu"]):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical GPU {entry['gpu']}")
    return auth


def model_contract(model: Any) -> dict[str, Any]:
    c = model.config
    allowed = {"mistral", "olmo2", "gemma2"}
    if c.model_type not in allowed:
        raise RuntimeError(f"Architecture not allowed by design: {c.model_type}")
    layers = getattr(model.model, "layers", None)
    if layers is None or len(layers) != c.num_hidden_layers:
        raise RuntimeError("model.model.layers contract failed")
    for index, layer in enumerate(layers):
        for dotted in ("self_attn.o_proj", "mlp.down_proj"):
            current = layer
            for part in dotted.split("."):
                if not hasattr(current, part):
                    raise RuntimeError(f"Missing {dotted} at layer {index}")
                current = getattr(current, part)
    return {
        "model_type": c.model_type,
        "num_hidden_layers": int(c.num_hidden_layers),
        "hidden_size": int(c.hidden_size),
        "intermediate_size": int(c.intermediate_size),
        "num_attention_heads": int(c.num_attention_heads),
        "num_key_value_heads": int(c.num_key_value_heads),
        "head_dim": int(getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)),
        "post_normalized_component_outputs": c.model_type in {"olmo2", "gemma2"},
    }


def load_model(entry: dict[str, Any], design: dict[str, Any], device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    path = resolve(WORKSPACE, entry["model_path"])
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval().to(device)
    if model.training or next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("Model must be frozen eval FP32")
    return model, tokenizer, model_contract(model)


def release_model(model: Any | None) -> None:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def eligible_layers(num_layers: int, scope: str) -> list[int]:
    if scope != "all_transformer_blocks":
        raise RuntimeError(f"Unsupported layer scope: {scope}")
    return list(range(num_layers))


def depth_band(relative_depth: float, design: dict[str, Any]) -> str:
    if relative_depth <= float(design["depth_bins"]["early_max"]):
        return "early"
    if relative_depth <= float(design["depth_bins"]["middle_max"]):
        return "middle"
    return "late"


def batches(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def encode(tokenizer: Any, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    result = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=False, return_token_type_ids=False)
    if not bool(torch.all(result["attention_mask"][:, -1] == 1)):
        raise RuntimeError("Last index is not the final prompt token")
    return {k: v.to(device) for k, v in result.items()}


class DiscoveryCapture:
    def __init__(self, model: Any, indices: list[int]) -> None:
        self.model = model
        self.indices = indices
        self.handles: list[Any] = []
        self.reset()

    def reset(self) -> None:
        # Hooks close over the pre_o/z dictionary objects at installation.
        # Preserve those objects across batches and clear only their contents.
        # Rebinding them here would make the hooks write into stale dictionaries.
        if not hasattr(self, "h_out"):
            self.h_out: dict[int, torch.Tensor] = {}
            self.pre_o: dict[int, torch.Tensor] = {}
            self.z: dict[int, torch.Tensor] = {}
        else:
            self.h_out.clear()
            self.pre_o.clear()
            self.z.clear()

    @staticmethod
    def last(value: torch.Tensor) -> torch.Tensor:
        return value.detach()[:, -1, :].clone()

    def install(self) -> None:
        for index in self.indices:
            layer = self.model.model.layers[index]
            self.handles.append(layer.register_forward_hook(self._post(index)))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._pre(self.pre_o, index)))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._pre(self.z, index)))

    def _post(self, index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.h_out[index] = self.last(tensor)
        return hook

    def _pre(self, target: dict[int, torch.Tensor], index: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            target[index] = self.last(args[0])
        return hook

    def validate(self) -> None:
        expected = set(self.indices)
        if set(self.h_out) != expected or set(self.pre_o) != expected or set(self.z) != expected:
            raise RuntimeError("Incomplete discovery capture")

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class FinalCapture:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.value: torch.Tensor | None = None
        self.handle: Any | None = None

    def install(self) -> None:
        index = self.model.config.num_hidden_layers - 1
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.value = tensor.detach()[:, -1, :].clone()
        self.handle = self.model.model.layers[index].register_forward_hook(hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def grouped_add(target_sum: np.ndarray, target_sumsq: np.ndarray, values: torch.Tensor, labels: torch.Tensor) -> None:
    for group in (GROUP_MEMORY, GROUP_REASONING):
        selected = values[labels == group]
        if len(selected):
            target_sum[group] += selected.double().sum(dim=0).cpu().numpy()
            target_sumsq[group] += selected.double().square().sum(dim=0).cpu().numpy()


@torch.inference_mode()
def discovery_pass(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    data: dict[str, Any],
    indices: list[int],
    layer_indices: list[int],
    batch_size: int,
) -> dict[str, Any]:
    c = model.config
    n_late = len(layer_indices)
    attn_width = int(model.model.layers[layer_indices[0]].self_attn.o_proj.in_features)
    h_sum = np.zeros((2, n_late, c.hidden_size), dtype=np.float64)
    h_sumsq = np.zeros_like(h_sum)
    pre_sum = np.zeros((2, n_late, attn_width), dtype=np.float64)
    pre_sumsq = np.zeros_like(pre_sum)
    z_sum = np.zeros((2, n_late, c.intermediate_size), dtype=np.float64)
    z_sumsq = np.zeros_like(z_sum)
    counts = np.zeros(2, dtype=np.int64)
    capture = DiscoveryCapture(model, layer_indices)
    capture.install()
    try:
        for number, batch_indices in enumerate(batches(indices, batch_size), 1):
            capture.reset()
            encoded = encode(tokenizer, [data["prompts"][i] for i in batch_indices], device)
            model(**encoded, use_cache=False, return_dict=True)
            capture.validate()
            h = torch.stack([capture.h_out[i] for i in layer_indices], dim=1)
            pre = torch.stack([capture.pre_o[i] for i in layer_indices], dim=1)
            z = torch.stack([capture.z[i] for i in layer_indices], dim=1)
            labels = torch.as_tensor(data["labels"][batch_indices], device=device)
            for group in (GROUP_MEMORY, GROUP_REASONING):
                counts[group] += int((labels == group).sum())
            grouped_add(h_sum, h_sumsq, h, labels)
            grouped_add(pre_sum, pre_sumsq, pre, labels)
            grouped_add(z_sum, z_sumsq, z, labels)
            if number % 100 == 0:
                print(f"  discovery batches: {number}", flush=True)
            del encoded, h, pre, z, labels
    finally:
        capture.remove()
    h_mean = h_sum / counts[:, None, None]
    raw = h_mean[GROUP_REASONING] - h_mean[GROUP_MEMORY]
    norms = np.linalg.norm(raw, axis=1)
    if not bool(np.all(np.isfinite(norms) & (norms > 1e-8))):
        raise RuntimeError("Invalid Discovery LiReF direction")
    return {
        "counts": counts,
        "h_sum": h_sum,
        "h_sumsq": h_sumsq,
        "pre_sum": pre_sum,
        "pre_sumsq": pre_sumsq,
        "z_sum": z_sum,
        "z_sumsq": z_sumsq,
        "raw_directions": raw,
        "unit_directions": raw / norms[:, None],
    }


def direction_alignment(model_name: str, design: dict[str, Any], layers: list[int], directions: np.ndarray) -> dict[str, Any]:
    path = resolve(WORKSPACE, design["layerwise_root"]) / model_name / "liref_vectors_heldout.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    mapping = {
        int(block) - 1: artifact["normalized_liref"][row].double().numpy()
        for row, block in enumerate(artifact["transformer_block_numbers"])
    }
    rows = []
    for offset, layer in enumerate(layers):
        if layer not in mapping:
            rows.append({"module_index": layer, "available": False, "cosine": None})
            continue
        a, b = directions[offset], mapping[layer]
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        rows.append({"module_index": layer, "available": True, "cosine": cosine})
    available = [row["cosine"] for row in rows if row["available"]]
    minimum = float(min(available)) if available else math.nan
    threshold = float(design["direction_alignment_cosine_min"])
    return {"artifact": str(path), "rows": rows, "min_available_cosine": minimum, "threshold": threshold, "pass": bool(available and minimum >= threshold)}


def component_frames(model: Any, layers: list[int], discovery: dict[str, Any], design: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = discovery["counts"]
    total = int(counts.sum())
    head_rows, neuron_rows = [], []
    for offset, layer_index in enumerate(layers):
        layer = model.model.layers[layer_index]
        direction = torch.as_tensor(discovery["unit_directions"][offset], device=layer.self_attn.o_proj.weight.device, dtype=torch.float64)
        o_weight = layer.self_attn.o_proj.weight.detach().double()
        heads = int(model.config.num_attention_heads)
        head_dim = int(o_weight.shape[1] // heads)
        q = torch.einsum("ohd,o->hd", o_weight.reshape(o_weight.shape[0], heads, head_dim), direction).cpu().numpy()
        pre_mean = discovery["pre_sum"][:, offset] / counts[:, None]
        pre_delta = (pre_mean[GROUP_REASONING] - pre_mean[GROUP_MEMORY]).reshape(heads, head_dim)
        pre_m2 = discovery["pre_sumsq"][:, offset].sum(axis=0) / total
        head_scale = np.sqrt((pre_m2.reshape(heads, head_dim) * np.square(q)).sum(axis=1))
        head_delta = (pre_delta * q).sum(axis=1)
        for index in range(heads):
            relative_depth = (layer_index + 1) / model.config.num_hidden_layers
            head_rows.append({
                "component_id": f"L{layer_index:02d}H{index:05d}", "component_type": "head",
                "module_index": layer_index, "component_index": index,
                "relative_layer_depth": relative_depth, "depth_band": depth_band(relative_depth, design),
                "Delta_discovery": float(head_delta[index]), "abs_Delta_discovery": float(abs(head_delta[index])),
                "writer_scale_proxy": float(head_scale[index]), "pooled_activation_mean": None,
            })
        neuron_projection = torch.mv(layer.mlp.down_proj.weight.detach().double().T, direction).cpu().numpy()
        z_mean = discovery["z_sum"][:, offset] / counts[:, None]
        z_delta = z_mean[GROUP_REASONING] - z_mean[GROUP_MEMORY]
        z_m2 = discovery["z_sumsq"][:, offset].sum(axis=0) / total
        pooled_mean = discovery["z_sum"][:, offset].sum(axis=0) / total
        neuron_delta = z_delta * neuron_projection
        neuron_scale = np.sqrt(z_m2) * np.abs(neuron_projection)
        for index in range(model.config.intermediate_size):
            relative_depth = (layer_index + 1) / model.config.num_hidden_layers
            neuron_rows.append({
                "component_id": f"L{layer_index:02d}N{index:05d}", "component_type": "neuron",
                "module_index": layer_index, "component_index": index,
                "relative_layer_depth": relative_depth, "depth_band": depth_band(relative_depth, design),
                "Delta_discovery": float(neuron_delta[index]), "abs_Delta_discovery": float(abs(neuron_delta[index])),
                "writer_scale_proxy": float(neuron_scale[index]), "pooled_activation_mean": float(pooled_mean[index]),
            })
    return pd.DataFrame(head_rows), pd.DataFrame(neuron_rows)


def deterministic_rng(seed: int, key: str) -> np.random.Generator:
    suffix = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng((seed + suffix) % (2**32))


def select_candidates_and_controls(frames: list[pd.DataFrame], design: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_components = pd.concat(frames, ignore_index=True)
    selected_parts = []
    topk = int(design["discovery_max_candidates_per_component_type"])
    for component_type in ("head", "neuron"):
        positive = all_components[(all_components["component_type"] == component_type) & (all_components["Delta_discovery"] > 0)]
        chosen = positive.sort_values(["Delta_discovery", "component_id"], ascending=[False, True]).head(topk).copy()
        # This is a global cap, not a quota per depth band.  A component type
        # may contribute fewer than five candidates, including zero.
        chosen["discovery_rank_within_type"] = np.arange(1, len(chosen) + 1)
        selected_parts.append(chosen)
    candidates = pd.concat(selected_parts, ignore_index=True)
    selected_ids = set(candidates["component_id"])
    controls = []
    seed = int(design["seed"])
    n_random = int(design["random_controls_per_candidate"])
    for candidate in candidates.to_dict("records"):
        same = all_components[
            (all_components["component_type"] == candidate["component_type"])
            & (all_components["module_index"] == candidate["module_index"])
            & (~all_components["component_id"].isin(selected_ids))
        ].copy()
        cutoff = float(same["abs_Delta_discovery"].median())
        pool = same[same["abs_Delta_discovery"] <= cutoff].copy()
        if len(pool) < n_random + 1:
            raise RuntimeError(f"Control pool too small for {candidate['component_id']}")
        eps = 1e-12
        pool["scale_distance"] = np.abs(np.log((pool["writer_scale_proxy"] + eps) / (float(candidate["writer_scale_proxy"]) + eps)))
        matched = pool.sort_values(["scale_distance", "component_id"]).iloc[0]
        controls.append({"owner_candidate_id": candidate["component_id"], "control_kind": "matched", **matched.to_dict()})
        remaining = pool[pool["component_id"] != matched["component_id"]]
        rng = deterministic_rng(seed, candidate["component_id"])
        chosen_indices = rng.choice(remaining.index.to_numpy(), size=n_random, replace=False)
        for index in chosen_indices:
            controls.append({"owner_candidate_id": candidate["component_id"], "control_kind": "random", **remaining.loc[index].to_dict()})
    control_columns = ["owner_candidate_id", "control_kind", *all_components.columns]
    return candidates, pd.DataFrame(controls, columns=control_columns)


def bh(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(values)
    output[order] = np.clip(ranked, 0, 1)
    return output


def moments_frame(sum_: np.ndarray, sumsq: np.ndarray, counts: np.ndarray, ids: list[str]) -> pd.DataFrame:
    means = sum_ / counts[:, None]
    variances = (sumsq - np.square(sum_) / counts[:, None]) / (counts[:, None] - 1)
    variances = np.maximum(variances, 0)
    delta = means[GROUP_REASONING] - means[GROUP_MEMORY]
    se2 = variances[GROUP_MEMORY] / counts[GROUP_MEMORY] + variances[GROUP_REASONING] / counts[GROUP_REASONING]
    se = np.sqrt(se2)
    denom = np.square(variances[GROUP_MEMORY] / counts[GROUP_MEMORY]) / (counts[GROUP_MEMORY] - 1)
    denom += np.square(variances[GROUP_REASONING] / counts[GROUP_REASONING]) / (counts[GROUP_REASONING] - 1)
    df = np.square(se2) / denom
    t = delta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    return pd.DataFrame({
        "component_id": ids, "memory_mean_validation": means[GROUP_MEMORY],
        "reasoning_mean_validation": means[GROUP_REASONING], "Delta_validation": delta,
        "welch_p_validation": p,
    })


@torch.inference_mode()
def validation_pass(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    data: dict[str, Any],
    indices: list[int],
    layers: list[int],
    directions: np.ndarray,
    candidates: pd.DataFrame,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    heads = int(model.config.num_attention_heads)
    intermediate = int(model.config.intermediate_size)
    n_late = len(layers)
    head_projection, neuron_projection = [], []
    for offset, layer_index in enumerate(layers):
        layer = model.model.layers[layer_index]
        direction = torch.as_tensor(directions[offset], device=device, dtype=torch.float32)
        o = layer.self_attn.o_proj.weight
        head_dim = o.shape[1] // heads
        head_projection.append(torch.einsum("ohd,o->hd", o.reshape(o.shape[0], heads, head_dim), direction))
        neuron_projection.append(torch.mv(layer.mlp.down_proj.weight.T, direction))
    head_projection_t = torch.stack(head_projection)
    neuron_projection_t = torch.stack(neuron_projection)
    h_sum = np.zeros((2, n_late * heads), dtype=np.float64)
    h_sumsq = np.zeros_like(h_sum)
    n_sum = np.zeros((2, n_late * intermediate), dtype=np.float64)
    n_sumsq = np.zeros_like(n_sum)
    counts = np.zeros(2, dtype=np.int64)
    baseline_rows = []
    capture = DiscoveryCapture(model, layers)
    capture.install()
    final_offset = layers.index(model.config.num_hidden_layers - 1)
    try:
        for batch_indices in batches(indices, batch_size):
            capture.reset()
            encoded = encode(tokenizer, [data["prompts"][i] for i in batch_indices], device)
            model(**encoded, use_cache=False, return_dict=True)
            capture.validate()
            pre = torch.stack([capture.pre_o[i] for i in layers], dim=1).reshape(-1, n_late, heads, head_projection_t.shape[-1])
            z = torch.stack([capture.z[i] for i in layers], dim=1)
            head_scores = (pre * head_projection_t.unsqueeze(0)).sum(dim=-1).reshape(len(batch_indices), -1)
            neuron_scores = (z * neuron_projection_t.unsqueeze(0)).reshape(len(batch_indices), -1)
            labels = torch.as_tensor(data["labels"][batch_indices], device=device)
            for group in (GROUP_MEMORY, GROUP_REASONING):
                counts[group] += int((labels == group).sum())
            grouped_add(h_sum, h_sumsq, head_scores, labels)
            grouped_add(n_sum, n_sumsq, neuron_scores, labels)
            final_direction = torch.as_tensor(directions[final_offset], device=device, dtype=torch.float32)
            final_scores = torch.mv(capture.h_out[layers[final_offset]], final_direction).float().cpu().numpy()
            for local, row_index in enumerate(batch_indices):
                baseline_rows.append({"row_index": row_index, "label": int(data["labels"][row_index]), "score": float(final_scores[local])})
            del encoded, pre, z, head_scores, neuron_scores, labels
    finally:
        capture.remove()
    head_ids = [f"L{layer:02d}H{index:05d}" for layer in layers for index in range(heads)]
    neuron_ids = [f"L{layer:02d}N{index:05d}" for layer in layers for index in range(intermediate)]
    stats_all = pd.concat([
        moments_frame(h_sum, h_sumsq, counts, head_ids),
        moments_frame(n_sum, n_sumsq, counts, neuron_ids),
    ], ignore_index=True)
    selected = candidates.merge(stats_all, on="component_id", how="left", validate="one_to_one")
    selected["heldout_bh_q_candidates"] = bh(selected["welch_p_validation"].to_numpy()) if len(selected) else np.asarray([])
    selected["heldout_positive_same_sign"] = (selected["Delta_validation"] > 0) & (selected["Delta_discovery"] > 0)
    return selected, pd.DataFrame(baseline_rows).sort_values("row_index"), stats_all


def parse_component(component_id: str) -> tuple[str, int, int]:
    match = COMPONENT_RE.fullmatch(component_id)
    if match is None:
        raise RuntimeError(f"Invalid component ID: {component_id}")
    return ("head" if match.group(2) == "H" else "neuron", int(match.group(1)), int(match.group(3)))


class Intervention:
    def __init__(self, model: Any, component_id: str, alpha: float, mean: float | None) -> None:
        self.model = model
        self.component_id = component_id
        self.alpha = alpha
        self.mean = mean
        self.handle: Any | None = None

    def install(self) -> None:
        component_type, layer_index, index = parse_component(self.component_id)
        layer = self.model.model.layers[layer_index]
        module = layer.self_attn.o_proj if component_type == "head" else layer.mlp.down_proj
        head_dim = int(layer.self_attn.o_proj.in_features // self.model.config.num_attention_heads)
        def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            values = args[0].clone()
            if component_type == "head":
                start = index * head_dim
                values[:, -1, start : start + head_dim] *= 1.0 - self.alpha
            else:
                reference = torch.as_tensor(float(self.mean), device=values.device, dtype=values.dtype)
                values[:, -1, index] = (1.0 - self.alpha) * values[:, -1, index] + self.alpha * reference
            return (values, *args[1:])
        self.handle = module.register_forward_pre_hook(hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.inference_mode()
def infer_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    direction: np.ndarray,
    batch_size: int,
    component_id: str | None = None,
    alpha: float = 0.0,
    mean: float | None = None,
    baseline_logits: torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor, np.ndarray, np.ndarray]:
    capture = FinalCapture(model)
    capture.install()
    intervention = None
    if component_id is not None:
        intervention = Intervention(model, component_id, alpha, mean)
        intervention.install()
    direction_t = torch.as_tensor(direction, device=device, dtype=torch.float32)
    scores, logits_parts, kl_parts, top1_parts = [], [], [], []
    offset = 0
    try:
        for batch_prompts in [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]:
            encoded = encode(tokenizer, batch_prompts, device)
            output = model(**encoded, use_cache=False, return_dict=True)
            if capture.value is None:
                raise RuntimeError("Final layer capture failed")
            logits = output.logits[:, -1, :]
            scores.extend(torch.mv(capture.value, direction_t).float().cpu().tolist())
            if baseline_logits is None:
                kl_parts.extend([0.0] * len(batch_prompts))
                top1_parts.extend([0] * len(batch_prompts))
            else:
                base = baseline_logits[offset : offset + len(batch_prompts)].to(device=device, dtype=torch.float32)
                p = F.softmax(base, dim=-1)
                kl = (p * (F.log_softmax(base, dim=-1) - F.log_softmax(logits.float(), dim=-1))).sum(dim=-1)
                kl_parts.extend(kl.cpu().tolist())
                top1_parts.extend((base.argmax(dim=-1) != logits.argmax(dim=-1)).to(torch.int8).cpu().tolist())
            logits_parts.append(logits.detach().cpu().to(torch.float16))
            offset += len(batch_prompts)
            capture.value = None
            del encoded, output, logits
    finally:
        if intervention is not None:
            intervention.remove()
        capture.remove()
    return np.asarray(scores), torch.cat(logits_parts), np.asarray(kl_parts), np.asarray(top1_parts)


def gap(base: np.ndarray, changed: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    g0 = float(base[labels == GROUP_REASONING].mean() - base[labels == GROUP_MEMORY].mean())
    g1 = float(changed[labels == GROUP_REASONING].mean() - changed[labels == GROUP_MEMORY].mean())
    return {"G_base": g0, "G_intervention": g1, "delta_G": g1 - g0, "abs_G": abs(g1), "gap_reduction": abs(g0) - abs(g1)}


def stratified_draws(labels: np.ndarray, reps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    memory = np.flatnonzero(labels == GROUP_MEMORY)
    reasoning = np.flatnonzero(labels == GROUP_REASONING)
    return rng.choice(memory, (reps, len(memory)), replace=True), rng.choice(reasoning, (reps, len(reasoning)), replace=True)


def boot_reduction(base: np.ndarray, changed: np.ndarray, memory: np.ndarray, reasoning: np.ndarray) -> np.ndarray:
    g0 = base[reasoning].mean(axis=1) - base[memory].mean(axis=1)
    g1 = changed[reasoning].mean(axis=1) - changed[memory].mean(axis=1)
    return np.abs(g0) - np.abs(g1)


def permutation_p(change: np.ndarray, labels: np.ndarray, reps: int, seed: int) -> float:
    observed = abs(float(change[labels == GROUP_REASONING].mean() - change[labels == GROUP_MEMORY].mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(reps):
        shuffled = rng.permutation(labels)
        value = abs(float(change[shuffled == GROUP_REASONING].mean() - change[shuffled == GROUP_MEMORY].mean()))
        exceed += value >= observed
    return (exceed + 1) / (reps + 1)


def intervention_pass(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    data: dict[str, Any],
    design: dict[str, Any],
    final_direction: np.ndarray,
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
    validation: pd.DataFrame,
    output: Path,
    batch_size: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if candidates.empty:
        empty = pd.DataFrame(columns=["component_id", "functional_homologue_pass"])
        atomic_csv(output / "tables" / "intervention_responses.csv.gz", pd.DataFrame(columns=[
            "row_index", "label", "condition_id", "owner_candidate_id", "component_role",
            "control_kind", "component_id", "alpha", "score", "next_token_kl", "top1_changed",
        ]))
        return empty, []
    indices = [int(i) for i in data["indices"]["validation"]]
    indices.sort(key=lambda i: len(tokenizer(data["prompts"][i], add_special_tokens=True)["input_ids"]))
    prompts = [data["prompts"][i] for i in indices]
    labels = np.asarray([data["labels"][i] for i in indices], dtype=np.int8)
    baseline_scores, baseline_logits, _, _ = infer_condition(model, tokenizer, device, prompts, final_direction, batch_size)
    meta = pd.concat([candidates, controls], ignore_index=True).drop_duplicates("component_id").set_index("component_id")
    logical_conditions = []
    for candidate in candidates["component_id"]:
        for alpha in design["candidate_alphas"]:
            logical_conditions.append({"owner": candidate, "role": "candidate", "kind": "candidate", "component_id": candidate, "alpha": float(alpha)})
        owned = controls[controls["owner_candidate_id"] == candidate]
        for row in owned.to_dict("records"):
            logical_conditions.append({"owner": candidate, "role": "control", "kind": row["control_kind"], "component_id": row["component_id"], "alpha": float(design["control_alpha"])})
    cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    response_frames = [pd.DataFrame({
        "row_index": indices, "label": labels, "condition_id": "baseline", "owner_candidate_id": "baseline",
        "component_role": "baseline", "control_kind": "baseline", "component_id": "baseline", "alpha": 0.0,
        "score": baseline_scores, "next_token_kl": np.zeros(len(indices)), "top1_changed": np.zeros(len(indices), dtype=np.int8),
    })]
    for number, condition in enumerate(logical_conditions, 1):
        key = (condition["component_id"], condition["alpha"])
        if key not in cache:
            row = meta.loc[condition["component_id"]]
            mean = None if row["component_type"] == "head" else float(row["pooled_activation_mean"])
            print(f"  intervention {number}/{len(logical_conditions)}: {key[0]} alpha={key[1]}", flush=True)
            scores, _, kl, top1 = infer_condition(
                model, tokenizer, device, prompts, final_direction, batch_size,
                component_id=key[0], alpha=key[1], mean=mean, baseline_logits=baseline_logits,
            )
            cache[key] = (scores, kl, top1)
        scores, kl, top1 = cache[key]
        response_frames.append(pd.DataFrame({
            "row_index": indices, "label": labels,
            "condition_id": f"{condition['role']}::{condition['owner']}::{condition['kind']}::{condition['component_id']}::a{condition['alpha']:g}",
            "owner_candidate_id": condition["owner"], "component_role": condition["role"], "control_kind": condition["kind"],
            "component_id": condition["component_id"], "alpha": condition["alpha"], "score": scores,
            "next_token_kl": kl, "top1_changed": top1,
        }))
    responses = pd.concat(response_frames, ignore_index=True)
    atomic_csv(output / "tables" / "intervention_responses.csv.gz", responses)
    reps = int(design["bootstrap_replicates"])
    memory, reasoning = stratified_draws(labels, reps, int(design["seed"]))
    # Use the physical inference cache rather than logical response rows.  A
    # single control may legitimately be assigned to more than one candidate;
    # grouping the logical rows would duplicate those 600 held-out scores.
    score_map = {key: value[0] for key, value in cache.items()}
    rows, cards = [], []
    permutation_values = []
    for candidate_number, candidate in enumerate(candidates["component_id"]):
        full = score_map[(candidate, 1.0)]
        p = permutation_p(full - baseline_scores, labels, int(design["permutation_replicates"]), int(design["seed"]) + candidate_number)
        permutation_values.append(p)
    permutation_q = bh(np.asarray(permutation_values))
    validation_map = validation.set_index("component_id")
    for candidate_number, candidate in enumerate(candidates["component_id"]):
        half, full = score_map[(candidate, 0.5)], score_map[(candidate, 1.0)]
        metric = gap(baseline_scores, full, labels)
        reduction = boot_reduction(baseline_scores, full, memory, reasoning)
        owned = controls[controls["owner_candidate_id"] == candidate]
        matched_id = owned[owned["control_kind"] == "matched"]["component_id"].iloc[0]
        random_ids = owned[owned["control_kind"] == "random"]["component_id"].tolist()
        matched_reduction = boot_reduction(baseline_scores, score_map[(matched_id, 1.0)], memory, reasoning)
        random_reductions = np.stack([boot_reduction(baseline_scores, score_map[(value, 1.0)], memory, reasoning) for value in random_ids])
        candidate_minus_matched = reduction - matched_reduction
        candidate_minus_random = reduction - random_reductions.mean(axis=0)
        g0 = abs(metric["G_base"])
        ghalf = abs(gap(baseline_scores, half, labels)["G_intervention"])
        gfull = abs(metric["G_intervention"])
        v = validation_map.loc[candidate]
        pass_checks = {
            "heldout_positive_same_sign": bool(v["heldout_positive_same_sign"]),
            "heldout_bh_q": bool(v["heldout_bh_q_candidates"] < design["fdr_alpha"]),
            "gap_reduction_ci": bool(np.quantile(reduction, 0.025) > 0),
            "permutation_bh_q": bool(permutation_q[candidate_number] < design["fdr_alpha"]),
            "dose_monotonic": bool(g0 >= ghalf >= gfull),
            "candidate_minus_matched_ci": bool(np.quantile(candidate_minus_matched, 0.025) > 0),
            "candidate_minus_random_ci": bool(np.quantile(candidate_minus_random, 0.025) > 0),
        }
        card = {
            "component_id": candidate,
            "component_type": parse_component(candidate)[0],
            "module_index": parse_component(candidate)[1],
            "relative_layer_depth": float(candidates.set_index("component_id").loc[candidate, "relative_layer_depth"]),
            "depth_band": str(candidates.set_index("component_id").loc[candidate, "depth_band"]),
            "Delta_discovery": float(v["Delta_discovery"]),
            "Delta_validation": float(v["Delta_validation"]),
            "heldout_bh_q": float(v["heldout_bh_q_candidates"]),
            **metric,
            "gap_reduction_ci": [float(np.quantile(reduction, 0.025)), float(np.quantile(reduction, 0.975))],
            "permutation_p": float(permutation_values[candidate_number]),
            "permutation_bh_q": float(permutation_q[candidate_number]),
            "abs_G_alpha_0": g0, "abs_G_alpha_0_5": ghalf, "abs_G_alpha_1": gfull,
            "matched_control": matched_id, "random_controls": random_ids,
            "candidate_minus_matched": float(candidate_minus_matched.mean()),
            "candidate_minus_matched_ci": [float(np.quantile(candidate_minus_matched, 0.025)), float(np.quantile(candidate_minus_matched, 0.975))],
            "candidate_minus_random_mean": float(candidate_minus_random.mean()),
            "candidate_minus_random_mean_ci": [float(np.quantile(candidate_minus_random, 0.025)), float(np.quantile(candidate_minus_random, 0.975))],
            "mean_next_token_kl": float(responses[(responses["component_id"] == candidate) & (responses["alpha"] == 1.0)]["next_token_kl"].mean()),
            "top1_change_rate": float(responses[(responses["component_id"] == candidate) & (responses["alpha"] == 1.0)]["top1_changed"].mean()),
            "checks": pass_checks,
            "functional_homologue_pass": bool(all(pass_checks.values())),
        }
        cards.append(card)
        rows.append(card)
        write_json(output / "candidate_cards" / f"{candidate}.json", card)
    return pd.DataFrame(rows), cards


def run_model(args: argparse.Namespace, design: dict[str, Any], design_path: Path) -> None:
    if not args.model or args.authorization is None:
        raise RuntimeError("--model and --authorization are required for model phase")
    entry = model_entry(design, args.model)
    if args.device != "cuda:0":
        raise RuntimeError("Frozen execution exposes one physical GPU and requires logical device cuda:0")
    validate_authorization(
        args.authorization,
        design,
        design_path,
        Path(__file__),
        SCRIPT_DIR / "STATIC_REVIEW_V1_1.md",
        entry,
    )
    output = resolve(WORKSPACE, design["output_root"]) / args.model
    if (output / "summary.json").exists():
        raise RuntimeError(f"Completed output already exists; refusing overwrite: {output}")
    for name in ("manifests", "tables", "candidate_cards", "status"):
        (output / name).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    config = {
        "dataset_path": str(resolve(WORKSPACE, design["dataset_path"])),
        "split_path": str(resolve(WORKSPACE, design["split_path"])),
        "prompt_template": design["prompt_template"], "score_threshold": design["score_threshold"],
    }
    data = load_dataset_and_split(config)
    model = None
    try:
        model, tokenizer, contract = load_model(entry, design, device)
        layers = eligible_layers(contract["num_hidden_layers"], design["layer_scope"])
        discovery_indices = [int(i) for i in data["indices"]["discovery"]]
        validation_indices = [int(i) for i in data["indices"]["validation"]]
        discovery_indices.sort(key=lambda i: len(tokenizer(data["prompts"][i], add_special_tokens=True)["input_ids"]))
        print(f"[{args.model}] Discovery: {len(discovery_indices)} items, layers={layers}", flush=True)
        discovery = discovery_pass(model, tokenizer, device, data, discovery_indices, layers, int(entry["batch_size"]))
        alignment = direction_alignment(args.model, design, layers, discovery["unit_directions"])
        if not alignment["pass"]:
            write_json(output / "status" / "direction_alignment_failure.json", alignment)
            raise RuntimeError(f"Direction alignment gate failed: {alignment['min_available_cosine']}")
        heads, neurons = component_frames(model, layers, discovery, design)
        candidates, controls = select_candidates_and_controls([heads, neurons], design)
        atomic_csv(output / "tables" / "discovery_heads.csv.gz", heads)
        atomic_csv(output / "tables" / "discovery_neurons.csv.gz", neurons)
        atomic_csv(output / "manifests" / "frozen_candidates.csv", candidates)
        atomic_csv(output / "manifests" / "frozen_controls.csv", controls)
        np.savez_compressed(output / "manifests" / "discovery_directions.npz", layers=np.asarray(layers), raw=discovery["raw_directions"], unit=discovery["unit_directions"])
        write_json(output / "manifests" / "direction_alignment.json", alignment)
        print(f"[{args.model}] Held-out component validation", flush=True)
        validation, baseline, validation_all = validation_pass(
            model, tokenizer, device, data, validation_indices, layers, discovery["unit_directions"], candidates, int(entry["batch_size"]),
        )
        all_metadata = pd.concat([heads, neurons], ignore_index=True)[
            ["component_id", "component_type", "module_index", "component_index", "relative_layer_depth", "depth_band"]
        ]
        validation_all = all_metadata.merge(validation_all, on="component_id", how="left", validate="one_to_one")
        atomic_csv(output / "tables" / "candidate_validation.csv", validation)
        atomic_csv(output / "tables" / "all_component_validation.csv.gz", validation_all)
        atomic_csv(output / "tables" / "baseline_validation_scores.csv", baseline)
        survivors = validation[
            validation["heldout_positive_same_sign"]
            & (validation["heldout_bh_q_candidates"] < float(design["heldout_survival"]["candidate_family_bh_q_lt"]))
        ].copy()
        survivor_controls = controls[controls["owner_candidate_id"].isin(survivors["component_id"])].copy()
        atomic_csv(output / "manifests" / "heldout_survivors.csv", survivors)
        print(f"[{args.model}] Causal suppression: {len(survivors)} held-out survivors", flush=True)
        final_direction = discovery["unit_directions"][layers.index(contract["num_hidden_layers"] - 1)]
        causal, cards = intervention_pass(
            model, tokenizer, device, data, design, final_direction, survivors, survivor_controls, validation,
            output, int(entry["batch_size"]),
        )
        atomic_csv(output / "tables" / "causal_candidate_results.csv", causal)
        pass_heads = sum(row["functional_homologue_pass"] and row["component_type"] == "head" for row in cards)
        pass_neurons = sum(row["functional_homologue_pass"] and row["component_type"] == "neuron" for row in cards)
        discovery_depth_counts = candidates.groupby(["component_type", "depth_band"]).size().to_dict()
        survivor_depth_counts = survivors.groupby(["component_type", "depth_band"]).size().to_dict()
        pass_depth_counts: dict[str, int] = {}
        for row in cards:
            if row["functional_homologue_pass"]:
                key = f"{row['component_type']}::{row['depth_band']}"
                pass_depth_counts[key] = pass_depth_counts.get(key, 0) + 1
        summary = {
            "study_id": design["study_id"], "model": args.model, "status": "COMPLETE",
            "contract": contract, "eligible_layers": layers,
            "discovery_candidate_count": len(candidates), "heldout_survivor_count": len(survivors),
            "intervention_candidate_count": len(cards),
            "discovery_depth_counts": {f"{key[0]}::{key[1]}": int(value) for key, value in discovery_depth_counts.items()},
            "survivor_depth_counts": {f"{key[0]}::{key[1]}": int(value) for key, value in survivor_depth_counts.items()},
            "functional_pass_depth_counts": pass_depth_counts,
            "functional_pass_heads": pass_heads, "functional_pass_neurons": pass_neurons,
            "class_recurrence": bool(pass_heads >= design["model_level"]["class_recurrence_min_heads"] and pass_neurons >= design["model_level"]["class_recurrence_min_neurons"]),
            "meta_like_sparse_pattern": bool(pass_heads >= design["model_level"]["meta_like_min_heads"] and pass_neurons >= design["model_level"]["meta_like_min_neurons"]),
            "passing_components": [row["component_id"] for row in cards if row["functional_homologue_pass"]],
            "direction_alignment_min_cosine": alignment["min_available_cosine"],
            "architecture_caveat": design["architecture_caveat"],
            "interpretation": "Functional recurrence in this fixed R/M dataset and prompt only; not weight-level neuron alignment or a universal reasoning mechanism.",
            "hashes": {
                "design": sha256_file(design_path), "implementation": sha256_file(Path(__file__)),
                "dataset": sha256_file(resolve(WORKSPACE, design["dataset_path"])),
                "split": sha256_file(resolve(WORKSPACE, design["split_path"])),
                "candidate_manifest": sha256_file(output / "manifests" / "frozen_candidates.csv"),
                "control_manifest": sha256_file(output / "manifests" / "frozen_controls.csv"),
                "causal_results": sha256_file(output / "tables" / "causal_candidate_results.csv"),
            },
        }
        write_json(output / "summary.json", summary)
        write_json(output / "status" / "complete.json", {"status": "COMPLETE", "summary_sha256": sha256_file(output / "summary.json")})
        print(f"[{args.model}] COMPLETE: heads={pass_heads}, neurons={pass_neurons}", flush=True)
    except Exception as exc:
        write_json(output / "status" / "error.json", {"status": "ERROR", "error": str(exc)})
        raise
    finally:
        release_model(model)


def combined_report(design: dict[str, Any], design_path: Path) -> None:
    root = resolve(WORKSPACE, design["output_root"])
    summaries = []
    for entry in design["models"]:
        path = root / entry["name"] / "summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing model summary: {path}")
        summaries.append(read_json(path))
    all_class = all(row["class_recurrence"] for row in summaries)
    all_meta = all(row["meta_like_sparse_pattern"] for row in summaries)
    combined = {
        "study_id": design["study_id"], "status": "COMPLETE", "models": summaries,
        "class_recurrence_in_all_three_models": all_class,
        "meta_like_sparse_pattern_in_all_three_models": all_meta,
        "claim_boundary": "This study tests functional recurrence of late component classes. It does not identify exact cross-model neuron/head correspondences or the input feature that creates R/M separation.",
        "result_pdf_updated": False,
        "hashes": {"design": sha256_file(design_path)},
    }
    lines = [
        "# Cross-model Functional Homologue Search v1.1 결과", "", "상태: **COMPLETE**", "",
        "## 모델별 결과", "", "| Model | Survivors | PASS Heads | PASS Neurons | PASS depth | Class recurrence | Meta-like 3H+1N | Passing IDs |",
        "|---|---:|---:|---:|---|:---:|:---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model']} | {row['heldout_survivor_count']} | {row['functional_pass_heads']} | {row['functional_pass_neurons']} | "
            f"{json.dumps(row['functional_pass_depth_counts'], ensure_ascii=False, sort_keys=True)} | "
            f"{'YES' if row['class_recurrence'] else 'NO'} | {'YES' if row['meta_like_sparse_pattern'] else 'NO'} | "
            f"{', '.join(row['passing_components']) if row['passing_components'] else 'none'} |"
        )
    lines.extend([
        "", "## 전체 판정", "",
        f"- 세 모델 모두에서 head+neuron class recurrence: **{'YES' if all_class else 'NO'}**",
        f"- 세 모델 모두에서 Meta-like 3-head+1-neuron sparse pattern: **{'YES' if all_meta else 'NO'}**",
        "", "## 해석 제한", "",
        "- 같은 번호 또는 weight-level neuron alignment를 의미하지 않는다.",
        "- 모든 layer를 검색했고 Early/Middle/Late는 결과 표지일 뿐 후보 할당량이 아니다.",
        "- Gemma-2와 OLMo-2의 discovery contribution은 post-normalization 이전 screening proxy다.",
        "- 최종 판정은 held-out 재현, dose response, matched/random control보다 큰 최종-layer R/M gap 감소를 모두 요구했다.",
        "- 이 결과는 R/M을 다르게 만드는 입력 Feature를 찾는 실험이 아니다.",
        "- `result.pdf`는 이 실행에서 자동 수정하지 않았다.",
    ])
    (root / "RESULTS_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    combined["hashes"]["results_ko"] = sha256_file(root / "RESULTS_KO.md")
    write_json(root / "combined_summary.json", combined)


def main() -> None:
    args = parse_args()
    design_path = args.design.resolve()
    design = load_design(design_path)
    if args.phase == "model":
        run_model(args, design, design_path)
    else:
        combined_report(design, design_path)


if __name__ == "__main__":
    main()
