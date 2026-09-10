"""Model capture and decomposition primitives for LiReF Stage A."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from stats import GroupMoments


GROUP_MEMORY = 0
GROUP_REASONING = 1
PROMPT_TEMPLATE_ID = "liref_question_only_q_a_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(payload: Any) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["config_path"] = str(path.resolve())
    config["config_hash"] = canonical_hash({k: v for k, v in config.items() if k not in {"config_path", "config_hash"}})
    return config


def code_state(experiment_dir: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in sorted(experiment_dir.glob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".sh", ".md"}:
            files[path.name] = sha256_file(path)
    git_state = "unavailable"
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=experiment_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0:
            git_state = proc.stdout.strip()
    except Exception:
        pass
    return {"file_sha256": files, "code_hash": canonical_hash(files), "git_commit": git_state}


def load_dataset_and_split(config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(config["dataset_path"])
    split_path = Path(config["split_path"])
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("dataset_sha256") != sha256_file(dataset_path):
        raise RuntimeError("Dataset SHA-256 does not match split_ids.json")
    if len(records) != 3000:
        raise RuntimeError(f"Expected 3000 records, found {len(records)}")
    required = {"question_id", "memory_reason_score", "category", "question"}
    for row_index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise KeyError(f"Dataset row {row_index} missing {sorted(missing)}")

    labels = np.asarray(
        [float(record["memory_reason_score"]) > float(config["score_threshold"]) for record in records],
        dtype=np.int8,
    )
    question_ids = [str(record["question_id"]) for record in records]
    if len(set(question_ids)) != len(question_ids):
        raise RuntimeError("question_id is not unique")

    mapping = {"discovery": "train", "validation": "heldout"}
    indices: dict[str, np.ndarray] = {}
    for analysis_name, source_name in mapping.items():
        values = np.asarray(split[source_name]["row_indices"], dtype=np.int64)
        expected_ids = [str(value) for value in split[source_name]["question_ids"]]
        actual_ids = [question_ids[int(index)] for index in values]
        if actual_ids != expected_ids:
            raise RuntimeError(f"question_id mapping mismatch in {source_name}")
        if int((labels[values] == GROUP_REASONING).sum()) != int(split[source_name]["n_reasoning"]):
            raise RuntimeError(f"Reasoning count mismatch in {source_name}")
        if int((labels[values] == GROUP_MEMORY).sum()) != int(split[source_name]["n_memory"]):
            raise RuntimeError(f"Memory count mismatch in {source_name}")
        indices[analysis_name] = values
    if set(indices["discovery"]) & set(indices["validation"]):
        raise RuntimeError("Discovery and validation overlap")
    if set(indices["discovery"]) | set(indices["validation"]) != set(range(len(records))):
        raise RuntimeError("Discovery and validation do not cover the dataset")
    prompts = [config["prompt_template"].format(question=record["question"]) for record in records]
    return {
        "records": records,
        "labels": labels,
        "question_ids": question_ids,
        "indices": indices,
        "prompts": prompts,
        "split": split,
        "mapping": mapping,
        "dataset_sha256": sha256_file(dataset_path),
        "split_sha256": sha256_file(split_path),
    }


def build_identity(config: dict[str, Any], data: dict[str, Any], experiment_dir: Path) -> dict[str, Any]:
    model_path = Path(config["model_path"])
    code = code_state(experiment_dir)
    identity = {
        "run_id": config["run_id"],
        "config_hash": config["config_hash"],
        "code_hash": code["code_hash"],
        "code_files": code["file_sha256"],
        "git_commit": code["git_commit"],
        "model_path": str(model_path.resolve()),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "dataset_path": str(Path(config["dataset_path"]).resolve()),
        "dataset_sha256": data["dataset_sha256"],
        "split_path": str(Path(config["split_path"]).resolve()),
        "split_sha256": data["split_sha256"],
        "split_seed": int(data["split"]["random_seed"]),
        "source_to_analysis_split": {"train": "discovery", "heldout": "validation"},
        "discovery_row_ids_hash": canonical_hash(data["indices"]["discovery"].tolist()),
        "validation_row_ids_hash": canonical_hash(data["indices"]["validation"].tolist()),
        "prompt_template": config["prompt_template"],
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "prompt_template_sha256": sha256_text(config["prompt_template"]),
        "dtype": config["dtype"],
        "model_training": False,
    }
    identity["identity_hash"] = canonical_hash(identity)
    return identity


def environment_payload(device: torch.device | None = None) -> dict[str, Any]:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device is not None and device.type == "cuda":
        result["device"] = str(device)
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return result


def freeze_run_inputs(output_root: Path, config: dict[str, Any], identity: dict[str, Any]) -> None:
    checkpoints = output_root / "a_core" / "checkpoints"
    frozen_config_path = checkpoints / "frozen_config.json"
    manifest_path = checkpoints / "input_manifest.json"
    if frozen_config_path.exists():
        existing_config = json.loads(frozen_config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise RuntimeError("Existing frozen_config.json differs from current config")
    else:
        atomic_json(frozen_config_path, config)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("Existing input_manifest.json differs; refusing stale checkpoint reuse")
    else:
        atomic_json(manifest_path, identity)


def choose_sanity_indices(data: dict[str, Any], per_group: int) -> np.ndarray:
    discovery = data["indices"]["discovery"]
    labels = data["labels"]
    memory = discovery[labels[discovery] == GROUP_MEMORY][:per_group]
    reasoning = discovery[labels[discovery] == GROUP_REASONING][:per_group]
    if len(memory) != per_group or len(reasoning) != per_group:
        raise RuntimeError("Insufficient discovery samples for sanity subset")
    return np.sort(np.concatenate([memory, reasoning]))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name}")
    return mapping[name]


def load_model_and_tokenizer(config: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    dtype = resolve_dtype(config["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()
    model.to(device)
    if model.training:
        raise RuntimeError("Model must be in eval mode")
    return model, tokenizer


def validate_model_contract(model: Any, tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
    c = model.config
    if c.model_type != "llama" or c.num_hidden_layers != 32:
        raise RuntimeError(f"Unexpected model architecture: {c.model_type}, layers={c.num_hidden_layers}")
    if c.hidden_size != 4096 or c.intermediate_size != 14336:
        raise RuntimeError("Unexpected Meta-Llama-3-8B dimensions")
    if c.num_attention_heads != 32 or c.num_key_value_heads != 8:
        raise RuntimeError("Unexpected attention head configuration")
    return {
        "model_type": c.model_type,
        "num_hidden_layers": c.num_hidden_layers,
        "hidden_size": c.hidden_size,
        "intermediate_size": c.intermediate_size,
        "num_attention_heads": c.num_attention_heads,
        "num_key_value_heads": c.num_key_value_heads,
        "head_dim": getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
        "attention_bias": bool(c.attention_bias),
        "mlp_bias": bool(c.mlp_bias),
        "attention_backend": c._attn_implementation,
        "model_config_dtype": str(c.torch_dtype),
        "analysis_dtype": config["dtype"],
        "tokenizer_padding_side": tokenizer.padding_side,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
    }


class LayerCapture:
    def __init__(self, model: Any, components: bool) -> None:
        self.model = model
        self.components = components
        self.handles: list[Any] = []
        self.reset()

    def reset(self) -> None:
        # Hooks close over these dictionaries when installed. Clear them in place
        # between batches so the hook targets remain valid.
        for name in ("h_in", "h_out", "attn_pre_o", "attn_out", "mlp_input", "z", "mlp_out"):
            if hasattr(self, name):
                getattr(self, name).clear()
            else:
                setattr(self, name, {})

    @staticmethod
    def _last(value: torch.Tensor) -> torch.Tensor:
        return value.detach()[:, -1, :].clone()

    def install(self) -> None:
        if self.handles:
            raise RuntimeError("Hooks already installed")
        layers = self.model.model.layers
        for index, layer in enumerate(layers):
            self.handles.append(layer.register_forward_pre_hook(self._layer_pre(index)))
            self.handles.append(layer.register_forward_hook(self._layer_post(index)))
            if self.components:
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._store_pre(self.attn_pre_o, index)))
                self.handles.append(layer.self_attn.o_proj.register_forward_hook(self._store_post(self.attn_out, index)))
                self.handles.append(layer.mlp.register_forward_pre_hook(self._store_pre(self.mlp_input, index)))
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._store_pre(self.z, index)))
                self.handles.append(layer.mlp.down_proj.register_forward_hook(self._store_post(self.mlp_out, index)))

    def _layer_pre(self, index: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            self.h_in[index] = self._last(args[0])
        return hook

    def _layer_post(self, index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.h_out[index] = self._last(tensor)
        return hook

    def _store_pre(self, target: dict[int, torch.Tensor], index: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            target[index] = self._last(args[0])
        return hook

    def _store_post(self, target: dict[int, torch.Tensor], index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            target[index] = self._last(output)
        return hook

    def validate_complete(self) -> None:
        expected = set(range(self.model.config.num_hidden_layers))
        fields = {"h_in": self.h_in, "h_out": self.h_out}
        if self.components:
            fields.update({
                "attn_pre_o": self.attn_pre_o,
                "attn_out": self.attn_out,
                "mlp_input": self.mlp_input,
                "z": self.z,
                "mlp_out": self.mlp_out,
            })
        for name, mapping in fields.items():
            if set(mapping) != expected:
                raise RuntimeError(f"Incomplete hook capture for {name}: {sorted(mapping)}")

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def encode_batch(tokenizer: Any, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        truncation=False,
        return_token_type_ids=False,
    )
    if not bool(torch.all(encoded["attention_mask"][:, -1] == 1)):
        raise RuntimeError("Index -1 is not the final prompt token for every sample")
    return {key: value.to(device) for key, value in encoded.items()}


def _group_reductions(values: torch.Tensor, labels: torch.Tensor) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    output = []
    for group in (GROUP_MEMORY, GROUP_REASONING):
        selected = values[labels == group]
        count = int(selected.shape[0])
        if count == 0:
            shape = tuple(values.shape[1:])
            output.append((group, 0, np.zeros(shape), np.zeros(shape)))
            continue
        value_sum = selected.double().sum(dim=0).cpu().numpy()
        value_sumsq = selected.double().square().sum(dim=0).cpu().numpy()
        output.append((group, count, value_sum, value_sumsq))
    return output


@torch.inference_mode()
def direction_pass(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    data: dict[str, Any],
    indices: np.ndarray,
    batch_size: int,
    epsilon: float,
    capture_samples: bool = False,
) -> dict[str, Any]:
    layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    sums = np.zeros((2, layers, hidden), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    sample_h_out: dict[int, np.ndarray] = {}
    capture = LayerCapture(model, components=False)
    capture.install()
    try:
        for batch_indices in iter_batches(indices, batch_size):
            capture.reset()
            prompts = [data["prompts"][int(index)] for index in batch_indices]
            encoded = encode_batch(tokenizer, prompts, device)
            model(**encoded, use_cache=False, output_hidden_states=False, return_dict=True)
            capture.validate_complete()
            stacked = torch.stack([capture.h_out[layer] for layer in range(layers)], dim=1)
            labels = torch.as_tensor(data["labels"][batch_indices], device=device)
            for group, count, value_sum, _ in _group_reductions(stacked, labels):
                counts[group] += count
                sums[group] += value_sum
            if capture_samples:
                values = stacked.float().cpu().numpy()
                for offset, row_index in enumerate(batch_indices):
                    sample_h_out[int(row_index)] = values[offset]
            del encoded, stacked, labels
    finally:
        capture.remove()
    means = sums / counts[:, None, None]
    raw = means[GROUP_REASONING] - means[GROUP_MEMORY]
    norms = np.linalg.norm(raw, axis=1)
    unit = np.zeros_like(raw)
    valid = np.isfinite(norms) & (norms > epsilon)
    unit[valid] = raw[valid] / norms[valid, None]
    return {
        "counts": counts,
        "group_means": means,
        "raw_directions": raw,
        "direction_norms": norms,
        "unit_directions": unit,
        "valid_direction": valid,
        "sample_h_out": sample_h_out,
    }


def projection_weights(model: Any, unit_directions: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    head_weights = []
    neuron_weights = []
    dtype = next(model.parameters()).dtype
    for layer_index, layer in enumerate(model.model.layers):
        direction = torch.as_tensor(unit_directions[layer_index], device=device, dtype=dtype)
        o_weight = layer.self_attn.o_proj.weight
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        blocks = o_weight.reshape(model.config.hidden_size, model.config.num_attention_heads, head_dim)
        q = torch.einsum("ohd,o->hd", blocks, direction)
        head_weights.append(q)
        neuron_weights.append(torch.mv(layer.mlp.down_proj.weight.T, direction))
    return torch.stack(head_weights), torch.stack(neuron_weights)


@torch.inference_mode()
def component_pass(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    data: dict[str, Any],
    indices: np.ndarray,
    batch_size: int,
    unit_directions: np.ndarray,
    sanity_manual_z: bool = False,
    capture_samples: bool = False,
) -> dict[str, Any]:
    layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    heads = model.config.num_attention_heads
    intermediate = model.config.intermediate_size
    head_dim = hidden // heads
    head_projection, neuron_projection = projection_weights(model, unit_directions, device)
    head_moments = GroupMoments((layers, heads))
    neuron_moments = GroupMoments((layers, intermediate))
    activation_sums = np.zeros((2, layers, intermediate), dtype=np.float64)
    activation_counts = np.zeros(2, dtype=np.int64)
    vector_sums = {name: np.zeros((2, layers, hidden), dtype=np.float64) for name in ("input", "attention", "mlp", "output")}
    vector_counts = np.zeros(2, dtype=np.int64)
    z_sums = np.zeros((2, layers, intermediate), dtype=np.float64)
    reconstruction = {name: {"sum_abs": 0.0, "max_abs": 0.0, "n": 0} for name in ("layer_vector", "layer_scalar", "head_scalar", "ffn_scalar", "gated_activation")}
    sample_h_out: dict[int, np.ndarray] = {}
    capture = LayerCapture(model, components=True)
    capture.install()
    try:
        for batch_indices in iter_batches(indices, batch_size):
            capture.reset()
            prompts = [data["prompts"][int(index)] for index in batch_indices]
            encoded = encode_batch(tokenizer, prompts, device)
            model(**encoded, use_cache=False, output_hidden_states=False, return_dict=True)
            capture.validate_complete()
            h_in = torch.stack([capture.h_in[i] for i in range(layers)], dim=1)
            h_out = torch.stack([capture.h_out[i] for i in range(layers)], dim=1)
            attn_out = torch.stack([capture.attn_out[i] for i in range(layers)], dim=1)
            mlp_out = torch.stack([capture.mlp_out[i] for i in range(layers)], dim=1)
            pre_o = torch.stack([capture.attn_pre_o[i] for i in range(layers)], dim=1).reshape(-1, layers, heads, head_dim)
            z = torch.stack([capture.z[i] for i in range(layers)], dim=1)
            labels = torch.as_tensor(data["labels"][batch_indices], device=device)

            layer_error = (h_out - (h_in + attn_out + mlp_out)).abs()
            _update_error(reconstruction["layer_vector"], layer_error)
            directions = torch.as_tensor(unit_directions, device=device, dtype=h_out.dtype)
            input_score = torch.einsum("blh,lh->bl", h_in, directions)
            attention_score = torch.einsum("blh,lh->bl", attn_out, directions)
            mlp_score = torch.einsum("blh,lh->bl", mlp_out, directions)
            output_score = torch.einsum("blh,lh->bl", h_out, directions)
            scalar_error = (output_score - input_score - attention_score - mlp_score).abs()
            _update_error(reconstruction["layer_scalar"], scalar_error)

            head_score = (pre_o * head_projection.unsqueeze(0)).sum(dim=-1)
            _update_error(reconstruction["head_scalar"], (head_score.sum(dim=-1) - attention_score).abs())
            neuron_score = z * neuron_projection.unsqueeze(0)
            _update_error(reconstruction["ffn_scalar"], (neuron_score.sum(dim=-1) - mlp_score).abs())

            if sanity_manual_z:
                mlp_inputs = torch.stack([capture.mlp_input[i] for i in range(layers)], dim=1)
                manual_parts = []
                for layer_index, layer in enumerate(model.model.layers):
                    x = mlp_inputs[:, layer_index]
                    manual_parts.append(layer.mlp.act_fn(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x))
                manual_z = torch.stack(manual_parts, dim=1)
                _update_error(reconstruction["gated_activation"], (manual_z - z).abs())
                del mlp_inputs, manual_parts, manual_z

            for group, count, value_sum, value_sumsq in _group_reductions(head_score, labels):
                head_moments.update_reduced(group, count, value_sum, value_sumsq)
            for group, count, value_sum, value_sumsq in _group_reductions(neuron_score, labels):
                neuron_moments.update_reduced(group, count, value_sum, value_sumsq)
            for group in (GROUP_MEMORY, GROUP_REASONING):
                mask = labels == group
                count = int(mask.sum())
                if count == 0:
                    continue
                activation_counts[group] += count
                z_sums[group] += z[mask].double().sum(dim=0).cpu().numpy()
                activation_sums[group] += z[mask].double().sum(dim=0).cpu().numpy()
                vector_counts[group] += count
                for name, values in (("input", h_in), ("attention", attn_out), ("mlp", mlp_out), ("output", h_out)):
                    vector_sums[name][group] += values[mask].double().sum(dim=0).cpu().numpy()

            if capture_samples:
                values = h_out.float().cpu().numpy()
                for offset, row_index in enumerate(batch_indices):
                    sample_h_out[int(row_index)] = values[offset]
            del encoded, labels, h_in, h_out, attn_out, mlp_out, pre_o, z, head_score, neuron_score
    finally:
        capture.remove()

    vector_means = {name: values / vector_counts[:, None, None] for name, values in vector_sums.items()}
    vector_deltas = {name: values[GROUP_REASONING] - values[GROUP_MEMORY] for name, values in vector_means.items()}
    delta_z = z_sums[GROUP_REASONING] / activation_counts[GROUP_REASONING] - z_sums[GROUP_MEMORY] / activation_counts[GROUP_MEMORY]
    ffn_reconstructed = []
    for layer_index, layer in enumerate(model.model.layers):
        weight = layer.mlp.down_proj.weight.detach().double().cpu().numpy()
        ffn_reconstructed.append(weight @ delta_z[layer_index])
    ffn_reconstructed_array = np.stack(ffn_reconstructed)
    ffn_vector_error = np.abs(ffn_reconstructed_array - vector_deltas["mlp"])
    reconstruction["ffn_vector"] = {
        "sum_abs": float(ffn_vector_error.sum()),
        "max_abs": float(ffn_vector_error.max()),
        "n": int(ffn_vector_error.size),
    }
    return {
        "head_moments": head_moments.state_dict(),
        "neuron_moments": neuron_moments.state_dict(),
        "activation_sums": activation_sums,
        "activation_counts": activation_counts,
        "vector_counts": vector_counts,
        "vector_means": vector_means,
        "vector_deltas": vector_deltas,
        "delta_z": delta_z,
        "ffn_mlp_vector_reconstructed": ffn_reconstructed_array,
        "reconstruction": reconstruction,
        "sample_h_out": sample_h_out,
    }


def _update_error(target: dict[str, Any], values: torch.Tensor) -> None:
    detached = values.detach().double()
    target["sum_abs"] += float(detached.sum().item())
    target["max_abs"] = max(float(target["max_abs"]), float(detached.max().item()))
    target["n"] += int(detached.numel())


def finalize_errors(payload: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    output = {}
    for name, values in payload.items():
        n = int(values.get("n", 0))
        output[name] = {
            "mean_abs_error": float(values.get("sum_abs", 0.0)) / n if n else math.nan,
            "max_abs_error": float(values.get("max_abs", math.nan)),
            "n_values": n,
        }
    return output


def release_model(model: Any | None) -> None:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
