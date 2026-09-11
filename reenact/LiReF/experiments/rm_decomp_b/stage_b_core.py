"""Core gates, feature extraction, controls, and model capture for LiReF Stage B."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_A_DIR = SCRIPT_DIR.parent / "rm_decomp"
if str(STAGE_A_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_A_DIR))

from core import (  # noqa: E402
    GROUP_MEMORY,
    GROUP_REASONING,
    atomic_csv,
    atomic_json,
    canonical_hash,
    load_dataset_and_split,
    load_model_and_tokenizer,
    release_model,
    sha256_file,
    sha256_text,
)


CONTROLLED_COLUMNS = [
    "pair_id",
    "hypothesis_id",
    "feature_family",
    "template_id",
    "template_family",
    "split",
    "original_text",
    "modified_text",
    "changed_span_original",
    "changed_span_modified",
    "semantic_role",
    "changed_feature",
    "invariant_features",
    "token_length_original",
    "token_length_modified",
    "number_spans_original",
    "number_spans_modified",
    "generation_rule_id",
    "reviewer_id",
    "human_validated",
    "approved",
    "expected_answer_original",
    "expected_answer_modified",
]


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return jsonable(value.tolist() if isinstance(value, np.ndarray) else value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    atomic_json(path, jsonable(value))


def composite_file_hash(paths: Iterable[Path]) -> tuple[str, dict[str, str]]:
    values = {str(path.resolve()): sha256_file(path) for path in sorted(paths)}
    return canonical_hash(values), values


def model_parameter_checksum(model_path: Path) -> tuple[str, dict[str, str]]:
    files = []
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        files.extend(model_path.glob(pattern))
    if not files:
        raise RuntimeError(f"No model parameter files found in {model_path}")
    return composite_file_hash(files)


def code_checksum() -> tuple[str, dict[str, str]]:
    return composite_file_hash(
        path for path in SCRIPT_DIR.iterdir() if path.is_file() and path.suffix in {".py", ".json", ".sh", ".md"}
    )


def require_status(output_root: Path, phase: str) -> dict[str, Any]:
    path = output_root / "status" / f"{phase}.json"
    if not path.exists():
        raise RuntimeError(f"Required phase has not run: {phase}")
    payload = read_json(path)
    if payload.get("status") != "PASS":
        raise RuntimeError(f"Required phase is not PASS: {phase}")
    return payload


def write_status(output_root: Path, phase: str, status: str, **values: Any) -> None:
    write_json(
        output_root / "status" / f"{phase}.json",
        {"phase": phase, "status": status, **values},
    )


def validate_backend_tolerances(config: dict[str, Any]) -> None:
    required = {
        "logit_max_abs_tolerance",
        "hidden_state_max_abs_tolerance",
        "hidden_state_cosine_tolerance",
        "head_reconstruction_mean_tolerance",
        "head_reconstruction_max_tolerance",
        "source_reconstruction_mean_tolerance",
        "source_reconstruction_max_tolerance",
    }
    values = config.get("backend_tolerances", {})
    if set(values) != required:
        raise RuntimeError(f"backend_tolerances must contain exactly {sorted(required)}")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"Backend tolerance must be a finite number: {key}")
    if not 0.0 <= float(values["hidden_state_cosine_tolerance"]) <= 1.0:
        raise RuntimeError("hidden_state_cosine_tolerance must be in [0, 1]")
    for key, value in values.items():
        if key != "hidden_state_cosine_tolerance" and float(value) < 0:
            raise RuntimeError(f"Backend error tolerance cannot be negative: {key}")


def _reject_placeholders(value: Any, location: str) -> None:
    if value is None:
        raise RuntimeError(f"Feature schema contains null: {location}")
    if isinstance(value, str) and (not value.strip() or "..." in value):
        raise RuntimeError(f"Feature schema contains empty/ellipsis placeholder: {location}")
    if isinstance(value, (list, dict)) and not value:
        raise RuntimeError(f"Feature schema contains an empty collection: {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_placeholders(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_placeholders(item, f"{location}[{index}]")


def validate_feature_schema(schema: dict[str, Any]) -> None:
    if schema.get("approved") is not True:
        raise RuntimeError("feature_schema.json must have approved=true")
    required = {
        "numeric_digit_span",
        "numeric_word_lexicon",
        "relation_lexicon",
        "symbolic_operator_set",
        "unicode_operator_set",
        "textual_operator_lexicon",
        "domain_source",
        "token_length_definition",
    }
    missing = required.difference(schema)
    if missing:
        raise RuntimeError(f"Feature schema missing fields: {sorted(missing)}")
    _reject_placeholders(schema, "feature_schema")
    re.compile(schema["numeric_digit_span"]["pattern"], re.UNICODE)


def validate_confirmatory_power(config: dict[str, Any], require_count: bool) -> None:
    required = {
        "target_power",
        "two_sided_alpha",
        "minimum_effect_of_interest_dz",
        "multiplicity_adjustment_plan",
        "expected_template_exclusion_rate",
        "planned_confirmatory_template_count",
        "maximum_template_count",
        "sample_size_method",
    }
    values = config.get("confirmatory_power", {})
    if set(values) != required:
        raise RuntimeError(f"confirmatory_power must contain exactly {sorted(required)}")
    for key in ("target_power", "two_sided_alpha", "minimum_effect_of_interest_dz", "expected_template_exclusion_rate"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"Invalid confirmatory power value: {key}")
    if not 0.0 < float(values["target_power"]) < 1.0:
        raise RuntimeError("target_power must be in (0, 1)")
    if not 0.0 < float(values["two_sided_alpha"]) < 1.0:
        raise RuntimeError("two_sided_alpha must be in (0, 1)")
    if float(values["minimum_effect_of_interest_dz"]) <= 0.0:
        raise RuntimeError("minimum_effect_of_interest_dz must be positive")
    if not 0.0 <= float(values["expected_template_exclusion_rate"]) < 1.0:
        raise RuntimeError("expected_template_exclusion_rate must be in [0, 1)")
    if require_count and (not isinstance(values["planned_confirmatory_template_count"], int) or values["planned_confirmatory_template_count"] <= 1):
        raise RuntimeError("planned_confirmatory_template_count must be frozen to an integer >1")
    if not isinstance(values["maximum_template_count"], int) or values["maximum_template_count"] <= 1:
        raise RuntimeError("maximum_template_count must be an integer >1")


@dataclass
class FeatureExtractor:
    schema: dict[str, Any]

    def __post_init__(self) -> None:
        normalization = self.schema["normalization"]
        self.unicode_form = normalization["unicode"]
        self.casefold = bool(normalization["casefold"])
        self.digit_pattern = re.compile(self.schema["numeric_digit_span"]["pattern"], re.UNICODE)
        self.number_words = set(self.schema["numeric_word_lexicon"])
        self.relation_words = {
            word: family for family, words in self.schema["relation_lexicon"].items() for word in words
        }
        self.textual_operators = {
            word: family for family, words in self.schema["textual_operator_lexicon"].items() for word in words
        }
        self.symbolic_operators = set(self.schema["symbolic_operator_set"] + self.schema["unicode_operator_set"])
        self.word_pattern = re.compile(r"\b[\w'-]+\b", re.UNICODE)

    def normalize(self, text: str) -> str:
        result = unicodedata.normalize(self.unicode_form, text)
        return result.casefold() if self.casefold else result

    def spans(self, text: str) -> list[dict[str, Any]]:
        normalized = self.normalize(text)
        output: list[dict[str, Any]] = []
        occupied: set[tuple[int, int, str]] = set()

        def add(start: int, end: int, role: str, family: str, value: str) -> None:
            key = (start, end, role)
            if key not in occupied:
                occupied.add(key)
                output.append({"start": start, "end": end, "semantic_role": role, "family": family, "text": value})

        for match in self.digit_pattern.finditer(normalized):
            add(match.start(), match.end(), "numeric", "numeric_digit", match.group())
        for match in self.word_pattern.finditer(normalized):
            word = match.group()
            if word in self.number_words:
                add(match.start(), match.end(), "numeric", "numeric_word", word)
            if word in self.relation_words:
                add(match.start(), match.end(), "relation", self.relation_words[word], word)
            if word in self.textual_operators:
                add(match.start(), match.end(), "operator", self.textual_operators[word], word)
        for index, character in enumerate(normalized):
            if character in self.symbolic_operators:
                add(index, index + 1, "operator", "symbolic", character)
        return sorted(output, key=lambda item: (item["start"], item["end"], item["semantic_role"]))

    def summarize(self, text: str) -> dict[str, Any]:
        spans = self.spans(text)
        counts = Counter(item["semantic_role"] for item in spans)
        return {
            "numeric_span_count": int(counts["numeric"]),
            "has_numeric": bool(counts["numeric"]),
            "relation_span_count": int(counts["relation"]),
            "has_relation": bool(counts["relation"]),
            "operator_span_count": int(counts["operator"]),
            "has_operator": bool(counts["operator"]),
            "feature_spans": spans,
        }


def token_semantic_role(start: int, end: int, spans: list[dict[str, Any]], prompt_prefix_end: int, prompt_suffix_start: int) -> tuple[str, str]:
    if end <= prompt_prefix_end:
        return "prompt_prefix", "prompt_prefix"
    if start >= prompt_suffix_start:
        return "prompt_suffix", "prompt_suffix"
    overlaps = [item for item in spans if max(start, item["start"] + prompt_prefix_end) < min(end, item["end"] + prompt_prefix_end)]
    if not overlaps:
        return "other", "other"
    priority = {"numeric": 0, "relation": 1, "operator": 2}
    item = sorted(overlaps, key=lambda row: priority.get(row["semantic_role"], 99))[0]
    return str(item["semantic_role"]), str(item["family"])


def load_stage_a_assets(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["stage_a_root"])
    status = read_json(root / "stage_status.json")
    if status.get("status") != "PASS":
        raise RuntimeError("Stage A full status is not PASS")
    identity = read_json(root / "checkpoints" / "input_manifest.json")
    manifest_path = root / "manifests" / "candidate_manifest.json"
    manifest = read_json(manifest_path)
    direction_path = root / "checkpoints" / "discovery_liref_directions.pt"
    if manifest.get("identity_hash") != identity.get("identity_hash"):
        raise RuntimeError("Stage A candidate and input identity hashes differ")
    if manifest.get("direction_artifact_sha256") != sha256_file(direction_path):
        raise RuntimeError("Stage A direction artifact hash mismatch")
    prompt_hash = sha256_text(config["prompt_template"])
    if identity.get("prompt_template") != config["prompt_template"] or identity.get("prompt_template_sha256") != prompt_hash:
        raise RuntimeError("Stage A and Stage B prompt templates differ")
    if sha256_file(Path(config["dataset_path"])) != identity.get("dataset_sha256"):
        raise RuntimeError("Dataset hash differs from Stage A")
    if sha256_file(Path(config["split_path"])) != identity.get("split_sha256"):
        raise RuntimeError("Split hash differs from Stage A")
    if sha256_file(Path(config["model_path"]) / "config.json") != identity.get("model_config_sha256"):
        raise RuntimeError("Model config hash differs from Stage A")
    return {
        "root": root,
        "status": status,
        "identity": identity,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "direction_path": direction_path,
    }


def frozen_candidates(config: dict[str, Any], assets: dict[str, Any]) -> list[dict[str, Any]]:
    expected = config["expected_candidates"]
    output: list[dict[str, Any]] = []
    for component_type, filename in (
        ("head", "head_candidate_validation.csv"),
        ("neuron", "neuron_candidate_validation.csv"),
    ):
        source = [row for row in assets["manifest"]["candidates"][component_type] if row.get("detailed_candidate") is True]
        table = pd.read_csv(assets["root"] / "tables" / filename)
        if len(source) != int(expected[component_type]):
            raise RuntimeError(f"Unexpected detailed {component_type} candidate count: {len(source)}")
        if len({row["component_id"] for row in source}) != len(source):
            raise RuntimeError(f"Duplicate {component_type} candidate")
        by_id = table.set_index("component_id", drop=False)
        for row in source:
            component_id = row["component_id"]
            if component_id not in by_id.index:
                raise RuntimeError(f"Candidate missing validation metadata: {component_id}")
            metadata = jsonable(by_id.loc[component_id].to_dict())
            if metadata.get("reproduced") is not True:
                raise RuntimeError(f"Detailed candidate failed Stage A validation: {component_id}")
            output.append({**row, "component_type": component_type, "stage_a_metadata": metadata})
        signs = Counter(row["sign_group_discovery"] for row in source)
        for sign in ("positive", "negative"):
            if signs[sign] != int(expected["positive_per_type"] if sign == "positive" else expected["negative_per_type"]):
                raise RuntimeError(f"Unexpected {component_type} {sign} candidate count")
    if len({row["component_id"] for row in output}) != len(output):
        raise RuntimeError("Duplicate component ID across frozen candidates")
    return sorted(output, key=lambda row: (row["component_type"], row["module_index"], row["component_index"]))


def load_directions(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = np.asarray(payload["result"]["unit_directions"], dtype=np.float64)
    if result.shape != (32, 4096) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"Unexpected direction array: {result.shape}")
    return result


def pooled_stats(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    n0 = result["n_memory"].astype(float)
    n1 = result["n_reasoning"].astype(float)
    m0 = result["memory_mean"].astype(float)
    m1 = result["reasoning_mean"].astype(float)
    v0 = result["memory_variance"].astype(float)
    v1 = result["reasoning_variance"].astype(float)
    n = n0 + n1
    mean = (n0 * m0 + n1 * m1) / n
    ss = (n0 - 1) * v0 + (n1 - 1) * v1 + n0 * (m0 - mean) ** 2 + n1 * (m1 - mean) ** 2
    result["pooled_contribution_mean"] = mean
    result["pooled_contribution_variance"] = ss / (n - 1)
    if "memory_activation_mean" in result:
        result["pooled_activation_mean"] = (n0 * result["memory_activation_mean"] + n1 * result["reasoning_activation_mean"]) / n
    return result


@torch.inference_mode()
def projection_metadata(model: Any, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    head_norms = []
    neuron_abs = []
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    for module_index, layer in enumerate(model.model.layers):
        direction = torch.as_tensor(directions[module_index], device=device, dtype=dtype)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        blocks = layer.self_attn.o_proj.weight.reshape(model.config.hidden_size, model.config.num_attention_heads, head_dim)
        q = torch.einsum("ohd,o->hd", blocks, direction)
        p = torch.mv(layer.mlp.down_proj.weight.T, direction)
        head_norms.append(q.double().norm(dim=-1).cpu().numpy())
        neuron_abs.append(p.double().abs().cpu().numpy())
    return np.stack(head_norms), np.stack(neuron_abs)


def build_controls(
    config: dict[str, Any],
    assets: dict[str, Any],
    candidates: list[dict[str, Any]],
    head_norms: np.ndarray,
    neuron_abs: np.ndarray,
) -> dict[str, Any]:
    all_candidate_ids = {
        row["component_id"]
        for kind in ("head", "neuron")
        for row in assets["manifest"]["candidates"][kind]
    }
    records: list[dict[str, Any]] = []
    dropped_variables: dict[str, list[dict[str, str]]] = {"head": [], "neuron": []}
    random_count = int(config["controls"]["random_per_candidate"])
    seed = int(config["seed"])

    for component_type, filename, projection_name, projection_values in (
        ("head", "discovery_head_statistics.csv.gz", "projection_norm", head_norms),
        ("neuron", "discovery_neuron_statistics.csv.gz", "projection_abs", neuron_abs),
    ):
        pool = pooled_stats(pd.read_csv(assets["root"] / "tables" / filename, low_memory=False))
        pool[projection_name] = projection_values[pool["module_index"].to_numpy(int), pool["component_index"].to_numpy(int)]
        variables = ["pooled_contribution_mean", "pooled_contribution_variance", projection_name]
        if component_type == "neuron" and "pooled_activation_mean" in pool:
            variables.append("pooled_activation_mean")
        usable = []
        standardized: dict[str, np.ndarray] = {}
        for variable in variables:
            values = pool[variable].to_numpy(float)
            finite = np.isfinite(values)
            std = float(np.std(values[finite], ddof=0)) if finite.any() else math.nan
            if not math.isfinite(std) or std == 0.0:
                dropped_variables[component_type].append({"variable": variable, "reason": "zero_or_nonfinite_pool_standard_deviation"})
                continue
            mean = float(values[finite].mean())
            standardized[variable] = (values - mean) / std
            usable.append(variable)
        if not usable:
            raise RuntimeError(f"No usable control matching variables for {component_type}")
        index_by_id = {value: index for index, value in enumerate(pool["component_id"].astype(str))}
        for candidate in [row for row in candidates if row["component_type"] == component_type]:
            cid = candidate["component_id"]
            ci = index_by_id[cid]
            same_layer = (pool["module_index"].to_numpy(int) == int(candidate["module_index"]))
            eligible = same_layer & ~pool["component_id"].isin(all_candidate_ids).to_numpy()
            distances = np.zeros(len(pool), dtype=np.float64)
            valid = eligible.copy()
            used_for_candidate = []
            for variable in usable:
                values = standardized[variable]
                if not math.isfinite(float(values[ci])):
                    continue
                valid &= np.isfinite(values)
                distances += (values - values[ci]) ** 2
                used_for_candidate.append(variable)
            if not used_for_candidate:
                raise RuntimeError(f"Candidate has no finite matching variables: {cid}")
            eligible_indices = np.flatnonzero(valid)
            if not len(eligible_indices):
                raise RuntimeError(f"No eligible matched control for {cid}")
            order = sorted(eligible_indices.tolist(), key=lambda i: (float(distances[i]), int(pool.iloc[i]["component_index"])))
            matched_index = order[0]
            matched_id = str(pool.iloc[matched_index]["component_id"])
            records.append(
                {
                    "candidate_id": cid,
                    "control_id": matched_id,
                    "control_kind": "matched",
                    "component_type": component_type,
                    "module_index": int(candidate["module_index"]),
                    "component_index": int(pool.iloc[matched_index]["component_index"]),
                    "matching_variables": used_for_candidate,
                    "candidate_matching_values": {name: float(pool.iloc[ci][name]) for name in used_for_candidate},
                    "control_matching_values": {name: float(pool.iloc[matched_index][name]) for name in used_for_candidate},
                    "standardized_euclidean_distance": float(math.sqrt(distances[matched_index])),
                    "seed": seed,
                }
            )
            random_pool = [i for i in eligible_indices.tolist() if str(pool.iloc[i]["component_id"]) != matched_id]
            if len(random_pool) < random_count:
                raise RuntimeError(f"Insufficient random controls for {cid}")
            candidate_seed = seed + int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(candidate_seed)
            for random_index in rng.choice(random_pool, size=random_count, replace=False).tolist():
                records.append(
                    {
                        "candidate_id": cid,
                        "control_id": str(pool.iloc[random_index]["component_id"]),
                        "control_kind": "random",
                        "component_type": component_type,
                        "module_index": int(candidate["module_index"]),
                        "component_index": int(pool.iloc[random_index]["component_index"]),
                        "matching_variables": [],
                        "candidate_matching_values": {},
                        "control_matching_values": {},
                        "standardized_euclidean_distance": None,
                        "seed": candidate_seed,
                    }
                )
    reuse = Counter(record["control_id"] for record in records)
    for record in records:
        record["reused_across_candidates"] = reuse[record["control_id"]] > 1
        record["control_reuse_count"] = reuse[record["control_id"]]
    return {
        "matching_normalization": "z-score over the full Stage A Discovery component pool",
        "distance": "standardized Euclidean",
        "tie_break": "component_index ascending",
        "missing_values_imputed": False,
        "dropped_variables": dropped_variables,
        "controls": records,
    }


def component_lookup(candidates_payload: dict[str, Any], controls_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {
        row["component_id"]: {
            "component_id": row["component_id"],
            "component_type": row["component_type"],
            "module_index": int(row["module_index"]),
            "component_index": int(row["component_index"]),
            "role": "candidate",
            "candidate_id": row["component_id"],
            "control_kind": "candidate",
        }
        for row in candidates_payload["candidates"]
    }
    for row in controls_payload["controls"]:
        key = f"{row['candidate_id']}::{row['control_kind']}::{row['control_id']}"
        output[key] = {
            "component_id": row["control_id"],
            "component_type": row["component_type"],
            "module_index": int(row["module_index"]),
            "component_index": int(row["component_index"]),
            "role": "control",
            "candidate_id": row["candidate_id"],
            "control_kind": row["control_kind"],
        }
    return output


def unique_components(lookup: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["component_id"]: row for row in lookup.values()}


class StageBCapture:
    def __init__(self, model: Any, components: dict[str, dict[str, Any]], capture_sources: bool) -> None:
        self.model = model
        self.components = components
        self.capture_sources = capture_sources
        self.handles: list[Any] = []
        self.head_layers = sorted({row["module_index"] for row in components.values() if row["component_type"] == "head"})
        self.neuron_layers = sorted({row["module_index"] for row in components.values() if row["component_type"] == "neuron"})
        self.all_layers = sorted(set(self.head_layers) | set(self.neuron_layers))
        self.reset()

    def reset(self) -> None:
        self.pre_o: dict[int, torch.Tensor] = {}
        self.z: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}
        self.h_out: dict[int, torch.Tensor] = {}

    @staticmethod
    def _last(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach()[:, -1, :].clone()

    def install(self) -> None:
        for module_index in self.head_layers:
            layer = self.model.model.layers[module_index]
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._pre_o_hook(module_index)))
            if self.capture_sources:
                self.handles.append(layer.self_attn.v_proj.register_forward_hook(self._value_hook(module_index)))
        for module_index in self.neuron_layers:
            layer = self.model.model.layers[module_index]
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._z_hook(module_index)))
        for module_index in self.all_layers:
            self.handles.append(self.model.model.layers[module_index].register_forward_hook(self._h_hook(module_index)))

    def _pre_o_hook(self, module_index: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            self.pre_o[module_index] = self._last(args[0])
        return hook

    def _z_hook(self, module_index: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            self.z[module_index] = self._last(args[0])
        return hook

    def _value_hook(self, module_index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            self.values[module_index] = output.detach().clone()
        return hook

    def _h_hook(self, module_index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.h_out[module_index] = self._last(tensor)
        return hook

    def validate(self) -> None:
        if set(self.pre_o) != set(self.head_layers):
            raise RuntimeError("Incomplete attention pre-o capture")
        if set(self.z) != set(self.neuron_layers):
            raise RuntimeError("Incomplete neuron activation capture")
        if set(self.h_out) != set(self.all_layers):
            raise RuntimeError("Incomplete hidden-state capture")
        if self.capture_sources and set(self.values) != set(self.head_layers):
            raise RuntimeError("Incomplete attention value capture")

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def projections_for_components(
    model: Any,
    directions: np.ndarray,
    components: dict[str, dict[str, Any]],
) -> dict[str, torch.Tensor]:
    output = {}
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    for component_id, row in components.items():
        layer = model.model.layers[row["module_index"]]
        direction = torch.as_tensor(directions[row["module_index"]], device=device, dtype=dtype)
        index = row["component_index"]
        if row["component_type"] == "neuron":
            output[component_id] = torch.dot(layer.mlp.down_proj.weight[:, index], direction)
        else:
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            block = layer.self_attn.o_proj.weight[:, index * head_dim : (index + 1) * head_dim]
            output[component_id] = torch.mv(block.T, direction)
    return output


class AtomicCsvSink:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        if path.suffix == ".gz":
            self.handle = gzip.open(self.temporary, "wt", encoding="utf-8", newline="")
        else:
            self.handle = self.temporary.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()

    def writerow(self, row: dict[str, Any]) -> None:
        self.writer.writerow(jsonable(row))

    def close(self, commit: bool = True) -> None:
        self.handle.close()
        if commit:
            os.replace(self.temporary, self.path)
        elif self.temporary.exists():
            self.temporary.unlink()


def validate_controlled_manifest(frame: pd.DataFrame, split: str, require_approved: bool) -> None:
    missing = set(CONTROLLED_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"Controlled manifest missing columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError("Controlled manifest is empty")
    if frame["pair_id"].astype(str).duplicated().any():
        raise RuntimeError("Controlled pair_id must be unique")
    if set(frame["split"].astype(str)) != {split}:
        raise RuntimeError(f"Controlled manifest must contain only split={split}")
    nonempty = [
        "pair_id", "hypothesis_id", "feature_family", "template_id", "template_family",
        "original_text", "modified_text", "changed_span_original", "changed_span_modified",
        "semantic_role", "changed_feature", "invariant_features", "generation_rule_id",
        "expected_answer_original", "expected_answer_modified",
    ]
    for column in nonempty:
        if frame[column].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"Every controlled row requires a non-empty {column}")
    if (frame["original_text"].astype(str) == frame["modified_text"].astype(str)).any():
        raise RuntimeError("Controlled original_text and modified_text must differ")
    for column in ("token_length_original", "token_length_modified"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or (numeric <= 0).any():
            raise RuntimeError(f"Every controlled row requires a positive integer {column}")
    if require_approved:
        approved = frame["approved"].astype(str).str.lower().isin({"true", "1"})
        validated = frame["human_validated"].astype(str).str.lower().isin({"true", "1"})
        reviewers = frame["reviewer_id"].astype(str).str.strip().ne("")
        if not bool((approved & validated & reviewers).all()):
            raise RuntimeError("Every controlled row must be human_validated=true, approved=true, and have reviewer_id")


def frozen_provenance(
    config: dict[str, Any],
    assets: dict[str, Any],
    feature_schema_path: Path,
    controls_path: Path | None = None,
    hypothesis_path: Path | None = None,
    controlled_path: Path | None = None,
    confirmatory_design_path: Path | None = None,
) -> dict[str, Any]:
    code_hash, code_files = code_checksum()
    parameter_hash, parameter_files = model_parameter_checksum(Path(config["model_path"]))
    values: dict[str, Any] = {
        "hash_algorithm": "SHA-256",
        "stage_b_run_id": config["stage_b_run_id"],
        "dataset_sha256": sha256_file(Path(config["dataset_path"])),
        "split_sha256": sha256_file(Path(config["split_path"])),
        "prompt_sha256": sha256_text(config["prompt_template"]),
        "model_config_sha256": sha256_file(Path(config["model_path"]) / "config.json"),
        "model_parameter_checksum": parameter_hash,
        "model_parameter_file_sha256": parameter_files,
        "stage_a_identity_hash": assets["identity"]["identity_hash"],
        "stage_a_config_hash": assets["identity"]["config_hash"],
        "discovery_direction_sha256": sha256_file(assets["direction_path"]),
        "candidate_manifest_sha256": sha256_file(assets["manifest_path"]),
        "head_validation_table_sha256": sha256_file(assets["root"] / "tables" / "head_candidate_validation.csv"),
        "neuron_validation_table_sha256": sha256_file(assets["root"] / "tables" / "neuron_candidate_validation.csv"),
        "feature_schema_sha256": sha256_file(feature_schema_path),
        "statistical_config_sha256": canonical_hash(config["statistics"]),
        "confirmatory_power_config_sha256": canonical_hash(config["confirmatory_power"]),
        "stage_b_code_hash": code_hash,
        "stage_b_code_files": code_files,
        "random_seeds": {"global": config["seed"], "statistics": config["statistics"]["random_seed"]},
        "model_training": False,
    }
    for key, path in (
        ("frozen_control_manifest_sha256", controls_path),
        ("hypothesis_manifest_sha256", hypothesis_path),
        ("controlled_pair_manifest_sha256", controlled_path),
        ("confirmatory_design_sha256", confirmatory_design_path),
    ):
        values[key] = sha256_file(path) if path is not None and path.exists() else None
    return values
