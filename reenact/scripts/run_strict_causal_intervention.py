#!/usr/bin/env python3
"""Run the pre-registered single-layer, single-prefill-last-token LiReF intervention.

This is a controlled extension of the public LiReF intervention code.  It does
not replace or modify the official implementation under ``liref/``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import gzip
import hashlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import torch
import transformers
from datasets import DatasetDict, load_from_disk
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

from compute_layerwise_liref import (
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    load_model_config,
    sha256_file,
)


SCRIPT_PATH = Path(__file__).resolve()
REENACT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_DATASET_DIR = REENACT_ROOT / "liref" / "dataset"
DEFAULT_MODEL_DIR = REENACT_ROOT / "liref_models"
DEFAULT_CACHE_DIR = REENACT_ROOT / "liref_outputs" / "hidden_states"
DEFAULT_LIREF_DIR = REENACT_ROOT / "liref_outputs" / "layerwise_liref"
DEFAULT_OUTPUT_DIR = REENACT_ROOT / "liref_outputs" / "strict_causal_intervention"
DEFAULT_PUBLIC_FEATURES = (
    REENACT_ROOT
    / "liref"
    / "reasoning_representation"
    / "Intervention"
    / "features_intervention.py"
)
DEFAULT_PUBLIC_UTILS = DEFAULT_PUBLIC_FEATURES.parent / "utils.py"
DEFAULT_EXTRACTION_NOTEBOOK = (
    REENACT_ROOT
    / "liref"
    / "reasoning_representation"
    / "LiReFs_storing_hs.ipynb"
)

EXPECTED_MODEL = "Meta-Llama-3-8B"
EXPECTED_TOTAL = 3000
EXPECTED_TRAIN = 2400
EXPECTED_HELDOUT = 600
EXPECTED_TRAIN_REASONING = 1103
EXPECTED_TRAIN_MEMORY = 1297
EXPECTED_HELDOUT_REASONING = 276
EXPECTED_HELDOUT_MEMORY = 324
EXPECTED_PRIMARY_CACHE_INDEX = 12
EXPECTED_LAYERS = (4, 12, 28)
EXPECTED_ALPHAS = (-0.10, -0.05, 0.0, 0.05, 0.10)
PRIMARY_ALPHA_MAGNITUDE = 0.10
PRIMARY_SOURCE = "mmlu2400_train"
SENSITIVITY_SOURCE = "mmlu3000_full"
PRIMARY_VECTOR_FILE = "liref_vectors_heldout.pt"
SENSITIVITY_VECTOR_FILE = "liref_vectors_in_sample.pt"
SCORE_THRESHOLD = 0.5
DEFAULT_ORTHOGONAL_CONTROLS = 3
DEFAULT_LABEL_SHUFFLED_CONTROLS = 1
EPS = 1e-12
LAYER_TIE_TOLERANCE = 1e-12
PARSER_PATTERN = re.compile(r"answer is \(?([ABCDEFGHIJ])\)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "prepare",
            "sanity",
            "pilot",
            "freeze",
            "full",
            "control",
            "sensitivity",
            "analyze",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--model", default=EXPECTED_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--liref-dir", type=Path, default=DEFAULT_LIREF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--layers", type=int, nargs="+", default=list(EXPECTED_LAYERS))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(EXPECTED_ALPHAS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--orthogonal-controls", type=int, default=DEFAULT_ORTHOGONAL_CONTROLS
    )
    parser.add_argument(
        "--label-shuffled-controls",
        type=int,
        default=DEFAULT_LABEL_SHUFFLED_CONTROLS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.skip_existing and args.overwrite:
        parser.error("--skip-existing and --overwrite are mutually exclusive")
    if args.model != EXPECTED_MODEL:
        parser.error(
            f"The pre-registered primary run is fixed to {EXPECTED_MODEL}; got {args.model}"
        )
    if tuple(args.layers) != EXPECTED_LAYERS:
        parser.error(f"Pre-registered layers must be {EXPECTED_LAYERS}")
    if len(args.alphas) != len(EXPECTED_ALPHAS) or not np.allclose(
        args.alphas, EXPECTED_ALPHAS, atol=0.0, rtol=0.0
    ):
        parser.error(f"Pre-registered alpha grid must be {EXPECTED_ALPHAS}")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch size and max new tokens must be positive")
    if args.bootstrap_replicates < 1:
        parser.error("bootstrap replicates must be positive")
    if args.orthogonal_controls != DEFAULT_ORTHOGONAL_CONTROLS:
        parser.error(
            f"Frozen design requires {DEFAULT_ORTHOGONAL_CONTROLS} orthogonal controls"
        )
    if args.label_shuffled_controls != DEFAULT_LABEL_SHUFFLED_CONTROLS:
        parser.error(
            f"Frozen design requires {DEFAULT_LABEL_SHUFFLED_CONTROLS} label-shuffled control"
        )
    return args


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "dataset_dir",
        "model_dir",
        "cache_dir",
        "liref_dir",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def atomic_write_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
    os.replace(temporary, path)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_info(path: Path, include_sha256: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }
    if include_sha256:
        result["sha256"] = sha256_file(resolved)
    return result


def alpha_slug(alpha: float) -> str:
    sign = "p" if alpha >= 0 else "m"
    return f"{sign}{abs(alpha):.3f}".replace(".", "p")


def canonical_sample_hash(frame: pd.DataFrame) -> str:
    payload = [
        {"dataset_row_index": int(row.dataset_row_index), "question_id": str(row.question_id)}
        for row in frame.itertuples(index=False)
    ]
    return stable_json_sha256(payload)


def form_options(options: list[str]) -> str:
    letters = list("ABCDEFGHIJ")
    value = "Options are:\n"
    for option, letter in zip(options, letters):
        value += f"({letter}): {option}\n"
    return value


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset_dir / "mmlu-pro-3000samples.json"
    split_path = args.liref_dir / "split_ids.json"
    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected 3000 records, got {len(records)}")
    if split["dataset_sha256"] != sha256_file(dataset_path):
        raise RuntimeError("Dataset SHA does not match split_ids.json")
    train_indices = np.asarray(split["train"]["row_indices"], dtype=np.int64)
    heldout_indices = np.asarray(split["heldout"]["row_indices"], dtype=np.int64)
    if len(train_indices) != EXPECTED_TRAIN or len(heldout_indices) != EXPECTED_HELDOUT:
        raise RuntimeError("Unexpected train/held-out split sizes")
    if set(train_indices) & set(heldout_indices):
        raise RuntimeError("Train and held-out row indices overlap")
    if set(train_indices) | set(heldout_indices) != set(range(EXPECTED_TOTAL)):
        raise RuntimeError("Train and held-out rows do not cover exactly 3000 records")
    question_ids = [str(record["question_id"]) for record in records]
    if len(set(question_ids)) != EXPECTED_TOTAL:
        raise RuntimeError("question_id is not unique")
    for split_name, indices in (("train", train_indices), ("heldout", heldout_indices)):
        expected_ids = [str(value) for value in split[split_name]["question_ids"]]
        observed_ids = [question_ids[index] for index in indices]
        if observed_ids != expected_ids:
            raise RuntimeError(f"question_id mapping mismatch in {split_name}")
    labels = np.asarray(
        [float(record["memory_reason_score"]) > SCORE_THRESHOLD for record in records],
        dtype=bool,
    )
    counts = {
        "train_reasoning": int(labels[train_indices].sum()),
        "train_memory": int(len(train_indices) - labels[train_indices].sum()),
        "heldout_reasoning": int(labels[heldout_indices].sum()),
        "heldout_memory": int(len(heldout_indices) - labels[heldout_indices].sum()),
    }
    expected_counts = {
        "train_reasoning": EXPECTED_TRAIN_REASONING,
        "train_memory": EXPECTED_TRAIN_MEMORY,
        "heldout_reasoning": EXPECTED_HELDOUT_REASONING,
        "heldout_memory": EXPECTED_HELDOUT_MEMORY,
    }
    if counts != expected_counts:
        raise RuntimeError(f"Unexpected group counts: {counts}")
    return {
        "records": records,
        "labels": labels,
        "train_indices": train_indices,
        "heldout_indices": heldout_indices,
        "dataset_path": dataset_path.resolve(),
        "split_path": split_path.resolve(),
        "counts": counts,
    }


def build_behavior_prompts(
    records: list[dict[str, Any]], dataset_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    mmlu_disk_path = dataset_dir / "mmlu-pro"
    dataset = load_from_disk(str(mmlu_disk_path))
    if not isinstance(dataset, DatasetDict) or "validation" not in dataset:
        raise RuntimeError("MMLU-Pro validation split is unavailable")
    categories = (
        "computer science",
        "math",
        "chemistry",
        "engineering",
        "law",
        "biology",
        "health",
        "physics",
        "business",
        "philosophy",
        "economics",
        "other",
        "psychology",
        "history",
    )
    prefixes = {category: "" for category in categories}
    validation_questions: set[str] = set()
    for entry in dataset["validation"]:
        category = str(entry["category"])
        if category not in prefixes:
            raise RuntimeError(f"Unexpected validation category: {category}")
        validation_questions.add(str(entry["question"]).strip())
        prefixes[category] += (
            "Q: "
            + str(entry["question"])
            + "\n"
            + form_options(list(entry["options"]))
            + "\n"
            + str(entry["cot_content"])
            + "\n\n"
        )
    counts = {
        category: sum(
            1 for entry in dataset["validation"] if str(entry["category"]) == category
        )
        for category in categories
    }
    if set(counts.values()) != {5}:
        raise RuntimeError(f"Expected exactly five demonstrations per category: {counts}")
    prompts = []
    for record in records:
        category = str(record["category"])
        prompt = (
            prefixes[category]
            + "Q: "
            + str(record["question"])
            + "\n"
            + form_options(list(record["options"]))
            + "\n\nA: Let's think step by step. "
        )
        prompts.append(prompt)
    heldout_overlap = sum(
        str(record["question"]).strip() in validation_questions for record in records
    )
    return prompts, {
        "dataset_path": str(mmlu_disk_path.resolve()),
        "dataset_fingerprint": getattr(dataset["validation"], "_fingerprint", None),
        "validation_samples": len(dataset["validation"]),
        "demonstrations_per_category": counts,
        "exact_question_overlap_with_3000": heldout_overlap,
        "template": (
            "category-specific 5-shot CoT prefix + Q: {question} + options + "
            "A: Let's think step by step. "
        ),
        "chat_template_used": False,
    }


def deterministic_sample_sets(
    inputs: dict[str, Any], seed: int
) -> dict[str, np.ndarray]:
    labels = inputs["labels"]
    train = inputs["train_indices"]
    train_reasoning = train[labels[train]]
    train_memory = train[~labels[train]]
    rng = np.random.default_rng(seed)
    pilot_reasoning = np.sort(rng.choice(train_reasoning, size=100, replace=False))
    pilot_memory = np.sort(rng.choice(train_memory, size=100, replace=False))
    pilot = np.concatenate((pilot_reasoning, pilot_memory))
    sanity = np.concatenate((pilot_reasoning[:2], pilot_memory[:2]))
    return {
        "sanity_train4": sanity,
        "pilot_train200": pilot,
        "locked_heldout600": inputs["heldout_indices"].copy(),
    }


def make_eval_samples(
    inputs: dict[str, Any], prompts: list[str], sample_sets: dict[str, np.ndarray]
) -> pd.DataFrame:
    membership = {
        name: set(int(index) for index in indices) for name, indices in sample_sets.items()
    }
    train_set = set(int(index) for index in inputs["train_indices"])
    rows = []
    for index, record in enumerate(inputs["records"]):
        rows.append(
            {
                "dataset_row_index": index,
                "question_id": str(record["question_id"]),
                "split": "train" if index in train_set else "locked_heldout",
                "group": (
                    "Reasoning"
                    if float(record["memory_reason_score"]) > SCORE_THRESHOLD
                    else "Memory"
                ),
                "memory_reason_score": float(record["memory_reason_score"]),
                "category": str(record["category"]),
                "gold_answer": str(record["answer"]),
                "question": str(record["question"]),
                "options_json": json.dumps(record["options"], ensure_ascii=False),
                "behavior_prompt_sha256": text_sha256(prompts[index]),
                "in_sanity_train4": index in membership["sanity_train4"],
                "in_pilot_train200": index in membership["pilot_train200"],
                "in_locked_heldout600": index in membership["locked_heldout600"],
            }
        )
    return pd.DataFrame(rows)


def load_vector_payloads(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    model_root = args.liref_dir / args.model
    specifications = {
        PRIMARY_SOURCE: PRIMARY_VECTOR_FILE,
        SENSITIVITY_SOURCE: SENSITIVITY_VECTOR_FILE,
    }
    payloads = {}
    for source, filename in specifications.items():
        path = model_root / filename
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        indices = [int(value) for value in payload["cache_indices"].tolist()]
        if indices != list(range(1, int(config["num_hidden_layers"]))):
            raise RuntimeError(f"Unexpected LiReF cache indices in {path}")
        expected_analysis = "heldout" if source == PRIMARY_SOURCE else "in_sample"
        if payload["metadata"]["analysis_type"] != expected_analysis:
            raise RuntimeError(f"LiReF source mapping mismatch: {source}, {path}")
        expected_direction_counts = (
            (EXPECTED_TRAIN_REASONING, EXPECTED_TRAIN_MEMORY)
            if source == PRIMARY_SOURCE
            else (1379, 1621)
        )
        observed_direction_counts = (
            int(payload["metadata"]["direction_reasoning_count"]),
            int(payload["metadata"]["direction_memory_count"]),
        )
        if observed_direction_counts != expected_direction_counts:
            raise RuntimeError(
                f"Direction construction count mismatch: {source}, {observed_direction_counts}"
            )
        if tuple(payload["raw_liref"].shape) != (
            len(indices),
            int(config["hidden_size"]),
        ):
            raise RuntimeError(f"LiReF vector shape mismatch: {path}")
        if payload["raw_liref"].dtype != torch.float64:
            raise RuntimeError(f"LiReF vectors must be float64: {path}")
        payloads[source] = {"path": path.resolve(), "payload": payload, "indices": indices}
    return payloads


def vector_at(payloads: dict[str, Any], source: str, cache_index: int) -> torch.Tensor:
    entry = payloads[source]
    try:
        position = entry["indices"].index(cache_index)
    except ValueError as exc:
        raise KeyError(f"LiReF has no cache index {cache_index}") from exc
    vector = entry["payload"]["raw_liref"][position].detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(vector).all()) or float(vector.norm()) <= EPS:
        raise RuntimeError(f"Invalid LiReF vector at {source}/cache {cache_index}")
    return vector


def load_mmlu_cache(args: argparse.Namespace, config: dict[str, Any]) -> tuple[dict[int, torch.Tensor], Path]:
    partial = args.cache_dir / ".partial" / args.model / "mmlu-pro_3000samples.pt"
    if not partial.is_file():
        raise FileNotFoundError(f"Required MMLU partial cache not found: {partial}")
    cache = torch.load(partial, map_location="cpu", mmap=True, weights_only=True)
    if sorted(cache) != list(range(int(config["num_hidden_layers"]))):
        raise RuntimeError("MMLU activation cache index mismatch")
    expected_shape = (EXPECTED_TOTAL, int(config["hidden_size"]))
    for cache_index, tensor in cache.items():
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.float32:
            raise RuntimeError(
                f"MMLU cache mismatch at {cache_index}: {tensor.shape}, {tensor.dtype}"
            )
    return cache, partial.resolve()


def compute_train_only_layer_selection(
    args: argparse.Namespace,
    config: dict[str, Any],
    inputs: dict[str, Any],
    payloads: dict[str, Any],
) -> tuple[pd.DataFrame, dict[int, torch.Tensor], Path]:
    cache, cache_path = load_mmlu_cache(args, config)
    train = torch.from_numpy(inputs["train_indices"].astype(np.int64, copy=False))
    train_labels = inputs["labels"][inputs["train_indices"]]
    reasoning = torch.from_numpy(train_labels)
    memory = torch.from_numpy(~train_labels)
    rows = []
    for cache_index in range(1, int(config["num_hidden_layers"])):
        hidden = cache[cache_index].index_select(0, train).to(torch.float64)
        direction = vector_at(payloads, PRIMARY_SOURCE, cache_index)
        unit = direction / direction.norm()
        projections = hidden @ unit
        norms = hidden.norm(dim=1)
        cosine = projections / norms
        cosine_gap = float(cosine[reasoning].mean() - cosine[memory].mean())
        rows.append(
            {
                "model": args.model,
                "direction_source": PRIMARY_SOURCE,
                "selection_split": "train2400",
                "cache_index": cache_index,
                "target_module_index": cache_index - 1,
                "relative_layer_depth": cache_index / int(config["num_hidden_layers"]),
                "cosine_gap": cosine_gap,
                "raw_liref_norm": float(direction.norm()),
                "hidden_norm_mean": float(norms.mean()),
                "hidden_norm_median": float(torch.quantile(norms, 0.5)),
                "alpha_0p1_injected_norm": float(PRIMARY_ALPHA_MAGNITUDE * direction.norm()),
                "alpha_0p1_relative_to_hidden_median": float(
                    PRIMARY_ALPHA_MAGNITUDE * direction.norm() / torch.quantile(norms, 0.5)
                ),
                "n_reasoning": int(reasoning.sum()),
                "n_memory": int(memory.sum()),
            }
        )
        del hidden, projections, norms, cosine
    frame = pd.DataFrame(rows)
    target = float(frame["cosine_gap"].max())
    selected = frame[
        np.abs(frame["cosine_gap"] - target) <= LAYER_TIE_TOLERANCE
    ].sort_values("cache_index").iloc[0]
    if int(selected["cache_index"]) != EXPECTED_PRIMARY_CACHE_INDEX:
        raise RuntimeError(
            f"Train-only layer selection changed: {int(selected['cache_index'])}"
        )
    return frame, cache, cache_path


def make_control_vectors(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    payloads: dict[str, Any],
    cache: dict[int, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    primary = vector_at(payloads, PRIMARY_SOURCE, EXPECTED_PRIMARY_CACHE_INDEX)
    norm = primary.norm()
    unit = primary / norm
    controls: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for control_index in range(args.orthogonal_controls):
        seed = 42001 + control_index
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_vector = torch.randn(
            len(primary), generator=generator, dtype=torch.float64
        )
        orthogonal = random_vector - torch.dot(random_vector, unit) * unit
        orthogonal = orthogonal / orthogonal.norm() * norm
        control_id = f"orthogonal_{control_index + 1:02d}"
        controls[control_id] = orthogonal
        rows.append(
            {
                "control_id": control_id,
                "control_type": "gaussian_orthogonal_matched_norm",
                "seed": seed,
                "cache_index": EXPECTED_PRIMARY_CACHE_INDEX,
                "norm": float(orthogonal.norm()),
                "target_liref_norm": float(norm),
                "dot_with_liref": float(torch.dot(orthogonal, primary)),
                "cosine_with_liref": float(
                    torch.dot(orthogonal, primary) / (orthogonal.norm() * norm)
                ),
                "vector_sha256": tensor_sha256(orthogonal),
            }
        )
    train_indices = inputs["train_indices"]
    train_labels = inputs["labels"][train_indices].copy()
    hidden = cache[EXPECTED_PRIMARY_CACHE_INDEX].index_select(
        0, torch.from_numpy(train_indices.astype(np.int64, copy=False))
    ).to(torch.float64)
    for control_index in range(args.label_shuffled_controls):
        seed = 42011 + control_index
        shuffled = np.random.default_rng(seed).permutation(train_labels)
        shuffled_tensor = torch.from_numpy(shuffled)
        raw = hidden[shuffled_tensor].mean(dim=0) - hidden[~shuffled_tensor].mean(dim=0)
        if float(raw.norm()) <= EPS:
            raise RuntimeError("Label-shuffled direction has zero norm")
        matched = raw / raw.norm() * norm
        control_id = f"label_shuffled_{control_index + 1:02d}"
        controls[control_id] = matched
        rows.append(
            {
                "control_id": control_id,
                "control_type": "train_label_shuffled_mean_difference_matched_norm",
                "seed": seed,
                "cache_index": EXPECTED_PRIMARY_CACHE_INDEX,
                "norm": float(matched.norm()),
                "target_liref_norm": float(norm),
                "dot_with_liref": float(torch.dot(matched, primary)),
                "cosine_with_liref": float(
                    torch.dot(matched, primary) / (matched.norm() * norm)
                ),
                "vector_sha256": tensor_sha256(matched),
            }
        )
    del hidden
    return controls, pd.DataFrame(rows)


def build_condition_plan(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        phase: str,
        sample_set: str,
        source: str,
        direction_type: str,
        control_id: str,
        cache_index: int | None,
        alpha: float,
        hook_installed: bool,
    ) -> None:
        layer_slug = "none" if cache_index is None else f"cache{cache_index:03d}"
        condition_id = "__".join(
            (
                phase,
                source,
                direction_type,
                control_id or "none",
                layer_slug,
                f"alpha_{alpha_slug(alpha)}",
            )
        )
        rows.append(
            {
                "condition_id": condition_id,
                "phase": phase,
                "sample_set": sample_set,
                "direction_source": source,
                "intervention_direction": direction_type,
                "control_id": control_id,
                "target_cache_index": cache_index,
                "target_module_index": (
                    None if cache_index is None else cache_index - 1
                ),
                "alpha": alpha,
                "hook_installed": hook_installed,
            }
        )

    for phase, sample_set in (
        ("sanity", "sanity_train4"),
        ("pilot", "pilot_train200"),
        ("full", "locked_heldout600"),
    ):
        add(phase, sample_set, "none", "none", "", None, 0.0, False)
        layers = (EXPECTED_PRIMARY_CACHE_INDEX,) if phase == "sanity" else EXPECTED_LAYERS
        alphas = (-0.10, 0.0, 0.10) if phase == "sanity" else EXPECTED_ALPHAS
        for cache_index in layers:
            for alpha in alphas:
                add(
                    phase,
                    sample_set,
                    PRIMARY_SOURCE,
                    "liref",
                    "",
                    cache_index,
                    alpha,
                    True,
                )
    for control_index in range(args.orthogonal_controls):
        control_id = f"orthogonal_{control_index + 1:02d}"
        for alpha in (-PRIMARY_ALPHA_MAGNITUDE, PRIMARY_ALPHA_MAGNITUDE):
            add(
                "control",
                "locked_heldout600",
                PRIMARY_SOURCE,
                "orthogonal",
                control_id,
                EXPECTED_PRIMARY_CACHE_INDEX,
                alpha,
                True,
            )
    for control_index in range(args.label_shuffled_controls):
        control_id = f"label_shuffled_{control_index + 1:02d}"
        for alpha in (-PRIMARY_ALPHA_MAGNITUDE, PRIMARY_ALPHA_MAGNITUDE):
            add(
                "control",
                "locked_heldout600",
                PRIMARY_SOURCE,
                "label_shuffled",
                control_id,
                EXPECTED_PRIMARY_CACHE_INDEX,
                alpha,
                True,
            )
    for alpha in (-PRIMARY_ALPHA_MAGNITUDE, PRIMARY_ALPHA_MAGNITUDE):
        add(
            "sensitivity",
            "locked_heldout600",
            SENSITIVITY_SOURCE,
            "liref",
            "",
            EXPECTED_PRIMARY_CACHE_INDEX,
            alpha,
            True,
        )
    plan = pd.DataFrame(rows)
    if plan["condition_id"].duplicated().any():
        raise RuntimeError("Condition identifiers are not unique")
    return plan


def prepare_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_model_config(args.model_dir, args.model)
    inputs = load_inputs(args)
    prompts, prompt_metadata = build_behavior_prompts(inputs["records"], args.dataset_dir)
    sample_sets = deterministic_sample_sets(inputs, args.seed)
    eval_samples = make_eval_samples(inputs, prompts, sample_sets)
    payloads = load_vector_payloads(args, config)
    layer_selection, cache, cache_path = compute_train_only_layer_selection(
        args, config, inputs, payloads
    )
    controls, control_metadata = make_control_vectors(args, inputs, payloads, cache)
    condition_plan = build_condition_plan(args)
    primary_peak = int(
        layer_selection.sort_values(
            ["cosine_gap", "cache_index"], ascending=[False, True]
        ).iloc[0]["cache_index"]
    )
    if primary_peak != EXPECTED_PRIMARY_CACHE_INDEX:
        raise RuntimeError("Train-only selected layer is not the pre-registered layer 12")

    atomic_write_csv(args.output_dir / "eval_samples.csv", eval_samples)
    atomic_write_csv(
        args.output_dir / "layer_selection_train2400.csv", layer_selection
    )
    atomic_write_csv(args.output_dir / "control_vector_metadata.csv", control_metadata)
    atomic_torch_save(
        args.output_dir / "control_vectors.pt",
        {
            "vectors": controls,
            "metadata": control_metadata.to_dict(orient="records"),
        },
    )
    atomic_write_csv(args.output_dir / "condition_plan.csv", condition_plan)
    config_path = args.output_dir / "experiment_config.json"
    created_at = datetime.now().astimezone().isoformat()
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            previous_config = json.load(handle)
        created_at = str(previous_config.get("created_at", created_at))
    experiment_config = {
        "experiment_name": "single-layer, single-prefill-last-token raw-LiReF causal intervention",
        "classification": "controlled causal intervention extension; not an exact reproduction of the paper intervention scope",
        "model": args.model,
        "dataset": file_info(inputs["dataset_path"]),
        "split_ids": file_info(inputs["split_path"]),
        "locked_set_disclosure": (
            "The 600 rows were previously used in representation/baseline analyses; "
            "they are locked against tuning for this strict intervention, not historically unseen."
        ),
        "split_counts": inputs["counts"],
        "sample_sets": {
            name: {
                "rows": len(indices),
                "sha256": canonical_sample_hash(
                    eval_samples.iloc[indices][["dataset_row_index", "question_id"]]
                ),
            }
            for name, indices in sample_sets.items()
        },
        "primary_direction": {
            "source": PRIMARY_SOURCE,
            "vector_file": file_info(payloads[PRIMARY_SOURCE]["path"]),
            "construction": "train 2400: Reasoning mean - Memory mean",
        },
        "leakage_sensitivity_direction": {
            "source": SENSITIVITY_SOURCE,
            "vector_file": file_info(payloads[SENSITIVITY_SOURCE]["path"]),
            "construction": "full 3000: Reasoning mean - Memory mean",
        },
        "activation_cache": file_info(cache_path, include_sha256=False),
        "train_only_selected_cache_index": primary_peak,
        "cache_module_mapping": "cache_index k is output of one-based block k; module index is k-1",
        "layers": list(EXPECTED_LAYERS),
        "alpha_grid": list(EXPECTED_ALPHAS),
        "primary_alpha_magnitude": PRIMARY_ALPHA_MAGNITUDE,
        "primary_estimands": [
            "Reasoning: accuracy(+0.10) - accuracy(-0.10) at cache 12",
            "Memory: accuracy(-0.10) - accuracy(+0.10) at cache 12",
            "Reasoning: accuracy(+0.10) - no-hook baseline at cache 12",
            "Memory: accuracy(-0.10) - no-hook baseline at cache 12",
        ],
        "secondary_estimands": "layers 4/28 and |alpha|=0.05 dose-response",
        "orthogonal_controls": args.orthogonal_controls,
        "orthogonal_control_seeds": list(range(42001, 42001 + args.orthogonal_controls)),
        "label_shuffled_controls": args.label_shuffled_controls,
        "label_shuffled_control_seeds": list(
            range(42011, 42011 + args.label_shuffled_controls)
        ),
        "prompt": {
            "hidden_state_extraction": "Q: {question}\\nA: ",
            "behavioral_evaluation": prompt_metadata,
            "mismatch_disclosure": (
                "LiReF was extracted from a question-only no-CoT prompt without options, "
                "whereas behavior is evaluated with category-specific 5-shot CoT prompts "
                "and options; this is cross-prompt transfer, not intervention on the exact "
                "same prompt representation."
            ),
        },
        "parser": {
            "regex": PARSER_PATTERN.pattern,
            "case_sensitive": True,
            "random_fallback": False,
            "failure_policy": "parsed_answer=None, parse_ok=False, correct=False",
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "padding_side": "left",
            "temperature": None,
            "top_p": None,
            "chat_template": False,
        },
        "hook": {
            "module": "model.model.layers[target_cache_index - 1]",
            "location": "transformer block output residual stream",
            "token_scope": "last token of the first prefill call only",
            "decode_scope": "no intervention during generated-token decoding",
            "equation": "h' = h + alpha * raw_liref",
        },
        "statistics": {
            "unit": "question-level paired outcomes",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.seed,
            "bootstrap_ci": "percentile 95%",
            "mcnemar": "exact binomial test on discordant pairs",
            "fdr": "Benjamini-Hochberg within phase x comparison_family x group",
        },
        "condition_plan_sha256": stable_json_sha256(
            condition_plan.to_dict(orient="records")
        ),
        "public_code": {
            "features": file_info(DEFAULT_PUBLIC_FEATURES),
            "utils": file_info(DEFAULT_PUBLIC_UTILS),
            "difference": (
                "Public code hooks every layer plus attention/MLP and repeatedly applies "
                "projection amplification; this experiment adds raw LiReF once to one "
                "block output and one prefill token."
            ),
        },
        "extraction_notebook": file_info(DEFAULT_EXTRACTION_NOTEBOOK),
        "seed": args.seed,
        "created_at": created_at,
    }
    atomic_write_json(config_path, experiment_config)
    return {
        "config": config,
        "inputs": inputs,
        "prompts": prompts,
        "sample_sets": sample_sets,
        "eval_samples": eval_samples,
        "payloads": payloads,
        "controls": controls,
        "control_metadata": control_metadata,
        "condition_plan": condition_plan,
        "experiment_config": experiment_config,
    }


def parse_answer(text: str) -> str | None:
    """Parse the public evaluator's explicit answer phrase without guessing."""
    match = PARSER_PATTERN.search(text)
    return None if match is None else match.group(1)


def load_generation_model(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, Any, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen intervention run")
    if args.device_id < 0 or args.device_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {args.device_id} is unavailable; count={torch.cuda.device_count()}"
        )
    device = torch.device(f"cuda:{args.device_id}")
    model_path = args.model_dir / args.model
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    )
    if "llama" in str(model.config.model_type).lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad_token_id")
    tokenizer.padding_side = "left"
    model.eval().to(device)
    return model, tokenizer, device


@dataclass
class HookDiagnostics:
    hook_call_count: int = 0
    prefill_apply_events: int = 0
    applied_sample_count: int = 0
    before_last_norms: list[float] | None = None
    delta_last_norms: list[float] | None = None
    relative_injection_norms: list[float] | None = None
    delta_cosines: list[float] | None = None
    delta_expected_max_abs_errors: list[float] | None = None
    other_token_max_abs_diff: float = 0.0

    def __post_init__(self) -> None:
        self.before_last_norms = []
        self.delta_last_norms = []
        self.relative_injection_norms = []
        self.delta_cosines = []
        self.delta_expected_max_abs_errors = []


class SinglePrefillLastTokenHook:
    """Apply raw LiReF once to one block output at the last prompt token."""

    def __init__(
        self,
        module: torch.nn.Module,
        direction: torch.Tensor,
        alpha: float,
    ) -> None:
        self.direction = direction
        self.alpha = float(alpha)
        self.applied = False
        self.diagnostics = HookDiagnostics()
        self.handle = module.register_forward_hook(self._hook)

    def _hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        del module, inputs
        self.diagnostics.hook_call_count += 1
        if self.applied:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3:
            raise RuntimeError(f"Unexpected block output shape: {tuple(hidden.shape)}")
        direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        if direction.ndim != 1 or direction.shape[0] != hidden.shape[-1]:
            raise RuntimeError("Intervention direction does not match hidden size")
        changed = hidden.clone()
        before = hidden[:, -1, :]
        expected_delta = self.alpha * direction
        changed[:, -1, :] = before + expected_delta
        delta = changed[:, -1, :] - before
        before_norm = before.float().norm(dim=-1)
        delta_norm = delta.float().norm(dim=-1)
        expected_norm = expected_delta.float().norm().clamp_min(EPS)
        expected_batch = expected_delta.float().unsqueeze(0).expand_as(delta.float())
        cosine = torch.nn.functional.cosine_similarity(
            delta.float(), expected_batch, dim=-1, eps=EPS
        )
        max_error = (delta.float() - expected_batch).abs().amax(dim=-1)
        if hidden.shape[1] > 1:
            other_diff = float((changed[:, :-1, :] - hidden[:, :-1, :]).abs().max())
        else:
            other_diff = 0.0
        self.diagnostics.before_last_norms.extend(before_norm.detach().cpu().tolist())
        self.diagnostics.delta_last_norms.extend(delta_norm.detach().cpu().tolist())
        self.diagnostics.relative_injection_norms.extend(
            (delta_norm / before_norm.clamp_min(EPS)).detach().cpu().tolist()
        )
        if float(expected_norm) > EPS:
            self.diagnostics.delta_cosines.extend(cosine.detach().cpu().tolist())
        else:
            self.diagnostics.delta_cosines.extend([1.0] * hidden.shape[0])
        self.diagnostics.delta_expected_max_abs_errors.extend(
            max_error.detach().cpu().tolist()
        )
        self.diagnostics.other_token_max_abs_diff = max(
            self.diagnostics.other_token_max_abs_diff, other_diff
        )
        self.diagnostics.prefill_apply_events += 1
        self.diagnostics.applied_sample_count += int(hidden.shape[0])
        self.applied = True
        if isinstance(output, tuple):
            return (changed,) + output[1:]
        return changed

    def remove(self) -> None:
        self.handle.remove()


def model_block(model: torch.nn.Module, cache_index: int) -> torch.nn.Module:
    module_index = cache_index - 1
    try:
        return model.model.layers[module_index]
    except (AttributeError, IndexError) as exc:
        raise RuntimeError(
            f"Cannot map cache index {cache_index} to model.model.layers[{module_index}]"
        ) from exc


def condition_vector(
    row: pd.Series,
    prepared: dict[str, Any],
) -> torch.Tensor | None:
    if not bool(row["hook_installed"]):
        return None
    if str(row["intervention_direction"]) == "liref":
        return vector_at(
            prepared["payloads"],
            str(row["direction_source"]),
            int(row["target_cache_index"]),
        )
    control_id = str(row["control_id"])
    if control_id not in prepared["controls"]:
        raise RuntimeError(f"Unknown control vector: {control_id}")
    return prepared["controls"][control_id]


def generate_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    direction: torch.Tensor | None,
    cache_index: int | None,
    alpha: float,
    max_new_tokens: int,
) -> tuple[list[str], list[dict[str, Any]], HookDiagnostics]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    if not bool((encoded["attention_mask"][:, -1] == 1).all()):
        raise RuntimeError("Last padded position is not the final prompt token")
    model_limit = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    prompt_tokens = int(encoded["input_ids"].shape[1])
    if model_limit and prompt_tokens + max_new_tokens > model_limit:
        raise RuntimeError(
            f"Prompt plus generation exceeds context: {prompt_tokens}+{max_new_tokens}>{model_limit}"
        )
    inputs = {key: value.to(device) for key, value in encoded.items()}
    hook: SinglePrefillLastTokenHook | None = None
    diagnostics = HookDiagnostics()
    try:
        if direction is not None:
            if cache_index is None:
                raise RuntimeError("Hooked condition is missing cache index")
            hook = SinglePrefillLastTokenHook(
                model_block(model, cache_index), direction, alpha
            )
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        last_ids = inputs["input_ids"][:, -1].detach().cpu().tolist()
        metadata = []
        for index, prompt in enumerate(prompts):
            unpadded_count = int(inputs["attention_mask"][index].sum())
            metadata.append(
                {
                    "behavior_prompt_sha256_runtime": text_sha256(prompt),
                    "input_token_count": unpadded_count,
                    "last_prompt_token_id": int(last_ids[index]),
                    "last_prompt_token_text": tokenizer.decode([int(last_ids[index])]),
                    "generated_token_count": int(
                        (new_tokens[index] != tokenizer.pad_token_id).sum().item()
                    ),
                }
            )
        diagnostics = hook.diagnostics if hook is not None else diagnostics
        return responses, metadata, diagnostics
    finally:
        if hook is not None:
            hook.remove()
        del inputs, encoded


def phase_rows(prepared: dict[str, Any], phase: str) -> pd.DataFrame:
    frame = prepared["condition_plan"]
    selected = frame[frame["phase"] == phase].copy()
    if selected.empty:
        raise RuntimeError(f"No conditions registered for phase {phase}")
    return selected.reset_index(drop=True)


def sample_indices_for_condition(
    condition: pd.Series, prepared: dict[str, Any]
) -> np.ndarray:
    sample_set = str(condition["sample_set"])
    if sample_set not in prepared["sample_sets"]:
        raise RuntimeError(f"Unknown sample set: {sample_set}")
    return prepared["sample_sets"][sample_set]


def condition_paths(args: argparse.Namespace, phase: str, condition_id: str) -> tuple[Path, Path]:
    root = args.output_dir / ".partial" / phase
    return root / f"{condition_id}.csv.gz", root / f"{condition_id}.manifest.json"


def condition_identity(
    args: argparse.Namespace,
    condition: pd.Series,
    indices: np.ndarray,
    prepared: dict[str, Any],
    vector: torch.Tensor | None,
) -> dict[str, Any]:
    sample_frame = prepared["eval_samples"].iloc[indices][
        ["dataset_row_index", "question_id"]
    ]
    identity = {
        "condition": {
            key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for key, value in condition.to_dict().items()
        },
        "sample_set_sha256": canonical_sample_hash(sample_frame),
        "sample_count": int(len(indices)),
        "dataset_sha256": prepared["experiment_config"]["dataset"]["sha256"],
        "script_sha256": sha256_file(SCRIPT_PATH),
        "experiment_config_sha256": sha256_file(args.output_dir / "experiment_config.json"),
        "vector_sha256": None if vector is None else tensor_sha256(vector),
        "generation": prepared["experiment_config"]["generation"],
        "parser": prepared["experiment_config"]["parser"],
    }
    identity["identity_sha256"] = stable_json_sha256(identity)
    return identity


def validate_saved_condition(
    data_path: Path,
    manifest_path: Path,
    identity: dict[str, Any],
) -> pd.DataFrame:
    if not data_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Incomplete existing condition output: {data_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("identity_sha256") != identity["identity_sha256"]:
        raise RuntimeError(f"Existing condition identity mismatch: {data_path}")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Existing condition is not complete: {data_path}")
    frame = pd.read_csv(data_path, compression="gzip")
    if len(frame) != identity["sample_count"]:
        raise RuntimeError(f"Existing condition row count mismatch: {data_path}")
    if sha256_file(data_path) != manifest.get("data_sha256"):
        raise RuntimeError(f"Existing condition data hash mismatch: {data_path}")
    return frame


def run_condition(
    args: argparse.Namespace,
    condition: pd.Series,
    prepared: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> pd.DataFrame:
    phase = str(condition["phase"])
    condition_id = str(condition["condition_id"])
    indices = sample_indices_for_condition(condition, prepared)
    vector = condition_vector(condition, prepared)
    identity = condition_identity(args, condition, indices, prepared, vector)
    data_path, manifest_path = condition_paths(args, phase, condition_id)
    if data_path.exists() or manifest_path.exists():
        if args.overwrite:
            pass
        elif args.skip_existing:
            print(f"[SKIP validated] {condition_id}", flush=True)
            return validate_saved_condition(data_path, manifest_path, identity)
        else:
            raise FileExistsError(
                f"Condition output exists; use --skip-existing after validation: {data_path}"
            )

    records = prepared["inputs"]["records"]
    prompts = prepared["prompts"]
    result_rows: list[dict[str, Any]] = []
    print(f"[START {phase}] {condition_id}: n={len(indices)}", flush=True)
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        batch_prompts = [prompts[int(index)] for index in batch_indices]
        responses, runtime_metadata, diagnostics = generate_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=batch_prompts,
            direction=vector,
            cache_index=(
                None
                if pd.isna(condition["target_cache_index"])
                else int(condition["target_cache_index"])
            ),
            alpha=float(condition["alpha"]),
            max_new_tokens=args.max_new_tokens,
        )
        if vector is not None:
            if diagnostics.prefill_apply_events != 1:
                raise RuntimeError("Hook did not apply exactly once during the batch prefill")
            if diagnostics.applied_sample_count != len(batch_indices):
                raise RuntimeError("Hook sample count mismatch")
            if diagnostics.other_token_max_abs_diff != 0.0:
                raise RuntimeError("Hook altered a token other than the last prompt token")
        for offset, (index, response, runtime) in enumerate(
            zip(batch_indices, responses, runtime_metadata)
        ):
            record = records[int(index)]
            parsed = parse_answer(response)
            result_rows.append(
                {
                    "condition_id": condition_id,
                    "phase": phase,
                    "sample_set": str(condition["sample_set"]),
                    "dataset_row_index": int(index),
                    "question_id": str(record["question_id"]),
                    "group": (
                        "Reasoning"
                        if float(record["memory_reason_score"]) > SCORE_THRESHOLD
                        else "Memory"
                    ),
                    "category": str(record["category"]),
                    "memory_reason_score": float(record["memory_reason_score"]),
                    "gold_answer": str(record["answer"]),
                    "parsed_answer": parsed,
                    "parse_ok": parsed is not None,
                    "correct": parsed == str(record["answer"]),
                    "raw_response": response,
                    "direction_source": str(condition["direction_source"]),
                    "intervention_direction": str(condition["intervention_direction"]),
                    "control_id": str(condition["control_id"]),
                    "target_cache_index": (
                        None
                        if pd.isna(condition["target_cache_index"])
                        else int(condition["target_cache_index"])
                    ),
                    "target_module_index": (
                        None
                        if pd.isna(condition["target_module_index"])
                        else int(condition["target_module_index"])
                    ),
                    "alpha": float(condition["alpha"]),
                    "hook_installed": bool(condition["hook_installed"]),
                    "hook_call_count_batch": diagnostics.hook_call_count,
                    "hook_prefill_apply_events_batch": diagnostics.prefill_apply_events,
                    "hook_applied_sample_count_batch": diagnostics.applied_sample_count,
                    "hidden_norm_before": (
                        None
                        if vector is None
                        else diagnostics.before_last_norms[offset]
                    ),
                    "injection_norm": (
                        0.0 if vector is None else diagnostics.delta_last_norms[offset]
                    ),
                    "relative_injection_norm": (
                        0.0
                        if vector is None
                        else diagnostics.relative_injection_norms[offset]
                    ),
                    "delta_expected_cosine": (
                        None if vector is None else diagnostics.delta_cosines[offset]
                    ),
                    "delta_expected_max_abs_error": (
                        None
                        if vector is None
                        else diagnostics.delta_expected_max_abs_errors[offset]
                    ),
                    "other_token_max_abs_diff_batch": diagnostics.other_token_max_abs_diff,
                    "vector_sha256": None if vector is None else tensor_sha256(vector),
                    **runtime,
                }
            )
        completed = min(start + len(batch_indices), len(indices))
        print(f"  {completed}/{len(indices)}", flush=True)
    result = pd.DataFrame(result_rows)
    if len(result) != len(indices) or result["question_id"].duplicated().any():
        raise RuntimeError(f"Condition integrity failure: {condition_id}")
    atomic_write_csv_gz(data_path, result)
    manifest = {
        **identity,
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "rows": len(result),
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "question_ids_sha256": stable_json_sha256(result["question_id"].tolist()),
        "parse_failures": int((~result["parse_ok"]).sum()),
    }
    atomic_write_json(manifest_path, manifest)
    print(f"[DONE {phase}] {condition_id}", flush=True)
    return result


def run_activation_sanity(
    args: argparse.Namespace,
    prepared: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, Any]:
    direction = vector_at(
        prepared["payloads"], PRIMARY_SOURCE, EXPECTED_PRIMARY_CACHE_INDEX
    )
    indices = prepared["sample_sets"]["sanity_train4"]
    cases: list[dict[str, Any]] = []
    for index in indices:
        prompt = prepared["prompts"][int(index)]
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        model_limit = int(getattr(model.config, "max_position_embeddings", 0) or 0)
        if model_limit and int(encoded["input_ids"].shape[1]) > model_limit:
            raise RuntimeError("Sanity prompt exceeds model context")
        inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            baseline = model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        for alpha in (-PRIMARY_ALPHA_MAGNITUDE, 0.0, PRIMARY_ALPHA_MAGNITUDE):
            hook = SinglePrefillLastTokenHook(
                model_block(model, EXPECTED_PRIMARY_CACHE_INDEX), direction, alpha
            )
            try:
                with torch.inference_mode():
                    intervened = model(
                        **inputs,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
            finally:
                hook.remove()
            previous_equal = torch.equal(
                baseline.hidden_states[EXPECTED_PRIMARY_CACHE_INDEX - 1],
                intervened.hidden_states[EXPECTED_PRIMARY_CACHE_INDEX - 1],
            )
            target_baseline = baseline.hidden_states[EXPECTED_PRIMARY_CACHE_INDEX]
            target_changed = intervened.hidden_states[EXPECTED_PRIMARY_CACHE_INDEX]
            observed_delta = (
                target_changed[:, -1, :] - target_baseline[:, -1, :]
            ).float()
            expected_delta = (
                alpha * direction.to(device=device, dtype=target_baseline.dtype)
            ).float().unsqueeze(0)
            other_equal = torch.equal(
                target_baseline[:, :-1, :], target_changed[:, :-1, :]
            )
            max_error = float((observed_delta - expected_delta).abs().max())
            expected_norm = float(expected_delta.norm())
            cosine = (
                1.0
                if expected_norm <= EPS
                else float(
                    torch.nn.functional.cosine_similarity(
                        observed_delta, expected_delta, dim=-1, eps=EPS
                    )[0]
                )
            )
            logits_exact = torch.equal(baseline.logits, intervened.logits)
            case = {
                "dataset_row_index": int(index),
                "question_id": str(
                    prepared["inputs"]["records"][int(index)]["question_id"]
                ),
                "alpha": alpha,
                "previous_layer_exact": previous_equal,
                "target_other_tokens_exact": other_equal,
                "target_last_delta_norm": float(observed_delta.norm()),
                "expected_delta_norm": expected_norm,
                "delta_expected_cosine": cosine,
                "delta_expected_max_abs_error": max_error,
                "alpha_zero_logits_exact": logits_exact if alpha == 0.0 else None,
                "hook_call_count": hook.diagnostics.hook_call_count,
                "prefill_apply_events": hook.diagnostics.prefill_apply_events,
                "applied_sample_count": hook.diagnostics.applied_sample_count,
            }
            if not previous_equal or not other_equal:
                raise RuntimeError(f"Activation locality sanity failed: {case}")
            if max_error > 1e-5 or abs(cosine - 1.0) > 1e-5:
                raise RuntimeError(f"Raw LiReF addition sanity failed: {case}")
            if alpha == 0.0 and not logits_exact:
                raise RuntimeError(f"Alpha-zero forward equality failed: {case}")
            if hook.diagnostics.prefill_apply_events != 1:
                raise RuntimeError(f"Sanity hook application count failed: {case}")
            cases.append(case)
        del inputs, encoded, baseline, intervened
        torch.cuda.empty_cache()
    payload = {
        "status": "passed",
        "model": args.model,
        "cache_index": EXPECTED_PRIMARY_CACHE_INDEX,
        "module_index": EXPECTED_PRIMARY_CACHE_INDEX - 1,
        "equation": "h' = h + alpha * raw_liref",
        "token_scope": "last prompt token",
        "cases": cases,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    atomic_write_json(args.output_dir / "sanity_checks.json", payload)
    return payload


def load_phase_outputs(
    args: argparse.Namespace,
    prepared: dict[str, Any],
    phase: str,
) -> pd.DataFrame:
    frames = []
    for _, condition in phase_rows(prepared, phase).iterrows():
        indices = sample_indices_for_condition(condition, prepared)
        vector = condition_vector(condition, prepared)
        identity = condition_identity(args, condition, indices, prepared, vector)
        data_path, manifest_path = condition_paths(
            args, phase, str(condition["condition_id"])
        )
        frames.append(validate_saved_condition(data_path, manifest_path, identity))
    result = pd.concat(frames, ignore_index=True)
    expected = sum(
        len(sample_indices_for_condition(row, prepared))
        for _, row in phase_rows(prepared, phase).iterrows()
    )
    if len(result) != expected:
        raise RuntimeError(f"{phase} aggregate row count mismatch")
    return result


def check_alpha_zero_equality(frame: pd.DataFrame, phase: str) -> dict[str, Any]:
    nohook = frame[
        (frame["intervention_direction"] == "none") & (~frame["hook_installed"])
    ].set_index("question_id")
    if nohook.empty:
        raise RuntimeError(f"{phase} has no no-hook baseline")
    checks = []
    hooked_zero = frame[(frame["hook_installed"]) & (frame["alpha"] == 0.0)]
    for condition_id, condition_frame in hooked_zero.groupby("condition_id"):
        current = condition_frame.set_index("question_id").loc[nohook.index]
        raw_equal = bool((current["raw_response"] == nohook["raw_response"]).all())
        parsed_equal = bool(
            current["parsed_answer"].fillna("<NONE>").equals(
                nohook["parsed_answer"].fillna("<NONE>")
            )
        )
        correct_equal = bool(current["correct"].equals(nohook["correct"]))
        checks.append(
            {
                "condition_id": condition_id,
                "rows": len(current),
                "raw_response_exact": raw_equal,
                "parsed_answer_exact": parsed_equal,
                "correct_exact": correct_equal,
            }
        )
        if not (raw_equal and parsed_equal and correct_equal):
            raise RuntimeError(
                f"{phase} no-hook versus alpha-zero generation equality failed: {condition_id}"
            )
    if len(checks) == 0:
        raise RuntimeError(f"{phase} has no alpha-zero hook condition")
    return {"phase": phase, "status": "passed", "checks": checks}


def run_phase(
    args: argparse.Namespace,
    prepared: dict[str, Any],
    phase: str,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> pd.DataFrame:
    for _, condition in phase_rows(prepared, phase).iterrows():
        run_condition(args, condition, prepared, model, tokenizer, device)
    frame = load_phase_outputs(args, prepared, phase)
    if phase in {"sanity", "pilot", "full"}:
        check = check_alpha_zero_equality(frame, phase)
        atomic_write_json(args.output_dir / f"{phase}_alpha_zero_check.json", check)
    return frame


def freeze_experiment(args: argparse.Namespace, prepared: dict[str, Any]) -> dict[str, Any]:
    sanity_path = args.output_dir / "sanity_checks.json"
    if not sanity_path.is_file():
        raise RuntimeError("Activation sanity must pass before freeze")
    with sanity_path.open("r", encoding="utf-8") as handle:
        sanity = json.load(handle)
    if sanity.get("status") != "passed":
        raise RuntimeError("Activation sanity did not pass")
    sanity_frame = load_phase_outputs(args, prepared, "sanity")
    pilot_frame = load_phase_outputs(args, prepared, "pilot")
    sanity_zero = check_alpha_zero_equality(sanity_frame, "sanity")
    pilot_zero = check_alpha_zero_equality(pilot_frame, "pilot")
    freeze_payload = {
        "experiment_name": prepared["experiment_config"]["experiment_name"],
        "freeze_policy": (
            "Pilot results are an engineering check only and are not used to change "
            "alpha, layer, vector source, controls, parser, prompts, or conditions."
        ),
        "locked_set_disclosure": prepared["experiment_config"][
            "locked_set_disclosure"
        ],
        "primary_contrast": {
            "cache_index": EXPECTED_PRIMARY_CACHE_INDEX,
            "module_index": EXPECTED_PRIMARY_CACHE_INDEX - 1,
            "alpha_magnitude": PRIMARY_ALPHA_MAGNITUDE,
            "direction_source": PRIMARY_SOURCE,
        },
        "layers": list(EXPECTED_LAYERS),
        "alpha_grid": list(EXPECTED_ALPHAS),
        "orthogonal_controls": args.orthogonal_controls,
        "orthogonal_control_seeds": list(
            range(42001, 42001 + args.orthogonal_controls)
        ),
        "label_shuffled_controls": args.label_shuffled_controls,
        "label_shuffled_control_seeds": list(
            range(42011, 42011 + args.label_shuffled_controls)
        ),
        "fdr_scope": "phase x comparison_family x group",
        "parser": prepared["experiment_config"]["parser"],
        "prompt": prepared["experiment_config"]["prompt"],
        "generation": prepared["experiment_config"]["generation"],
        "hook": prepared["experiment_config"]["hook"],
        "all_conditions": [
            {
                key: (
                    None
                    if pd.isna(value)
                    else value.item()
                    if hasattr(value, "item")
                    else value
                )
                for key, value in row.items()
            }
            for row in prepared["condition_plan"].to_dict(orient="records")
        ],
        "integrity": {
            "script_sha256": sha256_file(SCRIPT_PATH),
            "experiment_config_sha256": sha256_file(
                args.output_dir / "experiment_config.json"
            ),
            "condition_plan_sha256": sha256_file(
                args.output_dir / "condition_plan.csv"
            ),
            "eval_samples_sha256": sha256_file(args.output_dir / "eval_samples.csv"),
            "control_vector_metadata_sha256": sha256_file(
                args.output_dir / "control_vector_metadata.csv"
            ),
            "primary_vector_tensor_sha256": tensor_sha256(
                vector_at(
                    prepared["payloads"],
                    PRIMARY_SOURCE,
                    EXPECTED_PRIMARY_CACHE_INDEX,
                )
            ),
            "sensitivity_vector_tensor_sha256": tensor_sha256(
                vector_at(
                    prepared["payloads"],
                    SENSITIVITY_SOURCE,
                    EXPECTED_PRIMARY_CACHE_INDEX,
                )
            ),
            "control_vector_tensor_sha256": {
                key: tensor_sha256(value)
                for key, value in sorted(prepared["controls"].items())
            },
        },
        "checks": {
            "activation_sanity": sanity,
            "sanity_alpha_zero": sanity_zero,
            "pilot_alpha_zero": pilot_zero,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "cuda": torch.version.cuda,
            "device_id": args.device_id,
        },
        "frozen_at": datetime.now().astimezone().isoformat(),
    }
    frozen_path = args.output_dir / "frozen_experiment_config.json"
    atomic_write_json(frozen_path, freeze_payload)
    manifest = {
        "status": "frozen",
        "frozen_config_path": str(frozen_path.resolve()),
        "frozen_config_sha256": sha256_file(frozen_path),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "frozen_at": freeze_payload["frozen_at"],
    }
    atomic_write_json(args.output_dir / "freeze_manifest.json", manifest)
    return freeze_payload


def verify_freeze(args: argparse.Namespace, prepared: dict[str, Any]) -> dict[str, Any]:
    frozen_path = args.output_dir / "frozen_experiment_config.json"
    manifest_path = args.output_dir / "freeze_manifest.json"
    if not frozen_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Frozen config is required before locked evaluation")
    with frozen_path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if sha256_file(frozen_path) != manifest.get("frozen_config_sha256"):
        raise RuntimeError("Frozen config SHA mismatch")
    current = {
        "script_sha256": sha256_file(SCRIPT_PATH),
        "experiment_config_sha256": sha256_file(
            args.output_dir / "experiment_config.json"
        ),
        "condition_plan_sha256": sha256_file(args.output_dir / "condition_plan.csv"),
        "eval_samples_sha256": sha256_file(args.output_dir / "eval_samples.csv"),
        "control_vector_metadata_sha256": sha256_file(
            args.output_dir / "control_vector_metadata.csv"
        ),
        "primary_vector_tensor_sha256": tensor_sha256(
            vector_at(
                prepared["payloads"], PRIMARY_SOURCE, EXPECTED_PRIMARY_CACHE_INDEX
            )
        ),
        "sensitivity_vector_tensor_sha256": tensor_sha256(
            vector_at(
                prepared["payloads"], SENSITIVITY_SOURCE, EXPECTED_PRIMARY_CACHE_INDEX
            )
        ),
        "control_vector_tensor_sha256": {
            key: tensor_sha256(value)
            for key, value in sorted(prepared["controls"].items())
        },
    }
    if current != frozen["integrity"]:
        differences = {
            key: {"frozen": frozen["integrity"].get(key), "current": value}
            for key, value in current.items()
            if frozen["integrity"].get(key) != value
        }
        raise RuntimeError(f"Frozen experiment integrity failure: {differences}")
    return frozen


def bootstrap_mean_ci(
    values: np.ndarray, replicates: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap requires a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 200):
        count = min(200, replicates - start)
        draws = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[draws].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def stable_seed(base_seed: int, *parts: Any) -> int:
    suffix = "|".join(str(part) for part in parts)
    return int((base_seed + int(text_sha256(suffix)[:8], 16)) % (2**32 - 1))


def exact_mcnemar(
    reference: np.ndarray, treatment: np.ndarray
) -> tuple[int, int, float]:
    reference = np.asarray(reference, dtype=bool)
    treatment = np.asarray(treatment, dtype=bool)
    correct_to_wrong = int(np.sum(reference & ~treatment))
    wrong_to_correct = int(np.sum(~reference & treatment))
    discordant = correct_to_wrong + wrong_to_correct
    p_value = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                min(correct_to_wrong, wrong_to_correct),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )
    return correct_to_wrong, wrong_to_correct, p_value


def add_bh_fdr(
    frame: pd.DataFrame,
    p_column: str = "mcnemar_p_value",
    scope: tuple[str, ...] = ("phase", "comparison_family", "group"),
) -> pd.DataFrame:
    result = frame.copy()
    result["fdr_q_value"] = np.nan
    result["fdr_reject_0p05"] = False
    for _, positions in result.groupby(list(scope), dropna=False).groups.items():
        positions = list(positions)
        p_values = result.loc[positions, p_column].to_numpy(dtype=np.float64)
        order = np.argsort(p_values)
        ranked = p_values[order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        unsorted = np.empty_like(adjusted)
        unsorted[order] = adjusted
        result.loc[positions, "fdr_q_value"] = unsorted
        result.loc[positions, "fdr_reject_0p05"] = unsorted <= 0.05
    return result


def summarize_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_id, condition in frame.groupby("condition_id", sort=False):
        first = condition.iloc[0]
        for group in ("Overall", "Reasoning", "Memory"):
            subset = condition if group == "Overall" else condition[condition["group"] == group]
            rows.append(
                {
                    "condition_id": condition_id,
                    "phase": first["phase"],
                    "group": group,
                    "direction_source": first["direction_source"],
                    "intervention_direction": first["intervention_direction"],
                    "control_id": first["control_id"],
                    "target_cache_index": first["target_cache_index"],
                    "target_module_index": first["target_module_index"],
                    "alpha": first["alpha"],
                    "n": len(subset),
                    "n_correct": int(subset["correct"].astype(bool).sum()),
                    "accuracy": float(subset["correct"].astype(bool).mean()),
                    "parse_failures": int((~subset["parse_ok"].astype(bool)).sum()),
                    "parse_failure_rate": float((~subset["parse_ok"].astype(bool)).mean()),
                    "accuracy_among_parsed": (
                        float(
                            subset.loc[subset["parse_ok"].astype(bool), "correct"]
                            .astype(bool)
                            .mean()
                        )
                        if subset["parse_ok"].astype(bool).any()
                        else np.nan
                    ),
                    "mean_relative_injection_norm": float(
                        subset["relative_injection_norm"].fillna(0.0).mean()
                    ),
                    "unique_raw_responses": int(subset["raw_response"].nunique()),
                    "empty_response_rate": float(
                        subset["raw_response"].fillna("").str.strip().eq("").mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_baseline_effects(
    frame: pd.DataFrame, baseline: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_indexed = baseline.set_index("question_id")
    effect_rows = []
    flip_rows = []
    for condition_id, condition in frame.groupby("condition_id", sort=False):
        if str(condition.iloc[0]["intervention_direction"]) == "none":
            continue
        current = condition.set_index("question_id").loc[baseline_indexed.index]
        for group in ("Overall", "Reasoning", "Memory"):
            mask = (
                np.ones(len(current), dtype=bool)
                if group == "Overall"
                else current["group"].to_numpy() == group
            )
            base_correct = baseline_indexed["correct"].astype(bool).to_numpy()[mask]
            current_correct = current["correct"].astype(bool).to_numpy()[mask]
            differences = current_correct.astype(float) - base_correct.astype(float)
            seed = stable_seed(args.seed, condition_id, group, "baseline")
            ci_low, ci_high = bootstrap_mean_ci(
                differences, args.bootstrap_replicates, seed
            )
            ctw, wtc, p_value = exact_mcnemar(base_correct, current_correct)
            first = condition.iloc[0]
            effect_rows.append(
                {
                    "condition_id": condition_id,
                    "phase": first["phase"],
                    "comparison_family": "condition_vs_nohook_baseline",
                    "group": group,
                    "direction_source": first["direction_source"],
                    "intervention_direction": first["intervention_direction"],
                    "control_id": first["control_id"],
                    "target_cache_index": first["target_cache_index"],
                    "alpha": first["alpha"],
                    "n": int(mask.sum()),
                    "baseline_accuracy": float(base_correct.mean()),
                    "condition_accuracy": float(current_correct.mean()),
                    "accuracy_delta": float(differences.mean()),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "correct_to_wrong": ctw,
                    "wrong_to_correct": wtc,
                    "mcnemar_p_value": p_value,
                }
            )
        joined = current[
            ["dataset_row_index", "group", "category", "gold_answer", "parsed_answer", "correct"]
        ].copy()
        joined["condition_id"] = condition_id
        joined["baseline_parsed_answer"] = baseline_indexed.loc[
            joined.index, "parsed_answer"
        ].to_numpy()
        joined["baseline_correct"] = baseline_indexed.loc[
            joined.index, "correct"
        ].astype(bool).to_numpy()
        joined["answer_changed"] = (
            joined["parsed_answer"].fillna("<NONE>")
            != joined["baseline_parsed_answer"].fillna("<NONE>")
        )
        joined["wrong_to_correct"] = (~joined["baseline_correct"]) & joined[
            "correct"
        ].astype(bool)
        joined["correct_to_wrong"] = joined["baseline_correct"] & ~joined[
            "correct"
        ].astype(bool)
        joined.reset_index(names="question_id", inplace=True)
        flip_rows.append(joined)
    effects = add_bh_fdr(pd.DataFrame(effect_rows))
    flips = pd.concat(flip_rows, ignore_index=True)
    return effects, flips


def directional_effects(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    keys = [
        "phase",
        "direction_source",
        "intervention_direction",
        "control_id",
        "target_cache_index",
    ]
    rows = []
    working = frame[frame["hook_installed"].astype(bool) & (frame["alpha"] != 0.0)].copy()
    working["alpha_magnitude"] = working["alpha"].abs()
    for key_values, pair in working.groupby(keys + ["alpha_magnitude"], dropna=False):
        positive = pair[pair["alpha"] > 0]
        negative = pair[pair["alpha"] < 0]
        if positive["condition_id"].nunique() != 1 or negative["condition_id"].nunique() != 1:
            continue
        plus = positive.set_index("question_id")
        minus = negative.set_index("question_id").loc[plus.index]
        for group in ("Overall", "Reasoning", "Memory"):
            mask = (
                np.ones(len(plus), dtype=bool)
                if group == "Overall"
                else plus["group"].to_numpy() == group
            )
            plus_correct = plus["correct"].astype(bool).to_numpy()[mask]
            minus_correct = minus["correct"].astype(bool).to_numpy()[mask]
            plus_minus = plus_correct.astype(float) - minus_correct.astype(float)
            expected = -plus_minus if group == "Memory" else plus_minus
            seed = stable_seed(args.seed, *key_values, group, "directional")
            ci_low, ci_high = bootstrap_mean_ci(
                expected, args.bootstrap_replicates, seed
            )
            if group == "Memory":
                ctw, wtc, p_value = exact_mcnemar(plus_correct, minus_correct)
            else:
                ctw, wtc, p_value = exact_mcnemar(minus_correct, plus_correct)
            rows.append(
                {
                    **dict(zip(keys, key_values[:-1])),
                    "alpha_magnitude": float(key_values[-1]),
                    "comparison_family": "positive_vs_negative_direction",
                    "group": group,
                    "n": int(mask.sum()),
                    "negative_alpha_accuracy": float(minus_correct.mean()),
                    "positive_alpha_accuracy": float(plus_correct.mean()),
                    "plus_minus_accuracy": float(plus_minus.mean()),
                    "expected_directional_effect": float(expected.mean()),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "opposite_to_expected_flips": ctw,
                    "expected_direction_flips": wtc,
                    "mcnemar_p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    return add_bh_fdr(result) if not result.empty else result


def control_specificity(
    full: pd.DataFrame, control: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    primary = full[
        (full["target_cache_index"] == EXPECTED_PRIMARY_CACHE_INDEX)
        & (full["alpha"].abs() == PRIMARY_ALPHA_MAGNITUDE)
        & (full["intervention_direction"] == "liref")
    ]
    primary_plus = primary[primary["alpha"] > 0].set_index("question_id")
    primary_minus = primary[primary["alpha"] < 0].set_index("question_id")
    rows = []
    for control_id, control_frame in control.groupby("control_id"):
        control_plus = control_frame[control_frame["alpha"] > 0].set_index("question_id")
        control_minus = control_frame[control_frame["alpha"] < 0].set_index("question_id")
        for group in ("Reasoning", "Memory"):
            ids = primary_plus[primary_plus["group"] == group].index
            sign = 1.0 if group == "Reasoning" else -1.0
            primary_effect = sign * (
                primary_plus.loc[ids, "correct"].astype(float).to_numpy()
                - primary_minus.loc[ids, "correct"].astype(float).to_numpy()
            )
            control_effect = sign * (
                control_plus.loc[ids, "correct"].astype(float).to_numpy()
                - control_minus.loc[ids, "correct"].astype(float).to_numpy()
            )
            paired_difference = primary_effect - control_effect
            ci_low, ci_high = bootstrap_mean_ci(
                paired_difference,
                args.bootstrap_replicates,
                stable_seed(args.seed, control_id, group, "specificity"),
            )
            rows.append(
                {
                    "control_id": control_id,
                    "control_type": str(control_frame.iloc[0]["intervention_direction"]),
                    "group": group,
                    "n": len(ids),
                    "liref_expected_directional_effect": float(primary_effect.mean()),
                    "control_expected_directional_effect": float(control_effect.mean()),
                    "liref_minus_control_paired_effect": float(paired_difference.mean()),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def representation_causal_table(
    args: argparse.Namespace,
    prepared: dict[str, Any],
    directional: pd.DataFrame,
) -> pd.DataFrame:
    representation = prepared["eval_samples"]
    del representation
    layer_metrics = prepared["output_layer_selection"].copy() if "output_layer_selection" in prepared else pd.read_csv(
        args.output_dir / "layer_selection_train2400.csv"
    )
    causal = directional[
        (directional["phase"] == "full")
        & (directional["intervention_direction"] == "liref")
        & (directional["alpha_magnitude"] == PRIMARY_ALPHA_MAGNITUDE)
        & (directional["group"].isin(["Reasoning", "Memory"]))
    ].pivot(index="target_cache_index", columns="group", values="expected_directional_effect")
    causal.columns = [f"causal_effect_{str(column).lower()}" for column in causal.columns]
    selected = layer_metrics[layer_metrics["cache_index"].isin(EXPECTED_LAYERS)].copy()
    selected = selected.merge(causal, left_on="cache_index", right_index=True, how="left")
    return selected


def validate_analysis_inputs(
    sanity: pd.DataFrame,
    pilot: pd.DataFrame,
    full: pd.DataFrame,
    control: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> dict[str, Any]:
    expected = {
        "sanity": 16,
        "pilot": 3200,
        "full": 9600,
        "control": 4800,
        "sensitivity": 1200,
    }
    observed = {
        "sanity": len(sanity),
        "pilot": len(pilot),
        "full": len(full),
        "control": len(control),
        "sensitivity": len(sensitivity),
    }
    if observed != expected:
        raise RuntimeError(f"Analysis row-count failure: {observed} != {expected}")
    for phase, frame in (
        ("sanity", sanity),
        ("pilot", pilot),
        ("full", full),
        ("control", control),
        ("sensitivity", sensitivity),
    ):
        duplicate = frame.duplicated(["condition_id", "question_id"]).any()
        if duplicate:
            raise RuntimeError(f"Duplicate condition-question pair in {phase}")
        hooked = frame[frame["hook_installed"].astype(bool)]
        if not hooked.empty:
            if not bool((hooked["hook_prefill_apply_events_batch"] == 1).all()):
                raise RuntimeError(f"Hook apply event mismatch in {phase}")
            if float(hooked["other_token_max_abs_diff_batch"].max()) != 0.0:
                raise RuntimeError(f"Non-target token was modified in {phase}")
    return {"status": "passed", "expected_rows": expected, "observed_rows": observed}


def make_figures(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    directional: pd.DataFrame,
    specificity: pd.DataFrame,
) -> list[str]:
    if args.no_figures:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    full = summary[
        (summary["phase"] == "full")
        & (summary["intervention_direction"].isin(["liref", "none"]))
        & (summary["group"].isin(["Reasoning", "Memory"]))
    ]
    for group in ("Reasoning", "Memory"):
        figure, axis = plt.subplots(figsize=(8, 5))
        subset = full[(full["group"] == group) & (full["intervention_direction"] == "liref")]
        for cache_index, layer_frame in subset.groupby("target_cache_index"):
            layer_frame = layer_frame.sort_values("alpha")
            axis.plot(
                layer_frame["alpha"],
                100 * layer_frame["accuracy"],
                marker="o",
                label=f"Layer {int(cache_index)}",
            )
        baseline = full[(full["group"] == group) & (full["intervention_direction"] == "none")]
        axis.axhline(
            100 * float(baseline.iloc[0]["accuracy"]),
            color="black",
            linestyle="--",
            label="No-hook baseline",
        )
        axis.set(xlabel="alpha", ylabel="Accuracy (%)", title=f"Locked MMLU-Pro: {group}")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = figure_dir / f"dose_response_{group.lower()}.pdf"
        figure.savefig(path)
        plt.close(figure)
        paths.append(str(path.resolve()))

    primary = directional[
        (directional["phase"] == "full")
        & (directional["target_cache_index"] == EXPECTED_PRIMARY_CACHE_INDEX)
        & (directional["alpha_magnitude"] == PRIMARY_ALPHA_MAGNITUDE)
        & (directional["group"].isin(["Reasoning", "Memory"]))
    ]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        primary["group"],
        100 * primary["expected_directional_effect"],
        color=["#e59432", "#879f72"],
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set(ylabel="Expected directional effect (pp)", title="Primary LiReF contrast")
    figure.tight_layout()
    path = figure_dir / "primary_directional_effect.pdf"
    figure.savefig(path)
    plt.close(figure)
    paths.append(str(path.resolve()))

    if not specificity.empty:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        labels = specificity["control_id"] + " / " + specificity["group"]
        axis.bar(labels, 100 * specificity["liref_minus_control_paired_effect"])
        axis.axhline(0, color="black", linewidth=1)
        axis.tick_params(axis="x", rotation=45)
        axis.set(ylabel="LiReF - control paired effect (pp)", title="Control specificity")
        figure.tight_layout()
        path = figure_dir / "control_specificity.pdf"
        figure.savefig(path)
        plt.close(figure)
        paths.append(str(path.resolve()))
    return paths


def analyze_experiment(args: argparse.Namespace, prepared: dict[str, Any]) -> dict[str, Any]:
    verify_freeze(args, prepared)
    sanity = load_phase_outputs(args, prepared, "sanity")
    pilot = load_phase_outputs(args, prepared, "pilot")
    full = load_phase_outputs(args, prepared, "full")
    control = load_phase_outputs(args, prepared, "control")
    sensitivity = load_phase_outputs(args, prepared, "sensitivity")
    integrity = validate_analysis_inputs(sanity, pilot, full, control, sensitivity)
    baseline = full[
        (full["intervention_direction"] == "none") & (~full["hook_installed"].astype(bool))
    ].copy()
    if len(baseline) != EXPECTED_HELDOUT:
        raise RuntimeError("Locked no-hook baseline must contain exactly 600 rows")
    locked_results = pd.concat((full, control, sensitivity), ignore_index=True)
    atomic_write_csv_gz(args.output_dir / "baseline_results.csv.gz", baseline)
    atomic_write_csv_gz(args.output_dir / "sample_results.csv.gz", locked_results)
    atomic_write_csv_gz(args.output_dir / "pilot_results.csv.gz", pilot)

    summary = summarize_conditions(pd.concat((pilot, locked_results), ignore_index=True))
    effects, flips = paired_baseline_effects(locked_results, baseline, args)
    directional = directional_effects(locked_results, args)
    specificity = control_specificity(full, control, args)
    representation_causal = representation_causal_table(args, prepared, directional)
    output_collapse = summary[
        (summary["parse_failure_rate"] >= 0.5)
        | (summary["empty_response_rate"] >= 0.5)
        | (summary["unique_raw_responses"] <= 1)
    ].copy()
    primary = directional[
        (directional["phase"] == "full")
        & (directional["direction_source"] == PRIMARY_SOURCE)
        & (directional["intervention_direction"] == "liref")
        & (directional["target_cache_index"] == EXPECTED_PRIMARY_CACHE_INDEX)
        & (directional["alpha_magnitude"] == PRIMARY_ALPHA_MAGNITUDE)
        & (directional["group"].isin(["Reasoning", "Memory"]))
    ].copy()
    leakage = directional[
        (directional["phase"] == "sensitivity")
        & (directional["group"].isin(["Reasoning", "Memory"]))
    ].copy()

    table_dir = args.output_dir / "tables"
    atomic_write_csv(table_dir / "condition_summary.csv", summary)
    atomic_write_csv(table_dir / "paired_baseline_effects.csv", effects)
    atomic_write_csv(table_dir / "directional_effects.csv", directional)
    atomic_write_csv(table_dir / "primary_contrasts.csv", primary)
    atomic_write_csv(table_dir / "flip_analysis.csv", flips)
    atomic_write_csv(table_dir / "control_specificity.csv", specificity)
    atomic_write_csv(table_dir / "leakage_sensitivity.csv", leakage)
    atomic_write_csv(table_dir / "representation_causal_layers.csv", representation_causal)
    atomic_write_csv(table_dir / "output_collapse_checks.csv", output_collapse)
    figure_paths = make_figures(args, summary, effects, directional, specificity)

    primary_records = primary[
        [
            "group",
            "negative_alpha_accuracy",
            "positive_alpha_accuracy",
            "expected_directional_effect",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "mcnemar_p_value",
            "fdr_q_value",
        ]
    ].to_dict(orient="records")
    interpretation = "Pending manual scientific interpretation"
    if len(primary) == 2:
        positive_groups = primary[primary["expected_directional_effect"] > 0]
        specific_positive = specificity[
            specificity["liref_minus_control_paired_effect"] > 0
        ]
        if len(positive_groups) == 2 and len(specific_positive) == len(specificity):
            interpretation = "Directional effects in both groups with descriptive control specificity"
        elif len(positive_groups) >= 1:
            interpretation = "Partial directional evidence"
        else:
            interpretation = "Null or opposite-direction primary result"
    metadata = {
        "status": "complete",
        "experiment_name": prepared["experiment_config"]["experiment_name"],
        "locked_set_disclosure": prepared["experiment_config"]["locked_set_disclosure"],
        "integrity": integrity,
        "primary_results": primary_records,
        "descriptive_interpretation": interpretation,
        "interpretation_warning": (
            "Three orthogonal controls and one label-shuffled control are limited negative "
            "controls; specificity should be interpreted descriptively unless paired CIs "
            "exclude zero consistently."
        ),
        "figure_paths": figure_paths,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    atomic_write_json(args.output_dir / "analysis_metadata.json", metadata)
    write_readme(args, prepared, metadata, summary, effects, directional, specificity)
    validate_output_manifest(args)
    return metadata


def classify_result(
    primary: pd.DataFrame, specificity: pd.DataFrame
) -> tuple[str, str]:
    if len(primary) != 2:
        return "Incomplete", "Primary Reasoning/Memory contrasts are incomplete."
    positive = primary["expected_directional_effect"] > 0
    ci_positive = primary["bootstrap_ci_low"] > 0
    control_positive = (
        specificity["liref_minus_control_paired_effect"] > 0
        if not specificity.empty
        else pd.Series(dtype=bool)
    )
    control_ci_positive = (
        specificity["bootstrap_ci_low"] > 0
        if not specificity.empty
        else pd.Series(dtype=bool)
    )
    if bool(ci_positive.all()) and not specificity.empty and bool(control_ci_positive.all()):
        return (
            "Strong",
            "Both pre-registered directional effects and all limited control-specificity contrasts have positive 95% bootstrap CIs.",
        )
    if bool(positive.all()) and not specificity.empty and not bool(control_positive.all()):
        return (
            "Non-specific",
            "Both directional point estimates are positive, but LiReF does not consistently exceed the limited controls.",
        )
    if bool(positive.any()):
        return (
            "Partial",
            "At least one pre-registered group has a positive directional point estimate, but the full directional/specificity criterion is not met.",
        )
    return (
        "Null",
        "Neither pre-registered group has a positive expected-direction point estimate.",
    )


def markdown_rows(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "(결과 없음)"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_readme(
    args: argparse.Namespace,
    prepared: dict[str, Any],
    metadata: dict[str, Any],
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    directional: pd.DataFrame,
    specificity: pd.DataFrame,
) -> None:
    primary = directional[
        (directional["phase"] == "full")
        & (directional["direction_source"] == PRIMARY_SOURCE)
        & (directional["intervention_direction"] == "liref")
        & (directional["target_cache_index"] == EXPECTED_PRIMARY_CACHE_INDEX)
        & (directional["alpha_magnitude"] == PRIMARY_ALPHA_MAGNITUDE)
        & (directional["group"].isin(["Reasoning", "Memory"]))
    ].copy()
    classification, rationale = classify_result(primary, specificity)
    baseline = summary[
        (summary["phase"] == "full")
        & (summary["intervention_direction"] == "none")
        & (summary["group"].isin(["Reasoning", "Memory", "Overall"]))
    ]
    primary_baseline = effects[
        (effects["phase"] == "full")
        & (effects["direction_source"] == PRIMARY_SOURCE)
        & (effects["intervention_direction"] == "liref")
        & (effects["target_cache_index"] == EXPECTED_PRIMARY_CACHE_INDEX)
        & (
            (
                (effects["group"] == "Reasoning")
                & (effects["alpha"] == PRIMARY_ALPHA_MAGNITUDE)
            )
            | (
                (effects["group"] == "Memory")
                & (effects["alpha"] == -PRIMARY_ALPHA_MAGNITUDE)
            )
        )
    ]
    parser_failures = int(
        summary.loc[
            summary["phase"].isin(["full", "control", "sensitivity"])
            & (summary["group"] == "Overall"),
            "parse_failures",
        ].sum()
    )
    text = f"""# Strict Causal Intervention

## 1. 실험 정체성

이 실험은 **Single-layer, single-prefill-last-token raw-LiReF causal intervention**이다. 공개 `features_intervention.py`를 수정하거나 대체하지 않은 통제된 확장 실험이며, 논문 intervention의 완전 재현이라고 부르지 않는다.

## 2. 모델·데이터·split

- Model: `{args.model}`
- Dataset: `mmlu-pro-3000samples.json` 3,000개
- Direction/engineering split: train 2,400개 (Reasoning 1,103 / Memory 1,297)
- Locked evaluation split: 600개 (Reasoning 276 / Memory 324)
- 주의: 이 600개는 과거 representation/baseline 분석에는 사용되었다. 이번 strict intervention의 조건 선택과 tuning에 사용하지 않은 **locked evaluation set**이지, 역사적으로 완전히 unseen인 데이터는 아니다.

## 3. train/locked mapping 검증

`split_ids.json`의 row index와 `question_id`를 모두 대조했으며, 두 split은 겹치지 않고 합집합이 정확히 3,000개이다. 상세 표본은 `eval_samples.csv`에 기록했다.

## 4. train-only layer 선택

LiReF 방향과 Reasoning/Memory cosine gap은 train 2,400개에서만 다시 계산했다. 최대 gap의 cache index는 `{EXPECTED_PRIMARY_CACHE_INDEX}`였고, transformer module index는 `{EXPECTED_PRIMARY_CACHE_INDEX - 1}`이다.

## 5. 방향 벡터

- Primary: `{PRIMARY_SOURCE}` = train 2,400개의 `mean(Reasoning) - mean(Memory)`
- Leakage sensitivity: `{SENSITIVITY_SOURCE}` = 전체 3,000개의 같은 평균 차이
- Primary contrast: Layer 12, `|alpha|=0.10`

## 6. cache index ↔ module index

`cache_index k`는 1-based k번째 block 출력이고 실제 hook 대상은 `model.model.layers[k-1]`이다. 따라서 Layer 12는 `model.model.layers[11]`이다.

## 7. prompt 차이

- Hidden-state extraction: `Q: {{question}}\\nA: ` (선택지 없음, no-CoT)
- Behavioral evaluation: category별 5-shot CoT + 질문 + 선택지 + `A: Let's think step by step. `
- 따라서 동일 prompt representation을 조작한 것이 아니라, question-only prompt에서 얻은 LiReF의 **cross-prompt transfer**를 검증한다.

## 8. parser

정규식 `{PARSER_PATTERN.pattern}`만 사용한다. 일치하지 않으면 `parsed_answer=None`, `parse_ok=False`, `correct=False`이며 random fallback은 없다.

## 9. hook과 intervention 범위

- 위치: 선택한 transformer block의 output residual stream
- token: 첫 prefill call의 마지막 prompt token 한 개
- decoding: 적용하지 않음
- 수식: `h' = h + alpha * raw_liref`
- 모든 batch에서 hook apply event가 정확히 1회인지 기록·검증했다.

## 10. 고정 조건

- Layers: `{list(EXPECTED_LAYERS)}`
- Alpha grid: `{list(EXPECTED_ALPHAS)}`
- Orthogonal controls: {args.orthogonal_controls}, seeds 42001–42003
- Label-shuffled control: {args.label_shuffled_controls}, seed 42011
- Pilot 결과로 layer, alpha 또는 분석 조건을 변경하지 않았다.

## 11. baseline

{markdown_rows(baseline, ['group', 'n', 'n_correct', 'accuracy', 'parse_failures'])}

## 12. Primary directional contrast

`expected_directional_effect`는 Reasoning에서 `Acc(+0.10)-Acc(-0.10)`, Memory에서 `Acc(-0.10)-Acc(+0.10)`이다.

{markdown_rows(primary, ['group', 'n', 'negative_alpha_accuracy', 'positive_alpha_accuracy', 'expected_directional_effect', 'bootstrap_ci_low', 'bootstrap_ci_high', 'mcnemar_p_value', 'fdr_q_value'])}

## 13. Primary 방향의 baseline 대비 효과

{markdown_rows(primary_baseline, ['group', 'alpha', 'baseline_accuracy', 'condition_accuracy', 'accuracy_delta', 'correct_to_wrong', 'wrong_to_correct', 'bootstrap_ci_low', 'bootstrap_ci_high', 'mcnemar_p_value'])}

## 14. paired 통계

모든 accuracy 차이는 동일 question의 paired binary outcome으로 계산했다. 95% CI는 question-level percentile bootstrap {args.bootstrap_replicates}회, 유의확률은 discordant pair의 exact McNemar binomial test이다. FDR은 `phase × comparison_family × group` 안에서 Benjamini–Hochberg 방식으로 보정했다.

## 15. control specificity

{markdown_rows(specificity, ['control_id', 'control_type', 'group', 'liref_expected_directional_effect', 'control_expected_directional_effect', 'liref_minus_control_paired_effect', 'bootstrap_ci_low', 'bootstrap_ci_high'])}

Orthogonal control 3개와 label-shuffled control 1개는 제한된 negative controls이므로 specificity는 보수적으로 해석해야 한다.

## 16. 결과 분류

- 분류: **{classification}**
- 근거: {rationale}

## 17. 오류·collapse 점검

- 집계된 parser failure 표시 합계(Overall 행 기준 해석 필요): {parser_failures}
- 조건별 parse failure, 빈 응답률, 고유 응답 수는 `tables/condition_summary.csv`와 `tables/output_collapse_checks.csv`에 기록했다.
- 기술적 generation error가 발생한 조건은 complete manifest를 만들지 않으므로 결과에 포함되지 않는다.

## 18. 말할 수 있는 결론

이 실험은 고정된 한 model과 locked MMLU-Pro split에서, 한 block의 한 prefill token에 raw LiReF를 더했을 때 paired 정답 행동이 방향성 있게 변하는지 검증한다. 실제 결론은 위 primary CI, McNemar 결과, control specificity를 함께 근거로 제한해야 한다.

## 19. 주장하면 안 되는 결론

- 전체 모델·전체 benchmark에 대한 일반화
- held-out 600개가 역사적으로 완전히 unseen이라는 주장
- 공개 `features_intervention.py`의 모든-layer intervention을 그대로 재현했다는 주장
- limited controls만으로 LiReF만의 유일한 인과 기제임이 증명됐다는 주장
- prompt mismatch가 없는 동일표현 개입이라는 주장

## 20. 주요 출력

- `experiment_config.json`: 사전 고정 설계
- `frozen_experiment_config.json`: pilot 이후 동결된 코드·조건·SHA
- `eval_samples.csv`: split 및 표본 ID
- `sanity_checks.json`: activation locality와 raw-add 검증
- `baseline_results.csv.gz`: locked 600 no-hook 결과
- `sample_results.csv.gz`: locked full/control/sensitivity 개별 답변
- `pilot_results.csv.gz`: train 200 engineering pilot
- `tables/condition_summary.csv`: 조건별 accuracy와 parser 통계
- `tables/paired_baseline_effects.csv`: baseline 대비 paired 효과
- `tables/directional_effects.csv`: `+alpha` 대 `-alpha`
- `tables/primary_contrasts.csv`: 사전 지정 primary 결과
- `tables/flip_analysis.csv`: wrong→correct / correct→wrong 문항
- `tables/control_specificity.csv`: LiReF-control paired specificity
- `tables/leakage_sensitivity.csv`: full-3000 direction 민감도
- `tables/representation_causal_layers.csv`: 표현 gap과 causal layer 결과
- `figures/`: dose response, primary, control PDF
- `output_integrity.json`: 출력 파일 SHA와 row-count 검증

## 21. 재현 해석 주의

공개 코드는 선택된 방향을 모든 layer의 residual input, attention output, MLP output에 반복 적용하고 모든 token/decoding에 영향을 줄 수 있다. 본 실험은 한 layer의 block output, 마지막 prefill token, 1회 raw addition으로 범위를 엄격히 제한했다. 그러므로 두 실험의 결과 크기를 직접 같은 intervention으로 비교하면 안 된다.
"""
    atomic_write_text(args.output_dir / "README.md", text)


def validate_output_manifest(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        "README.md",
        "experiment_config.json",
        "frozen_experiment_config.json",
        "freeze_manifest.json",
        "eval_samples.csv",
        "layer_selection_train2400.csv",
        "control_vector_metadata.csv",
        "control_vectors.pt",
        "condition_plan.csv",
        "sanity_checks.json",
        "sanity_alpha_zero_check.json",
        "pilot_alpha_zero_check.json",
        "full_alpha_zero_check.json",
        "baseline_results.csv.gz",
        "sample_results.csv.gz",
        "pilot_results.csv.gz",
        "analysis_metadata.json",
        "tables/condition_summary.csv",
        "tables/paired_baseline_effects.csv",
        "tables/directional_effects.csv",
        "tables/primary_contrasts.csv",
        "tables/flip_analysis.csv",
        "tables/control_specificity.csv",
        "tables/leakage_sensitivity.csv",
        "tables/representation_causal_layers.csv",
        "tables/output_collapse_checks.csv",
    ]
    missing = [name for name in required if not (args.output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Required output files are missing: {missing}")
    row_counts = {
        "baseline_results.csv.gz": len(
            pd.read_csv(args.output_dir / "baseline_results.csv.gz", compression="gzip")
        ),
        "sample_results.csv.gz": len(
            pd.read_csv(args.output_dir / "sample_results.csv.gz", compression="gzip")
        ),
        "pilot_results.csv.gz": len(
            pd.read_csv(args.output_dir / "pilot_results.csv.gz", compression="gzip")
        ),
    }
    expected = {
        "baseline_results.csv.gz": EXPECTED_HELDOUT,
        "sample_results.csv.gz": 15600,
        "pilot_results.csv.gz": 3200,
    }
    if row_counts != expected:
        raise RuntimeError(f"Final output row-count mismatch: {row_counts}")
    files = []
    integrity_path = args.output_dir / "output_integrity.json"
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path != integrity_path:
            files.append(
                {
                    "relative_path": str(path.relative_to(args.output_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "status": "passed",
        "row_counts": row_counts,
        "required_files": required,
        "file_count_excluding_self": len(files),
        "files": files,
        "validated_at": datetime.now().astimezone().isoformat(),
    }
    atomic_write_json(integrity_path, payload)
    return payload


def release_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = resolve_args(parse_args())
    prepared = prepare_experiment(args)
    print(f"Strict causal intervention: phase={args.phase}", flush=True)
    print(f"  output={args.output_dir}", flush=True)
    print(f"  locked set: 600 rows, locked against strict tuning (not historically unseen)", flush=True)
    if args.phase == "prepare":
        print("[DONE prepare]", flush=True)
        return 0
    if args.phase == "freeze":
        freeze_experiment(args, prepared)
        print("[DONE freeze]", flush=True)
        return 0
    if args.phase == "analyze":
        analyze_experiment(args, prepared)
        print("[DONE analyze]", flush=True)
        return 0

    locked_phases = {"full", "control", "sensitivity"}
    if args.phase in locked_phases:
        verify_freeze(args, prepared)

    model: torch.nn.Module | None = None
    try:
        model, tokenizer, device = load_generation_model(args)
        print(f"[MODEL loaded] {args.model} on {device}", flush=True)
        if args.phase == "sanity":
            run_activation_sanity(args, prepared, model, tokenizer, device)
            run_phase(args, prepared, "sanity", model, tokenizer, device)
        elif args.phase == "pilot":
            run_phase(args, prepared, "pilot", model, tokenizer, device)
        elif args.phase in locked_phases:
            run_phase(args, prepared, args.phase, model, tokenizer, device)
        elif args.phase == "all":
            run_activation_sanity(args, prepared, model, tokenizer, device)
            run_phase(args, prepared, "sanity", model, tokenizer, device)
            run_phase(args, prepared, "pilot", model, tokenizer, device)
            freeze_experiment(args, prepared)
            verify_freeze(args, prepared)
            for phase in ("full", "control", "sensitivity"):
                run_phase(args, prepared, phase, model, tokenizer, device)
        else:
            raise RuntimeError(f"Unsupported phase: {args.phase}")
    finally:
        release_model(model)

    if args.phase == "all":
        analyze_experiment(args, prepared)
    print(f"[DONE {args.phase}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
