#!/usr/bin/env python3
"""Compute MMLU-Pro layer-wise LiReF representation statistics from caches.

This script never loads a language model and never performs a forward pass. It
uses the existing ``mmlu-pro_3000samples`` hidden-state cache only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


SCRIPT_PATH = Path(__file__).resolve()
REENACT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_DATASET_PATH = REENACT_ROOT / "liref" / "dataset" / "mmlu-pro-3000samples.json"
DEFAULT_CACHE_DIR = REENACT_ROOT / "liref_outputs" / "hidden_states"
DEFAULT_MODEL_DIR = REENACT_ROOT / "liref_models"
DEFAULT_OUTPUT_DIR = REENACT_ROOT / "liref_outputs" / "layerwise_liref"

CACHE_SUFFIX = "-base_hs_cache_no_cot_all.pt"
CACHE_KEY = "mmlu-pro_3000samples"
EXPECTED_TOTAL = 3000
EXPECTED_REASONING = 1379
EXPECTED_MEMORY = 1621
EXPECTED_TRAIN_REASONING = 1103
EXPECTED_TRAIN_MEMORY = 1297
EXPECTED_HELDOUT_REASONING = 276
EXPECTED_HELDOUT_MEMORY = 324
SCORE_THRESHOLD = 0.5
SPLIT_SEED = 42
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MMLU-Pro LiReF directions and layer-wise representation metrics."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Cache-derived model names to process, or 'all' (default).",
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs for selected models.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose complete metric/vector outputs already exist.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Compute metrics without creating figures.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def discover_caches(cache_dir: Path) -> dict[str, Path]:
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")

    discovered: dict[str, Path] = {}
    for path in sorted(cache_dir.glob(f"*{CACHE_SUFFIX}")):
        model_name = path.name[: -len(CACHE_SUFFIX)]
        if model_name in discovered:
            raise RuntimeError(f"Duplicate cache model name: {model_name}")
        discovered[model_name] = path.resolve()

    if not discovered:
        raise FileNotFoundError(f"No LiReF cache files found in {cache_dir}")
    return discovered


def select_models(requested: list[str], caches: dict[str, Path]) -> list[str]:
    if requested == ["all"]:
        return sorted(caches)
    if "all" in requested:
        raise ValueError("Use either '--models all' or an explicit model list, not both.")
    missing = [model for model in requested if model not in caches]
    if missing:
        raise KeyError(
            f"Requested models have no discovered cache: {missing}. "
            f"Available models: {sorted(caches)}"
        )
    return requested


def load_dataset_info(dataset_path: Path) -> dict[str, Any]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError("MMLU-Pro JSON must contain a list of samples.")
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} samples, found {len(records)}")

    required_fields = {
        "question_id",
        "memory_reason_score",
        "category",
        "question",
    }
    for row_index, record in enumerate(records):
        missing = required_fields.difference(record)
        if missing:
            raise KeyError(f"Dataset row {row_index} is missing fields: {sorted(missing)}")

    question_ids = np.asarray([record["question_id"] for record in records])
    if len({str(value) for value in question_ids.tolist()}) != EXPECTED_TOTAL:
        raise RuntimeError("question_id values are not unique across all 3000 samples.")

    scores = np.asarray(
        [float(record["memory_reason_score"]) for record in records], dtype=np.float64
    )
    if not np.isfinite(scores).all():
        raise RuntimeError("memory_reason_score contains NaN or Inf.")
    labels = (scores > SCORE_THRESHOLD).astype(np.int8)
    reasoning_count = int(labels.sum())
    memory_count = int(len(labels) - reasoning_count)
    boundary_count = int(np.count_nonzero(scores == SCORE_THRESHOLD))
    if (reasoning_count, memory_count, boundary_count) != (
        EXPECTED_REASONING,
        EXPECTED_MEMORY,
        0,
    ):
        raise RuntimeError(
            "Unexpected MMLU-Pro label counts: "
            f"Reasoning={reasoning_count}, Memory={memory_count}, "
            f"score==0.5={boundary_count}"
        )

    return {
        "records": records,
        "question_ids": question_ids,
        "scores": scores,
        "labels": labels,
        "categories": np.asarray([str(record["category"]) for record in records]),
        "sha256": sha256_file(dataset_path),
    }


def create_or_validate_split(
    output_dir: Path,
    dataset_path: Path,
    dataset_info: dict[str, Any],
) -> dict[str, np.ndarray]:
    labels = dataset_info["labels"]
    all_indices = np.arange(len(labels), dtype=np.int64)
    train_indices, heldout_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=SPLIT_SEED,
        stratify=labels,
    )
    train_indices = np.sort(train_indices)
    heldout_indices = np.sort(heldout_indices)

    train_labels = labels[train_indices]
    heldout_labels = labels[heldout_indices]
    actual_counts = (
        int(train_labels.sum()),
        int(len(train_labels) - train_labels.sum()),
        int(heldout_labels.sum()),
        int(len(heldout_labels) - heldout_labels.sum()),
    )
    expected_counts = (
        EXPECTED_TRAIN_REASONING,
        EXPECTED_TRAIN_MEMORY,
        EXPECTED_HELDOUT_REASONING,
        EXPECTED_HELDOUT_MEMORY,
    )
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected stratified split counts: {actual_counts}; expected {expected_counts}"
        )

    question_ids = dataset_info["question_ids"]
    split_payload = {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_info["sha256"],
        "score_threshold": SCORE_THRESHOLD,
        "random_seed": SPLIT_SEED,
        "stratification_label": "memory_reason_score > 0.5",
        "train": {
            "row_indices": train_indices.tolist(),
            "question_ids": question_ids[train_indices].tolist(),
            "n_reasoning": EXPECTED_TRAIN_REASONING,
            "n_memory": EXPECTED_TRAIN_MEMORY,
        },
        "heldout": {
            "row_indices": heldout_indices.tolist(),
            "question_ids": question_ids[heldout_indices].tolist(),
            "n_reasoning": EXPECTED_HELDOUT_REASONING,
            "n_memory": EXPECTED_HELDOUT_MEMORY,
        },
    }

    split_path = output_dir / "split_ids.json"
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != split_payload:
            raise RuntimeError(
                f"Existing shared split does not match the fixed seed-42 split: {split_path}"
            )
    else:
        atomic_write_json(split_path, split_payload)

    return {"train": train_indices, "heldout": heldout_indices}


def load_model_config(model_dir: Path, model_name: str) -> dict[str, Any]:
    config_path = model_dir / model_name / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Model config is required for index interpretation, but was not found: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    try:
        num_hidden_layers = int(config["num_hidden_layers"])
        hidden_size = int(config["hidden_size"])
    except KeyError as exc:
        raise KeyError(f"Missing architecture value in {config_path}: {exc}") from exc
    return {
        "path": config_path.resolve(),
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "model_type": str(config.get("model_type", "unknown")),
    }


def tensor_indices(indices: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(indices.astype(np.int64, copy=False))


def direction_from_indices(
    hidden_states: torch.Tensor,
    reasoning_indices: torch.Tensor,
    memory_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    reasoning_mean = hidden_states.index_select(0, reasoning_indices).mean(dim=0)
    memory_mean = hidden_states.index_select(0, memory_indices).mean(dim=0)
    raw_liref = reasoning_mean - memory_mean
    liref_norm_tensor = torch.linalg.vector_norm(raw_liref)
    liref_norm = float(liref_norm_tensor.item())
    if not math.isfinite(liref_norm) or liref_norm <= EPS:
        raise RuntimeError(f"LiReF norm is invalid or <= {EPS}: {liref_norm}")
    normalized_liref = raw_liref / liref_norm_tensor
    normalized_norm = float(torch.linalg.vector_norm(normalized_liref).item())
    if abs(normalized_norm - 1.0) > 1e-10:
        raise RuntimeError(f"Normalized LiReF norm is not approximately one: {normalized_norm}")
    return reasoning_mean, memory_mean, raw_liref, normalized_liref, liref_norm


def pooled_cohen_d(
    reasoning_projection: np.ndarray,
    memory_projection: np.ndarray,
) -> float:
    n_reasoning = len(reasoning_projection)
    n_memory = len(memory_projection)
    reasoning_std = float(np.std(reasoning_projection, ddof=1))
    memory_std = float(np.std(memory_projection, ddof=1))
    pooled_variance = (
        (n_reasoning - 1) * reasoning_std**2
        + (n_memory - 1) * memory_std**2
    ) / (n_reasoning + n_memory - 2)
    if not math.isfinite(pooled_variance) or pooled_variance <= EPS**2:
        raise RuntimeError(f"Pooled projection variance is too small: {pooled_variance}")
    return float(
        (np.mean(reasoning_projection) - np.mean(memory_projection))
        / math.sqrt(pooled_variance)
    )


def compute_metric_row(
    *,
    model_name: str,
    analysis_type: str,
    cache_index: int,
    num_hidden_layers: int,
    hidden_size: int,
    liref_norm: float,
    normalized_liref_norm: float,
    projection: np.ndarray,
    cosine: np.ndarray,
    evaluation_labels: np.ndarray,
    direction_reasoning_count: int,
    direction_memory_count: int,
) -> dict[str, Any]:
    if not np.isfinite(projection).all() or not np.isfinite(cosine).all():
        raise RuntimeError(f"NaN or Inf at {model_name}, {analysis_type}, cache {cache_index}")
    if float(cosine.min()) < -1.0 - 1e-10 or float(cosine.max()) > 1.0 + 1e-10:
        raise RuntimeError(
            f"Cosine value outside [-1, 1] at {model_name}, {analysis_type}, cache {cache_index}"
        )

    reasoning_mask = evaluation_labels == 1
    memory_mask = ~reasoning_mask
    reasoning_projection = projection[reasoning_mask]
    memory_projection = projection[memory_mask]
    reasoning_cosine = cosine[reasoning_mask]
    memory_cosine = cosine[memory_mask]

    reasoning_projection_mean = float(reasoning_projection.mean())
    memory_projection_mean = float(memory_projection.mean())
    projection_gap = reasoning_projection_mean - memory_projection_mean
    reasoning_cosine_mean = float(reasoning_cosine.mean())
    memory_cosine_mean = float(memory_cosine.mean())
    auc = float(roc_auc_score(evaluation_labels, projection))
    cohen_d = pooled_cohen_d(reasoning_projection, memory_projection)

    identity_difference = abs(projection_gap - liref_norm)
    if analysis_type == "in_sample":
        identity_tolerance = max(1e-8, abs(liref_norm) * 1e-8)
        if identity_difference > identity_tolerance:
            raise RuntimeError(
                "In-sample projection gap does not match LiReF norm: "
                f"gap={projection_gap}, norm={liref_norm}, diff={identity_difference}"
            )

    return {
        "model": model_name,
        "analysis_type": analysis_type,
        "cache_index": cache_index,
        "representation_type": "embedding" if cache_index == 0 else "transformer_block",
        "transformer_block_number": np.nan if cache_index == 0 else cache_index,
        "relative_layer_depth": cache_index / num_hidden_layers,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "n_reasoning": int(reasoning_mask.sum()),
        "n_memory": int(memory_mask.sum()),
        "n_direction_reasoning": direction_reasoning_count,
        "n_direction_memory": direction_memory_count,
        "n_evaluation_reasoning": int(reasoning_mask.sum()),
        "n_evaluation_memory": int(memory_mask.sum()),
        "liref_norm": liref_norm,
        "normalized_liref_norm": normalized_liref_norm,
        "reasoning_cosine_mean": reasoning_cosine_mean,
        "reasoning_cosine_std": float(np.std(reasoning_cosine, ddof=1)),
        "memory_cosine_mean": memory_cosine_mean,
        "memory_cosine_std": float(np.std(memory_cosine, ddof=1)),
        "cosine_gap": reasoning_cosine_mean - memory_cosine_mean,
        "reasoning_projection_mean": reasoning_projection_mean,
        "reasoning_projection_std": float(np.std(reasoning_projection, ddof=1)),
        "memory_projection_mean": memory_projection_mean,
        "memory_projection_std": float(np.std(memory_projection, ddof=1)),
        "projection_gap": projection_gap,
        "projection_gap_liref_norm_abs_diff": identity_difference,
        "cohen_d": cohen_d,
        "auroc": auc,
    }


def make_sample_frame(
    *,
    model_name: str,
    analysis_type: str,
    cache_index: int,
    num_hidden_layers: int,
    row_indices: np.ndarray,
    dataset_info: dict[str, Any],
    projection: np.ndarray,
    cosine: np.ndarray,
) -> pd.DataFrame:
    labels = dataset_info["labels"][row_indices]
    transformer_block = pd.array(
        [pd.NA if cache_index == 0 else cache_index] * len(row_indices), dtype="Int64"
    )
    return pd.DataFrame(
        {
            "model": model_name,
            "analysis_type": analysis_type,
            "row_index": row_indices,
            "question_id": dataset_info["question_ids"][row_indices],
            "cache_index": cache_index,
            "representation_type": (
                "embedding" if cache_index == 0 else "transformer_block"
            ),
            "transformer_block_number": transformer_block,
            "relative_layer_depth": cache_index / num_hidden_layers,
            "memory_reason_score": dataset_info["scores"][row_indices],
            "label": np.where(labels == 1, "Reasoning", "Memory"),
            "category": dataset_info["categories"][row_indices],
            "cosine_similarity": cosine,
            "projection": projection,
        }
    )


def vector_payload(
    *,
    model_name: str,
    analysis_type: str,
    cache_path: Path,
    dataset_path: Path,
    dataset_sha256: str,
    config: dict[str, Any],
    raw_liref: list[torch.Tensor],
    normalized_liref: list[torch.Tensor],
    reasoning_means: list[torch.Tensor],
    memory_means: list[torch.Tensor],
    analyzed_cache_indices: list[int],
    total_cached_representations: int,
    direction_counts: tuple[int, int],
    evaluation_counts: tuple[int, int],
) -> dict[str, Any]:
    num_cached_depths = len(raw_liref)
    return {
        "raw_liref": torch.stack(raw_liref),
        "normalized_liref": torch.stack(normalized_liref),
        "reasoning_mean": torch.stack(reasoning_means),
        "memory_mean": torch.stack(memory_means),
        "cache_indices": torch.tensor(analyzed_cache_indices, dtype=torch.int64),
        "transformer_block_numbers": [
            None if cache_index == 0 else cache_index
            for cache_index in analyzed_cache_indices
        ],
        "metadata": {
            "model_name": model_name,
            "analysis_type": analysis_type,
            "cache_path": str(cache_path),
            "cache_key": CACHE_KEY,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": dataset_sha256,
            "num_hidden_layers": config["num_hidden_layers"],
            "num_cached_representations": total_cached_representations,
            "num_analyzed_representations": num_cached_depths,
            "analyzed_cache_indices": analyzed_cache_indices,
            "excluded_cache_indices": [
                cache_index
                for cache_index in range(total_cached_representations)
                if cache_index not in analyzed_cache_indices
            ],
            "hidden_size": config["hidden_size"],
            "model_type": config["model_type"],
            "score_threshold": SCORE_THRESHOLD,
            "label_definition": "Reasoning if memory_reason_score > 0.5; Memory otherwise",
            "direction_definition": "reasoning_mean - memory_mean",
            "direction_reasoning_count": direction_counts[0],
            "direction_memory_count": direction_counts[1],
            "evaluation_reasoning_count": evaluation_counts[0],
            "evaluation_memory_count": evaluation_counts[1],
            "source_tensor_dtype": "torch.float32",
            "computation_and_saved_vector_dtype": "torch.float64",
            "normalization_epsilon": EPS,
            "standard_deviation": "sample standard deviation (ddof=1)",
            "cohen_d": "(Reasoning mean - Memory mean) / pooled sample standard deviation",
            "layer_index_convention": (
                "cache_index 0 is embedding output; cache_index k>=1 is the output "
                "after one-based Transformer block k; final Transformer block output "
                "is absent from the existing cache"
            ),
            "relative_layer_depth_definition": "cache_index / num_hidden_layers",
        },
    }


def expected_model_outputs(model_output_dir: Path) -> list[Path]:
    return [
        model_output_dir / "liref_vectors_in_sample.pt",
        model_output_dir / "liref_vectors_heldout.pt",
        model_output_dir / "layer_metrics_in_sample.csv",
        model_output_dir / "layer_metrics_heldout.csv",
        model_output_dir / "sample_metrics_in_sample.csv",
        model_output_dir / "sample_metrics_heldout.csv",
    ]


def plot_model_metrics(
    model_name: str,
    analysis_type: str,
    metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    x = metrics["relative_layer_depth"].to_numpy()

    def save_figure(filename: str) -> None:
        output_path = figure_dir / filename
        temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
        plt.tight_layout()
        plt.savefig(temporary, dpi=200, bbox_inches="tight")
        plt.close()
        os.replace(temporary, output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(x, metrics["reasoning_cosine_mean"], marker="o", ms=3, label="Reasoning")
    plt.plot(x, metrics["memory_cosine_mean"], marker="o", ms=3, label="Memory")
    plt.xlabel("Relative representation depth")
    plt.ylabel("Mean cosine similarity")
    plt.title(f"{model_name} — Layer-wise Mean Cosine ({analysis_type})")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(f"mean_cosine_{analysis_type}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(x, metrics["cosine_gap"], marker="o", ms=3)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Relative representation depth")
    plt.ylabel("Reasoning mean cosine − Memory mean cosine")
    plt.title(f"{model_name} — Cosine Gap ({analysis_type})")
    plt.grid(alpha=0.3)
    save_figure(f"cosine_gap_{analysis_type}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        metrics["reasoning_projection_mean"],
        marker="o",
        ms=3,
        label="Reasoning",
    )
    plt.plot(
        x,
        metrics["memory_projection_mean"],
        marker="o",
        ms=3,
        label="Memory",
    )
    plt.xlabel("Relative representation depth")
    plt.ylabel("Mean projection")
    plt.title(f"{model_name} — Layer-wise Mean Projection ({analysis_type})")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(f"mean_projection_{analysis_type}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(x, metrics["cohen_d"], marker="o", ms=3)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Relative representation depth")
    plt.ylabel("Cohen's d (projection)")
    plt.title(f"{model_name} — Cohen's d ({analysis_type})")
    plt.grid(alpha=0.3)
    save_figure(f"cohen_d_{analysis_type}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(x, metrics["auroc"], marker="o", ms=3)
    plt.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Relative representation depth")
    plt.ylabel("AUROC (Reasoning = positive)")
    plt.title(f"{model_name} — AUROC ({analysis_type})")
    plt.grid(alpha=0.3)
    save_figure(f"auroc_{analysis_type}.png")


def print_representative_metrics(model_name: str, frame: pd.DataFrame) -> None:
    available_indices = sorted(frame["cache_index"].astype(int).unique().tolist())
    representative_indices = sorted(
        {
            available_indices[0],
            available_indices[min(1, len(available_indices) - 1)],
            available_indices[len(available_indices) // 2],
            available_indices[-1],
        }
    )
    columns = [
        "analysis_type",
        "cache_index",
        "representation_type",
        "liref_norm",
        "reasoning_cosine_mean",
        "memory_cosine_mean",
        "cosine_gap",
        "reasoning_projection_mean",
        "memory_projection_mean",
        "projection_gap",
        "cohen_d",
        "auroc",
    ]
    print(f"\n[{model_name}] representative cached depths", flush=True)
    print(
        frame[frame["cache_index"].isin(representative_indices)][columns].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ),
        flush=True,
    )


def process_model(
    *,
    model_name: str,
    cache_path: Path,
    model_dir: Path,
    output_dir: Path,
    dataset_path: Path,
    dataset_info: dict[str, Any],
    split: dict[str, np.ndarray],
    overwrite: bool,
    skip_existing: bool,
    make_figures: bool,
) -> pd.DataFrame | None:
    model_output_dir = output_dir / model_name
    expected_outputs = expected_model_outputs(model_output_dir)
    if all(path.is_file() for path in expected_outputs) and skip_existing:
        print(f"[SKIP] Complete outputs already exist for {model_name}", flush=True)
        return None
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Outputs already exist for {model_name}; use --overwrite or --skip-existing: {existing}"
        )

    config = load_model_config(model_dir, model_name)
    num_hidden_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]

    print(f"\n[START] {model_name}", flush=True)
    print(f"  cache path: {cache_path}", flush=True)
    print(f"  dataset path: {dataset_path.resolve()}", flush=True)
    print(f"  config path: {config['path']} (read only; model is not loaded)", flush=True)
    print(f"  num_hidden_layers: {num_hidden_layers}", flush=True)
    print(f"  hidden_size: {hidden_size}", flush=True)

    loaded = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if CACHE_KEY not in loaded:
        raise KeyError(f"Cache key '{CACHE_KEY}' is absent from {cache_path}")
    cache = loaded[CACHE_KEY]
    if not isinstance(cache, dict):
        raise TypeError(f"Cache key '{CACHE_KEY}' must map to a layer dictionary.")
    cache_indices = sorted(cache)
    expected_indices = list(range(num_hidden_layers))
    if cache_indices != expected_indices:
        raise RuntimeError(
            f"Unexpected cache indices for {model_name}: {cache_indices}; expected {expected_indices}"
        )

    first_tensor = cache[cache_indices[0]]
    expected_shape = (EXPECTED_TOTAL, hidden_size)
    if tuple(first_tensor.shape) != expected_shape or first_tensor.dtype != torch.float32:
        raise RuntimeError(
            f"Unexpected first cache tensor for {model_name}: "
            f"shape={tuple(first_tensor.shape)}, dtype={first_tensor.dtype}; "
            f"expected shape={expected_shape}, dtype=torch.float32"
        )
    print(f"  cache index range: {cache_indices[0]}..{cache_indices[-1]}", flush=True)
    print(f"  cache tensor shape: {tuple(first_tensor.shape)}", flush=True)
    print(f"  source dtype/device: {first_tensor.dtype}/{first_tensor.device}", flush=True)
    print(f"  total samples: {EXPECTED_TOTAL}", flush=True)
    print(f"  Reasoning/Memory: {EXPECTED_REASONING}/{EXPECTED_MEMORY}", flush=True)
    print("  cache_index 0: embedding output", flush=True)
    print(
        f"  final block output absent: architecture block {num_hidden_layers} is not cached",
        flush=True,
    )

    embedding_tensor = cache[0].to(dtype=torch.float64)
    embedding_max_row_difference = float(
        torch.max(torch.abs(embedding_tensor - embedding_tensor[0])).item()
    )
    embedding_reasoning_mean = embedding_tensor.index_select(
        0, tensor_indices(np.flatnonzero(dataset_info["labels"] == 1))
    ).mean(dim=0)
    embedding_memory_mean = embedding_tensor.index_select(
        0, tensor_indices(np.flatnonzero(dataset_info["labels"] == 0))
    ).mean(dim=0)
    embedding_liref_norm = float(
        torch.linalg.vector_norm(embedding_reasoning_mean - embedding_memory_mean).item()
    )
    if embedding_liref_norm <= EPS:
        analyzed_cache_indices = cache_indices[1:]
        print(
            "  cache_index 0 LiReF: undefined and excluded "
            f"(norm={embedding_liref_norm:.6g}, max row difference={embedding_max_row_difference:.6g})",
            flush=True,
        )
    else:
        analyzed_cache_indices = cache_indices
        print(
            "  cache_index 0 LiReF: defined "
            f"(norm={embedding_liref_norm:.6g}, max row difference={embedding_max_row_difference:.6g})",
            flush=True,
        )
    if not analyzed_cache_indices:
        raise RuntimeError(f"No cache representation has a defined LiReF for {model_name}")
    del embedding_tensor, embedding_reasoning_mean, embedding_memory_mean

    labels = dataset_info["labels"]
    all_indices = np.arange(EXPECTED_TOTAL, dtype=np.int64)
    full_reasoning = all_indices[labels == 1]
    full_memory = all_indices[labels == 0]
    train_indices = split["train"]
    heldout_indices = split["heldout"]
    train_labels = labels[train_indices]
    train_reasoning = train_indices[train_labels == 1]
    train_memory = train_indices[train_labels == 0]

    torch_full_reasoning = tensor_indices(full_reasoning)
    torch_full_memory = tensor_indices(full_memory)
    torch_train_reasoning = tensor_indices(train_reasoning)
    torch_train_memory = tensor_indices(train_memory)
    torch_heldout = tensor_indices(heldout_indices)

    vector_lists: dict[str, dict[str, list[torch.Tensor]]] = {
        "in_sample": {"raw": [], "normalized": [], "reasoning_mean": [], "memory_mean": []},
        "heldout": {"raw": [], "normalized": [], "reasoning_mean": [], "memory_mean": []},
    }
    metric_rows: list[dict[str, Any]] = []
    sample_frames: dict[str, list[pd.DataFrame]] = {"in_sample": [], "heldout": []}

    for position, cache_index in enumerate(analyzed_cache_indices):
        source_tensor = cache[cache_index]
        if tuple(source_tensor.shape) != expected_shape or source_tensor.dtype != torch.float32:
            raise RuntimeError(
                f"Invalid tensor at {model_name} cache {cache_index}: "
                f"shape={tuple(source_tensor.shape)}, dtype={source_tensor.dtype}"
            )
        hidden_states = source_tensor.to(dtype=torch.float64)
        if not torch.isfinite(hidden_states).all():
            raise RuntimeError(f"Hidden state contains NaN or Inf: {model_name}, cache {cache_index}")
        hidden_norms = torch.linalg.vector_norm(hidden_states, dim=1)
        if bool(torch.any(hidden_norms <= EPS)):
            raise RuntimeError(f"Zero/near-zero hidden-state norm: {model_name}, cache {cache_index}")

        in_reason_mean, in_memory_mean, in_raw, in_normalized, in_norm = direction_from_indices(
            hidden_states, torch_full_reasoning, torch_full_memory
        )
        held_reason_mean, held_memory_mean, held_raw, held_normalized, held_norm = (
            direction_from_indices(hidden_states, torch_train_reasoning, torch_train_memory)
        )

        in_projection_tensor = torch.mv(hidden_states, in_normalized)
        in_cosine_tensor = in_projection_tensor / hidden_norms
        held_hidden = hidden_states.index_select(0, torch_heldout)
        held_projection_tensor = torch.mv(held_hidden, held_normalized)
        held_cosine_tensor = held_projection_tensor / hidden_norms.index_select(0, torch_heldout)

        in_projection = in_projection_tensor.numpy()
        in_cosine = in_cosine_tensor.numpy()
        held_projection = held_projection_tensor.numpy()
        held_cosine = held_cosine_tensor.numpy()

        metric_rows.append(
            compute_metric_row(
                model_name=model_name,
                analysis_type="in_sample",
                cache_index=cache_index,
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                liref_norm=in_norm,
                normalized_liref_norm=float(torch.linalg.vector_norm(in_normalized).item()),
                projection=in_projection,
                cosine=in_cosine,
                evaluation_labels=labels,
                direction_reasoning_count=EXPECTED_REASONING,
                direction_memory_count=EXPECTED_MEMORY,
            )
        )
        metric_rows.append(
            compute_metric_row(
                model_name=model_name,
                analysis_type="heldout",
                cache_index=cache_index,
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                liref_norm=held_norm,
                normalized_liref_norm=float(torch.linalg.vector_norm(held_normalized).item()),
                projection=held_projection,
                cosine=held_cosine,
                evaluation_labels=labels[heldout_indices],
                direction_reasoning_count=EXPECTED_TRAIN_REASONING,
                direction_memory_count=EXPECTED_TRAIN_MEMORY,
            )
        )

        sample_frames["in_sample"].append(
            make_sample_frame(
                model_name=model_name,
                analysis_type="in_sample",
                cache_index=cache_index,
                num_hidden_layers=num_hidden_layers,
                row_indices=all_indices,
                dataset_info=dataset_info,
                projection=in_projection,
                cosine=in_cosine,
            )
        )
        sample_frames["heldout"].append(
            make_sample_frame(
                model_name=model_name,
                analysis_type="heldout",
                cache_index=cache_index,
                num_hidden_layers=num_hidden_layers,
                row_indices=heldout_indices,
                dataset_info=dataset_info,
                projection=held_projection,
                cosine=held_cosine,
            )
        )

        for name, values in (
            ("in_sample", (in_raw, in_normalized, in_reason_mean, in_memory_mean)),
            ("heldout", (held_raw, held_normalized, held_reason_mean, held_memory_mean)),
        ):
            vector_lists[name]["raw"].append(values[0].clone())
            vector_lists[name]["normalized"].append(values[1].clone())
            vector_lists[name]["reasoning_mean"].append(values[2].clone())
            vector_lists[name]["memory_mean"].append(values[3].clone())

        if (
            position == 0
            or (position + 1) % 5 == 0
            or position == len(analyzed_cache_indices) - 1
        ):
            print(
                f"  [{position + 1:02d}/{len(analyzed_cache_indices):02d}] cache_index={cache_index} "
                f"in_norm={in_norm:.6f} heldout_norm={held_norm:.6f}",
                flush=True,
            )

        del (
            hidden_states,
            hidden_norms,
            held_hidden,
            in_projection_tensor,
            in_cosine_tensor,
            held_projection_tensor,
            held_cosine_tensor,
            in_projection,
            in_cosine,
            held_projection,
            held_cosine,
            in_reason_mean,
            in_memory_mean,
            in_raw,
            in_normalized,
            held_reason_mean,
            held_memory_mean,
            held_raw,
            held_normalized,
        )
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    in_metrics = metrics[metrics["analysis_type"] == "in_sample"].copy()
    heldout_metrics = metrics[metrics["analysis_type"] == "heldout"].copy()
    in_samples = pd.concat(sample_frames["in_sample"], ignore_index=True)
    heldout_samples = pd.concat(sample_frames["heldout"], ignore_index=True)

    model_output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(model_output_dir / "layer_metrics_in_sample.csv", in_metrics)
    atomic_write_csv(model_output_dir / "layer_metrics_heldout.csv", heldout_metrics)
    atomic_write_csv(model_output_dir / "sample_metrics_in_sample.csv", in_samples)
    atomic_write_csv(model_output_dir / "sample_metrics_heldout.csv", heldout_samples)

    atomic_torch_save(
        model_output_dir / "liref_vectors_in_sample.pt",
        vector_payload(
            model_name=model_name,
            analysis_type="in_sample",
            cache_path=cache_path,
            dataset_path=dataset_path,
            dataset_sha256=dataset_info["sha256"],
            config=config,
            raw_liref=vector_lists["in_sample"]["raw"],
            normalized_liref=vector_lists["in_sample"]["normalized"],
            reasoning_means=vector_lists["in_sample"]["reasoning_mean"],
            memory_means=vector_lists["in_sample"]["memory_mean"],
            analyzed_cache_indices=analyzed_cache_indices,
            total_cached_representations=len(cache_indices),
            direction_counts=(EXPECTED_REASONING, EXPECTED_MEMORY),
            evaluation_counts=(EXPECTED_REASONING, EXPECTED_MEMORY),
        ),
    )
    atomic_torch_save(
        model_output_dir / "liref_vectors_heldout.pt",
        vector_payload(
            model_name=model_name,
            analysis_type="heldout",
            cache_path=cache_path,
            dataset_path=dataset_path,
            dataset_sha256=dataset_info["sha256"],
            config=config,
            raw_liref=vector_lists["heldout"]["raw"],
            normalized_liref=vector_lists["heldout"]["normalized"],
            reasoning_means=vector_lists["heldout"]["reasoning_mean"],
            memory_means=vector_lists["heldout"]["memory_mean"],
            analyzed_cache_indices=analyzed_cache_indices,
            total_cached_representations=len(cache_indices),
            direction_counts=(EXPECTED_TRAIN_REASONING, EXPECTED_TRAIN_MEMORY),
            evaluation_counts=(EXPECTED_HELDOUT_REASONING, EXPECTED_HELDOUT_MEMORY),
        ),
    )

    if make_figures:
        figure_dir = model_output_dir / "figures"
        plot_model_metrics(model_name, "in_sample", in_metrics, figure_dir)
        plot_model_metrics(model_name, "heldout", heldout_metrics, figure_dir)

    print(
        f"  raw LiReF shape: ({len(analyzed_cache_indices)}, {hidden_size})",
        flush=True,
    )
    print(
        f"  normalized LiReF shape: ({len(analyzed_cache_indices)}, {hidden_size})",
        flush=True,
    )
    print_representative_metrics(model_name, metrics)
    print(f"[DONE] {model_name}: {model_output_dir}", flush=True)

    del loaded, cache, in_samples, heldout_samples, vector_lists, sample_frames
    gc.collect()
    return metrics


def collect_all_metrics(output_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        for analysis_type in ("in_sample", "heldout"):
            path = model_dir / f"layer_metrics_{analysis_type}.csv"
            if path.is_file():
                frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("No model layer metric CSV files were created.")
    summary = pd.concat(frames, ignore_index=True)
    atomic_write_csv(output_dir / "all_models_layer_metrics.csv", summary)
    return summary


def plot_all_models(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "cosine_gap": "Cosine gap",
        "cohen_d": "Cohen's d (projection)",
        "auroc": "AUROC (projection)",
    }
    for analysis_type in ("in_sample", "heldout"):
        subset = summary[summary["analysis_type"] == analysis_type]
        if subset.empty:
            continue
        for metric, ylabel in metrics.items():
            plt.figure(figsize=(10, 6))
            for model_name, model_frame in subset.groupby("model", sort=True):
                model_frame = model_frame.sort_values("relative_layer_depth")
                plt.plot(
                    model_frame["relative_layer_depth"],
                    model_frame[metric],
                    marker="o",
                    ms=2,
                    linewidth=1.2,
                    label=model_name,
                )
            if metric == "auroc":
                plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
                plt.ylim(0.0, 1.0)
            else:
                plt.axhline(0.0, color="black", linewidth=0.8)
            plt.xlabel("Relative representation depth")
            plt.ylabel(ylabel)
            plt.title(f"All models — {ylabel} ({analysis_type})")
            plt.legend(fontsize=7, ncol=2)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            output_path = figure_dir / f"all_models_{metric}_{analysis_type}.png"
            temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
            plt.savefig(temporary, dpi=200, bbox_inches="tight")
            plt.close()
            os.replace(temporary, output_path)


def make_readme(
    *,
    output_dir: Path,
    dataset_path: Path,
    dataset_sha256: str,
    caches: dict[str, Path],
) -> None:
    cache_lines = "\n".join(f"- `{name}`: `{path}`" for name, path in sorted(caches.items()))
    readme = f"""# MMLU-Pro Layer-wise LiReF Representation Analysis

이 디렉터리는 기존 hidden-state cache를 재사용한 MMLU-Pro 3,000문항 기반 확장 분석 결과다. 모델을 로드하거나 forward하지 않았으며, 기존 notebook·dataset·cache를 수정하지 않았다.

## 분석 범위

- Dataset: `{dataset_path.resolve()}`
- Dataset SHA-256: `{dataset_sha256}`
- Cache key: `{CACHE_KEY}`
- Reasoning: `memory_reason_score > {SCORE_THRESHOLD}` (1,379개)
- Memory: `memory_reason_score <= {SCORE_THRESHOLD}` (1,621개)
- LiReF: `mean(Reasoning hidden states) - mean(Memory hidden states)`
- Projection: `h · normalized_LiReF`
- Cosine: `(h · normalized_LiReF) / ||h||`

이 분석은 논문의 기존 Figure를 그대로 다시 계산한 것이 아니라, 공식 intervention 코드와 동일하게 MMLU-Pro 3,000개를 기준으로 만든 layer-wise 확장 분석이다.

## Cache index convention

`cache_index=0`은 embedding output이다. `cache_index=k (k>=1)`는 one-based Transformer block `k`를 지난 출력이다. 기존 추출 loop가 `range(num_hidden_layers)`만 저장했으므로 **final Transformer block output is not present in the existing cache**. Relative depth는 `cache_index / num_hidden_layers`로 계산한다.

모든 prompt가 동일한 마지막 입력 token으로 끝나므로 확인 결과 embedding output(`cache_index=0`)은 sample 사이에서 동일하고 LiReF norm이 0이다. 따라서 이 지점에서는 normalized LiReF가 수학적으로 정의되지 않으며 layer/sample metric과 vector에서 제외한다. 분석 가능한 범위는 `cache_index=1`부터 마지막 cached Transformer block output까지다.

MMLU-Pro cache는 JSON을 shuffle 없이 순서대로 읽고 batch 결과를 동일 순서로 concatenate한 추출 코드에서 생성됐다. 따라서 row index를 JSON row index와 대응시켰다. 다만 cache 자체에는 question ID가 없으므로 독립적인 ID-to-cache verification은 불가능하다.

## In-sample과 held-out

- `in_sample`: 전체 3,000개로 direction을 만들고 같은 3,000개에서 평가한다.
- `heldout`: seed 42의 고정 stratified split을 사용한다. Train 2,400개(R 1,103/M 1,297)로 direction을 만들고 held-out 600개(R 276/M 324)에서만 평가한다.
- 모든 모델은 루트의 `split_ids.json`에 저장된 동일 row/question ID split을 공유한다.

In-sample에서는 `projection_gap == ||LiReF||`가 수학적 항등식이다. 두 값을 독립적인 발견으로 해석해서는 안 된다. In-sample Cohen's d와 AUROC도 direction 생성 데이터에서 계산되므로 일반화 지표가 아니다.

## 수치 정의

- Cache dtype: float32
- 계산 및 저장 vector dtype: float64 (공개 notebook의 mean-direction 계산과 일치)
- Normalization epsilon: `{EPS}`
- 표준편차: sample standard deviation (`ddof=1`)
- Cohen's d: `(Reasoning projection mean - Memory projection mean) / pooled sample standard deviation`
- AUROC: Reasoning=1, Memory=0, projection이 클수록 Reasoning으로 정의

LiReF 방향은 두 집단의 상대적 차이를 나타낼 뿐이다. Reasoning projection이 반드시 양수이거나 Memory projection이 반드시 음수인 것은 아니다. Analyzable Transformer block output에서 LiReF norm이 `{EPS}` 이하이면 오류로 중단한다.

## 해석상 한계

결과는 `memory_reason_score`와 0.5 threshold에 따른 MMLU-Pro 내부 분리다. Category 구성 차이가 direction에 일부 반영될 수 있다. Held-out에서도 여러 cached depth를 탐색해 peak를 선택하므로 현재 결과는 탐색적 representation evidence이며 causal evidence가 아니다. 이후 다른 데이터셋 projection과 intervention으로 별도 검증해야 한다.

## 출력

각 모델 디렉터리에는 in-sample/held-out LiReF vectors, layer metrics, sample metrics, figures가 저장된다. `liref_vectors_*.pt`에는 raw/normalized LiReF, 두 집단 평균, cache index와 해석 metadata가 포함된다. `all_models_layer_metrics.csv`는 완료된 모델의 layer metric을 합친 파일이다.

## 발견된 cache

{cache_lines}

## 실행

```bash
/home/jinhyun/.conda/envs/torch/bin/python {SCRIPT_PATH} --models Meta-Llama-3-8B
/home/jinhyun/.conda/envs/torch/bin/python {SCRIPT_PATH} --models all --skip-existing
```

Generated: {datetime.now().astimezone().isoformat(timespec="seconds")}
"""
    atomic_write_text(output_dir / "README.md", readme)


def print_peaks(summary: pd.DataFrame) -> None:
    print("\nPeak cached representations", flush=True)
    for (model_name, analysis_type), frame in summary.groupby(
        ["model", "analysis_type"], sort=True
    ):
        parts = []
        for metric in ("cosine_gap", "projection_gap", "cohen_d", "auroc"):
            peak = frame.loc[frame[metric].idxmax()]
            parts.append(
                f"{metric}=cache {int(peak['cache_index'])} ({float(peak[metric]):.6f})"
            )
        print(f"  {model_name} [{analysis_type}]: " + "; ".join(parts), flush=True)


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()
    cache_dir = args.cache_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    dataset_info = load_dataset_info(dataset_path)
    split = create_or_validate_split(output_dir, dataset_path, dataset_info)
    caches = discover_caches(cache_dir)
    models = select_models(args.models, caches)

    print("MMLU-Pro Layer-wise LiReF analysis", flush=True)
    print(f"  output: {output_dir}", flush=True)
    print(f"  discovered caches: {len(caches)}", flush=True)
    print(f"  selected models: {models}", flush=True)
    print("  computation: CPU float64; no model load/forward; no GPU", flush=True)
    print(
        "  fixed split: train R/M=1103/1297, heldout R/M=276/324",
        flush=True,
    )

    for model_name in models:
        process_model(
            model_name=model_name,
            cache_path=caches[model_name],
            model_dir=model_dir,
            output_dir=output_dir,
            dataset_path=dataset_path,
            dataset_info=dataset_info,
            split=split,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
            make_figures=not args.no_figures,
        )

    summary = collect_all_metrics(output_dir)
    if not args.no_figures:
        plot_all_models(summary, output_dir)
    make_readme(
        output_dir=output_dir,
        dataset_path=dataset_path,
        dataset_sha256=dataset_info["sha256"],
        caches=caches,
    )
    print_peaks(summary)
    print(f"\nAll requested work completed: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user; existing source caches were not modified.", file=sys.stderr)
        raise
