#!/usr/bin/env python3
"""Project GSM8K and PopQA caches onto existing MMLU-Pro LiReF vectors.

The script is read-only with respect to models, datasets, activation caches,
and the existing layerwise_liref results. It never loads a language model and
never performs a forward pass.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.metrics import roc_auc_score

from compute_layerwise_liref import (
    CACHE_KEY as MMLU_CACHE_KEY,
    CACHE_SUFFIX,
    EPS,
    atomic_write_csv,
    atomic_write_text,
    discover_caches,
    load_model_config,
)


SCRIPT_PATH = Path(__file__).resolve()
REENACT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_DATASET_DIR = REENACT_ROOT / "liref" / "dataset"
DEFAULT_CACHE_DIR = REENACT_ROOT / "liref_outputs" / "hidden_states"
DEFAULT_MODEL_DIR = REENACT_ROOT / "liref_models"
DEFAULT_LIREF_DIR = REENACT_ROOT / "liref_outputs" / "layerwise_liref"
DEFAULT_OUTPUT_DIR = REENACT_ROOT / "liref_outputs" / "cross_dataset_projection"

GSM8K_CACHE_KEY = "gsm8k"
POPQA_CACHE_KEY = "popqa"
GSM8K_EXPECTED_SAMPLES = 1319
POPQA_EXPECTED_SAMPLES = 14267

DIRECTION_SPECS = {
    "mmlu3000_full": {
        "vector_file": "liref_vectors_in_sample.pt",
        "reference_metrics_file": "layer_metrics_in_sample.csv",
        "reference_analysis_type": "in_sample",
        "construction_samples": 3000,
    },
    "mmlu2400_train": {
        "vector_file": "liref_vectors_heldout.pt",
        "reference_metrics_file": "layer_metrics_heldout.csv",
        "reference_analysis_type": "heldout",
        "construction_samples": 2400,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project GSM8K and PopQA hidden states onto MMLU-Pro LiReF directions."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Cache-derived model names to process, or 'all' (default).",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--liref-dir", type=Path, default=DEFAULT_LIREF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def select_models(requested: list[str], caches: dict[str, Path]) -> list[str]:
    if requested == ["all"]:
        return sorted(caches)
    if "all" in requested:
        raise ValueError("Use either '--models all' or explicit model names, not both.")
    missing = [model for model in requested if model not in caches]
    if missing:
        raise KeyError(f"No activation cache found for {missing}; available={sorted(caches)}")
    return requested


def load_external_datasets(dataset_dir: Path) -> dict[str, Any]:
    gsm_path = dataset_dir / "gsm8k" / "main"
    popqa_path = dataset_dir / "PopQA" / "test"
    gsm_object = load_from_disk(str(gsm_path))
    popqa_object = load_from_disk(str(popqa_path))

    if not isinstance(gsm_object, DatasetDict) or "test" not in gsm_object:
        raise TypeError(f"Expected a DatasetDict with test split at {gsm_path}")
    gsm_test = gsm_object["test"]
    if not isinstance(popqa_object, Dataset):
        raise TypeError(f"Expected a Dataset at {popqa_path}")
    if len(gsm_test) != GSM8K_EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected {GSM8K_EXPECTED_SAMPLES} GSM8K test rows, found {len(gsm_test)}"
        )
    if len(popqa_object) != POPQA_EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected {POPQA_EXPECTED_SAMPLES} PopQA rows, found {len(popqa_object)}"
        )
    if gsm_test.column_names != ["question", "answer"]:
        raise RuntimeError(f"Unexpected GSM8K fields: {gsm_test.column_names}")

    required_popqa = {
        "id",
        "subj",
        "prop",
        "obj",
        "s_pop",
        "o_pop",
        "question",
        "possible_answers",
    }
    missing_popqa = required_popqa.difference(popqa_object.column_names)
    if missing_popqa:
        raise RuntimeError(f"PopQA is missing required fields: {sorted(missing_popqa)}")
    popqa_ids = [str(value) for value in popqa_object["id"]]
    if len(set(popqa_ids)) != POPQA_EXPECTED_SAMPLES:
        raise RuntimeError("PopQA id values are not unique.")

    gsm_row_indices = np.arange(len(gsm_test), dtype=np.int64)
    popqa_row_indices = np.arange(len(popqa_object), dtype=np.int64)
    return {
        "gsm8k": {
            "path": gsm_path.resolve(),
            "split": "test",
            "dataset": gsm_test,
            "row_indices": gsm_row_indices,
            "sample_ids": np.asarray(
                [f"gsm8k_test_{index}" for index in gsm_row_indices], dtype=object
            ),
        },
        "popqa": {
            "path": popqa_path.resolve(),
            "split": "test",
            "dataset": popqa_object,
            "row_indices": popqa_row_indices,
            "sample_ids": np.asarray(
                [f"popqa_{sample_id}" for sample_id in popqa_ids], dtype=object
            ),
        },
    }


def write_dataset_metadata(output_dir: Path, datasets: dict[str, Any]) -> None:
    gsm = datasets["gsm8k"]
    gsm_dataset = gsm["dataset"]
    gsm_frame = pd.DataFrame(
        {
            "dataset": "gsm8k",
            "split": "test",
            "row_index": gsm["row_indices"],
            "sample_id": gsm["sample_ids"],
            "question": gsm_dataset["question"],
            "answer": gsm_dataset["answer"],
        }
    )
    pop = datasets["popqa"]
    pop_dataset = pop["dataset"]
    pop_frame = pd.DataFrame(
        {
            "dataset": "popqa",
            "split": "test",
            "row_index": pop["row_indices"],
            "sample_id": pop["sample_ids"],
            "id": pop_dataset["id"],
            "subj": pop_dataset["subj"],
            "prop": pop_dataset["prop"],
            "obj": pop_dataset["obj"],
            "s_pop": pop_dataset["s_pop"],
            "o_pop": pop_dataset["o_pop"],
            "question": pop_dataset["question"],
            "possible_answers": pop_dataset["possible_answers"],
        }
    )
    atomic_write_csv(output_dir / "gsm8k_sample_metadata.csv", gsm_frame)
    atomic_write_csv(output_dir / "popqa_sample_metadata.csv", pop_frame)


def load_direction_data(
    liref_dir: Path,
    model_name: str,
    num_hidden_layers: int,
    hidden_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    direction_data: dict[str, dict[str, Any]] = {}
    reference_metrics: dict[str, pd.DataFrame] = {}
    expected_indices = list(range(1, num_hidden_layers))

    for direction_type, spec in DIRECTION_SPECS.items():
        vector_path = liref_dir / model_name / spec["vector_file"]
        metrics_path = liref_dir / model_name / spec["reference_metrics_file"]
        if not vector_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(
                f"Required MMLU-Pro LiReF result is missing: {vector_path} or {metrics_path}"
            )
        payload = torch.load(vector_path, map_location="cpu", weights_only=True)
        cache_indices = [int(value) for value in payload["cache_indices"].tolist()]
        normalized = payload["normalized_liref"]
        metadata = payload["metadata"]
        if cache_indices != expected_indices:
            raise RuntimeError(
                f"LiReF cache index mismatch for {model_name}/{direction_type}: {cache_indices}"
            )
        if tuple(normalized.shape) != (len(expected_indices), hidden_size):
            raise RuntimeError(
                f"LiReF shape mismatch for {model_name}/{direction_type}: {tuple(normalized.shape)}"
            )
        if normalized.dtype != torch.float64:
            raise RuntimeError(
                f"Expected float64 LiReF vectors for {model_name}, got {normalized.dtype}"
            )
        if metadata["model_name"] != model_name or metadata["cache_key"] != MMLU_CACHE_KEY:
            raise RuntimeError(f"LiReF metadata mismatch in {vector_path}")
        max_norm_error = float((normalized.norm(dim=1) - 1.0).abs().max().item())
        if max_norm_error > 1e-10:
            raise RuntimeError(f"Non-unit LiReF vector in {vector_path}: error={max_norm_error}")

        metrics = pd.read_csv(metrics_path).sort_values("cache_index").reset_index(drop=True)
        if metrics["cache_index"].astype(int).tolist() != expected_indices:
            raise RuntimeError(f"Reference metric index mismatch in {metrics_path}")
        if set(metrics["analysis_type"]) != {spec["reference_analysis_type"]}:
            raise RuntimeError(f"Reference analysis type mismatch in {metrics_path}")

        direction_data[direction_type] = {
            "path": vector_path.resolve(),
            "vectors": normalized,
            "cache_indices": cache_indices,
            "metadata": metadata,
        }
        reference_metrics[direction_type] = metrics
    return direction_data, reference_metrics


def describe_scores(values: np.ndarray, prefix: str) -> dict[str, float]:
    if not np.isfinite(values).all():
        raise RuntimeError(f"{prefix} values contain NaN or Inf.")
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)),
        f"{prefix}_median": float(np.median(values)),
        f"positive_{prefix}_ratio": float(np.mean(values > 0.0)),
        f"negative_{prefix}_ratio": float(np.mean(values < 0.0)),
        f"neutral_{prefix}_ratio": float(np.mean(values == 0.0)),
    }


def select_peak_row(
    frame: pd.DataFrame,
    metric: str,
    *,
    mode: str = "max",
    absolute_tolerance: float = 1e-12,
) -> pd.Series:
    """Return the earliest cached depth among numerically tied extrema."""
    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported peak mode: {mode}")
    target = float(frame[metric].max() if mode == "max" else frame[metric].min())
    tied = frame[np.abs(frame[metric].to_numpy(dtype=np.float64) - target) <= absolute_tolerance]
    if tied.empty:
        raise RuntimeError(f"Could not select {mode} row for {metric}")
    return tied.sort_values("cache_index").iloc[0]


def dataset_metric_row(
    *,
    model_name: str,
    direction_type: str,
    dataset_name: str,
    cache_index: int,
    num_hidden_layers: int,
    n_samples: int,
    cosine: np.ndarray,
    projection: np.ndarray,
) -> dict[str, Any]:
    if float(cosine.min()) < -1.0 - 1e-10 or float(cosine.max()) > 1.0 + 1e-10:
        raise RuntimeError(
            f"Cosine outside [-1,1]: {model_name}/{direction_type}/{dataset_name}/{cache_index}"
        )
    row: dict[str, Any] = {
        "model": model_name,
        "direction_type": direction_type,
        "dataset": dataset_name,
        "cache_index": cache_index,
        "representation_type": "transformer_block",
        "transformer_block_number": cache_index,
        "relative_layer_depth": cache_index / num_hidden_layers,
        "n_samples": n_samples,
    }
    row.update(describe_scores(cosine, "cosine"))
    row.update(describe_scores(projection, "projection"))
    return row


def comparison_metric_row(
    *,
    model_name: str,
    direction_type: str,
    cache_index: int,
    num_hidden_layers: int,
    gsm_cosine: np.ndarray,
    pop_cosine: np.ndarray,
    gsm_projection: np.ndarray,
    pop_projection: np.ndarray,
    mmlu_reference_row: pd.Series,
    mmlu_reference_peak_cache_index: int,
    mmlu_reference_analysis_type: str,
) -> dict[str, Any]:
    labels = np.concatenate(
        [np.ones(len(gsm_cosine), dtype=np.int8), np.zeros(len(pop_cosine), dtype=np.int8)]
    )
    cosine_scores = np.concatenate([gsm_cosine, pop_cosine])
    projection_scores = np.concatenate([gsm_projection, pop_projection])
    return {
        "model": model_name,
        "direction_type": direction_type,
        "cache_index": cache_index,
        "representation_type": "transformer_block",
        "transformer_block_number": cache_index,
        "relative_layer_depth": cache_index / num_hidden_layers,
        "n_gsm8k": len(gsm_cosine),
        "n_popqa": len(pop_cosine),
        "gsm8k_cosine_mean": float(gsm_cosine.mean()),
        "popqa_cosine_mean": float(pop_cosine.mean()),
        "cosine_gap": float(gsm_cosine.mean() - pop_cosine.mean()),
        "cosine_auroc": float(roc_auc_score(labels, cosine_scores)),
        "gsm8k_projection_mean": float(gsm_projection.mean()),
        "popqa_projection_mean": float(pop_projection.mean()),
        "projection_gap": float(gsm_projection.mean() - pop_projection.mean()),
        "projection_auroc": float(roc_auc_score(labels, projection_scores)),
        "mmlu_reference_analysis_type": mmlu_reference_analysis_type,
        "mmlu_reference_cosine_peak_cache_index": mmlu_reference_peak_cache_index,
        "mmlu_reasoning_cosine_mean": float(mmlu_reference_row["reasoning_cosine_mean"]),
        "mmlu_memory_cosine_mean": float(mmlu_reference_row["memory_cosine_mean"]),
        "mmlu_cosine_gap": float(mmlu_reference_row["cosine_gap"]),
        "mmlu_reasoning_projection_mean": float(
            mmlu_reference_row["reasoning_projection_mean"]
        ),
        "mmlu_memory_projection_mean": float(mmlu_reference_row["memory_projection_mean"]),
        "mmlu_projection_gap": float(mmlu_reference_row["projection_gap"]),
    }


def sample_metric_frame(
    *,
    model_name: str,
    direction_type: str,
    dataset_name: str,
    cache_index: int,
    num_hidden_layers: int,
    dataset_info: dict[str, Any],
    cosine: np.ndarray,
    projection: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model_name,
            "direction_type": direction_type,
            "dataset": dataset_name,
            "row_index": dataset_info["row_indices"],
            "sample_id": dataset_info["sample_ids"],
            "cache_index": cache_index,
            "representation_type": "transformer_block",
            "transformer_block_number": cache_index,
            "relative_layer_depth": cache_index / num_hidden_layers,
            "cosine_similarity": cosine,
            "projection": projection,
        }
    )


def model_output_files(model_output_dir: Path) -> list[Path]:
    return [
        model_output_dir / "layer_metrics.csv",
        model_output_dir / "dataset_comparison_metrics.csv",
        model_output_dir / "sample_metrics.csv.gz",
    ]


def plot_model_results(
    model_name: str,
    layer_metrics: pd.DataFrame,
    comparison_metrics: pd.DataFrame,
    reference_metrics: dict[str, pd.DataFrame],
    figure_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)

    def save(filename: str) -> None:
        path = figure_dir / filename
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        plt.tight_layout()
        plt.savefig(temporary, dpi=200, bbox_inches="tight")
        plt.close()
        os.replace(temporary, path)

    for direction_type in DIRECTION_SPECS:
        dataset_frame = layer_metrics[layer_metrics["direction_type"] == direction_type]
        comparison_frame = comparison_metrics[
            comparison_metrics["direction_type"] == direction_type
        ].sort_values("cache_index")
        gsm = dataset_frame[dataset_frame["dataset"] == "gsm8k"].sort_values("cache_index")
        pop = dataset_frame[dataset_frame["dataset"] == "popqa"].sort_values("cache_index")
        x = gsm["relative_layer_depth"].to_numpy()

        plt.figure(figsize=(8, 5))
        plt.plot(x, gsm["cosine_mean"], marker="o", ms=3, label="GSM8K")
        plt.plot(x, pop["cosine_mean"], marker="o", ms=3, label="PopQA")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Relative representation depth")
        plt.ylabel("Mean cosine similarity")
        plt.title(f"{model_name} — External Mean Cosine ({direction_type})")
        plt.legend()
        plt.grid(alpha=0.3)
        save(f"mean_cosine_{direction_type}.png")

        plt.figure(figsize=(8, 5))
        plt.plot(x, comparison_frame["cosine_gap"], marker="o", ms=3)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Relative representation depth")
        plt.ylabel("GSM8K mean cosine − PopQA mean cosine")
        plt.title(f"{model_name} — External Cosine Gap ({direction_type})")
        plt.grid(alpha=0.3)
        save(f"cosine_gap_{direction_type}.png")

        plt.figure(figsize=(8, 5))
        plt.plot(x, comparison_frame["cosine_auroc"], marker="o", ms=3)
        plt.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        plt.ylim(0.0, 1.0)
        plt.xlabel("Relative representation depth")
        plt.ylabel("AUROC (GSM8K=1, PopQA=0)")
        plt.title(f"{model_name} — External Cosine AUROC ({direction_type})")
        plt.grid(alpha=0.3)
        save(f"cosine_auroc_{direction_type}.png")

        plt.figure(figsize=(8, 5))
        plt.plot(x, gsm["projection_mean"], marker="o", ms=3, label="GSM8K")
        plt.plot(x, pop["projection_mean"], marker="o", ms=3, label="PopQA")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Relative representation depth")
        plt.ylabel("Mean projection")
        plt.title(f"{model_name} — External Mean Projection ({direction_type})")
        plt.legend()
        plt.grid(alpha=0.3)
        save(f"mean_projection_{direction_type}.png")

        mmlu_reference = reference_metrics[direction_type]
        peak_values = {
            "MMLU R/M\ncosine gap": int(
                select_peak_row(mmlu_reference, "cosine_gap")["cache_index"]
            ),
            "GSM8K-PopQA\ncosine gap": int(
                select_peak_row(comparison_frame, "cosine_gap")["cache_index"]
            ),
            "GSM8K max\nmean cosine": int(
                select_peak_row(gsm, "cosine_mean")["cache_index"]
            ),
            "PopQA min\nmean cosine": int(
                select_peak_row(pop, "cosine_mean", mode="min")["cache_index"]
            ),
        }
        plt.figure(figsize=(8, 5))
        labels = list(peak_values)
        depths = [peak_values[label] / int(gsm["num_hidden_layers"].iloc[0]) if "num_hidden_layers" in gsm else peak_values[label] / (float(gsm["cache_index"].max()) + 1.0) for label in labels]
        plt.bar(labels, depths)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Relative representation depth")
        plt.title(f"{model_name} — Peak Depth Comparison ({direction_type})")
        plt.xticks(rotation=10)
        plt.grid(axis="y", alpha=0.3)
        save(f"peak_depth_comparison_{direction_type}.png")


def process_model(
    *,
    model_name: str,
    cache_path: Path,
    model_dir: Path,
    liref_dir: Path,
    output_dir: Path,
    datasets: dict[str, Any],
    overwrite: bool,
    skip_existing: bool,
    make_figures: bool,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    model_output_dir = output_dir / model_name
    expected_outputs = model_output_files(model_output_dir)
    if all(path.is_file() for path in expected_outputs) and skip_existing:
        print(f"[SKIP] Complete outputs already exist for {model_name}", flush=True)
        return None
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Outputs exist for {model_name}; use --overwrite or --skip-existing: {existing}"
        )

    config = load_model_config(model_dir, model_name)
    num_hidden_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    direction_data, reference_metrics = load_direction_data(
        liref_dir, model_name, num_hidden_layers, hidden_size
    )
    analyzed_indices = direction_data["mmlu3000_full"]["cache_indices"]

    print(f"\n[START] {model_name}", flush=True)
    print(f"  activation cache: {cache_path}", flush=True)
    for direction_type, data in direction_data.items():
        print(f"  {direction_type} LiReF: {data['path']}", flush=True)
    print(f"  GSM8K cache key/count: {GSM8K_CACHE_KEY}/{GSM8K_EXPECTED_SAMPLES}", flush=True)
    print(f"  PopQA cache key/count: {POPQA_CACHE_KEY}/{POPQA_EXPECTED_SAMPLES}", flush=True)
    print(f"  cache indices: {analyzed_indices[0]}..{analyzed_indices[-1]}", flush=True)
    print(f"  hidden size: {hidden_size}", flush=True)
    print(
        f"  LiReF shape: {tuple(direction_data['mmlu3000_full']['vectors'].shape)}",
        flush=True,
    )

    loaded = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    for cache_key in (GSM8K_CACHE_KEY, POPQA_CACHE_KEY):
        if cache_key not in loaded or not isinstance(loaded[cache_key], dict):
            raise KeyError(f"Missing/invalid cache key '{cache_key}' in {cache_path}")
    cache_by_dataset = {
        "gsm8k": loaded[GSM8K_CACHE_KEY],
        "popqa": loaded[POPQA_CACHE_KEY],
    }
    expected_counts = {"gsm8k": GSM8K_EXPECTED_SAMPLES, "popqa": POPQA_EXPECTED_SAMPLES}
    expected_cache_indices = list(range(num_hidden_layers))
    for dataset_name, cache in cache_by_dataset.items():
        if sorted(cache) != expected_cache_indices:
            raise RuntimeError(f"Cache index mismatch for {model_name}/{dataset_name}")
        first = cache[0]
        expected_shape = (expected_counts[dataset_name], hidden_size)
        if tuple(first.shape) != expected_shape or first.dtype != torch.float32:
            raise RuntimeError(
                f"Cache shape/dtype mismatch for {model_name}/{dataset_name}: "
                f"{tuple(first.shape)}/{first.dtype}"
            )

    model_output_dir.mkdir(parents=True, exist_ok=True)
    sample_output_path = model_output_dir / "sample_metrics.csv.gz"
    sample_temporary_path = model_output_dir / "sample_metrics.csv.gz.tmp"
    if sample_temporary_path.exists():
        sample_temporary_path.unlink()

    layer_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    sample_header = True
    try:
        with gzip.open(sample_temporary_path, "wt", encoding="utf-8", newline="") as sample_handle:
            for position, cache_index in enumerate(analyzed_indices):
                scores: dict[str, dict[str, dict[str, np.ndarray]]] = {
                    direction_type: {} for direction_type in DIRECTION_SPECS
                }
                for dataset_name in ("gsm8k", "popqa"):
                    source = cache_by_dataset[dataset_name][cache_index]
                    expected_shape = (expected_counts[dataset_name], hidden_size)
                    if tuple(source.shape) != expected_shape or source.dtype != torch.float32:
                        raise RuntimeError(
                            f"Invalid tensor at {model_name}/{dataset_name}/cache {cache_index}: "
                            f"{tuple(source.shape)}/{source.dtype}"
                        )
                    hidden_states = source.to(dtype=torch.float64)
                    if not torch.isfinite(hidden_states).all():
                        raise RuntimeError(
                            f"NaN/Inf hidden states at {model_name}/{dataset_name}/{cache_index}"
                        )
                    hidden_norms = torch.linalg.vector_norm(hidden_states, dim=1)
                    if bool(torch.any(hidden_norms <= EPS)):
                        raise RuntimeError(
                            f"Zero hidden-state norm at {model_name}/{dataset_name}/{cache_index}"
                        )

                    for direction_type, data in direction_data.items():
                        direction = data["vectors"][position]
                        projection_tensor = torch.mv(hidden_states, direction)
                        cosine_tensor = projection_tensor / hidden_norms
                        projection = projection_tensor.numpy().copy()
                        cosine = cosine_tensor.numpy().copy()
                        if not np.isfinite(projection).all() or not np.isfinite(cosine).all():
                            raise RuntimeError(
                                f"NaN/Inf score at {model_name}/{direction_type}/"
                                f"{dataset_name}/{cache_index}"
                            )
                        if float(cosine.min()) < -1.0 - 1e-10 or float(cosine.max()) > 1.0 + 1e-10:
                            raise RuntimeError(
                                f"Cosine out of range at {model_name}/{direction_type}/"
                                f"{dataset_name}/{cache_index}"
                            )
                        scores[direction_type][dataset_name] = {
                            "projection": projection,
                            "cosine": cosine,
                        }
                        del projection_tensor, cosine_tensor
                    del hidden_states, hidden_norms
                    gc.collect()

                for direction_type, spec in DIRECTION_SPECS.items():
                    reference_frame = reference_metrics[direction_type]
                    reference_row = reference_frame[
                        reference_frame["cache_index"] == cache_index
                    ].iloc[0]
                    reference_peak = int(
                        select_peak_row(reference_frame, "cosine_gap")["cache_index"]
                    )
                    for dataset_name in ("gsm8k", "popqa"):
                        dataset_scores = scores[direction_type][dataset_name]
                        layer_rows.append(
                            {
                                **dataset_metric_row(
                                    model_name=model_name,
                                    direction_type=direction_type,
                                    dataset_name=dataset_name,
                                    cache_index=cache_index,
                                    num_hidden_layers=num_hidden_layers,
                                    n_samples=expected_counts[dataset_name],
                                    cosine=dataset_scores["cosine"],
                                    projection=dataset_scores["projection"],
                                ),
                                "num_hidden_layers": num_hidden_layers,
                                "hidden_size": hidden_size,
                                "mmlu_direction_construction_samples": spec[
                                    "construction_samples"
                                ],
                            }
                        )
                        sample_frame = sample_metric_frame(
                            model_name=model_name,
                            direction_type=direction_type,
                            dataset_name=dataset_name,
                            cache_index=cache_index,
                            num_hidden_layers=num_hidden_layers,
                            dataset_info=datasets[dataset_name],
                            cosine=dataset_scores["cosine"],
                            projection=dataset_scores["projection"],
                        )
                        sample_frame.to_csv(
                            sample_handle,
                            index=False,
                            header=sample_header,
                        )
                        sample_header = False

                    comparison_rows.append(
                        comparison_metric_row(
                            model_name=model_name,
                            direction_type=direction_type,
                            cache_index=cache_index,
                            num_hidden_layers=num_hidden_layers,
                            gsm_cosine=scores[direction_type]["gsm8k"]["cosine"],
                            pop_cosine=scores[direction_type]["popqa"]["cosine"],
                            gsm_projection=scores[direction_type]["gsm8k"]["projection"],
                            pop_projection=scores[direction_type]["popqa"]["projection"],
                            mmlu_reference_row=reference_row,
                            mmlu_reference_peak_cache_index=reference_peak,
                            mmlu_reference_analysis_type=spec["reference_analysis_type"],
                        )
                    )
                del scores
                gc.collect()
                if (
                    position == 0
                    or (position + 1) % 5 == 0
                    or position == len(analyzed_indices) - 1
                ):
                    latest = comparison_rows[-2]
                    print(
                        f"  [{position + 1:02d}/{len(analyzed_indices):02d}] "
                        f"cache={cache_index} primary cosine gap={latest['cosine_gap']:.6f} "
                        f"AUROC={latest['cosine_auroc']:.6f}",
                        flush=True,
                    )
        os.replace(sample_temporary_path, sample_output_path)
    except Exception:
        if sample_temporary_path.exists():
            sample_temporary_path.unlink()
        raise

    layer_metrics = pd.DataFrame(layer_rows)
    comparison_metrics = pd.DataFrame(comparison_rows)
    atomic_write_csv(model_output_dir / "layer_metrics.csv", layer_metrics)
    atomic_write_csv(
        model_output_dir / "dataset_comparison_metrics.csv", comparison_metrics
    )
    if make_figures:
        plot_model_results(
            model_name,
            layer_metrics,
            comparison_metrics,
            reference_metrics,
            model_output_dir / "figures",
        )

    print_representative_results(model_name, layer_metrics, comparison_metrics, reference_metrics)
    print(f"[DONE] {model_name}: {model_output_dir}", flush=True)
    del loaded, cache_by_dataset, direction_data, reference_metrics
    gc.collect()
    return layer_metrics, comparison_metrics


def print_representative_results(
    model_name: str,
    layer_metrics: pd.DataFrame,
    comparison_metrics: pd.DataFrame,
    reference_metrics: dict[str, pd.DataFrame],
) -> None:
    print(f"\n[{model_name}] peak summary", flush=True)
    for direction_type in DIRECTION_SPECS:
        layer = layer_metrics[layer_metrics["direction_type"] == direction_type]
        comp = comparison_metrics[comparison_metrics["direction_type"] == direction_type]
        gsm = layer[layer["dataset"] == "gsm8k"]
        pop = layer[layer["dataset"] == "popqa"]
        mmlu = reference_metrics[direction_type]
        mmlu_peak = int(select_peak_row(mmlu, "cosine_gap")["cache_index"])
        gap_peak = select_peak_row(comp, "cosine_gap")
        auroc_peak = select_peak_row(comp, "cosine_auroc")
        gsm_peak = select_peak_row(gsm, "cosine_mean")
        pop_min = select_peak_row(pop, "cosine_mean", mode="min")
        print(
            f"  {direction_type}: MMLU peak={mmlu_peak}; "
            f"external gap peak={int(gap_peak['cache_index'])} ({gap_peak['cosine_gap']:.6f}); "
            f"AUROC peak={int(auroc_peak['cache_index'])} ({auroc_peak['cosine_auroc']:.6f}); "
            f"GSM max={int(gsm_peak['cache_index'])} ({gsm_peak['cosine_mean']:.6f}); "
            f"Pop min={int(pop_min['cache_index'])} ({pop_min['cosine_mean']:.6f})",
            flush=True,
        )


def collect_all_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    layer_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []
    for path in sorted(item for item in output_dir.iterdir() if item.is_dir()):
        layer_path = path / "layer_metrics.csv"
        comparison_path = path / "dataset_comparison_metrics.csv"
        if layer_path.is_file() and comparison_path.is_file():
            layer_frames.append(pd.read_csv(layer_path))
            comparison_frames.append(pd.read_csv(comparison_path))
    if not layer_frames or not comparison_frames:
        raise RuntimeError("No completed cross-dataset model outputs were found.")
    layers = pd.concat(layer_frames, ignore_index=True)
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    atomic_write_csv(output_dir / "all_models_dataset_layer_metrics.csv", layers)
    atomic_write_csv(output_dir / "all_models_cross_dataset_metrics.csv", comparisons)
    return layers, comparisons


def plot_all_models(comparisons: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for direction_type in DIRECTION_SPECS:
        frame = comparisons[comparisons["direction_type"] == direction_type]
        for metric, ylabel in (
            ("cosine_gap", "GSM8K − PopQA cosine gap"),
            ("cosine_auroc", "Cosine AUROC"),
        ):
            plt.figure(figsize=(10, 6))
            for model_name, model_frame in frame.groupby("model", sort=True):
                model_frame = model_frame.sort_values("relative_layer_depth")
                plt.plot(
                    model_frame["relative_layer_depth"],
                    model_frame[metric],
                    marker="o",
                    ms=2,
                    linewidth=1.2,
                    label=model_name,
                )
            if metric == "cosine_auroc":
                plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
                plt.ylim(0.0, 1.0)
            else:
                plt.axhline(0.0, color="black", linewidth=0.8)
            plt.xlabel("Relative representation depth")
            plt.ylabel(ylabel)
            plt.title(f"All models — {ylabel} ({direction_type})")
            plt.legend(fontsize=7, ncol=2)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            output_path = figure_dir / f"all_models_{metric}_{direction_type}.png"
            temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
            plt.savefig(temporary, dpi=200, bbox_inches="tight")
            plt.close()
            os.replace(temporary, output_path)


def write_readme(output_dir: Path, datasets: dict[str, Any], caches: dict[str, Path]) -> None:
    cache_lines = "\n".join(f"- `{name}`: `{path}`" for name, path in sorted(caches.items()))
    readme = f"""# GSM8K + PopQA Cross-dataset LiReF Projection

이 결과는 기존 MMLU-Pro LiReF를 변경하거나 새로 계산하지 않고, 기존 GSM8K와 PopQA hidden-state cache를 같은 모델·같은 cached depth의 normalized MMLU-Pro LiReF에 projection한 확장 representation analysis다. 모델 loading, forward, hidden-state 재추출, intervention은 수행하지 않았다.

## 데이터와 정렬

- GSM8K: `{datasets['gsm8k']['path']}`, `test` 1,319개, cache key `{GSM8K_CACHE_KEY}`
- PopQA: `{datasets['popqa']['path']}`, 14,267개, cache key `{POPQA_CACHE_KEY}`
- 공개 extraction notebook은 두 데이터셋을 shuffle 없이 원본 순서대로 순회하고 같은 순서로 tensor를 concatenate한다.
- Cache에는 sample ID가 없으므로 독립적인 ID-to-cache verification은 불가능하다.
- GSM8K에는 공식 ID가 없어 `gsm8k_test_<row_index>`를 안정적인 sample ID로 사용한다. PopQA는 CSV dtype 혼동을 막기 위해 `popqa_<실제 id>`를 namespaced sample ID로 사용하며 실제 숫자 `id`도 metadata에 별도 보존한다.

## Direction

- PRIMARY `mmlu3000_full`: `liref_vectors_in_sample.pt`의 MMLU-Pro 3,000개 direction
- SECONDARY `mmlu2400_train`: `liref_vectors_heldout.pt`의 MMLU-Pro train 2,400개 direction
- PRIMARY external 결과는 이전 `layer_metrics_in_sample.csv`와 비교한다.
- SECONDARY external 결과는 이전 `layer_metrics_heldout.csv`와 비교한다.
- GSM8K/PopQA로 새로운 direction을 만들지 않았다.

`cache_index=0`은 norm 0인 동일 last-token embedding이라 제외한다. 기존 cache에 final Transformer block output은 없다. 분석 범위는 32-layer 모델 index 1~31, Gemma index 1~41이며 relative depth는 `cache_index / num_hidden_layers`다.

## 계산

- Projection: `h · normalized_MMLU_LiReF`
- Cosine: `(h · normalized_MMLU_LiReF) / ||h||`
- 계산 dtype: float64
- Epsilon: `{EPS}`
- 표준편차: sample standard deviation (`ddof=1`)
- Dataset comparison AUROC: GSM8K=1, PopQA=0
- Peak tie rule: 값 차이가 `1e-12` 이하인 depth들은 동률로 보고 가장 이른 cache index를 대표로 보고

Cosine을 primary 방향 정렬 지표로 사용하고 magnitude가 포함된 projection은 보조 지표로 사용한다. Score 0은 MMLU-Pro Reasoning/Memory 결정 경계가 아니다. Positive/negative ratio는 기술 통계일 뿐이며 `positive=Reasoning`, `negative=Memory`로 해석하지 않는다.

## 해석 제한

GSM8K와 PopQA는 reasoning/memory 성격뿐 아니라 domain, 문장 형식, 길이와 데이터 출처가 다르다. 따라서 gap이나 AUROC가 크다는 것은 MMLU-Pro LiReF score가 두 데이터셋을 구분한다는 cross-dataset alignment evidence이며, 순수 reasoning/memory 개념의 task-invariant 증명이나 causal evidence가 아니다.

## 출력

- `gsm8k_sample_metadata.csv`, `popqa_sample_metadata.csv`: raw text/metadata를 한 번만 저장
- `<model>/sample_metrics.csv.gz`: question text를 반복하지 않은 압축 sample-depth score
- `<model>/layer_metrics.csv`: dataset별 depth 통계
- `<model>/dataset_comparison_metrics.csv`: GSM8K-PopQA gap/AUROC와 matched MMLU reference
- `all_models_dataset_layer_metrics.csv`, `all_models_cross_dataset_metrics.csv`: 전체 모델 통합

## 발견된 activation cache

{cache_lines}

## 실행

```bash
/home/jinhyun/.conda/envs/torch/bin/python {SCRIPT_PATH} --models Meta-Llama-3-8B
/home/jinhyun/.conda/envs/torch/bin/python {SCRIPT_PATH} --models all --skip-existing
```

Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}
"""
    atomic_write_text(output_dir / "README.md", readme)


def print_global_peaks(comparisons: pd.DataFrame) -> None:
    print("\nCross-dataset peak summary", flush=True)
    for (model_name, direction_type), frame in comparisons.groupby(
        ["model", "direction_type"], sort=True
    ):
        gap = select_peak_row(frame, "cosine_gap")
        auc = select_peak_row(frame, "cosine_auroc")
        print(
            f"  {model_name} [{direction_type}]: "
            f"MMLU peak={int(gap['mmlu_reference_cosine_peak_cache_index'])}; "
            f"external gap peak={int(gap['cache_index'])} ({gap['cosine_gap']:.6f}); "
            f"AUROC peak={int(auc['cache_index'])} ({auc['cosine_auroc']:.6f})",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    model_dir = args.model_dir.resolve()
    liref_dir = args.liref_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    datasets = load_external_datasets(dataset_dir)
    write_dataset_metadata(output_dir, datasets)
    caches = discover_caches(cache_dir)
    models = select_models(args.models, caches)

    print("GSM8K + PopQA Cross-dataset LiReF Projection", flush=True)
    print(f"  output: {output_dir}", flush=True)
    print(f"  selected models: {models}", flush=True)
    print("  directions: mmlu3000_full (PRIMARY), mmlu2400_train (SECONDARY)", flush=True)
    print("  computation: CPU float64; no model load/forward; no GPU", flush=True)
    print("  sample text is stored once; sample-depth metrics use compressed CSV", flush=True)

    for model_name in models:
        process_model(
            model_name=model_name,
            cache_path=caches[model_name],
            model_dir=model_dir,
            liref_dir=liref_dir,
            output_dir=output_dir,
            datasets=datasets,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
            make_figures=not args.no_figures,
        )

    layers, comparisons = collect_all_outputs(output_dir)
    if not args.no_figures:
        plot_all_models(comparisons, output_dir)
    write_readme(output_dir, datasets, caches)
    print_global_peaks(comparisons)
    print(f"\nCompleted: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; source datasets, caches, and LiReF vectors were not modified.", file=sys.stderr)
        raise
