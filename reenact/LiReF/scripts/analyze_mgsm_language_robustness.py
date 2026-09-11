#!/usr/bin/env python3
"""Analyze MGSM language robustness using completed logical caches and MMLU LiReF.

The script performs no language-model forward pass.  It composes each model's
logical 11-language MGSM cache from the immutable legacy MGSM/GSM8K caches and
the supplements created by ``prepare_mgsm_11lang_cache.py``.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from compute_layerwise_liref import atomic_write_csv, atomic_write_text, load_model_config


SCRIPT_PATH = Path(__file__).resolve()
REENACT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_MODEL_DIR = REENACT_ROOT / "liref_models"
DEFAULT_CACHE_DIR = REENACT_ROOT / "liref_outputs" / "hidden_states"
DEFAULT_LOGICAL_CACHE_DIR = DEFAULT_CACHE_DIR / "mgsm_11lang"
DEFAULT_LIREF_DIR = REENACT_ROOT / "liref_outputs" / "layerwise_liref"
DEFAULT_CROSS_DIR = REENACT_ROOT / "liref_outputs" / "cross_dataset_projection"
DEFAULT_CHARACTERISTIC_DIR = REENACT_ROOT / "liref_outputs" / "problem_characteristics"
DEFAULT_OUTPUT_DIR = REENACT_ROOT / "liref_outputs" / "mgsm_language_robustness"

LANGUAGES = ("en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te")
LEGACY_LANGUAGE_ORDER = ("zh", "de", "bn", "ja", "te")
FULL_SUPPLEMENT_LANGUAGES = ("es", "fr", "ru", "sw", "th")
EXPECTED_PROBLEMS = 250
EXPECTED_LANGUAGE_SAMPLES = 2750
EPS = 1e-12
PEAK_TOLERANCE = 1e-12

DIRECTION_SPECS = {
    "mmlu3000_full": {
        "vector_file": "liref_vectors_in_sample.pt",
        "mmlu_analysis_type": "in_sample",
    },
    "mmlu2400_train": {
        "vector_file": "liref_vectors_heldout.pt",
        "mmlu_analysis_type": "heldout",
    },
}

RESULT_TABLES = (
    "language_layer_metrics.csv",
    "problem_language_variability.csv.gz",
    "english_language_comparison.csv",
    "language_pair_correlations.csv",
    "step_correlation_by_language_layer.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--logical-cache-dir", type=Path, default=DEFAULT_LOGICAL_CACHE_DIR)
    parser.add_argument("--liref-dir", type=Path, default=DEFAULT_LIREF_DIR)
    parser.add_argument("--cross-dir", type=Path, default=DEFAULT_CROSS_DIR)
    parser.add_argument("--characteristic-dir", type=Path, default=DEFAULT_CHARACTERISTIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def discover_models(logical_cache_dir: Path) -> list[str]:
    models = sorted(path.parent.name for path in logical_cache_dir.glob("*/manifest.json"))
    if not models:
        raise FileNotFoundError(f"No complete MGSM manifest found under {logical_cache_dir}")
    return models


def select_models(requested: list[str], available: list[str]) -> list[str]:
    if requested == ["all"]:
        return available
    if "all" in requested:
        raise ValueError("Use '--models all' or explicit names, not both.")
    missing = [model for model in requested if model not in available]
    if missing:
        raise KeyError(f"No complete logical cache for {missing}; available={available}")
    return requested


def atomic_write_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def load_metadata(logical_cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_path = logical_cache_dir / "mgsm_sample_metadata.csv"
    mapping_path = logical_cache_dir / "mgsm_problem_mapping.csv"
    sample = pd.read_csv(sample_path, dtype={"sample_id": "string", "problem_id": "string"})
    mapping = pd.read_csv(mapping_path, dtype={"mgsm_problem_id": "string"})
    required = {
        "problem_id",
        "language",
        "row_index_within_language",
        "sample_id",
        "gsm8k_row_index",
        "solution_calculation_steps",
        "step_group",
    }
    if required.difference(sample.columns):
        raise RuntimeError(f"MGSM metadata missing fields: {sorted(required.difference(sample.columns))}")
    if len(sample) != EXPECTED_LANGUAGE_SAMPLES or sample["sample_id"].nunique() != 2750:
        raise RuntimeError("MGSM metadata is not 2750 unique language-samples.")
    if set(sample["language"]) != set(LANGUAGES):
        raise RuntimeError(f"Unexpected languages: {sorted(sample['language'].unique())}")
    if sample.groupby("language").size().ne(250).any():
        raise RuntimeError("Each MGSM language must contain exactly 250 rows.")
    if sample.groupby("problem_id")["language"].nunique().ne(11).any():
        raise RuntimeError("Each problem must contain exactly 11 languages.")
    if len(mapping) != 250 or mapping["gsm8k_row_index"].nunique() != 250:
        raise RuntimeError("English-GSM8K mapping is not 250/250 one-to-one.")
    return sample, mapping


def load_manifest(logical_cache_dir: Path, model_name: str) -> dict[str, Any]:
    path = logical_cache_dir / model_name / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["model"] != model_name or manifest["languages"] != list(LANGUAGES):
        raise RuntimeError(f"Manifest identity mismatch: {path}")
    if manifest["total_language_samples"] != EXPECTED_LANGUAGE_SAMPLES:
        raise RuntimeError(f"Manifest sample count mismatch: {path}")
    return manifest


class LogicalMGSMCache:
    def __init__(
        self,
        manifest: dict[str, Any],
        mapping: pd.DataFrame,
    ) -> None:
        self.manifest = manifest
        self.num_hidden_layers = int(manifest["num_hidden_layers"])
        self.hidden_size = int(manifest["hidden_size"])
        self.mapping_indices = torch.tensor(
            mapping.sort_values("mgsm_en_row_index")["gsm8k_row_index"].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )
        sources = manifest["language_sources"]
        self.gsm8k = torch.load(
            sources["en"]["path"], map_location="cpu", mmap=True, weights_only=True
        )
        legacy_path = sources[LEGACY_LANGUAGE_ORDER[0]]["legacy_path"]
        self.legacy_mgsm = torch.load(
            legacy_path, map_location="cpu", mmap=True, weights_only=True
        )
        self.supplements: dict[str, dict[int, torch.Tensor]] = {}
        for language in (*LEGACY_LANGUAGE_ORDER, *FULL_SUPPLEMENT_LANGUAGES):
            self.supplements[language] = torch.load(
                sources[language]["supplement_path"],
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        self._validate()

    def _validate(self) -> None:
        expected_indices = list(range(self.num_hidden_layers))
        if sorted(self.gsm8k) != expected_indices or sorted(self.legacy_mgsm) != expected_indices:
            raise RuntimeError("Legacy cache indices do not match the manifest.")
        if tuple(self.gsm8k[0].shape) != (1319, self.hidden_size):
            raise RuntimeError("GSM8K source shape mismatch.")
        if tuple(self.legacy_mgsm[0].shape) != (1245, self.hidden_size):
            raise RuntimeError("Legacy MGSM source shape mismatch.")
        for language, payload in self.supplements.items():
            rows = 1 if language in LEGACY_LANGUAGE_ORDER else 250
            if sorted(payload) != expected_indices:
                raise RuntimeError(f"Supplement cache indices mismatch: {language}")
            if {tuple(payload[index].shape) for index in payload} != {(rows, self.hidden_size)}:
                raise RuntimeError(f"Supplement shape mismatch: {language}")

    def layer(self, language: str, cache_index: int) -> torch.Tensor:
        if cache_index < 0 or cache_index >= self.num_hidden_layers:
            raise IndexError(cache_index)
        if language == "en":
            result = self.gsm8k[cache_index].index_select(0, self.mapping_indices)
        elif language in LEGACY_LANGUAGE_ORDER:
            source = self.manifest["language_sources"][language]
            start = int(source["legacy_slice_start"])
            stop = int(source["legacy_slice_stop"])
            result = torch.cat(
                [self.supplements[language][cache_index], self.legacy_mgsm[cache_index][start:stop]],
                dim=0,
            )
        elif language in FULL_SUPPLEMENT_LANGUAGES:
            result = self.supplements[language][cache_index]
        else:
            raise KeyError(language)
        if tuple(result.shape) != (250, self.hidden_size):
            raise RuntimeError(f"Logical cache shape mismatch: {language}/{cache_index}")
        return result


def load_directions(
    liref_dir: Path,
    model_name: str,
    num_hidden_layers: int,
    hidden_size: int,
) -> dict[str, torch.Tensor]:
    directions: dict[str, torch.Tensor] = {}
    expected_cache_indices = list(range(1, num_hidden_layers))
    for direction_type, spec in DIRECTION_SPECS.items():
        path = liref_dir / model_name / spec["vector_file"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        indices = [int(value) for value in payload["cache_indices"].tolist()]
        vectors = payload["normalized_liref"]
        if indices != expected_cache_indices or tuple(vectors.shape) != (
            len(expected_cache_indices),
            hidden_size,
        ):
            raise RuntimeError(f"LiReF index/shape mismatch: {path}")
        if payload["metadata"]["model_name"] != model_name:
            raise RuntimeError(f"LiReF model mismatch: {path}")
        norm_error = float((vectors.norm(dim=1) - 1.0).abs().max())
        if norm_error > 1e-10:
            raise RuntimeError(f"LiReF vectors are not unit normalized: {path}")
        directions[direction_type] = vectors
    return directions


def score_hidden(hidden: torch.Tensor, direction: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    hidden64 = hidden.to(torch.float64)
    projection = torch.mv(hidden64, direction)
    norms = torch.linalg.vector_norm(hidden64, dim=1)
    if torch.any(norms <= EPS):
        raise RuntimeError("A hidden-state vector has a zero/invalid norm.")
    cosine = projection / norms
    projection_np = projection.numpy()
    cosine_np = cosine.numpy()
    if not np.isfinite(projection_np).all() or not np.isfinite(cosine_np).all():
        raise RuntimeError("Projection/cosine contains NaN or Inf.")
    if cosine_np.min() < -1.0 - 1e-10 or cosine_np.max() > 1.0 + 1e-10:
        raise RuntimeError("Cosine is outside [-1, 1].")
    return cosine_np, projection_np


def compute_model_scores(
    *,
    model_name: str,
    config: dict[str, Any],
    cache: LogicalMGSMCache,
    directions: dict[str, torch.Tensor],
    sample_metadata: pd.DataFrame,
) -> pd.DataFrame:
    metadata_by_language = {
        language: sample_metadata[sample_metadata["language"] == language]
        .sort_values("row_index_within_language")
        .reset_index(drop=True)
        for language in LANGUAGES
    }
    frames: list[pd.DataFrame] = []
    num_hidden_layers = config["num_hidden_layers"]
    for cache_index in range(1, num_hidden_layers):
        language_hidden = {
            language: cache.layer(language, cache_index) for language in LANGUAGES
        }
        for direction_type, vector_matrix in directions.items():
            direction = vector_matrix[cache_index - 1]
            for language in LANGUAGES:
                cosine, projection = score_hidden(language_hidden[language], direction)
                metadata = metadata_by_language[language]
                frames.append(
                    pd.DataFrame(
                        {
                            "model": model_name,
                            "direction_type": direction_type,
                            "problem_id": metadata["problem_id"].astype(str).to_numpy(),
                            "language": language,
                            "row_index_within_language": metadata[
                                "row_index_within_language"
                            ].to_numpy(dtype=np.int64),
                            "sample_id": metadata["sample_id"].astype(str).to_numpy(),
                            "solution_calculation_steps": metadata[
                                "solution_calculation_steps"
                            ].to_numpy(dtype=np.int64),
                            "step_group": metadata["step_group"].astype(str).to_numpy(),
                            "cache_index": cache_index,
                            "representation_type": "transformer_block",
                            "transformer_block_number": cache_index,
                            "relative_layer_depth": cache_index / num_hidden_layers,
                            "cosine_similarity": cosine,
                            "projection": projection,
                        }
                    )
                )
        del language_hidden
        if cache_index == 1 or cache_index % 10 == 0 or cache_index == num_hidden_layers - 1:
            print(
                f"  [{model_name}] cache {cache_index}/{num_hidden_layers - 1}",
                flush=True,
            )
        gc.collect()
    result = pd.concat(frames, ignore_index=True)
    expected_rows = 2 * (num_hidden_layers - 1) * EXPECTED_LANGUAGE_SAMPLES
    if len(result) != expected_rows or result["sample_id"].isna().any():
        raise RuntimeError(f"Unexpected sample score rows for {model_name}: {len(result)}")
    return result


def describe(values: np.ndarray, prefix: str) -> dict[str, float]:
    if len(values) < 2 or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid values for {prefix}: n={len(values)}")
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)),
        f"{prefix}_median": float(np.median(values)),
    }


def safe_correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Invalid paired values for correlation.")
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    values = (float(pearson.statistic), float(pearson.pvalue), float(spearman.statistic), float(spearman.pvalue))
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("A correlation is NaN or Inf.")
    return values


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    if len(pvalues) == 0 or not np.isfinite(pvalues).all():
        raise RuntimeError("Invalid p-values for FDR.")
    order = np.argsort(pvalues, kind="mergesort")
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def analyze_scores(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    language_rows: list[dict[str, Any]] = []
    variability_rows: list[dict[str, Any]] = []
    english_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    model_name = str(scores["model"].iloc[0])

    for direction_type in DIRECTION_SPECS:
        direction_frame = scores[scores["direction_type"] == direction_type]
        for cache_index in sorted(direction_frame["cache_index"].unique()):
            layer = direction_frame[direction_frame["cache_index"] == cache_index]
            relative_depth = float(layer["relative_layer_depth"].iloc[0])
            pivot_cosine = layer.pivot(
                index="problem_id", columns="language", values="cosine_similarity"
            ).sort_index()
            pivot_projection = layer.pivot(
                index="problem_id", columns="language", values="projection"
            ).sort_index()
            if pivot_cosine.shape != (250, 11) or pivot_cosine.isna().any().any():
                raise RuntimeError("MGSM same-problem pivot is incomplete.")

            for language in LANGUAGES:
                subset = layer[layer["language"] == language].sort_values("problem_id")
                cosine = subset["cosine_similarity"].to_numpy(dtype=np.float64)
                projection = subset["projection"].to_numpy(dtype=np.float64)
                language_rows.append(
                    {
                        "model": model_name,
                        "direction_type": direction_type,
                        "language": language,
                        "cache_index": int(cache_index),
                        "relative_layer_depth": relative_depth,
                        "n": len(subset),
                        **describe(cosine, "cosine"),
                        **describe(projection, "projection"),
                    }
                )
                steps = subset["solution_calculation_steps"].to_numpy(dtype=np.float64)
                for metric_name, values in (
                    ("cosine_similarity", cosine),
                    ("projection", projection),
                ):
                    correlation = spearmanr(steps, values)
                    rho = float(correlation.statistic)
                    pvalue = float(correlation.pvalue)
                    if not math.isfinite(rho) or not math.isfinite(pvalue):
                        raise RuntimeError("Invalid step Spearman result.")
                    step_rows.append(
                        {
                            "model": model_name,
                            "direction_type": direction_type,
                            "language": language,
                            "cache_index": int(cache_index),
                            "relative_layer_depth": relative_depth,
                            "n": len(subset),
                            "metric": metric_name,
                            "spearman_rho": rho,
                            "abs_spearman_rho": abs(rho),
                            "spearman_pvalue": pvalue,
                        }
                    )

            cosine_values = pivot_cosine[list(LANGUAGES)].to_numpy(dtype=np.float64)
            projection_values = pivot_projection[list(LANGUAGES)].to_numpy(dtype=np.float64)
            for row_number, problem_id in enumerate(pivot_cosine.index):
                cos_row = cosine_values[row_number]
                proj_row = projection_values[row_number]
                variability_rows.append(
                    {
                        "model": model_name,
                        "direction_type": direction_type,
                        "problem_id": problem_id,
                        "cache_index": int(cache_index),
                        "relative_layer_depth": relative_depth,
                        "n_languages": 11,
                        "mean_cosine_across_languages": float(np.mean(cos_row)),
                        "std_cosine_across_languages": float(np.std(cos_row, ddof=1)),
                        "range_cosine_across_languages": float(np.ptp(cos_row)),
                        "mean_projection_across_languages": float(np.mean(proj_row)),
                        "std_projection_across_languages": float(np.std(proj_row, ddof=1)),
                        "range_projection_across_languages": float(np.ptp(proj_row)),
                    }
                )

            english_cosine = pivot_cosine["en"].to_numpy(dtype=np.float64)
            for language in LANGUAGES:
                if language == "en":
                    continue
                values = pivot_cosine[language].to_numpy(dtype=np.float64)
                delta = values - english_cosine
                pr, pp, sr, sp = safe_correlations(english_cosine, values)
                english_rows.append(
                    {
                        "model": model_name,
                        "direction_type": direction_type,
                        "language": language,
                        "cache_index": int(cache_index),
                        "relative_layer_depth": relative_depth,
                        "n": 250,
                        "mean_delta_cosine": float(np.mean(delta)),
                        "median_delta_cosine": float(np.median(delta)),
                        "mean_absolute_delta_cosine": float(np.mean(np.abs(delta))),
                        "std_delta_cosine": float(np.std(delta, ddof=1)),
                        "pearson_r": pr,
                        "pearson_pvalue": pp,
                        "spearman_rho": sr,
                        "spearman_pvalue": sp,
                    }
                )

            # Store the complete 11×11 matrix, including both symmetric cells.
            for language_a in LANGUAGES:
                for language_b in LANGUAGES:
                    if language_a == language_b:
                        pr = sr = 1.0
                        pp = sp = 0.0
                    else:
                        pr, pp, sr, sp = safe_correlations(
                            pivot_cosine[language_a].to_numpy(dtype=np.float64),
                            pivot_cosine[language_b].to_numpy(dtype=np.float64),
                        )
                    pair_rows.append(
                        {
                            "model": model_name,
                            "direction_type": direction_type,
                            "language_a": language_a,
                            "language_b": language_b,
                            "cache_index": int(cache_index),
                            "relative_layer_depth": relative_depth,
                            "n": 250,
                            "pearson_r": pr,
                            "pearson_pvalue": pp,
                            "spearman_rho": sr,
                            "spearman_pvalue": sp,
                        }
                    )

    step = pd.DataFrame(step_rows)
    step["spearman_qvalue_fdr_bh"] = np.nan
    for _, indices in step.groupby(
        ["model", "direction_type", "language", "metric"], sort=False
    ).groups.items():
        index_array = np.asarray(list(indices), dtype=np.int64)
        step.loc[index_array, "spearman_qvalue_fdr_bh"] = fdr_bh(
            step.loc[index_array, "spearman_pvalue"].to_numpy(dtype=np.float64)
        )
    return {
        "language_layer": pd.DataFrame(language_rows),
        "variability": pd.DataFrame(variability_rows),
        "english": pd.DataFrame(english_rows),
        "pair": pd.DataFrame(pair_rows),
        "step": step,
    }


def partial_dir(output_dir: Path, model_name: str) -> Path:
    return output_dir / ".partial" / model_name


def model_complete(output_dir: Path, model_name: str) -> bool:
    root = partial_dir(output_dir, model_name)
    files = [root / "sample_metrics.csv.gz", *(root / name for name in RESULT_TABLES)]
    return all(path.is_file() for path in files)


def write_model_results(
    output_dir: Path,
    model_name: str,
    scores: pd.DataFrame,
    analyzed: dict[str, pd.DataFrame],
) -> None:
    root = partial_dir(output_dir, model_name)
    atomic_write_csv_gz(root / "sample_metrics.csv.gz", scores)
    atomic_write_csv(root / "language_layer_metrics.csv", analyzed["language_layer"])
    atomic_write_csv_gz(root / "problem_language_variability.csv.gz", analyzed["variability"])
    atomic_write_csv(root / "english_language_comparison.csv", analyzed["english"])
    atomic_write_csv(root / "language_pair_correlations.csv", analyzed["pair"])
    atomic_write_csv(root / "step_correlation_by_language_layer.csv", analyzed["step"])


def concatenate_gzip_csv(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as destination:
        for path_index, path in enumerate(paths):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
                for line_index, line in enumerate(source):
                    if path_index > 0 and line_index == 0:
                        continue
                    destination.write(line)
    os.replace(temporary, output_path)


def aggregate_results(output_dir: Path, models: list[str]) -> dict[str, pd.DataFrame]:
    for model in models:
        if not model_complete(output_dir, model):
            raise RuntimeError(f"Incomplete model output: {model}")
    sample_paths = [partial_dir(output_dir, model) / "sample_metrics.csv.gz" for model in models]
    concatenate_gzip_csv(sample_paths, output_dir / "mgsm_sample_metrics.csv.gz")
    variability_paths = [
        partial_dir(output_dir, model) / "problem_language_variability.csv.gz" for model in models
    ]
    concatenate_gzip_csv(
        variability_paths, output_dir / "problem_language_variability.csv.gz"
    )

    table_mapping = {
        "language_layer": "language_layer_metrics.csv",
        "english": "english_language_comparison.csv",
        "pair": "language_pair_correlations.csv",
        "step": "step_correlation_by_language_layer.csv",
    }
    result: dict[str, pd.DataFrame] = {}
    for key, filename in table_mapping.items():
        frame = pd.concat(
            [pd.read_csv(partial_dir(output_dir, model) / filename) for model in models],
            ignore_index=True,
        )
        atomic_write_csv(output_dir / filename, frame)
        result[key] = frame
    result["variability"] = pd.concat(
        [pd.read_csv(path) for path in variability_paths], ignore_index=True
    )
    return result


def select_max(frame: pd.DataFrame, value_column: str) -> pd.Series:
    maximum = float(frame[value_column].max())
    tied = frame[np.abs(frame[value_column] - maximum) <= PEAK_TOLERANCE]
    return tied.sort_values("cache_index").iloc[0]


def select_min(frame: pd.DataFrame, value_column: str) -> pd.Series:
    minimum = float(frame[value_column].min())
    tied = frame[np.abs(frame[value_column] - minimum) <= PEAK_TOLERANCE]
    return tied.sort_values("cache_index").iloc[0]


def reference_peaks(
    model_name: str,
    direction_type: str,
    liref_dir: Path,
    cross_dir: Path,
    characteristic_dir: Path,
) -> dict[str, int]:
    analysis_type = DIRECTION_SPECS[direction_type]["mmlu_analysis_type"]
    mmlu = pd.read_csv(liref_dir / model_name / f"layer_metrics_{analysis_type}.csv")
    mmlu_peak = select_max(mmlu, "cosine_gap")
    cross = pd.read_csv(cross_dir / "all_models_cross_dataset_metrics.csv")
    cross = cross[(cross["model"] == model_name) & (cross["direction_type"] == direction_type)]
    cross_peak = select_max(cross, "cosine_gap")
    characteristic = pd.read_csv(
        characteristic_dir / "summary" / "characteristic_peak_summary.csv"
    )
    characteristic = characteristic[
        (characteristic["model"] == model_name)
        & (characteristic["direction_type"] == direction_type)
        & (characteristic["analysis"] == "gsm8k_solution_steps")
        & (characteristic["metric"] == "abs_spearman_cosine")
    ]
    if len(characteristic) != 1:
        raise RuntimeError("GSM8K characteristic reference peak is ambiguous/missing.")
    return {
        "mmlu_peak": int(mmlu_peak["cache_index"]),
        "cross_dataset_peak": int(cross_peak["cache_index"]),
        "gsm8k_step_peak": int(characteristic.iloc[0]["peak_cache_index"]),
    }


def create_summaries(
    tables: dict[str, pd.DataFrame],
    models: list[str],
    liref_dir: Path,
    cross_dir: Path,
    characteristic_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    language_layer = tables["language_layer"]
    english = tables["english"]
    pair = tables["pair"]
    step = tables["step"]
    variability = tables["variability"]

    for model_name in models:
        for direction_type in DIRECTION_SPECS:
            references = reference_peaks(
                model_name, direction_type, liref_dir, cross_dir, characteristic_dir
            )
            ll = language_layer[
                (language_layer["model"] == model_name)
                & (language_layer["direction_type"] == direction_type)
            ]
            en = english[
                (english["model"] == model_name) & (english["direction_type"] == direction_type)
            ]
            pr = pair[
                (pair["model"] == model_name)
                & (pair["direction_type"] == direction_type)
                & (pair["language_a"] != pair["language_b"])
            ]
            st = step[
                (step["model"] == model_name)
                & (step["direction_type"] == direction_type)
                & (step["metric"] == "cosine_similarity")
            ]
            var = variability[
                (variability["model"] == model_name)
                & (variability["direction_type"] == direction_type)
            ]
            var_by_depth = var.groupby(["cache_index", "relative_layer_depth"], as_index=False).agg(
                mean_within_problem_cosine_std=("std_cosine_across_languages", "mean"),
                median_within_problem_cosine_std=("std_cosine_across_languages", "median"),
            )
            en_by_depth = en.groupby(["cache_index", "relative_layer_depth"], as_index=False).agg(
                mean_english_other_spearman=("spearman_rho", "mean"),
                mean_absolute_delta=("mean_absolute_delta_cosine", "mean"),
            )
            pair_by_depth = pr.groupby(["cache_index", "relative_layer_depth"], as_index=False).agg(
                mean_pairwise_spearman=("spearman_rho", "mean")
            )
            step_by_depth = st.groupby(["cache_index", "relative_layer_depth"], as_index=False).agg(
                positive_step_rho_languages=("spearman_rho", lambda x: int((x > 0).sum())),
                negative_step_rho_languages=("spearman_rho", lambda x: int((x < 0).sum())),
                mean_step_rho=("spearman_rho", "mean"),
            )

            # Correlation between 11 language mean-cosine curves across depth.
            curve = ll.pivot(index="cache_index", columns="language", values="cosine_mean")
            curve_correlations = []
            for a_index, language_a in enumerate(LANGUAGES):
                for language_b in LANGUAGES[a_index + 1 :]:
                    curve_correlations.append(
                        float(spearmanr(curve[language_a], curve[language_b]).statistic)
                    )
            model_rows.append(
                {
                    "model": model_name,
                    "direction_type": direction_type,
                    "mean_language_profile_spearman": float(np.mean(curve_correlations)),
                    "min_language_profile_spearman": float(np.min(curve_correlations)),
                    "mean_pairwise_problem_spearman_across_depth": float(
                        pair_by_depth["mean_pairwise_spearman"].mean()
                    ),
                    "mean_within_problem_cosine_std_across_depth": float(
                        var_by_depth["mean_within_problem_cosine_std"].mean()
                    ),
                    "mean_english_other_spearman_across_depth": float(
                        en_by_depth["mean_english_other_spearman"].mean()
                    ),
                }
            )
            for reference_name, cache_index in references.items():
                var_row = var_by_depth[var_by_depth["cache_index"] == cache_index].iloc[0]
                en_row = en_by_depth[en_by_depth["cache_index"] == cache_index].iloc[0]
                pair_row = pair_by_depth[pair_by_depth["cache_index"] == cache_index].iloc[0]
                step_row = step_by_depth[step_by_depth["cache_index"] == cache_index].iloc[0]
                peak_rows.append(
                    {
                        "model": model_name,
                        "direction_type": direction_type,
                        "analysis": f"fixed_{reference_name}",
                        "cache_index": cache_index,
                        "relative_layer_depth": float(var_row["relative_layer_depth"]),
                        "mean_within_problem_cosine_std": float(
                            var_row["mean_within_problem_cosine_std"]
                        ),
                        "mean_english_other_spearman": float(
                            en_row["mean_english_other_spearman"]
                        ),
                        "mean_pairwise_language_spearman": float(
                            pair_row["mean_pairwise_spearman"]
                        ),
                        "positive_step_rho_languages": int(
                            step_row["positive_step_rho_languages"]
                        ),
                        "negative_step_rho_languages": int(
                            step_row["negative_step_rho_languages"]
                        ),
                        "mean_step_rho": float(step_row["mean_step_rho"]),
                    }
                )
            for analysis, selected in (
                ("minimum_cross_language_variability", select_min(var_by_depth, "mean_within_problem_cosine_std")),
                ("maximum_english_other_spearman", select_max(en_by_depth, "mean_english_other_spearman")),
                ("maximum_pairwise_language_spearman", select_max(pair_by_depth, "mean_pairwise_spearman")),
                ("maximum_positive_step_language_count", select_max(step_by_depth, "positive_step_rho_languages")),
            ):
                cache_index = int(selected["cache_index"])
                matching = {
                    "var": var_by_depth[var_by_depth["cache_index"] == cache_index].iloc[0],
                    "en": en_by_depth[en_by_depth["cache_index"] == cache_index].iloc[0],
                    "pair": pair_by_depth[pair_by_depth["cache_index"] == cache_index].iloc[0],
                    "step": step_by_depth[step_by_depth["cache_index"] == cache_index].iloc[0],
                }
                peak_rows.append(
                    {
                        "model": model_name,
                        "direction_type": direction_type,
                        "analysis": analysis,
                        "cache_index": cache_index,
                        "relative_layer_depth": float(selected["relative_layer_depth"]),
                        "mean_within_problem_cosine_std": float(matching["var"]["mean_within_problem_cosine_std"]),
                        "mean_english_other_spearman": float(matching["en"]["mean_english_other_spearman"]),
                        "mean_pairwise_language_spearman": float(matching["pair"]["mean_pairwise_spearman"]),
                        "positive_step_rho_languages": int(matching["step"]["positive_step_rho_languages"]),
                        "negative_step_rho_languages": int(matching["step"]["negative_step_rho_languages"]),
                        "mean_step_rho": float(matching["step"]["mean_step_rho"]),
                    }
                )

        # Direct PRIMARY/SECONDARY comparison at matched sample/layer/language.
        primary = language_layer[
            (language_layer["model"] == model_name)
            & (language_layer["direction_type"] == "mmlu3000_full")
        ]
        secondary = language_layer[
            (language_layer["model"] == model_name)
            & (language_layer["direction_type"] == "mmlu2400_train")
        ]
        merged = primary.merge(
            secondary,
            on=["model", "language", "cache_index", "relative_layer_depth"],
            suffixes=("_primary", "_secondary"),
            validate="one_to_one",
        )
        robustness_rows.append(
            {
                "model": model_name,
                "mean_absolute_language_mean_cosine_difference": float(
                    np.mean(np.abs(merged["cosine_mean_primary"] - merged["cosine_mean_secondary"]))
                ),
                "language_mean_curve_pearson": float(
                    pearsonr(merged["cosine_mean_primary"], merged["cosine_mean_secondary"]).statistic
                ),
                "language_mean_curve_spearman": float(
                    spearmanr(merged["cosine_mean_primary"], merged["cosine_mean_secondary"]).statistic
                ),
            }
        )
    return pd.DataFrame(model_rows), pd.DataFrame(peak_rows), pd.DataFrame(robustness_rows)


def plot_results(
    tables: dict[str, pd.DataFrame],
    models: list[str],
    output_dir: Path,
    liref_dir: Path,
    cross_dir: Path,
    characteristic_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    def save(path: Path) -> None:
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        plt.tight_layout()
        plt.savefig(temporary, dpi=180, bbox_inches="tight")
        plt.close()
        os.replace(temporary, path)

    for model_name in models:
        direction_type = "mmlu3000_full"
        ll = tables["language_layer"]
        ll = ll[(ll["model"] == model_name) & (ll["direction_type"] == direction_type)]
        plt.figure(figsize=(9, 5.5))
        for language in LANGUAGES:
            frame = ll[ll["language"] == language].sort_values("relative_layer_depth")
            plt.plot(frame["relative_layer_depth"], frame["cosine_mean"], label=language, linewidth=1.2)
        plt.xlabel("Relative representation depth")
        plt.ylabel("Mean cosine")
        plt.title(f"{model_name} — MGSM 11-language mean LiReF alignment")
        plt.grid(alpha=0.25)
        plt.legend(ncol=4, fontsize=8)
        save(figure_dir / f"{model_name}__language_mean_cosine.png")

        var = tables["variability"]
        var = var[(var["model"] == model_name) & (var["direction_type"] == direction_type)]
        var = var.groupby("relative_layer_depth", as_index=False)["std_cosine_across_languages"].mean()
        plt.figure(figsize=(8, 5))
        plt.plot(var["relative_layer_depth"], var["std_cosine_across_languages"], marker="o", ms=3)
        plt.xlabel("Relative representation depth")
        plt.ylabel("Mean within-problem cosine std")
        plt.title(f"{model_name} — same-problem cross-language variability")
        plt.grid(alpha=0.25)
        save(figure_dir / f"{model_name}__cross_language_variability.png")

        en = tables["english"]
        en = en[(en["model"] == model_name) & (en["direction_type"] == direction_type)]
        en = en.groupby("relative_layer_depth", as_index=False)["spearman_rho"].mean()
        plt.figure(figsize=(8, 5))
        plt.plot(en["relative_layer_depth"], en["spearman_rho"], marker="o", ms=3)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("Relative representation depth")
        plt.ylabel("Mean English–other Spearman rho")
        plt.title(f"{model_name} — paired problem ordering consistency")
        plt.grid(alpha=0.25)
        save(figure_dir / f"{model_name}__english_other_spearman.png")

        step = tables["step"]
        step = step[
            (step["model"] == model_name)
            & (step["direction_type"] == direction_type)
            & (step["metric"] == "cosine_similarity")
        ]
        plt.figure(figsize=(9, 5.5))
        for language in LANGUAGES:
            frame = step[step["language"] == language].sort_values("relative_layer_depth")
            plt.plot(frame["relative_layer_depth"], frame["spearman_rho"], label=language, linewidth=1.2)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("Relative representation depth")
        plt.ylabel("Step heuristic ↔ cosine Spearman rho")
        plt.title(f"{model_name} — step association across languages")
        plt.grid(alpha=0.25)
        plt.legend(ncol=4, fontsize=8)
        save(figure_dir / f"{model_name}__step_rho_by_language.png")

        reference = reference_peaks(
            model_name, direction_type, liref_dir, cross_dir, characteristic_dir
        )["mmlu_peak"]
        pair = tables["pair"]
        pair = pair[
            (pair["model"] == model_name)
            & (pair["direction_type"] == direction_type)
            & (pair["cache_index"] == reference)
        ]
        matrix = pd.DataFrame(np.eye(11), index=LANGUAGES, columns=LANGUAGES)
        for row in pair.itertuples(index=False):
            matrix.loc[row.language_a, row.language_b] = row.spearman_rho
            matrix.loc[row.language_b, row.language_a] = row.spearman_rho
        plt.figure(figsize=(7.5, 6.5))
        image = plt.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        plt.xticks(range(11), LANGUAGES, rotation=45)
        plt.yticks(range(11), LANGUAGES)
        plt.colorbar(image, label="Problem-level Spearman rho")
        plt.title(f"{model_name} — language correlation at MMLU peak cache {reference}")
        save(figure_dir / f"{model_name}__language_heatmap_mmlu_peak.png")


def write_readme(output_dir: Path, models: list[str]) -> None:
    text = f"""# MGSM Language Robustness Analysis

이 분석은 MMLU-Pro에서 만든 기존 LiReF direction에 동일한 250개 수학 문제의 11개 언어 hidden state를 정렬한다. MGSM 자체에서 direction을 만들지 않았으며 intervention이 아닌 representation robustness 분석이다.

## Cache 보완

공식 notebook의 기존 `mgsm` cache는 `zh, de, bn, ja, te` 5개 언어만 포함하고, header 없는 TSV를 기본 `pd.read_csv`로 읽어 언어별 첫 문제가 누락되어 있었다. 기존 cache는 수정하지 않았다.

- English 250개: 정확히 매칭된 기존 GSM8K cache 재사용
- `zh, de, bn, ja, te`: 기존 rows 1..249 재사용 + 누락 row 0만 보완
- `es, fr, ru, sw, th`: 250개 전체 보완
- 신규 forward: 모델당 1,255개
- 논리적 canonical cache: 11 × 250 = 2,750 language-samples

## 분석

- PRIMARY: `mmlu3000_full`
- SECONDARY: `mmlu2400_train`
- cache index 0 제외
- cosine은 주 지표, projection은 보조 지표
- 같은 problem ID를 이용한 paired language comparison
- solution step은 기존 GSM8K 정답 solution의 계산 구조 heuristic이며 실제 모델 reasoning depth가 아니다.
- Step correlation p-value는 `model × direction × language × metric` 안에서 depth별 Benjamini–Hochberg FDR 보정했다.
- cosine 0은 Reasoning/Memory classifier 경계로 해석하지 않는다.

## 출력

- `mgsm_sample_metadata.csv`, `mgsm_problem_mapping.csv`: canonical identity와 GSM8K mapping
- `mgsm_sample_metrics.csv.gz`: sample × language × depth × direction score source of truth
- `language_layer_metrics.csv`: 언어별 평균 layer profile
- `problem_language_variability.csv.gz`: 같은 문제의 11언어 variation
- `english_language_comparison.csv`: English 대비 paired delta와 correlation
- `language_pair_correlations.csv`: language pair별 250문제 correlation
- `step_correlation_by_language_layer.csv`: 언어별 step heuristic association와 FDR
- `summary/model_language_summary.csv`: 모델 단위 curve/paired 요약
- `summary/peak_comparison_summary.csv`: 기존 peak fixed-depth 및 MGSM 특징 depth
- `summary/primary_secondary_robustness.csv`: direction construction robustness

## 제한

MGSM의 11개 번역과 현재 prompt에서의 안정성을 검증하는 것이며 완전한 language invariance를 증명하지 않는다. 낮은 cross-language 표준편차는 alignment가 강하다는 의미가 아니라 언어 variation이 작다는 의미다. 따라서 paired correlation, absolute delta, problem 간 ordering과 함께 해석한다.

Models: {', '.join(models)}  
Generated: {datetime.now().astimezone().isoformat()}
"""
    atomic_write_text(output_dir / "README.md", text)


def validate_tables(tables: dict[str, pd.DataFrame], models: list[str]) -> None:
    for name, frame in tables.items():
        if frame.empty or set(frame["model"]) != set(models):
            raise RuntimeError(f"Model coverage mismatch in {name}")
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"NaN/Inf in {name}")
    if not tables["step"]["spearman_qvalue_fdr_bh"].between(0, 1).all():
        raise RuntimeError("Invalid FDR q-value.")


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    logical_cache_dir = args.logical_cache_dir.resolve()
    liref_dir = args.liref_dir.resolve()
    cross_dir = args.cross_dir.resolve()
    characteristic_dir = args.characteristic_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    sample_metadata, mapping = load_metadata(logical_cache_dir)
    atomic_write_csv(output_dir / "mgsm_sample_metadata.csv", sample_metadata)
    atomic_write_csv(output_dir / "mgsm_problem_mapping.csv", mapping)
    available = discover_models(logical_cache_dir)
    selected = select_models(args.models, available)
    print("MGSM Language Robustness Analysis", flush=True)
    print(f"  models: {selected}", flush=True)
    print("  paired data: 11 languages × 250 problems", flush=True)

    for model_name in selected:
        complete = model_complete(output_dir, model_name)
        if complete and args.skip_existing:
            print(f"[SKIP] {model_name}", flush=True)
            continue
        if complete and not args.overwrite:
            raise FileExistsError(f"Existing output for {model_name}; use --overwrite/--skip-existing")
        config = load_model_config(model_dir, model_name)
        manifest = load_manifest(logical_cache_dir, model_name)
        if (
            manifest["num_hidden_layers"] != config["num_hidden_layers"]
            or manifest["hidden_size"] != config["hidden_size"]
        ):
            raise RuntimeError(f"Manifest/config mismatch: {model_name}")
        print(f"\n[START] {model_name}", flush=True)
        logical_cache = LogicalMGSMCache(manifest, mapping)
        directions = load_directions(
            liref_dir, model_name, config["num_hidden_layers"], config["hidden_size"]
        )
        scores = compute_model_scores(
            model_name=model_name,
            config=config,
            cache=logical_cache,
            directions=directions,
            sample_metadata=sample_metadata,
        )
        analyzed = analyze_scores(scores)
        write_model_results(output_dir, model_name, scores, analyzed)
        print(f"[DONE] {model_name}: sample-depth rows={len(scores)}", flush=True)
        del logical_cache, directions, scores, analyzed
        gc.collect()

    completed_models = sorted(
        model for model in available if model_complete(output_dir, model)
    )
    tables = aggregate_results(output_dir, completed_models)
    validate_tables(tables, completed_models)
    model_summary, peak_summary, robustness = create_summaries(
        tables, completed_models, liref_dir, cross_dir, characteristic_dir
    )
    atomic_write_csv(output_dir / "summary" / "model_language_summary.csv", model_summary)
    atomic_write_csv(output_dir / "summary" / "peak_comparison_summary.csv", peak_summary)
    atomic_write_csv(
        output_dir / "summary" / "primary_secondary_robustness.csv", robustness
    )
    if not args.no_figures:
        plot_results(
            tables,
            completed_models,
            output_dir,
            liref_dir,
            cross_dir,
            characteristic_dir,
        )
    write_readme(output_dir, completed_models)
    print(f"\nCompleted models: {completed_models}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed model results remain resumable.", file=sys.stderr)
        raise
