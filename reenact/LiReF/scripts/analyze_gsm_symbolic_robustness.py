#!/usr/bin/env python3
"""Analyze GSM-Symbolic robustness against existing MMLU-Pro LiReF vectors.

This script never loads a language model and never performs a forward pass. It
uses the immutable GSM8K/GSM-Symbolic activation caches and the previously
computed MMLU-Pro layer-wise LiReF vectors.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch
from datasets import DatasetDict, load_from_disk
from scipy.stats import pearsonr, rankdata, spearmanr

from compute_layerwise_liref import (
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
DEFAULT_CROSS_DIR = REENACT_ROOT / "liref_outputs" / "cross_dataset_projection"
DEFAULT_OUTPUT_DIR = REENACT_ROOT / "liref_outputs" / "gsm_symbolic_robustness"
DEFAULT_NOTEBOOK = (
    REENACT_ROOT / "liref" / "reasoning_representation" / "LiReFs_storing_hs.ipynb"
)

EXPECTED_SYMBOLIC = 5000
EXPECTED_TEMPLATES = 100
EXPECTED_VARIANTS = 50
EXPECTED_GSM8K = 1319
EXPECTED_MODELS = (
    "Meta-Llama-3-8B",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-v0.3",
    "Mistral-7B-Instruct-v0.3",
    "OLMo-2-1124-7B",
    "OLMo-2-1124-7B-Instruct",
    "gemma-2-9b",
    "gemma-2-9b-it",
)
MODEL_PAIRS = (
    ("LLaMA", "Meta-Llama-3-8B", "Meta-Llama-3-8B-Instruct"),
    ("Mistral", "Mistral-7B-v0.3", "Mistral-7B-Instruct-v0.3"),
    ("Gemma", "gemma-2-9b", "gemma-2-9b-it"),
    ("OLMo", "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct"),
)
DIRECTION_SPECS = {
    "mmlu3000_full": {
        "vector_file": "liref_vectors_in_sample.pt",
        "analysis_type": "in_sample",
    },
    "mmlu2400_train": {
        "vector_file": "liref_vectors_heldout.pt",
        "analysis_type": "heldout",
    },
}
DIRECTION_ORDER = tuple(DIRECTION_SPECS)
EXPECTED_PEAKS = {
    "Meta-Llama-3-8B": 12,
    "Meta-Llama-3-8B-Instruct": 8,
    "Mistral-7B-v0.3": 15,
    "Mistral-7B-Instruct-v0.3": 15,
    "OLMo-2-1124-7B": 9,
    "OLMo-2-1124-7B-Instruct": 9,
    "gemma-2-9b": 16,
    "gemma-2-9b-it": 13,
}
EPS = 1e-12
NORM_TOLERANCE = 1e-10
PEAK_TOLERANCE = 1e-12
NUMERIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)
LEXICAL_PATTERN = re.compile(r"[a-z0-9]+")

MODEL_RESULT_FILES = (
    "sample_metrics.csv.gz",
    "template_level_summary.csv",
    "layer_robustness_metrics.csv",
    "original_variant_correlations.csv",
    "permutation_control.csv",
    "bootstrap_ci.csv",
    "variant_characteristic_correlations.csv",
    "manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--liref-dir", type=Path, default=DEFAULT_LIREF_DIR)
    parser.add_argument("--cross-dir", type=Path, default=DEFAULT_CROSS_DIR)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--permutation-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--row-chunk-size", type=int, default=512)
    args = parser.parse_args()
    if args.skip_existing and args.overwrite:
        parser.error("--skip-existing and --overwrite are mutually exclusive")
    if args.bootstrap_replicates < 1 or args.permutation_replicates < 1:
        parser.error("bootstrap/permutation replicates must be positive")
    if args.row_chunk_size < 1:
        parser.error("--row-chunk-size must be positive")
    return args


def atomic_write_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
    os.replace(temporary, path)


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


def file_metadata(path: Path, include_sha256: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    payload: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }
    if include_sha256:
        payload["sha256"] = sha256_file(resolved)
    return payload


def normalize_final_answer(value: str) -> str:
    return re.sub(r"[,\s]", "", value.strip().casefold())


def extract_final_answer(solution: str) -> str:
    matches = re.findall(r"####\s*([^\n\r]+)", str(solution))
    if not matches:
        raise ValueError("Solution does not contain a '####' final answer marker")
    return normalize_final_answer(matches[-1])


def numeric_tokens(text: str) -> list[str]:
    return NUMERIC_PATTERN.findall(str(text))


def lexical_tokens(text: str) -> set[str]:
    return set(LEXICAL_PATTERN.findall(str(text).casefold()))


def lexical_jaccard(left: str, right: str) -> float:
    left_tokens = lexical_tokens(left)
    right_tokens = lexical_tokens(right)
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)


def verify_prompt_convention(notebook_path: Path) -> dict[str, Any]:
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    checks = {
        "gsm8k_branch": "other_running_set_name == 'gsm8k'" in source,
        "gsm_symbolic_branch": "other_running_set_name == 'gsm_symbolic'" in source,
        "question_answer_prompt": "'Q: ' + entry['question'] + \"\\nA: \"" in source,
        "last_token_hidden_state": "[:, -1, :]" in source
        or "[: ,-1 , :]" in source
        or "[:, -1 , :]" in source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Extraction prompt/position verification failed: {checks}")
    return {
        "notebook": str(notebook_path.resolve()),
        "checks": checks,
        "prompt": "Q: {question}\\nA: ",
        "representation": "last input token before answer generation",
    }


def load_dataset_mapping(
    dataset_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    symbolic_path = dataset_dir / "gsm-symbolic_data" / "GSM_symbolic.jsonl"
    gsm8k_path = dataset_dir / "gsm8k" / "main"
    records: list[dict[str, Any]] = []
    with symbolic_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            record = json.loads(line)
            record["symbolic_row_index"] = row_index
            records.append(record)
    if len(records) != EXPECTED_SYMBOLIC:
        raise RuntimeError(f"Expected 5000 GSM-Symbolic rows, found {len(records)}")
    required = {
        "id",
        "instance",
        "question",
        "answer",
        "original_id",
        "original_question",
        "original_answer",
        "canary",
    }
    for row in records:
        missing = required.difference(row)
        if missing:
            raise KeyError(
                f"GSM-Symbolic row {row['symbolic_row_index']} missing {sorted(missing)}"
            )

    gsm_object = load_from_disk(str(gsm8k_path))
    if not isinstance(gsm_object, DatasetDict) or "test" not in gsm_object:
        raise TypeError(f"Expected DatasetDict with test split: {gsm8k_path}")
    gsm_test = gsm_object["test"]
    if len(gsm_test) != EXPECTED_GSM8K:
        raise RuntimeError(f"Expected 1319 GSM8K test rows, found {len(gsm_test)}")

    mapping_rows: list[dict[str, Any]] = []
    characteristic_rows: list[dict[str, Any]] = []
    for record in records:
        original_id = int(record["original_id"])
        if original_id < 0 or original_id >= len(gsm_test):
            raise IndexError(f"Invalid original_id: {original_id}")
        original_numeric = numeric_tokens(record["original_question"])
        variant_numeric = numeric_tokens(record["question"])
        original_final = extract_final_answer(record["original_answer"])
        variant_final = extract_final_answer(record["answer"])
        mapping_rows.append(
            {
                "symbolic_row_index": int(record["symbolic_row_index"]),
                "id": int(record["id"]),
                "instance": int(record["instance"]),
                "original_id": original_id,
                "gsm8k_row_index": original_id,
                "original_question_exact_match": bool(
                    record["original_question"] == gsm_test[original_id]["question"]
                ),
                "original_final_answer_match": bool(
                    original_final == extract_final_answer(gsm_test[original_id]["answer"])
                ),
                "original_answer_full_string_match": bool(
                    record["original_answer"] == gsm_test[original_id]["answer"]
                ),
            }
        )
        characteristic_rows.append(
            {
                "symbolic_row_index": int(record["symbolic_row_index"]),
                "id": int(record["id"]),
                "instance": int(record["instance"]),
                "original_id": original_id,
                "original_question": record["original_question"],
                "variant_question": record["question"],
                "original_final_answer": original_final,
                "variant_final_answer": variant_final,
                "original_question_char_count": len(record["original_question"]),
                "variant_question_char_count": len(record["question"]),
                "question_char_count_delta": len(record["question"])
                - len(record["original_question"]),
                "original_question_whitespace_token_count": len(
                    record["original_question"].split()
                ),
                "variant_question_whitespace_token_count": len(record["question"].split()),
                "whitespace_token_count_delta": len(record["question"].split())
                - len(record["original_question"].split()),
                "original_numeric_token_count": len(original_numeric),
                "variant_numeric_token_count": len(variant_numeric),
                "numeric_token_count_delta": len(variant_numeric) - len(original_numeric),
                "extracted_original_numeric_values": json.dumps(original_numeric),
                "extracted_variant_numeric_values": json.dumps(variant_numeric),
                "final_answer_changed": bool(original_final != variant_final),
                "lexical_jaccard_similarity": lexical_jaccard(
                    record["original_question"], record["question"]
                ),
            }
        )

    mapping = pd.DataFrame(mapping_rows).sort_values("symbolic_row_index").reset_index(drop=True)
    characteristics = (
        pd.DataFrame(characteristic_rows)
        .sort_values("symbolic_row_index")
        .reset_index(drop=True)
    )
    if mapping[["id", "instance"]].duplicated().any():
        raise RuntimeError("(id, instance) is not unique across 5000 rows")
    sizes = mapping.groupby("id", sort=True).size()
    if len(sizes) != EXPECTED_TEMPLATES or not sizes.eq(EXPECTED_VARIANTS).all():
        raise RuntimeError("Expected 100 ids with exactly 50 variants each")
    instance_sets = mapping.groupby("id")["instance"].apply(set)
    expected_instances = set(range(EXPECTED_VARIANTS))
    if not instance_sets.map(lambda values: values == expected_instances).all():
        raise RuntimeError("Each id must contain instances 0..49")
    identity = mapping[["id", "original_id"]].drop_duplicates()
    if len(identity) != 100 or identity["id"].nunique() != 100 or identity["original_id"].nunique() != 100:
        raise RuntimeError("id and original_id are not template-level one-to-one")
    template = (
        identity.sort_values("id")
        .assign(gsm8k_row_index=lambda frame: frame["original_id"])
        .reset_index(drop=True)
    )
    question_matches = int(
        mapping.groupby("id")["original_question_exact_match"].first().sum()
    )
    final_matches = int(mapping.groupby("id")["original_final_answer_match"].first().sum())
    strict_matches = int(
        mapping.groupby("id")["original_answer_full_string_match"].first().sum()
    )
    if (question_matches, final_matches, strict_matches) != (100, 100, 3):
        raise RuntimeError(
            "Unexpected original mapping validation: "
            f"question={question_matches}, final={final_matches}, strict={strict_matches}"
        )

    characteristic_template = characteristics.copy()
    for column in (
        "question_char_count_delta",
        "whitespace_token_count_delta",
        "numeric_token_count_delta",
    ):
        characteristic_template[f"abs_{column}"] = characteristic_template[column].abs()
    characteristic_template = (
        characteristic_template.groupby(["id", "original_id"], as_index=False)
        .agg(
            mean_abs_question_char_count_delta=("abs_question_char_count_delta", "mean"),
            mean_abs_whitespace_token_count_delta=("abs_whitespace_token_count_delta", "mean"),
            mean_abs_numeric_token_count_delta=("abs_numeric_token_count_delta", "mean"),
            mean_lexical_jaccard_similarity=("lexical_jaccard_similarity", "mean"),
            final_answer_changed_rate=("final_answer_changed", "mean"),
        )
        .sort_values("id")
        .reset_index(drop=True)
    )
    validation = {
        "symbolic_samples": len(mapping),
        "unique_ids": int(mapping["id"].nunique()),
        "unique_original_ids": int(mapping["original_id"].nunique()),
        "variants_per_id": sorted(int(value) for value in sizes.unique()),
        "instance_min": int(mapping["instance"].min()),
        "instance_max": int(mapping["instance"].max()),
        "original_id_min": int(mapping["original_id"].min()),
        "original_id_max": int(mapping["original_id"].max()),
        "question_exact_matches": question_matches,
        "final_answer_matches": final_matches,
        "full_answer_strict_matches": strict_matches,
        "symbolic_dataset": file_metadata(symbolic_path, include_sha256=True),
        "gsm8k_dataset_path": str(gsm8k_path.resolve()),
    }
    return mapping, characteristics, characteristic_template, validation


def select_models(requested: list[str], cache_dir: Path, liref_dir: Path) -> list[str]:
    available = []
    for model in EXPECTED_MODELS:
        partial = cache_dir / ".partial" / model
        integrated = cache_dir / f"{model}-base_hs_cache_no_cot_all.pt"
        vectors = liref_dir / model
        if (
            ((partial / "gsm8k.pt").is_file() and (partial / "gsm_symbolic.pt").is_file())
            or integrated.is_file()
        ) and all((vectors / spec["vector_file"]).is_file() for spec in DIRECTION_SPECS.values()):
            available.append(model)
    if requested == ["all"]:
        if set(available) != set(EXPECTED_MODELS):
            raise RuntimeError(
                f"Expected all 8 models; available={available}, missing={sorted(set(EXPECTED_MODELS)-set(available))}"
            )
        return [model for model in EXPECTED_MODELS if model in available]
    if "all" in requested:
        raise ValueError("Use --models all or explicit model names, not both")
    missing = [model for model in requested if model not in available]
    if missing:
        raise FileNotFoundError(f"Missing activation/LiReF inputs for {missing}")
    return requested


class PairedActivationCache:
    def __init__(self, cache_dir: Path, model_name: str, config: dict[str, Any]) -> None:
        partial_dir = cache_dir / ".partial" / model_name
        gsm_path = partial_dir / "gsm8k.pt"
        symbolic_path = partial_dir / "gsm_symbolic.pt"
        self.integrated_payload: dict[str, Any] | None = None
        if gsm_path.is_file() and symbolic_path.is_file():
            self.source_type = "dataset_partial"
            self.gsm_path = gsm_path.resolve()
            self.symbolic_path = symbolic_path.resolve()
            self.gsm8k = torch.load(
                gsm_path, map_location="cpu", mmap=True, weights_only=True
            )
            self.symbolic = torch.load(
                symbolic_path, map_location="cpu", mmap=True, weights_only=True
            )
        else:
            integrated = cache_dir / f"{model_name}-base_hs_cache_no_cot_all.pt"
            if not integrated.is_file():
                raise FileNotFoundError(f"No partial or integrated cache for {model_name}")
            self.source_type = "integrated_fallback"
            self.gsm_path = integrated.resolve()
            self.symbolic_path = integrated.resolve()
            self.integrated_payload = torch.load(
                integrated, map_location="cpu", mmap=True, weights_only=True
            )
            self.gsm8k = self.integrated_payload["gsm8k"]
            self.symbolic = self.integrated_payload["gsm_symbolic"]
        self.num_hidden_layers = int(config["num_hidden_layers"])
        self.hidden_size = int(config["hidden_size"])
        self._validate_structure()

    def _validate_structure(self) -> None:
        expected_keys = list(range(self.num_hidden_layers))
        if sorted(self.gsm8k) != expected_keys or sorted(self.symbolic) != expected_keys:
            raise RuntimeError("Activation cache keys do not match expected convention")
        for cache_index in expected_keys:
            gsm = self.gsm8k[cache_index]
            symbolic = self.symbolic[cache_index]
            if tuple(gsm.shape) != (EXPECTED_GSM8K, self.hidden_size):
                raise RuntimeError(f"GSM8K shape mismatch at cache {cache_index}: {gsm.shape}")
            if tuple(symbolic.shape) != (EXPECTED_SYMBOLIC, self.hidden_size):
                raise RuntimeError(
                    f"GSM-Symbolic shape mismatch at cache {cache_index}: {symbolic.shape}"
                )
            if gsm.dtype != torch.float32 or symbolic.dtype != torch.float32:
                raise RuntimeError(f"Expected float32 cache at index {cache_index}")

    def validate_finite(self, cache_index: int) -> None:
        if not bool(torch.isfinite(self.gsm8k[cache_index]).all()):
            raise RuntimeError(f"GSM8K NaN/Inf at cache index {cache_index}")
        if not bool(torch.isfinite(self.symbolic[cache_index]).all()):
            raise RuntimeError(f"GSM-Symbolic NaN/Inf at cache index {cache_index}")

    def metadata(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "gsm8k": file_metadata(self.gsm_path),
            "gsm_symbolic": file_metadata(self.symbolic_path),
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "keys": list(range(self.num_hidden_layers)),
        }


def load_directions(
    liref_dir: Path, model_name: str, num_hidden_layers: int, hidden_size: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    expected_indices = list(range(1, num_hidden_layers))
    directions: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {}
    for source, spec in DIRECTION_SPECS.items():
        path = liref_dir / model_name / spec["vector_file"]
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        indices = [int(value) for value in payload["cache_indices"].tolist()]
        vectors = payload["normalized_liref"]
        if indices != expected_indices:
            raise RuntimeError(f"LiReF cache index mismatch: {path}")
        if tuple(vectors.shape) != (len(expected_indices), hidden_size):
            raise RuntimeError(f"LiReF shape mismatch: {path}, {vectors.shape}")
        if vectors.dtype != torch.float64:
            raise RuntimeError(f"Expected float64 LiReF vectors: {path}")
        if payload["metadata"]["model_name"] != model_name:
            raise RuntimeError(f"LiReF model metadata mismatch: {path}")
        norm_error = float((vectors.norm(dim=1) - 1.0).abs().max().item())
        if norm_error > NORM_TOLERANCE:
            raise RuntimeError(f"Non-unit LiReF vector in {path}: {norm_error}")
        directions[source] = vectors
        metadata[source] = {
            **file_metadata(path),
            "vector_file": spec["vector_file"],
            "analysis_type": spec["analysis_type"],
            "cache_indices": expected_indices,
            "max_unit_norm_error": norm_error,
        }
    return directions, metadata


def select_peak(frame: pd.DataFrame, value_column: str, mode: str) -> pd.Series:
    if mode == "max":
        target = float(frame[value_column].max())
    elif mode == "min":
        target = float(frame[value_column].min())
    else:
        raise ValueError(mode)
    tied = frame[np.abs(frame[value_column] - target) <= PEAK_TOLERANCE]
    return tied.sort_values("cache_index").iloc[0]


def load_confirmatory_peak(liref_dir: Path, model_name: str) -> tuple[int, dict[str, Any]]:
    path = liref_dir / model_name / "layer_metrics_heldout.csv"
    frame = pd.read_csv(path)
    peak = select_peak(frame, "cosine_gap", "max")
    cache_index = int(peak["cache_index"])
    if cache_index != EXPECTED_PEAKS[model_name]:
        raise RuntimeError(
            f"Held-out peak mismatch for {model_name}: {cache_index} != {EXPECTED_PEAKS[model_name]}"
        )
    return cache_index, {
        **file_metadata(path),
        "cache_index": cache_index,
        "relative_layer_depth": float(peak["relative_layer_depth"]),
        "cosine_gap": float(peak["cosine_gap"]),
        "selection": "maximum held-out cosine_gap; tolerance 1e-12; earliest tie",
    }


def score_hidden_two_directions(
    hidden: torch.Tensor, direction_matrix: torch.Tensor, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    rows = int(hidden.shape[0])
    projections = np.empty((rows, len(DIRECTION_ORDER)), dtype=np.float64)
    cosines = np.empty_like(projections)
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        chunk = hidden[start:stop].to(torch.float64)
        if not bool(torch.isfinite(chunk).all()):
            raise RuntimeError("Activation chunk contains NaN or Inf")
        projection = torch.mm(chunk, direction_matrix.T)
        norms = torch.linalg.vector_norm(chunk, dim=1)
        if bool(torch.any(norms <= EPS)):
            raise RuntimeError("Activation contains a zero/invalid norm")
        cosine = projection / norms[:, None]
        projections[start:stop] = projection.numpy()
        cosines[start:stop] = cosine.numpy()
    if not np.isfinite(projections).all() or not np.isfinite(cosines).all():
        raise RuntimeError("Projection/cosine contains NaN or Inf")
    if cosines.min() < -1.0 - NORM_TOLERANCE or cosines.max() > 1.0 + NORM_TOLERANCE:
        raise RuntimeError("Cosine outside [-1, 1]")
    return cosines, projections


def safe_correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) != EXPECTED_TEMPLATES or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Invalid template-level correlation input")
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    values = {
        "pearson_r": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("Template-level correlation is undefined")
    return values


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    if len(pvalues) == 0 or not np.isfinite(pvalues).all():
        raise RuntimeError("Invalid p-values for FDR")
    order = np.argsort(pvalues, kind="mergesort")
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def apply_fdr(
    frame: pd.DataFrame, group_columns: list[str], p_column: str, q_column: str
) -> pd.DataFrame:
    result = frame.copy()
    result[q_column] = np.nan
    for _, indices in result.groupby(group_columns, sort=False, dropna=False).groups.items():
        index_array = np.asarray(list(indices), dtype=np.int64)
        valid_mask = result.loc[index_array, p_column].notna().to_numpy()
        valid_indices = index_array[valid_mask]
        if len(valid_indices):
            result.loc[valid_indices, q_column] = fdr_bh(
                result.loc[valid_indices, p_column].to_numpy(dtype=np.float64)
            )
    return result


def rowwise_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    result = np.full(len(left), np.nan, dtype=np.float64)
    valid = denominator > EPS
    result[valid] = numerator[valid] / denominator[valid]
    return result


def permutation_statistics(
    x: np.ndarray, y: np.ndarray, permutation_indices: np.ndarray
) -> list[dict[str, Any]]:
    correlations = safe_correlations(x, y)
    permuted_y = y[permutation_indices]
    x_matrix = np.broadcast_to(x, permuted_y.shape)
    null_pearson = rowwise_pearson(x_matrix, permuted_y)
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    null_spearman = rowwise_pearson(
        np.broadcast_to(x_rank, permuted_y.shape), y_rank[permutation_indices]
    )
    null_mae = np.mean(np.abs(x_matrix - permuted_y), axis=1)
    observed = {
        "pearson_r": correlations["pearson_r"],
        "spearman_rho": correlations["spearman_rho"],
        "mean_abs_paired_error": float(np.mean(np.abs(x - y))),
    }
    nulls = {
        "pearson_r": null_pearson,
        "spearman_rho": null_spearman,
        "mean_abs_paired_error": null_mae,
    }
    rows: list[dict[str, Any]] = []
    for statistic, observed_value in observed.items():
        null = nulls[statistic]
        if not np.isfinite(null).all():
            raise RuntimeError(f"Permutation null contains invalid values: {statistic}")
        if statistic == "mean_abs_paired_error":
            extreme = int(np.count_nonzero(null <= observed_value))
            tail = "lower"
        else:
            extreme = int(np.count_nonzero(np.abs(null) >= abs(observed_value)))
            tail = "two_sided_absolute"
        rows.append(
            {
                "statistic": statistic,
                "observed_statistic": observed_value,
                "null_mean": float(np.mean(null)),
                "null_std": float(np.std(null, ddof=1)),
                "null_min": float(np.min(null)),
                "null_max": float(np.max(null)),
                "p_value": float((1 + extreme) / (len(null) + 1)),
                "permutation_replicates": int(len(null)),
                "tail": tail,
            }
        )
    return rows


def bootstrap_statistics(
    template_abs_delta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> list[dict[str, Any]]:
    abs_boot = template_abs_delta[bootstrap_indices].mean(axis=1)
    x_boot = x[bootstrap_indices]
    y_boot = y[bootstrap_indices]
    pearson_boot = rowwise_pearson(x_boot, y_boot)
    spearman_boot = rowwise_pearson(
        rankdata(x_boot, method="average", axis=1),
        rankdata(y_boot, method="average", axis=1),
    )
    correlations = safe_correlations(x, y)
    metrics = {
        "mean_abs_delta_cosine": (float(np.mean(template_abs_delta)), abs_boot),
        "pearson_r": (correlations["pearson_r"], pearson_boot),
        "spearman_rho": (correlations["spearman_rho"], spearman_boot),
    }
    rows: list[dict[str, Any]] = []
    for metric, (observed, values) in metrics.items():
        valid = values[np.isfinite(values)]
        if not len(valid):
            raise RuntimeError(f"All bootstrap replicates invalid: {metric}")
        rows.append(
            {
                "metric": metric,
                "observed_statistic": observed,
                "ci_lower": float(np.percentile(valid, 2.5)),
                "ci_upper": float(np.percentile(valid, 97.5)),
                "ci_method": "percentile_95",
                "requested_replicates": int(len(values)),
                "valid_replicates": int(len(valid)),
                "invalid_replicates": int(len(values) - len(valid)),
            }
        )
    return rows


def icc_one_way(values: np.ndarray) -> dict[str, float]:
    if values.shape != (EXPECTED_TEMPLATES, EXPECTED_VARIANTS):
        raise RuntimeError(f"ICC input shape mismatch: {values.shape}")
    n, k = values.shape
    group_means = values.mean(axis=1)
    grand_mean = float(values.mean())
    ms_between = float(k * np.sum((group_means - grand_mean) ** 2) / (n - 1))
    ms_within = float(np.sum((values - group_means[:, None]) ** 2) / (n * (k - 1)))
    denominator = ms_between + (k - 1) * ms_within
    icc = float((ms_between - ms_within) / denominator) if denominator > EPS else math.nan
    within_variance = float(np.mean(np.var(values, axis=1, ddof=1)))
    between_variance = float(np.var(group_means, ddof=1))
    ratio = (
        float(within_variance / between_variance)
        if between_variance > EPS
        else math.nan
    )
    return {
        "within_template_variance": within_variance,
        "between_template_variance": between_variance,
        "within_between_variance_ratio": ratio,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "icc_1_1": icc,
    }


def characteristic_associations(
    model_name: str,
    source: str,
    cache_index: int,
    relative_depth: float,
    template_characteristics: pd.DataFrame,
    template_abs_delta: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> list[dict[str, Any]]:
    feature_columns = (
        "mean_abs_question_char_count_delta",
        "mean_abs_whitespace_token_count_delta",
        "mean_abs_numeric_token_count_delta",
        "mean_lexical_jaccard_similarity",
        "final_answer_changed_rate",
    )
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        values = template_characteristics[feature].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Invalid template characteristic: {feature}")
        if float(np.ptp(values)) <= EPS:
            rows.append(
                {
                    "model": model_name,
                    "liref_source": source,
                    "cache_index": cache_index,
                    "relative_layer_depth": relative_depth,
                    "characteristic": feature,
                    "spearman_rho": np.nan,
                    "p_value": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "requested_replicates": len(bootstrap_indices),
                    "valid_replicates": 0,
                    "invalid_replicates": len(bootstrap_indices),
                    "n_templates": EXPECTED_TEMPLATES,
                    "analysis_level": "template_n100",
                    "status": "undefined_constant_characteristic",
                }
            )
            continue
        correlation = spearmanr(values, template_abs_delta)
        observed = float(correlation.statistic)
        p_value = float(correlation.pvalue)
        if not math.isfinite(observed) or not math.isfinite(p_value):
            raise RuntimeError(f"Undefined characteristic association: {feature}")
        boot = rowwise_pearson(
            rankdata(values[bootstrap_indices], method="average", axis=1),
            rankdata(template_abs_delta[bootstrap_indices], method="average", axis=1),
        )
        valid = boot[np.isfinite(boot)]
        if not len(valid):
            raise RuntimeError(f"All characteristic bootstrap replicates invalid: {feature}")
        rows.append(
            {
                "model": model_name,
                "liref_source": source,
                "cache_index": cache_index,
                "relative_layer_depth": relative_depth,
                "characteristic": feature,
                "spearman_rho": observed,
                "p_value": p_value,
                "ci_lower": float(np.percentile(valid, 2.5)),
                "ci_upper": float(np.percentile(valid, 97.5)),
                "requested_replicates": len(boot),
                "valid_replicates": len(valid),
                "invalid_replicates": len(boot) - len(valid),
                "n_templates": EXPECTED_TEMPLATES,
                "analysis_level": "template_n100",
                "status": "ok",
            }
        )
    return rows


def analyze_model(
    model_name: str,
    config: dict[str, Any],
    cache: PairedActivationCache,
    directions: dict[str, torch.Tensor],
    confirmatory_peak: int,
    mapping: pd.DataFrame,
    template_characteristics: pd.DataFrame,
    permutation_indices: np.ndarray,
    bootstrap_indices: np.ndarray,
    row_chunk_size: int,
) -> dict[str, pd.DataFrame]:
    template_identity = (
        mapping[["id", "original_id", "gsm8k_row_index"]]
        .drop_duplicates()
        .sort_values("id")
        .reset_index(drop=True)
    )
    if len(template_identity) != EXPECTED_TEMPLATES:
        raise RuntimeError("Template identity table must have 100 rows")
    template_ids = template_identity["id"].to_numpy(dtype=np.int64)
    id_to_position = {int(value): index for index, value in enumerate(template_ids)}
    template_codes = mapping["id"].map(id_to_position).to_numpy(dtype=np.int64)
    if np.any(template_codes < 0):
        raise RuntimeError("Failed to encode a GSM-Symbolic template id")
    groups = [np.flatnonzero(template_codes == index) for index in range(EXPECTED_TEMPLATES)]
    if any(len(group) != EXPECTED_VARIANTS for group in groups):
        raise RuntimeError("Template group does not contain exactly 50 variant rows")
    original_indices = torch.tensor(
        template_identity["gsm8k_row_index"].to_numpy(dtype=np.int64), dtype=torch.long
    )
    template_characteristics = template_identity[["id", "original_id"]].merge(
        template_characteristics,
        on=["id", "original_id"],
        how="left",
        validate="one_to_one",
    )
    if template_characteristics.isna().any().any():
        raise RuntimeError("Template characteristic alignment produced missing values")

    sample_frames: list[pd.DataFrame] = []
    template_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    characteristic_rows: list[dict[str, Any]] = []
    num_hidden_layers = int(config["num_hidden_layers"])

    # Validate the excluded embedding output as part of cache integrity checks.
    cache.validate_finite(0)
    for cache_index in range(1, num_hidden_layers):
        if cache_index == 1 or cache_index % 10 == 0 or cache_index == num_hidden_layers - 1:
            print(
                f"  [{model_name}] cache {cache_index}/{num_hidden_layers - 1}",
                flush=True,
            )
        cache.validate_finite(cache_index)
        gsm_layer = cache.gsm8k[cache_index]
        symbolic_layer = cache.symbolic[cache_index]
        original_hidden = gsm_layer.index_select(0, original_indices)
        direction_matrix = torch.stack(
            [directions[source][cache_index - 1] for source in DIRECTION_ORDER], dim=0
        )
        original_cosines, original_projections = score_hidden_two_directions(
            original_hidden, direction_matrix, row_chunk_size
        )
        variant_cosines, variant_projections = score_hidden_two_directions(
            symbolic_layer, direction_matrix, row_chunk_size
        )
        relative_depth = cache_index / num_hidden_layers

        for source_position, source in enumerate(DIRECTION_ORDER):
            original_cosine = original_cosines[:, source_position]
            variant_cosine = variant_cosines[:, source_position]
            original_projection = original_projections[:, source_position]
            variant_projection = variant_projections[:, source_position]
            row_original_cosine = original_cosine[template_codes]
            row_original_projection = original_projection[template_codes]
            delta_cosine = variant_cosine - row_original_cosine
            delta_projection = variant_projection - row_original_projection

            sample_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "liref_source": source,
                        "cache_index": cache_index,
                        "relative_layer_depth": relative_depth,
                        "symbolic_row_index": mapping["symbolic_row_index"].to_numpy(),
                        "id": mapping["id"].to_numpy(),
                        "instance": mapping["instance"].to_numpy(),
                        "original_id": mapping["original_id"].to_numpy(),
                        "gsm8k_row_index": mapping["gsm8k_row_index"].to_numpy(),
                        "original_cosine": row_original_cosine,
                        "variant_cosine": variant_cosine,
                        "delta_cosine": delta_cosine,
                        "abs_delta_cosine": np.abs(delta_cosine),
                        "original_projection": row_original_projection,
                        "variant_projection": variant_projection,
                        "delta_projection": delta_projection,
                        "abs_delta_projection": np.abs(delta_projection),
                    }
                )
            )

            cosine_matrix = np.stack([variant_cosine[group] for group in groups])
            projection_matrix = np.stack([variant_projection[group] for group in groups])
            delta_matrix = cosine_matrix - original_cosine[:, None]
            projection_delta_matrix = projection_matrix - original_projection[:, None]
            variant_mean = cosine_matrix.mean(axis=1)
            template_delta = delta_matrix.mean(axis=1)
            template_abs_delta = np.abs(delta_matrix).mean(axis=1)
            variant_projection_mean = projection_matrix.mean(axis=1)
            icc_values = icc_one_way(cosine_matrix)

            for template_position, identity in template_identity.iterrows():
                values = cosine_matrix[template_position]
                projections = projection_matrix[template_position]
                deltas = delta_matrix[template_position]
                projection_deltas = projection_delta_matrix[template_position]
                template_rows.append(
                    {
                        "model": model_name,
                        "liref_source": source,
                        "cache_index": cache_index,
                        "relative_layer_depth": relative_depth,
                        "id": int(identity["id"]),
                        "original_id": int(identity["original_id"]),
                        "gsm8k_row_index": int(identity["gsm8k_row_index"]),
                        "n_variants": EXPECTED_VARIANTS,
                        "original_cosine": float(original_cosine[template_position]),
                        "variant_mean_cosine": float(np.mean(values)),
                        "variant_median_cosine": float(np.median(values)),
                        "variant_std_cosine": float(np.std(values, ddof=1)),
                        "variant_min_cosine": float(np.min(values)),
                        "variant_max_cosine": float(np.max(values)),
                        "variant_range_cosine": float(np.ptp(values)),
                        "mean_delta_cosine": float(np.mean(deltas)),
                        "median_delta_cosine": float(np.median(deltas)),
                        "mean_abs_delta_cosine": float(np.mean(np.abs(deltas))),
                        "median_abs_delta_cosine": float(np.median(np.abs(deltas))),
                        "original_projection": float(original_projection[template_position]),
                        "variant_mean_projection": float(np.mean(projections)),
                        "variant_median_projection": float(np.median(projections)),
                        "variant_std_projection": float(np.std(projections, ddof=1)),
                        "variant_min_projection": float(np.min(projections)),
                        "variant_max_projection": float(np.max(projections)),
                        "variant_range_projection": float(np.ptp(projections)),
                        "mean_delta_projection": float(np.mean(projection_deltas)),
                        "median_delta_projection": float(np.median(projection_deltas)),
                        "mean_abs_delta_projection": float(
                            np.mean(np.abs(projection_deltas))
                        ),
                        "median_abs_delta_projection": float(
                            np.median(np.abs(projection_deltas))
                        ),
                    }
                )

            layer_rows.append(
                {
                    "model": model_name,
                    "liref_source": source,
                    "cache_index": cache_index,
                    "relative_layer_depth": relative_depth,
                    "n_templates": EXPECTED_TEMPLATES,
                    "n_variants": EXPECTED_SYMBOLIC,
                    "mean_delta_cosine": float(np.mean(template_delta)),
                    "median_delta_cosine": float(np.median(template_delta)),
                    "std_delta_cosine": float(np.std(template_delta, ddof=1)),
                    "mean_abs_delta_cosine": float(np.mean(template_abs_delta)),
                    "median_abs_delta_cosine": float(np.median(template_abs_delta)),
                    "original_cosine_mean": float(np.mean(original_cosine)),
                    "original_cosine_std": float(np.std(original_cosine, ddof=1)),
                    "variant_cosine_mean": float(np.mean(variant_cosine)),
                    "variant_cosine_std": float(np.std(variant_cosine, ddof=1)),
                    "mean_delta_projection": float(
                        np.mean(projection_delta_matrix.mean(axis=1))
                    ),
                    "mean_abs_delta_projection": float(
                        np.mean(np.abs(projection_delta_matrix).mean(axis=1))
                    ),
                    "is_confirmatory_mmlu_peak": cache_index == confirmatory_peak,
                    **icc_values,
                }
            )

            correlations = safe_correlations(original_cosine, variant_mean)
            correlation_rows.append(
                {
                    "model": model_name,
                    "liref_source": source,
                    "cache_index": cache_index,
                    "relative_layer_depth": relative_depth,
                    "n_templates": EXPECTED_TEMPLATES,
                    "mean_abs_paired_error": float(
                        np.mean(np.abs(original_cosine - variant_mean))
                    ),
                    **correlations,
                }
            )

            for row in permutation_statistics(
                original_cosine, variant_mean, permutation_indices
            ):
                permutation_rows.append(
                    {
                        "model": model_name,
                        "liref_source": source,
                        "cache_index": cache_index,
                        "relative_layer_depth": relative_depth,
                        "n_templates": EXPECTED_TEMPLATES,
                        **row,
                    }
                )
            for row in bootstrap_statistics(
                template_abs_delta, original_cosine, variant_mean, bootstrap_indices
            ):
                bootstrap_rows.append(
                    {
                        "model": model_name,
                        "liref_source": source,
                        "cache_index": cache_index,
                        "relative_layer_depth": relative_depth,
                        "n_templates": EXPECTED_TEMPLATES,
                        **row,
                    }
                )
            characteristic_rows.extend(
                characteristic_associations(
                    model_name,
                    source,
                    cache_index,
                    relative_depth,
                    template_characteristics,
                    template_abs_delta,
                    bootstrap_indices,
                )
            )

        del original_hidden, original_cosines, original_projections
        del variant_cosines, variant_projections, direction_matrix
        gc.collect()

    samples = pd.concat(sample_frames, ignore_index=True)
    templates = pd.DataFrame(template_rows)
    layers = pd.DataFrame(layer_rows)
    correlations = pd.DataFrame(correlation_rows)
    permutations = apply_fdr(
        pd.DataFrame(permutation_rows),
        ["model", "liref_source", "statistic"],
        "p_value",
        "q_value",
    )
    bootstraps = pd.DataFrame(bootstrap_rows)
    characteristic = apply_fdr(
        pd.DataFrame(characteristic_rows),
        ["model", "liref_source", "characteristic"],
        "p_value",
        "q_value",
    )
    return {
        "sample": samples,
        "template": templates,
        "layer": layers,
        "correlation": correlations,
        "permutation": permutations,
        "bootstrap": bootstraps,
        "characteristic": characteristic,
    }


def model_partial_dir(output_dir: Path, model_name: str) -> Path:
    return output_dir / ".partial" / model_name


def expected_model_rows(num_hidden_layers: int) -> dict[str, int]:
    analyzed_layers = num_hidden_layers - 1
    combinations = len(DIRECTION_ORDER) * analyzed_layers
    return {
        "sample": combinations * EXPECTED_SYMBOLIC,
        "template": combinations * EXPECTED_TEMPLATES,
        "layer": combinations,
        "correlation": combinations,
        "permutation": combinations * 3,
        "bootstrap": combinations * 3,
        "characteristic": combinations * 5,
    }


def model_complete(output_dir: Path, model_name: str, config: dict[str, Any]) -> bool:
    root = model_partial_dir(output_dir, model_name)
    if not all((root / filename).is_file() for filename in MODEL_RESULT_FILES):
        return False
    try:
        with (root / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "complete"
        and manifest.get("model") == model_name
        and manifest.get("row_counts")
        == expected_model_rows(int(config["num_hidden_layers"]))
    )


def write_model_results(
    output_dir: Path,
    model_name: str,
    config: dict[str, Any],
    results: dict[str, pd.DataFrame],
    cache_metadata: dict[str, Any],
    direction_metadata: dict[str, Any],
    confirmatory_metadata: dict[str, Any],
) -> None:
    expected = expected_model_rows(int(config["num_hidden_layers"]))
    row_counts = {name: len(frame) for name, frame in results.items()}
    if row_counts != expected:
        raise RuntimeError(f"Unexpected model row counts for {model_name}: {row_counts} != {expected}")
    root = model_partial_dir(output_dir, model_name)
    atomic_write_csv_gz(root / "sample_metrics.csv.gz", results["sample"])
    atomic_write_csv(root / "template_level_summary.csv", results["template"])
    atomic_write_csv(root / "layer_robustness_metrics.csv", results["layer"])
    atomic_write_csv(root / "original_variant_correlations.csv", results["correlation"])
    atomic_write_csv(root / "permutation_control.csv", results["permutation"])
    atomic_write_csv(root / "bootstrap_ci.csv", results["bootstrap"])
    atomic_write_csv(
        root / "variant_characteristic_correlations.csv", results["characteristic"]
    )
    atomic_write_json(
        root / "manifest.json",
        {
            "status": "complete",
            "model": model_name,
            "completed_at": datetime.now().astimezone().isoformat(),
            "row_counts": row_counts,
            "config": {
                "path": str(config["path"]),
                "num_hidden_layers": int(config["num_hidden_layers"]),
                "hidden_size": int(config["hidden_size"]),
                "model_type": config["model_type"],
            },
            "cache": cache_metadata,
            "directions": direction_metadata,
            "confirmatory_layer": confirmatory_metadata,
        },
    )


def validate_model_results(
    model_name: str,
    config: dict[str, Any],
    results: dict[str, pd.DataFrame],
) -> None:
    expected = expected_model_rows(int(config["num_hidden_layers"]))
    actual = {name: len(frame) for name, frame in results.items()}
    if actual != expected:
        raise RuntimeError(f"Model result count mismatch: {actual} != {expected}")
    for name in ("sample", "template", "layer", "correlation", "permutation", "bootstrap"):
        frame = results[name]
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"NaN/Inf in {model_name}/{name}")
    characteristic = results["characteristic"]
    defined = characteristic[characteristic["status"] == "ok"]
    numeric = defined.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"NaN/Inf in defined characteristic rows: {model_name}")
    if not results["permutation"]["q_value"].between(0, 1).all():
        raise RuntimeError(f"Invalid permutation q-value: {model_name}")
    if not defined.empty and not defined["q_value"].between(0, 1).all():
        raise RuntimeError(f"Invalid characteristic q-value: {model_name}")
    sample = results["sample"]
    key_columns = ["liref_source", "cache_index", "symbolic_row_index"]
    if sample.duplicated(key_columns).any():
        raise RuntimeError(f"Duplicate sample metric key: {model_name}")
    consistency = sample.groupby(
        ["liref_source", "cache_index", "original_id"], sort=False
    )["original_cosine"].nunique(dropna=False)
    if not consistency.eq(1).all():
        raise RuntimeError(f"Original cosine is not constant within template: {model_name}")
    if set(sample["cache_index"].unique()) != set(range(1, int(config["num_hidden_layers"]))):
        raise RuntimeError(f"Unexpected analyzed cache indices: {model_name}")


def aggregate_results(
    output_dir: Path,
    models: list[str],
    configs: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    for model in models:
        if not model_complete(output_dir, model, configs[model]):
            raise RuntimeError(f"Incomplete model result: {model}")
    sample_paths = [
        model_partial_dir(output_dir, model) / "sample_metrics.csv.gz" for model in models
    ]
    concatenate_gzip_csv(sample_paths, output_dir / "sample_metrics.csv.gz")
    table_files = {
        "template": "template_level_summary.csv",
        "layer": "layer_robustness_metrics.csv",
        "correlation": "original_variant_correlations.csv",
        "permutation": "permutation_control.csv",
        "bootstrap": "bootstrap_ci.csv",
        "characteristic": "variant_characteristic_correlations.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in table_files.items():
        frame = pd.concat(
            [pd.read_csv(model_partial_dir(output_dir, model) / filename) for model in models],
            ignore_index=True,
        )
        atomic_write_csv(output_dir / filename, frame)
        tables[key] = frame
    return tables


def curve_correlations(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if len(left) < 3 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RuntimeError("Invalid curve correlation input")
    pearson = float(pearsonr(left, right).statistic)
    spearman = float(spearmanr(left, right).statistic)
    if not math.isfinite(pearson) or not math.isfinite(spearman):
        raise RuntimeError("Curve correlation is undefined")
    return pearson, spearman


def create_summaries(
    tables: dict[str, pd.DataFrame],
    models: list[str],
    confirmatory_peaks: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    layer = tables["layer"]
    correlation = tables["correlation"]
    permutation = tables["permutation"]
    bootstrap = tables["bootstrap"]
    model_rows: list[dict[str, Any]] = []
    primary_secondary_rows: list[dict[str, Any]] = []
    base_instruct_rows: list[dict[str, Any]] = []

    for model in models:
        peak_index = confirmatory_peaks[model]
        for source in DIRECTION_ORDER:
            model_layer = layer[
                (layer["model"] == model) & (layer["liref_source"] == source)
            ].sort_values("cache_index")
            confirm = model_layer[model_layer["cache_index"] == peak_index]
            if len(confirm) != 1:
                raise RuntimeError(f"Missing confirmatory layer for {model}/{source}")
            confirm = confirm.iloc[0]
            exploratory = select_peak(model_layer, "mean_abs_delta_cosine", "min")
            confirm_corr = correlation[
                (correlation["model"] == model)
                & (correlation["liref_source"] == source)
                & (correlation["cache_index"] == peak_index)
            ]
            if len(confirm_corr) != 1:
                raise RuntimeError(f"Missing confirmatory correlation: {model}/{source}")
            confirm_corr = confirm_corr.iloc[0]
            confirm_perm = permutation[
                (permutation["model"] == model)
                & (permutation["liref_source"] == source)
                & (permutation["cache_index"] == peak_index)
            ].set_index("statistic")
            confirm_boot = bootstrap[
                (bootstrap["model"] == model)
                & (bootstrap["liref_source"] == source)
                & (bootstrap["cache_index"] == peak_index)
            ].set_index("metric")
            model_rows.append(
                {
                    "model": model,
                    "liref_source": source,
                    "confirmatory_cache_index": peak_index,
                    "confirmatory_relative_layer_depth": float(
                        confirm["relative_layer_depth"]
                    ),
                    "confirmatory_mean_delta_cosine": float(
                        confirm["mean_delta_cosine"]
                    ),
                    "confirmatory_mean_abs_delta_cosine": float(
                        confirm["mean_abs_delta_cosine"]
                    ),
                    "confirmatory_mean_abs_delta_ci_lower": float(
                        confirm_boot.loc["mean_abs_delta_cosine", "ci_lower"]
                    ),
                    "confirmatory_mean_abs_delta_ci_upper": float(
                        confirm_boot.loc["mean_abs_delta_cosine", "ci_upper"]
                    ),
                    "confirmatory_original_variant_pearson": float(
                        confirm_corr["pearson_r"]
                    ),
                    "confirmatory_pearson_ci_lower": float(
                        confirm_boot.loc["pearson_r", "ci_lower"]
                    ),
                    "confirmatory_pearson_ci_upper": float(
                        confirm_boot.loc["pearson_r", "ci_upper"]
                    ),
                    "confirmatory_original_variant_spearman": float(
                        confirm_corr["spearman_rho"]
                    ),
                    "confirmatory_spearman_ci_lower": float(
                        confirm_boot.loc["spearman_rho", "ci_lower"]
                    ),
                    "confirmatory_spearman_ci_upper": float(
                        confirm_boot.loc["spearman_rho", "ci_upper"]
                    ),
                    "confirmatory_icc_1_1": float(confirm["icc_1_1"]),
                    "confirmatory_within_between_ratio": float(
                        confirm["within_between_variance_ratio"]
                    ),
                    "pearson_permutation_p": float(
                        confirm_perm.loc["pearson_r", "p_value"]
                    ),
                    "pearson_permutation_q": float(
                        confirm_perm.loc["pearson_r", "q_value"]
                    ),
                    "spearman_permutation_p": float(
                        confirm_perm.loc["spearman_rho", "p_value"]
                    ),
                    "spearman_permutation_q": float(
                        confirm_perm.loc["spearman_rho", "q_value"]
                    ),
                    "mae_permutation_p": float(
                        confirm_perm.loc["mean_abs_paired_error", "p_value"]
                    ),
                    "mae_permutation_q": float(
                        confirm_perm.loc["mean_abs_paired_error", "q_value"]
                    ),
                    "exploratory_cache_index": int(exploratory["cache_index"]),
                    "exploratory_relative_layer_depth": float(
                        exploratory["relative_layer_depth"]
                    ),
                    "exploratory_mean_abs_delta_cosine": float(
                        exploratory["mean_abs_delta_cosine"]
                    ),
                    "cache_index_distance": abs(
                        int(exploratory["cache_index"]) - peak_index
                    ),
                    "relative_depth_distance": abs(
                        float(exploratory["relative_layer_depth"])
                        - float(confirm["relative_layer_depth"])
                    ),
                }
            )

        primary = layer[
            (layer["model"] == model) & (layer["liref_source"] == "mmlu3000_full")
        ].sort_values("cache_index")
        secondary = layer[
            (layer["model"] == model) & (layer["liref_source"] == "mmlu2400_train")
        ].sort_values("cache_index")
        merged = primary.merge(
            secondary,
            on=["model", "cache_index", "relative_layer_depth"],
            suffixes=("_primary", "_secondary"),
            validate="one_to_one",
        )
        mae_pr, mae_sr = curve_correlations(
            merged["mean_abs_delta_cosine_primary"].to_numpy(),
            merged["mean_abs_delta_cosine_secondary"].to_numpy(),
        )
        primary_corr = correlation[
            (correlation["model"] == model)
            & (correlation["liref_source"] == "mmlu3000_full")
        ].sort_values("cache_index")
        secondary_corr = correlation[
            (correlation["model"] == model)
            & (correlation["liref_source"] == "mmlu2400_train")
        ].sort_values("cache_index")
        pearson_curve_pr, pearson_curve_sr = curve_correlations(
            primary_corr["pearson_r"].to_numpy(), secondary_corr["pearson_r"].to_numpy()
        )
        spearman_curve_pr, spearman_curve_sr = curve_correlations(
            primary_corr["spearman_rho"].to_numpy(),
            secondary_corr["spearman_rho"].to_numpy(),
        )
        primary_exploratory = select_peak(primary, "mean_abs_delta_cosine", "min")
        secondary_exploratory = select_peak(secondary, "mean_abs_delta_cosine", "min")
        confirm_pair = merged[merged["cache_index"] == peak_index].iloc[0]
        primary_secondary_rows.append(
            {
                "model": model,
                "mean_abs_delta_curve_pearson": mae_pr,
                "mean_abs_delta_curve_spearman": mae_sr,
                "pearson_correlation_curve_pearson": pearson_curve_pr,
                "pearson_correlation_curve_spearman": pearson_curve_sr,
                "spearman_correlation_curve_pearson": spearman_curve_pr,
                "spearman_correlation_curve_spearman": spearman_curve_sr,
                "confirmatory_cache_index": peak_index,
                "confirmatory_secondary_minus_primary_mean_abs_delta": float(
                    confirm_pair["mean_abs_delta_cosine_secondary"]
                    - confirm_pair["mean_abs_delta_cosine_primary"]
                ),
                "primary_exploratory_cache_index": int(
                    primary_exploratory["cache_index"]
                ),
                "secondary_exploratory_cache_index": int(
                    secondary_exploratory["cache_index"]
                ),
                "exploratory_cache_index_distance": abs(
                    int(primary_exploratory["cache_index"])
                    - int(secondary_exploratory["cache_index"])
                ),
            }
        )

    for family, base_model, instruct_model in MODEL_PAIRS:
        if base_model not in models or instruct_model not in models:
            continue
        for source in DIRECTION_ORDER:
            base = layer[
                (layer["model"] == base_model) & (layer["liref_source"] == source)
            ].sort_values("cache_index")
            instruct = layer[
                (layer["model"] == instruct_model) & (layer["liref_source"] == source)
            ].sort_values("cache_index")
            merged = base.merge(
                instruct,
                on=["cache_index", "relative_layer_depth"],
                suffixes=("_base", "_instruct"),
                validate="one_to_one",
            )
            base_correlation = correlation[
                (correlation["model"] == base_model)
                & (correlation["liref_source"] == source)
            ][["cache_index", "pearson_r", "spearman_rho"]]
            instruct_correlation = correlation[
                (correlation["model"] == instruct_model)
                & (correlation["liref_source"] == source)
            ][["cache_index", "pearson_r", "spearman_rho"]]
            merged = merged.merge(
                base_correlation,
                on="cache_index",
                how="left",
                validate="one_to_one",
            ).rename(
                columns={
                    "pearson_r": "pearson_r_base",
                    "spearman_rho": "spearman_rho_base",
                }
            )
            merged = merged.merge(
                instruct_correlation,
                on="cache_index",
                how="left",
                validate="one_to_one",
            ).rename(
                columns={
                    "pearson_r": "pearson_r_instruct",
                    "spearman_rho": "spearman_rho_instruct",
                }
            )
            curve_pr, curve_sr = curve_correlations(
                merged["mean_abs_delta_cosine_base"].to_numpy(),
                merged["mean_abs_delta_cosine_instruct"].to_numpy(),
            )
            for row in merged.itertuples(index=False):
                base_instruct_rows.append(
                    {
                        "family": family,
                        "base_model": base_model,
                        "instruct_model": instruct_model,
                        "liref_source": source,
                        "analysis_type": "matched_cache_index",
                        "cache_index": int(row.cache_index),
                        "relative_layer_depth": float(row.relative_layer_depth),
                        "base_mean_abs_delta_cosine": float(
                            row.mean_abs_delta_cosine_base
                        ),
                        "instruct_mean_abs_delta_cosine": float(
                            row.mean_abs_delta_cosine_instruct
                        ),
                        "instruct_minus_base_mean_abs_delta": float(
                            row.mean_abs_delta_cosine_instruct
                            - row.mean_abs_delta_cosine_base
                        ),
                        "base_icc_1_1": float(row.icc_1_1_base),
                        "instruct_icc_1_1": float(row.icc_1_1_instruct),
                        "instruct_minus_base_icc": float(
                            row.icc_1_1_instruct - row.icc_1_1_base
                        ),
                        "base_original_variant_pearson": float(row.pearson_r_base),
                        "instruct_original_variant_pearson": float(
                            row.pearson_r_instruct
                        ),
                        "instruct_minus_base_pearson": float(
                            row.pearson_r_instruct - row.pearson_r_base
                        ),
                        "base_original_variant_spearman": float(
                            row.spearman_rho_base
                        ),
                        "instruct_original_variant_spearman": float(
                            row.spearman_rho_instruct
                        ),
                        "instruct_minus_base_spearman": float(
                            row.spearman_rho_instruct - row.spearman_rho_base
                        ),
                        "mean_abs_delta_curve_pearson": curve_pr,
                        "mean_abs_delta_curve_spearman": curve_sr,
                        "base_confirmatory_cache_index": confirmatory_peaks[base_model],
                        "instruct_confirmatory_cache_index": confirmatory_peaks[
                            instruct_model
                        ],
                    }
                )
    model_summary = pd.DataFrame(model_rows)
    model_summary["descriptive_joint_robustness_score"] = np.nan
    model_summary["descriptive_joint_robustness_rank"] = np.nan
    primary_mask = model_summary["liref_source"].eq("mmlu3000_full")
    primary = model_summary.loc[primary_mask]
    if not primary.empty:
        rank_score = (
            primary["confirmatory_mean_abs_delta_cosine"].rank(
                ascending=True, method="average"
            )
            + primary["confirmatory_original_variant_pearson"].rank(
                ascending=False, method="average"
            )
            + primary["confirmatory_original_variant_spearman"].rank(
                ascending=False, method="average"
            )
            + primary["confirmatory_icc_1_1"].rank(
                ascending=False, method="average"
            )
        ) / 4.0
        model_summary.loc[primary.index, "descriptive_joint_robustness_score"] = rank_score
        model_summary.loc[primary.index, "descriptive_joint_robustness_rank"] = (
            rank_score.rank(ascending=True, method="min")
        )
    return (
        model_summary,
        pd.DataFrame(primary_secondary_rows),
        pd.DataFrame(base_instruct_rows),
    )


def atomic_save_figure(figure: Any, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, format=path.suffix.lstrip("."), dpi=dpi, bbox_inches="tight")
    os.replace(temporary, path)


def create_figures(
    tables: dict[str, pd.DataFrame],
    model_summary: pd.DataFrame,
    base_instruct: pd.DataFrame,
    models: list[str],
    confirmatory_peaks: dict[str, int],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_paths: list[str] = []
    colors = {"mmlu3000_full": "#1769aa", "mmlu2400_train": "#ef6c00"}
    labels = {"mmlu3000_full": "PRIMARY: mmlu3000_full", "mmlu2400_train": "SECONDARY: mmlu2400_train"}

    for model in models:
        layer = tables["layer"][tables["layer"]["model"] == model]
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        for source in DIRECTION_ORDER:
            subset = layer[layer["liref_source"] == source].sort_values("cache_index")
            axis.plot(
                subset["relative_layer_depth"],
                subset["mean_abs_delta_cosine"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=colors[source],
                label=labels[source],
            )
        peak_depth = confirmatory_peaks[model] / int(layer["cache_index"].max() + 1)
        axis.axvline(peak_depth, color="#555555", linestyle="--", linewidth=1, label="MMLU confirmatory peak")
        axis.set(title=model, xlabel="Relative layer depth", ylabel="Mean |ΔCosine| (template-level)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        path = figure_dir / f"{model}_layer_mean_abs_delta.png"
        atomic_save_figure(figure, path)
        plt.close(figure)
        figure_paths.append(str(path.resolve()))

        peak = confirmatory_peaks[model]
        template = tables["template"]
        confirm = template[
            (template["model"] == model)
            & (template["liref_source"] == "mmlu3000_full")
            & (template["cache_index"] == peak)
        ].sort_values("id")
        figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
        axes[0].scatter(
            confirm["original_cosine"], confirm["variant_mean_cosine"], s=22, alpha=0.75
        )
        low = min(confirm["original_cosine"].min(), confirm["variant_mean_cosine"].min())
        high = max(confirm["original_cosine"].max(), confirm["variant_mean_cosine"].max())
        axes[0].plot([low, high], [low, high], linestyle="--", color="#666666", linewidth=1)
        axes[0].set(xlabel="Original cosine", ylabel="Variant mean cosine", title="Original vs variant mean (n=100)")
        axes[1].hist(confirm["variant_std_cosine"], bins=18, color="#1769aa", alpha=0.8)
        axes[1].set(xlabel="Within-template variant SD", ylabel="Templates", title="Variant SD distribution")
        axes[2].hist(confirm["mean_abs_delta_cosine"], bins=18, color="#ef6c00", alpha=0.8)
        axes[2].set(xlabel="Template Mean |ΔCosine|", ylabel="Templates", title="Template change distribution")
        figure.suptitle(f"{model} — PRIMARY confirmatory cache index {peak}")
        figure.tight_layout()
        path = figure_dir / f"{model}_confirmatory_primary_diagnostics.png"
        atomic_save_figure(figure, path)
        plt.close(figure)
        figure_paths.append(str(path.resolve()))

    primary_summary = model_summary[
        model_summary["liref_source"] == "mmlu3000_full"
    ].set_index("model").loc[models]
    positions = np.arange(len(models))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    axis.bar(
        positions - width / 2,
        primary_summary["confirmatory_mean_abs_delta_cosine"],
        width,
        label="MMLU confirmatory peak",
        color="#1769aa",
    )
    axis.bar(
        positions + width / 2,
        primary_summary["exploratory_mean_abs_delta_cosine"],
        width,
        label="Exploratory min-change layer",
        color="#ef6c00",
    )
    axis.set_xticks(positions, models, rotation=35, ha="right")
    axis.set(ylabel="Mean |ΔCosine|", title="Confirmatory vs exploratory layer (PRIMARY)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    path = figure_dir / "all_models_confirmatory_vs_exploratory_primary.png"
    atomic_save_figure(figure, path)
    plt.close(figure)
    figure_paths.append(str(path.resolve()))

    if not base_instruct.empty:
        for family in base_instruct["family"].drop_duplicates():
            figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
            for axis, source in zip(axes, DIRECTION_ORDER):
                subset = base_instruct[
                    (base_instruct["family"] == family)
                    & (base_instruct["liref_source"] == source)
                ].sort_values("cache_index")
                axis.plot(
                    subset["relative_layer_depth"],
                    subset["base_mean_abs_delta_cosine"],
                    marker="o",
                    markersize=3,
                    label="Base",
                )
                axis.plot(
                    subset["relative_layer_depth"],
                    subset["instruct_mean_abs_delta_cosine"],
                    marker="o",
                    markersize=3,
                    label="Instruct/IT",
                )
                axis.set(title=labels[source], xlabel="Relative layer depth")
                axis.grid(alpha=0.25)
                axis.legend()
            axes[0].set_ylabel("Mean |ΔCosine|")
            figure.suptitle(f"{family}: matched cache-index Base vs Instruct")
            figure.tight_layout()
            path = figure_dir / f"{family}_base_vs_instruct_matched_index.png"
            atomic_save_figure(figure, path)
            plt.close(figure)
            figure_paths.append(str(path.resolve()))
    return figure_paths


def count_gzip_csv_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def expected_global_rows(configs: dict[str, dict[str, Any]], models: list[str]) -> dict[str, int]:
    combinations = sum(
        len(DIRECTION_ORDER) * (int(configs[model]["num_hidden_layers"]) - 1)
        for model in models
    )
    return {
        "mapping": EXPECTED_SYMBOLIC,
        "sample": combinations * EXPECTED_SYMBOLIC,
        "template": combinations * EXPECTED_TEMPLATES,
        "layer": combinations,
        "correlation": combinations,
        "permutation": combinations * 3,
        "bootstrap": combinations * 3,
        "characteristic": combinations * 5,
    }


def validate_global_results(
    output_dir: Path,
    liref_dir: Path,
    mapping: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    configs: dict[str, dict[str, Any]],
    models: list[str],
    confirmatory_peaks: dict[str, int],
    bootstrap_replicates: int,
    permutation_replicates: int,
) -> dict[str, Any]:
    expected = expected_global_rows(configs, models)
    actual = {
        "mapping": len(mapping),
        "sample": count_gzip_csv_rows(output_dir / "sample_metrics.csv.gz"),
        **{key: len(frame) for key, frame in tables.items()},
    }
    if actual != expected:
        raise RuntimeError(f"Global row-count mismatch: actual={actual}, expected={expected}")

    expected_combinations = {
        (model, source, cache_index)
        for model in models
        for source in DIRECTION_ORDER
        for cache_index in range(1, int(configs[model]["num_hidden_layers"]))
    }
    for key in ("layer", "correlation"):
        observed = set(
            tables[key][["model", "liref_source", "cache_index"]]
            .itertuples(index=False, name=None)
        )
        if observed != expected_combinations:
            raise RuntimeError(f"Coverage mismatch in {key}")
    for key, multiplier in (("permutation", 3), ("bootstrap", 3), ("characteristic", 5)):
        counts = tables[key].groupby(["model", "liref_source", "cache_index"]).size()
        if len(counts) != len(expected_combinations) or not counts.eq(multiplier).all():
            raise RuntimeError(f"Per-combination row mismatch in {key}")

    if not tables["permutation"]["permutation_replicates"].eq(permutation_replicates).all():
        raise RuntimeError("Permutation replicate count mismatch")
    bootstrap = tables["bootstrap"]
    if not bootstrap["requested_replicates"].eq(bootstrap_replicates).all():
        raise RuntimeError("Bootstrap replicate count mismatch")
    if not (bootstrap["valid_replicates"] + bootstrap["invalid_replicates"]).eq(
        bootstrap["requested_replicates"]
    ).all():
        raise RuntimeError("Bootstrap valid/invalid accounting mismatch")
    if not tables["permutation"]["q_value"].between(0, 1).all():
        raise RuntimeError("Permutation q-values outside [0,1]")

    actual_peaks = {}
    for model in models:
        heldout_peak, _ = load_confirmatory_peak(liref_dir, model)
        actual_peaks[model] = heldout_peak
        if heldout_peak != confirmatory_peaks[model]:
            raise RuntimeError(f"Confirmatory peak changed during validation: {model}")
    return {
        "status": "passed",
        "expected_row_counts": expected,
        "actual_row_counts": actual,
        "model_source_layer_combinations": len(expected_combinations),
        "confirmatory_peaks": actual_peaks,
        "bootstrap_requested_replicates": bootstrap_replicates,
        "bootstrap_valid_replicates_min": int(bootstrap["valid_replicates"].min()),
        "bootstrap_invalid_replicates_max": int(bootstrap["invalid_replicates"].max()),
        "permutation_replicates": permutation_replicates,
    }


def validate_saved_model(
    output_dir: Path, model_name: str, config: dict[str, Any]
) -> None:
    root = model_partial_dir(output_dir, model_name)
    expected = expected_model_rows(int(config["num_hidden_layers"]))
    actual = {
        "sample": count_gzip_csv_rows(root / "sample_metrics.csv.gz"),
        "template": len(pd.read_csv(root / "template_level_summary.csv")),
        "layer": len(pd.read_csv(root / "layer_robustness_metrics.csv")),
        "correlation": len(pd.read_csv(root / "original_variant_correlations.csv")),
        "permutation": len(pd.read_csv(root / "permutation_control.csv")),
        "bootstrap": len(pd.read_csv(root / "bootstrap_ci.csv")),
        "characteristic": len(
            pd.read_csv(root / "variant_characteristic_correlations.csv")
        ),
    }
    if actual != expected:
        raise RuntimeError(
            f"Saved partial validation failed for {model_name}: {actual} != {expected}"
        )


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)):
                return "NA"
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_metadata(
    args: argparse.Namespace,
    models: list[str],
    configs: dict[str, dict[str, Any]],
    dataset_validation: dict[str, Any],
    prompt_validation: dict[str, Any],
    validation: dict[str, Any],
    figure_paths: list[str],
) -> dict[str, Any]:
    manifests = {}
    for model in models:
        with (
            model_partial_dir(args.output_dir, model) / "manifest.json"
        ).open("r", encoding="utf-8") as handle:
            manifests[model] = json.load(handle)
    serializable_args = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "analysis": "GSM-Symbolic Representation Robustness Analysis",
        "causal_intervention": False,
        "model_forward_pass": False,
        "gpu_used": False,
        "created_at": datetime.now().astimezone().isoformat(),
        "execution": {
            "script": file_metadata(SCRIPT_PATH, include_sha256=True),
            "arguments": serializable_args,
            "seed": args.seed,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "dataset": dataset_validation,
        "prompt": prompt_validation,
        "models": {
            model: {
                "config": {
                    "path": str(configs[model]["path"]),
                    "num_hidden_layers": int(configs[model]["num_hidden_layers"]),
                    "hidden_size": int(configs[model]["hidden_size"]),
                    "model_type": configs[model]["model_type"],
                    "analyzed_cache_indices": list(
                        range(1, int(configs[model]["num_hidden_layers"]))
                    ),
                },
                "inputs": manifests[model],
            }
            for model in models
        },
        "definitions": {
            "primary_liref_source": "mmlu3000_full -> liref_vectors_in_sample.pt",
            "secondary_liref_source": "mmlu2400_train -> liref_vectors_heldout.pt",
            "primary_metric": "cosine",
            "secondary_metric": "projection onto unit LiReF",
            "statistical_unit": "100 original templates; 50 variants retained as a cluster",
            "confirmatory_layer": "maximum layer_metrics_heldout.csv cosine_gap; tolerance 1e-12; earliest tie",
            "exploratory_layer": "minimum GSM-Symbolic Mean |Delta Cosine|; tolerance 1e-12; earliest tie",
            "permutation_fdr_scope": "model x liref_source x statistic across layers",
            "bootstrap": "100-template cluster bootstrap; percentile 95% CI",
            "numeric_regex": NUMERIC_PATTERN.pattern,
            "lexical_overlap": "lowercase alphanumeric token-set Jaccard",
        },
        "cross_dataset_projection_reference": {
            "path": str(args.cross_dir.resolve()),
            "used_as_vector_source": False,
            "purpose": "metric/source naming and figure-style reference only",
        },
        "validation": validation,
        "figures": figure_paths,
    }


def write_readme(
    output_dir: Path,
    models: list[str],
    model_summary: pd.DataFrame,
    primary_secondary: pd.DataFrame,
    base_instruct: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    dataset_validation: dict[str, Any],
    prompt_validation: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    primary = model_summary[
        model_summary["liref_source"] == "mmlu3000_full"
    ].sort_values("descriptive_joint_robustness_rank")
    summary_rows = []
    for row in primary.itertuples(index=False):
        summary_rows.append(
            [
                row.model,
                row.confirmatory_cache_index,
                row.confirmatory_mean_abs_delta_cosine,
                f"[{row.confirmatory_mean_abs_delta_ci_lower:.5g}, {row.confirmatory_mean_abs_delta_ci_upper:.5g}]",
                row.confirmatory_original_variant_pearson,
                row.confirmatory_original_variant_spearman,
                row.confirmatory_icc_1_1,
                row.confirmatory_within_between_ratio,
                row.exploratory_cache_index,
                int(row.descriptive_joint_robustness_rank),
            ]
        )
    summary_table = markdown_table(
        [
            "Model",
            "Confirm layer",
            "Mean |ΔC|",
            "95% CI",
            "Pearson",
            "Spearman",
            "ICC(1,1)",
            "Within/Between",
            "Exploratory layer",
            "Joint rank*",
        ],
        summary_rows,
    )

    permutation = tables["permutation"]
    perm_rows = []
    for row in primary.itertuples(index=False):
        subset = permutation[
            (permutation["model"] == row.model)
            & (permutation["liref_source"] == "mmlu3000_full")
            & (permutation["cache_index"] == row.confirmatory_cache_index)
        ]
        for perm in subset.itertuples(index=False):
            perm_rows.append(
                [
                    row.model,
                    perm.statistic,
                    perm.observed_statistic,
                    perm.null_mean,
                    perm.null_std,
                    perm.p_value,
                    perm.q_value,
                ]
            )
    permutation_table = markdown_table(
        ["Model", "Statistic", "Observed", "Null mean", "Null SD", "p", "BH q"],
        perm_rows,
    )

    source_rows = [
        [
            row.model,
            row.mean_abs_delta_curve_pearson,
            row.mean_abs_delta_curve_spearman,
            row.pearson_correlation_curve_pearson,
            row.spearman_correlation_curve_spearman,
            row.confirmatory_secondary_minus_primary_mean_abs_delta,
            row.exploratory_cache_index_distance,
        ]
        for row in primary_secondary.itertuples(index=False)
    ]
    source_table = markdown_table(
        [
            "Model",
            "|ΔC| curve r",
            "|ΔC| curve ρ",
            "Pearson-curve r",
            "Spearman-curve ρ",
            "Secondary−Primary at confirm",
            "Exploratory index distance",
        ],
        source_rows,
    )

    pair_rows: list[list[Any]] = []
    if not base_instruct.empty:
        grouped = base_instruct.groupby(["family", "liref_source"], sort=False)
        for (family, source), frame in grouped:
            pair_rows.append(
                [
                    family,
                    source,
                    frame["mean_abs_delta_curve_pearson"].iloc[0],
                    frame["mean_abs_delta_curve_spearman"].iloc[0],
                    frame["instruct_minus_base_mean_abs_delta"].mean(),
                    frame["instruct_minus_base_pearson"].mean(),
                    frame["instruct_minus_base_spearman"].mean(),
                    frame["instruct_minus_base_icc"].mean(),
                ]
            )
    pair_table = markdown_table(
        [
            "Family",
            "Source",
            "Curve r",
            "Curve ρ",
            "Mean Instruct−Base |ΔC|",
            "Mean ΔPearson",
            "Mean ΔSpearman",
            "Mean ΔICC",
        ],
        pair_rows,
    )

    characteristics = tables["characteristic"]
    confirm_characteristics = []
    for row in primary.itertuples(index=False):
        subset = characteristics[
            (characteristics["model"] == row.model)
            & (characteristics["liref_source"] == "mmlu3000_full")
            & (characteristics["cache_index"] == row.confirmatory_cache_index)
            & (characteristics["status"] == "ok")
        ]
        confirm_characteristics.append(subset)
    confirm_characteristics_frame = pd.concat(confirm_characteristics, ignore_index=True)
    significant_characteristics = confirm_characteristics_frame[
        confirm_characteristics_frame["q_value"] <= 0.05
    ]
    strongest_characteristics = (
        confirm_characteristics_frame.assign(
            absolute_rho=lambda frame: frame["spearman_rho"].abs()
        )
        .sort_values("absolute_rho", ascending=False)
        .head(10)
    )
    characteristic_table = markdown_table(
        ["Model", "Characteristic", "ρ", "95% CI", "p", "BH q"],
        [
            [
                row.model,
                row.characteristic,
                row.spearman_rho,
                f"[{row.ci_lower:.5g}, {row.ci_upper:.5g}]",
                row.p_value,
                row.q_value,
            ]
            for row in strongest_characteristics.itertuples(index=False)
        ],
    )

    supported_models = primary[
        (primary["pearson_permutation_q"] <= 0.05)
        & (primary["spearman_permutation_q"] <= 0.05)
        & (primary["mae_permutation_q"] <= 0.05)
        & (primary["confirmatory_original_variant_pearson"] > 0)
        & (primary["confirmatory_original_variant_spearman"] > 0)
    ]["model"].tolist()
    best_rank = float(primary["descriptive_joint_robustness_rank"].min())
    worst_rank = float(primary["descriptive_joint_robustness_rank"].max())
    strongest_models = primary[
        primary["descriptive_joint_robustness_rank"].eq(best_rank)
    ]["model"].tolist()
    weakest_models = primary[
        primary["descriptive_joint_robustness_rank"].eq(worst_rank)
    ]["model"].tolist()
    strongest = ", ".join(strongest_models)
    weakest = ", ".join(weakest_models)
    expected = validation["expected_row_counts"]
    actual = validation["actual_row_counts"]
    row_table = markdown_table(
        ["Output", "Expected", "Actual"],
        [[key, expected[key], actual[key]] for key in expected],
    )

    readme = f"""# GSM-Symbolic Representation Robustness Analysis

## 1. Research Question

동일한 문제 해결 구조를 유지하면서 이름·개체·수치를 바꾼 GSM-Symbolic variant에서도 MMLU-Pro 기반 LiReF alignment가 얼마나 유지되는지를 분석했다. 이 결과는 **표현 안정성(representation robustness)** 분석이며 인과 개입 결과가 아니다.

## 2. Why GSM-Symbolic

GSM-Symbolic은 100개 GSM8K 원문 각각에 50개 변형을 제공하므로, 원문과 구조 보존 변형 사이의 LiReF alignment 변화량을 paired 방식으로 볼 수 있다. 5,000개 variant는 서로 독립인 5,000개 문제로 취급하지 않았다.

## 3. Dataset Structure

- GSM-Symbolic: {dataset_validation['symbolic_samples']:,} rows
- Unique templates: {dataset_validation['unique_ids']}
- Variants per template: {dataset_validation['variants_per_id']}
- `instance`: {dataset_validation['instance_min']}–{dataset_validation['instance_max']}
- `original_id`: {dataset_validation['original_id_min']}–{dataset_validation['original_id_max']}

## 4. Original–Variant Mapping

`symbolic_row_index`는 JSONL 실제 행, `gsm8k_row_index`는 `original_id`이다. `id`와 `original_id`는 혼동하지 않았다. Question exact match는 {dataset_validation['question_exact_matches']}/100, 정규화한 `####` final answer match는 {dataset_validation['final_answer_matches']}/100, full solution strict match는 {dataset_validation['full_answer_strict_matches']}/100이었다. 마지막 수치는 문장부호·줄바꿈 차이 때문에 오류로 보지 않는다.

## 5. Hidden-State Cache

각 모델의 `hidden_states/.partial/<model>/gsm8k.pt`와 `gsm_symbolic.pt`를 우선 사용했다. 모델별·layer별로 CPU mmap 로드했고 model weight는 로드하지 않았다. 각 모델의 정확한 cache 경로와 fallback 여부는 `metadata.json`에 있다.

## 6. Representation Measurement Point

검증한 prompt는 `{prompt_validation['prompt']}`이며 hidden state는 **답변 생성 전 마지막 입력 token**에서 추출된 값이다. 본 분석은 생성된 풀이 과정이나 정답 정확도를 측정하지 않으며, 답변 생성 직전 마지막 prompt token representation의 LiReF alignment를 분석한다.

## 7. LiReF Sources

- PRIMARY `mmlu3000_full`: `liref_vectors_in_sample.pt`
- SECONDARY `mmlu2400_train`: `liref_vectors_heldout.pt`

두 방향 모두 기존 MMLU-Pro LiReF를 재사용했다. GSM-Symbolic에서 새 LiReF를 만들지 않았다.

## 8. Statistical Unit

주 추론 단위는 100개 original template이다. 각 template의 50개 variant를 먼저 aggregate한 뒤 100개 template를 동일 가중치로 분석했다. Raw 5,000행은 추적 가능한 기술통계로만 저장했다.

## 9. Analysis Method

각 layer에서 original과 variant의 cosine 및 unit LiReF projection을 계산했다. `cache index 0`은 embedding output이므로 제외했다. LLaMA/Mistral/OLMo는 1–31, Gemma는 1–41을 분석했다. Projection은 hidden-state magnitude를 포함하므로 모델 간 절댓값 비교에 사용하지 않는다.

## 10. Main Robustness Metric

주 지표는 template별 50개 `|ΔCosine|`의 평균을 다시 100개 template에 평균한 **Mean |ΔCosine|**이다. 작을수록 변형에 따른 alignment 변화가 작지만, 이 값 하나만으로 robustness를 결론내리지 않았다.

## 11. Template-Level Correlation

각 layer에서 100개 `original_cosine`과 100개 `variant_mean_cosine` 사이 Pearson 및 Spearman을 계산했다. Original을 50번 반복한 5,000쌍 상관은 주 분석에 사용하지 않았다.

## 12. Within vs Between Variation

Within-template variance는 template별 50개 variant cosine의 표본분산 평균이고, between-template variance는 100개 variant mean의 표본분산이다. 분모가 `{EPS}` 이하일 때 ratio는 정의하지 않는다.

## 13. ICC(1,1)

`MS_between = k/(n-1) Σ(mean_i-grand_mean)^2`, `MS_within = 1/[n(k-1)] ΣΣ(x_ij-mean_i)^2`, `ICC(1,1) = (MS_between-MS_within)/(MS_between+(k-1)MS_within)`를 사용했다. `n=100`, `k=50`이며 음수 ICC도 truncate하지 않는다.

## 14. Permutation Control

Original–variant template pairing을 {validation['permutation_replicates']:,}회 무작위 shuffle한 null과 Pearson, Spearman, paired MAE를 비교했다. p-value에는 `(1+extreme)/(B+1)`을 사용했고 `model × source × statistic` 범위에서 layer별 BH-FDR을 적용했다.

{permutation_table}

## 15. Cluster Bootstrap

100개 template를 replacement 방식으로 {validation['bootstrap_requested_replicates']:,}회 resample하고 각 template의 50개 variant cluster를 함께 유지했다. Mean |ΔCosine|, Pearson, Spearman에 percentile 95% CI를 계산했으며 valid replicate 최솟값은 {validation['bootstrap_valid_replicates_min']:,}, invalid replicate 최댓값은 {validation['bootstrap_invalid_replicates_max']:,}였다. 전체 수치는 `bootstrap_ci.csv`에 기록했다.

## 16. Confirmatory MMLU Peak Result

Confirmatory layer는 GSM-Symbolic을 보지 않고 기존 held-out MMLU `cosine_gap` 최대 layer로 고정했다. 아래 표는 PRIMARY 결과다.

{summary_table}

\* Joint rank는 confirmatory Mean |ΔC|가 작고 Pearson·Spearman·ICC가 큰 방향의 네 순위를 평균한 **기술통계용 순위**다. 유의성 검정이나 절대적인 모델 우열이 아니다.

## 17. Exploratory Min-Change Layer

각 모델/source에서 Mean |ΔCosine|가 최소인 layer를 탐색적으로 기록했다. 이는 `Exploratory GSM-Symbolic robustness layer`이며 best reasoning layer, causal layer 또는 가장 중요한 reasoning layer가 아니다.

## 18. Primary vs Secondary

{source_table}

## 19. Base vs Instruct

각 architecture family에서 동일 cache index끼리 비교했다. 아래 차이는 layer 전체의 기술통계 평균이며 raw projection magnitude는 비교하지 않았다.

{pair_table}

## 20. Variant Characteristics

공식 perturbation type label을 만들지 않았다. 문자·공백 token·numeric token 수 변화, `####` final answer 변화, lowercase alphanumeric token-set Jaccard만 계산했다. Numeric regex는 `{NUMERIC_PATTERN.pattern}`이다. Template별 characteristic 평균과 Mean |ΔCosine| 사이 exploratory Spearman을 계산하고 source/model/characteristic 내 layer BH-FDR을 적용했다.

Confirmatory PRIMARY에서 q≤0.05인 characteristic association은 {len(significant_characteristics)}개였다. 절대 ρ가 큰 10개는 다음과 같다.

{characteristic_table}

## 21. Main Findings

- Confirmatory PRIMARY에서 Pearson·Spearman·paired-MAE permutation 조건을 모두 만족한 모델은 {len(supported_models)}/{len(models)}개였다: {', '.join(supported_models) if supported_models else '없음'}.
- 네 지표의 기술통계 joint rank 기준 공동 최상위 모델은 `{strongest}`, 상대적으로 가장 낮은 모델은 `{weakest}`였다.
- Mean |ΔCosine|, 상관, ICC와 permutation을 함께 보아야 하며 모델별 수치는 위 표에 모두 공개했다.

## 22. Model-Specific Exceptions

`{weakest}`가 기술통계 joint rank에서 가장 낮았다는 것은 이 8개 모델·고정 confirmatory layer·현재 지표 안에서의 상대 비교일 뿐이다. 개별 모델의 낮은 correlation, 낮은 ICC 또는 큰 Mean |ΔCosine|는 서로 다른 현상일 수 있으므로 `model_summary.csv`와 layer curve를 함께 확인해야 한다.

## 23. What We Can Conclude

실제 pairing이 null pairing보다 우세한 모델과 layer가 어디인지, 구조 보존 변형에 대한 LiReF cosine 변화량과 template 간 순서 보존 정도가 모델·layer별로 어떻게 달라지는지는 말할 수 있다. 안정성 정도는 model-dependent하게 보고한다.

## 24. What We Cannot Conclude

이 분석만으로 모델이 진짜 reasoning을 한다거나, LiReF가 완전히 invariant하다거나, 숫자 변화와 독립적이라거나, 동일 reasoning algorithm을 사용했다거나, 인과 효과가 확인됐다고 주장할 수 없다. Cosine 0도 Reasoning/Memory의 절대 경계가 아니다.

## 25. Validation Checks

모든 dataset mapping, cache key/shape/dtype/finite value, LiReF index/dimension/unit norm, confirmatory peak, original cosine의 template 내 동일성, replicate accounting 및 행 수 검증을 통과했다.

{row_table}

## 26. Output Files

- `gsm_symbolic_mapping.csv`: 5,000개 symbolic–original 행 mapping
- `variant_characteristics.csv`: 객관적으로 계산한 variant metadata
- `sample_metrics.csv.gz`: raw sample-level 기술통계 및 역추적 key
- `template_level_summary.csv`: 50 variants를 묶은 template-level summary
- `layer_robustness_metrics.csv`: Mean |ΔCosine|, variance, ICC 등
- `original_variant_correlations.csv`: n=100 template-level correlation
- `permutation_control.csv`: 3개 통계의 null/p/q
- `bootstrap_ci.csv`: cluster-bootstrap percentile CI
- `variant_characteristic_correlations.csv`: exploratory characteristic association
- `model_summary.csv`, `primary_secondary_comparison.csv`, `base_instruct_comparison.csv`
- `.partial/<model>/`: 모델별 atomic/resume output
- `figures/`: 요청된 layer curve, confirmatory diagnostic 및 모델 비교 그림
- `metadata.json`: 입력 provenance, 환경, 정의 및 검증 결과
"""
    atomic_write_text(output_dir / "README.md", readme)


def main() -> int:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.liref_dir = args.liref_dir.resolve()
    args.cross_dir = args.cross_dir.resolve()
    args.notebook = args.notebook.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.skip_existing and not args.overwrite:
            raise FileExistsError(
                f"Output is not empty: {args.output_dir}. Use --skip-existing or --overwrite."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("GSM-Symbolic Representation Robustness Analysis", flush=True)
    print("  CPU only: model loading=False, hidden-state extraction=False, GPU=False", flush=True)
    print(f"  output: {args.output_dir}", flush=True)

    prompt_validation = verify_prompt_convention(args.notebook)
    mapping, characteristics, template_characteristics, dataset_validation = (
        load_dataset_mapping(args.dataset_dir)
    )
    print(
        "  mapping: 5000 rows, 100 templates, 50 variants/template; "
        f"question/final/strict={dataset_validation['question_exact_matches']}/"
        f"{dataset_validation['final_answer_matches']}/"
        f"{dataset_validation['full_answer_strict_matches']}",
        flush=True,
    )
    atomic_write_csv(args.output_dir / "gsm_symbolic_mapping.csv", mapping)
    atomic_write_csv(args.output_dir / "variant_characteristics.csv", characteristics)

    models = select_models(args.models, args.cache_dir, args.liref_dir)
    configs = {model: load_model_config(args.model_dir, model) for model in models}
    confirmatory_peaks: dict[str, int] = {}
    rng_permutation = np.random.default_rng(args.seed)
    rng_bootstrap = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    permutation_indices = np.stack(
        [rng_permutation.permutation(EXPECTED_TEMPLATES) for _ in range(args.permutation_replicates)]
    )
    bootstrap_indices = rng_bootstrap.integers(
        0,
        EXPECTED_TEMPLATES,
        size=(args.bootstrap_replicates, EXPECTED_TEMPLATES),
        dtype=np.int64,
    )

    for model in models:
        peak, peak_metadata = load_confirmatory_peak(args.liref_dir, model)
        confirmatory_peaks[model] = peak
        if args.skip_existing and model_complete(args.output_dir, model, configs[model]):
            validate_saved_model(args.output_dir, model, configs[model])
            print(f"[SKIP verified complete] {model}", flush=True)
            continue
        print(f"\n[START] {model}; fixed confirmatory cache index={peak}", flush=True)
        cache = PairedActivationCache(args.cache_dir, model, configs[model])
        if cache.source_type == "integrated_fallback":
            print(f"  [FALLBACK] {model}: using integrated cache", flush=True)
        else:
            print(f"  [CACHE] {model}: using dataset partial caches", flush=True)
        directions, direction_metadata = load_directions(
            args.liref_dir,
            model,
            int(configs[model]["num_hidden_layers"]),
            int(configs[model]["hidden_size"]),
        )
        results = analyze_model(
            model,
            configs[model],
            cache,
            directions,
            peak,
            mapping,
            template_characteristics,
            permutation_indices,
            bootstrap_indices,
            args.row_chunk_size,
        )
        validate_model_results(model, configs[model], results)
        write_model_results(
            args.output_dir,
            model,
            configs[model],
            results,
            cache.metadata(),
            direction_metadata,
            peak_metadata,
        )
        validate_saved_model(args.output_dir, model, configs[model])
        print(f"[DONE] {model}: {expected_model_rows(int(configs[model]['num_hidden_layers']))}", flush=True)
        del results, directions, cache
        gc.collect()

    tables = aggregate_results(args.output_dir, models, configs)
    model_summary, primary_secondary, base_instruct = create_summaries(
        tables, models, confirmatory_peaks
    )
    atomic_write_csv(args.output_dir / "model_summary.csv", model_summary)
    atomic_write_csv(
        args.output_dir / "primary_secondary_comparison.csv", primary_secondary
    )
    atomic_write_csv(args.output_dir / "base_instruct_comparison.csv", base_instruct)
    validation = validate_global_results(
        args.output_dir,
        args.liref_dir,
        mapping,
        tables,
        configs,
        models,
        confirmatory_peaks,
        args.bootstrap_replicates,
        args.permutation_replicates,
    )
    figure_paths = []
    if not args.no_figures:
        figure_paths = create_figures(
            tables,
            model_summary,
            base_instruct,
            models,
            confirmatory_peaks,
            args.output_dir,
        )
    metadata = build_metadata(
        args,
        models,
        configs,
        dataset_validation,
        prompt_validation,
        validation,
        figure_paths,
    )
    atomic_write_json(args.output_dir / "metadata.json", metadata)
    write_readme(
        args.output_dir,
        models,
        model_summary,
        primary_secondary,
        base_instruct,
        tables,
        dataset_validation,
        prompt_validation,
        validation,
    )
    print(f"\n[COMPLETE] validation={validation['status']}", flush=True)
    print(f"  rows: {validation['actual_row_counts']}", flush=True)
    print(f"  output: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
