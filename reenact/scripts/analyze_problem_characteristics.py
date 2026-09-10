#!/usr/bin/env python3
"""Analyze GSM8K/PopQA characteristics against existing LiReF alignments.

This script treats ``cross_dataset_projection/<model>/sample_metrics.csv.gz``
as the source of truth for cosine/projection values. It does not load a model,
activation cache, or LiReF vector and does not recompute any alignment score.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from compute_layerwise_liref import atomic_write_csv, atomic_write_text


SCRIPT_PATH = Path(__file__).resolve()
REENACT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_DATASET_DIR = REENACT_ROOT / "liref" / "dataset"
DEFAULT_DATASET_ANALYSIS_DIR = REENACT_ROOT / "liref_outputs" / "dataset_analysis"
DEFAULT_LAYERWISE_DIR = REENACT_ROOT / "liref_outputs" / "layerwise_liref"
DEFAULT_CROSS_DIR = REENACT_ROOT / "liref_outputs" / "cross_dataset_projection"
DEFAULT_OUTPUT_DIR = REENACT_ROOT / "liref_outputs" / "problem_characteristics"

DIRECTION_TYPES = ("mmlu3000_full", "mmlu2400_train")
DIRECTION_TO_MMLU_ANALYSIS = {
    "mmlu3000_full": "in_sample",
    "mmlu2400_train": "heldout",
}
EXPECTED_GSM8K = 1319
EXPECTED_POPQA = 14267
EXPECTED_GSM_STEP_COUNTS = {
    "1-step": 3,
    "2-step": 333,
    "3-step": 381,
    "4-step 이상": 602,
}
EXPECTED_POPULARITY_COUNTS = {"Low": 4760, "Medium": 4752, "High": 4755}
POPULARITY_LOW_MAX = 375
POPULARITY_MEDIUM_MAX = 2879
RELATION_FIGURE_MIN_N = 100
RELATION_FIGURE_TOP_K = 8
RELATION_CORRELATION_MIN_N = 100
PEAK_TOLERANCE = 1e-12

TABLE_PATHS = {
    "gsm_step_group": Path("gsm8k/step_group_layer_metrics.csv"),
    "gsm_step_correlation": Path("gsm8k/step_correlation_by_layer.csv"),
    "pop_popularity_group": Path("popqa/popularity_group_layer_metrics.csv"),
    "pop_popularity_correlation": Path("popqa/popularity_correlation_by_layer.csv"),
    "pop_relation": Path("popqa/relation_layer_metrics.csv"),
    "pop_relation_popularity_correlation": Path(
        "popqa/popularity_correlation_by_relation.csv"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GSM8K/PopQA characteristics using existing LiReF scores."
    )
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--dataset-analysis-dir", type=Path, default=DEFAULT_DATASET_ANALYSIS_DIR
    )
    parser.add_argument("--layerwise-dir", type=Path, default=DEFAULT_LAYERWISE_DIR)
    parser.add_argument("--cross-dir", type=Path, default=DEFAULT_CROSS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def discover_models(cross_dir: Path) -> list[str]:
    models = sorted(
        path.name
        for path in cross_dir.iterdir()
        if path.is_dir() and (path / "sample_metrics.csv.gz").is_file()
    )
    if not models:
        raise FileNotFoundError(f"No model sample metrics found under {cross_dir}")
    return models


def select_models(requested: list[str], available: list[str]) -> list[str]:
    if requested == ["all"]:
        return available
    if "all" in requested:
        raise ValueError("Use either '--models all' or explicit model names, not both.")
    missing = [model for model in requested if model not in available]
    if missing:
        raise KeyError(f"Requested models are unavailable: {missing}; available={available}")
    return requested


_CALCULATION_VERBS = re.compile(
    r"\b(?:add|subtract|multiply|divide|solve|equals)\b", re.IGNORECASE
)
_ARITHMETIC_EXPRESSION = re.compile(r"\d(?:[\d,.]*\d)?\s*[+\-*/×÷]\s*\d")
_ANNOTATED_CALCULATION = re.compile(r"<<.*?=.*?>>")


def solution_calculation_steps(answer: str) -> int:
    """Reproduce the documented GSM8K solution-line calculation heuristic."""
    solution = answer.split("####", 1)[0]
    step_count = 0
    for line in solution.splitlines():
        blocks = _ANNOTATED_CALCULATION.findall(line)
        if blocks:
            step_count += len(blocks)
        elif (
            "=" in line
            or _ARITHMETIC_EXPRESSION.search(line)
            or _CALCULATION_VERBS.search(line)
        ):
            step_count += 1
    return max(step_count, 1)


def gsm_step_group(step_count: int) -> str:
    if step_count <= 1:
        return "1-step"
    if step_count == 2:
        return "2-step"
    if step_count == 3:
        return "3-step"
    return "4-step 이상"


def popqa_popularity_group(value: int) -> str:
    if value <= POPULARITY_LOW_MAX:
        return "Low"
    if value <= POPULARITY_MEDIUM_MAX:
        return "Medium"
    return "High"


def load_and_validate_metadata(
    cross_dir: Path,
    dataset_analysis_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    gsm_path = cross_dir / "gsm8k_sample_metadata.csv"
    pop_path = cross_dir / "popqa_sample_metadata.csv"
    if not gsm_path.is_file() or not pop_path.is_file():
        raise FileNotFoundError(f"Cross-dataset metadata is missing: {gsm_path} or {pop_path}")

    gsm = pd.read_csv(gsm_path, dtype={"sample_id": "string"})
    pop = pd.read_csv(pop_path, dtype={"sample_id": "string"})
    if len(gsm) != EXPECTED_GSM8K or len(pop) != EXPECTED_POPQA:
        raise RuntimeError(f"Unexpected metadata counts: GSM8K={len(gsm)}, PopQA={len(pop)}")
    if gsm["row_index"].tolist() != list(range(EXPECTED_GSM8K)):
        raise RuntimeError("GSM8K metadata row_index is not sequential 0..1318.")
    if pop["row_index"].tolist() != list(range(EXPECTED_POPQA)):
        raise RuntimeError("PopQA metadata row_index is not sequential 0..14266.")

    expected_gsm_ids = pd.Series(
        [f"gsm8k_test_{index}" for index in range(EXPECTED_GSM8K)], dtype="string"
    )
    expected_pop_ids = "popqa_" + pop["id"].astype(str)
    if not gsm["sample_id"].reset_index(drop=True).equals(expected_gsm_ids):
        raise RuntimeError("GSM8K stable sample IDs do not match row_index.")
    if not (pop["sample_id"].astype(str).to_numpy() == expected_pop_ids.to_numpy()).all():
        raise RuntimeError("PopQA namespaced sample IDs do not match actual id values.")
    if gsm["sample_id"].nunique() != EXPECTED_GSM8K or pop["sample_id"].nunique() != EXPECTED_POPQA:
        raise RuntimeError("Metadata sample IDs are not unique.")

    gsm["solution_calculation_steps"] = gsm["answer"].map(solution_calculation_steps)
    gsm["step_group"] = gsm["solution_calculation_steps"].map(gsm_step_group)
    gsm["question_char_length"] = gsm["question"].str.len()
    gsm["solution_char_length"] = gsm["answer"].str.split("####", n=1).str[0].str.len()
    actual_step_counts = gsm["step_group"].value_counts().to_dict()
    if actual_step_counts != EXPECTED_GSM_STEP_COUNTS:
        raise RuntimeError(
            f"GSM8K heuristic distribution mismatch: {actual_step_counts}; "
            f"expected {EXPECTED_GSM_STEP_COUNTS}"
        )

    documented_steps = dataset_analysis_dir / "gsm8k" / "reasoning_step_stats.csv"
    if not documented_steps.is_file():
        raise FileNotFoundError(f"Documented GSM8K step statistics missing: {documented_steps}")
    step_stats = pd.read_csv(documented_steps)
    documented_test = step_stats[
        (step_stats["split"] == "test")
        & (step_stats["distribution_type"] == "grouped_estimated_step")
    ]
    documented_counts = dict(
        zip(documented_test["label"], documented_test["sample_count"].astype(int))
    )
    if documented_counts != EXPECTED_GSM_STEP_COUNTS:
        raise RuntimeError(
            f"Documented GSM8K test distribution mismatch: {documented_counts}"
        )

    if pop["s_pop"].isna().any() or not np.isfinite(pop["s_pop"].to_numpy(dtype=float)).all():
        raise RuntimeError("PopQA s_pop contains missing/invalid values.")
    pop["s_pop"] = pop["s_pop"].astype(np.int64)
    pop["popularity_group"] = pop["s_pop"].map(popqa_popularity_group)
    actual_popularity_counts = pop["popularity_group"].value_counts().to_dict()
    if actual_popularity_counts != EXPECTED_POPULARITY_COUNTS:
        raise RuntimeError(
            f"PopQA popularity distribution mismatch: {actual_popularity_counts}; "
            f"expected {EXPECTED_POPULARITY_COUNTS}"
        )

    documented_popularity = dataset_analysis_dir / "popQa" / "popularity_stats.csv"
    if not documented_popularity.is_file():
        raise FileNotFoundError(
            f"Documented PopQA popularity statistics missing: {documented_popularity}"
        )
    popularity_stats = pd.read_csv(documented_popularity)
    documented_groups = popularity_stats[
        (popularity_stats["record_type"] == "tertile_group")
        & (popularity_stats["entity_side"] == "subject")
    ]
    label_map = {"low": "Low", "medium": "Medium", "high": "High"}
    documented_popularity_counts = {
        label_map[str(row.label)]: int(row.sample_count)
        for row in documented_groups.itertuples()
    }
    if documented_popularity_counts != EXPECTED_POPULARITY_COUNTS:
        raise RuntimeError(
            f"Documented PopQA tertile counts mismatch: {documented_popularity_counts}"
        )

    relation_counts = pop["prop"].value_counts().sort_values(ascending=False)
    if len(relation_counts) != 16 or int(relation_counts.sum()) != EXPECTED_POPQA:
        raise RuntimeError(f"Unexpected PopQA relation distribution: {relation_counts.to_dict()}")
    eligible_relations = relation_counts[
        relation_counts >= RELATION_CORRELATION_MIN_N
    ].index.tolist()
    if len(eligible_relations) != 15:
        raise RuntimeError(
            f"Expected 15 relations with n>={RELATION_CORRELATION_MIN_N}, "
            f"found {len(eligible_relations)}"
        )
    figure_relations = relation_counts[
        relation_counts >= RELATION_FIGURE_MIN_N
    ].head(RELATION_FIGURE_TOP_K).index.tolist()

    gsm_output = gsm[
        [
            "row_index",
            "sample_id",
            "solution_calculation_steps",
            "step_group",
            "question_char_length",
            "solution_char_length",
        ]
    ].copy()
    pop_output = pop[
        [
            "row_index",
            "sample_id",
            "id",
            "s_pop",
            "popularity_group",
            "prop",
            "subj",
            "obj",
        ]
    ].copy()
    metadata_summary = {
        "gsm_step_counts": actual_step_counts,
        "popularity_counts": actual_popularity_counts,
        "relation_counts": relation_counts.to_dict(),
        "eligible_relations": eligible_relations,
        "figure_relations": figure_relations,
        "gsm_source": str(gsm_path.resolve()),
        "popqa_source": str(pop_path.resolve()),
        "documented_step_stats": str(documented_steps.resolve()),
        "documented_popularity_stats": str(documented_popularity.resolve()),
    }
    return gsm_output, pop_output, metadata_summary


def describe_group(
    frame: pd.DataFrame,
    *,
    model_name: str,
    direction_type: str,
    dataset: str,
    characteristic: str,
    group_name: str,
    cache_index: int,
    relative_depth: float,
) -> dict[str, Any]:
    cosine = frame["cosine_similarity"].to_numpy(dtype=np.float64)
    projection = frame["projection"].to_numpy(dtype=np.float64)
    if len(frame) < 2 or not np.isfinite(cosine).all() or not np.isfinite(projection).all():
        raise RuntimeError(
            f"Invalid group at {model_name}/{direction_type}/{dataset}/"
            f"{characteristic}/{group_name}/cache {cache_index}"
        )
    return {
        "model": model_name,
        "direction_type": direction_type,
        "dataset": dataset,
        "characteristic": characteristic,
        "group": group_name,
        "cache_index": cache_index,
        "relative_layer_depth": relative_depth,
        "n": len(frame),
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std(ddof=1)),
        "cosine_median": float(np.median(cosine)),
        "projection_mean": float(projection.mean()),
        "projection_std": float(projection.std(ddof=1)),
        "projection_median": float(np.median(projection)),
    }


def spearman_row(
    *,
    model_name: str,
    direction_type: str,
    dataset: str,
    analysis: str,
    characteristic: str,
    metric: str,
    cache_index: int,
    relative_depth: float,
    x: np.ndarray,
    y: np.ndarray,
    relation: str | None = None,
) -> dict[str, Any]:
    if len(x) != len(y) or len(x) < 3:
        raise RuntimeError(f"Invalid Spearman inputs: n_x={len(x)}, n_y={len(y)}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError(
            f"Non-finite Spearman input at {model_name}/{direction_type}/{analysis}/{cache_index}"
        )
    result = spearmanr(x, y)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    if not math.isfinite(rho) or not math.isfinite(pvalue):
        raise RuntimeError(
            f"Undefined Spearman at {model_name}/{direction_type}/{analysis}/{cache_index}"
        )
    row = {
        "model": model_name,
        "direction_type": direction_type,
        "dataset": dataset,
        "analysis": analysis,
        "characteristic": characteristic,
        "metric": metric,
        "cache_index": cache_index,
        "relative_layer_depth": relative_depth,
        "n": len(x),
        "spearman_rho": rho,
        "abs_rho": abs(rho),
        "spearman_pvalue": pvalue,
        "spearman_qvalue_fdr_bh": np.nan,
        "fdr_family": "",
    }
    if relation is not None:
        row["relation"] = relation
    return row


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise RuntimeError("FDR input p-values must be finite and within [0,1].")
    count = len(values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * count / np.arange(1, count + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def apply_fdr(
    frame: pd.DataFrame,
    family_columns: list[str],
) -> pd.DataFrame:
    output = frame.copy()
    for family, indices in output.groupby(family_columns, sort=False).groups.items():
        family_indices = list(indices)
        output.loc[family_indices, "spearman_qvalue_fdr_bh"] = fdr_bh(
            output.loc[family_indices, "spearman_pvalue"].to_numpy()
        )
        family_tuple = family if isinstance(family, tuple) else (family,)
        output.loc[family_indices, "fdr_family"] = "|".join(map(str, family_tuple))
    if output["spearman_qvalue_fdr_bh"].isna().any():
        raise RuntimeError("FDR correction left missing q-values.")
    return output


def load_model_scores(
    cross_dir: Path,
    model_name: str,
    gsm_metadata: pd.DataFrame,
    pop_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    path = cross_dir / model_name / "sample_metrics.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Sample metrics missing: {path}")
    usecols = [
        "direction_type",
        "dataset",
        "row_index",
        "sample_id",
        "cache_index",
        "relative_layer_depth",
        "cosine_similarity",
        "projection",
    ]
    scores = pd.read_csv(path, usecols=usecols, dtype={"sample_id": "string"})
    if set(scores["direction_type"]) != set(DIRECTION_TYPES):
        raise RuntimeError(
            f"Unexpected direction types in {path}: {scores['direction_type'].unique()}"
        )
    if set(scores["dataset"]) != {"gsm8k", "popqa"}:
        raise RuntimeError(f"Unexpected datasets in {path}: {scores['dataset'].unique()}")
    numeric = scores[
        ["row_index", "cache_index", "relative_layer_depth", "cosine_similarity", "projection"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"NaN/Inf in existing sample metrics: {path}")
    if not scores["cosine_similarity"].between(-1.0, 1.0).all():
        raise RuntimeError(f"Existing cosine outside [-1,1]: {path}")

    cache_indices = sorted(scores["cache_index"].unique().astype(int).tolist())
    expected_indices = list(range(1, 42 if model_name.startswith("gemma") else 32))
    if cache_indices != expected_indices:
        raise RuntimeError(f"Unexpected cache indices for {model_name}: {cache_indices}")
    expected_rows = len(cache_indices) * len(DIRECTION_TYPES) * (
        EXPECTED_GSM8K + EXPECTED_POPQA
    )
    if len(scores) != expected_rows:
        raise RuntimeError(f"Unexpected sample metric rows for {model_name}: {len(scores)}")

    gsm_scores = scores[scores["dataset"] == "gsm8k"].copy()
    pop_scores = scores[scores["dataset"] == "popqa"].copy()
    gsm_joined = gsm_scores.merge(
        gsm_metadata,
        on=["row_index", "sample_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    pop_joined = pop_scores.merge(
        pop_metadata,
        on=["row_index", "sample_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not (gsm_joined["_merge"] == "both").all() or not (
        pop_joined["_merge"] == "both"
    ).all():
        raise RuntimeError(f"Metadata join failure for {model_name}")
    gsm_joined.drop(columns=["_merge"], inplace=True)
    pop_joined.drop(columns=["_merge"], inplace=True)

    for direction_type in DIRECTION_TYPES:
        for cache_index in cache_indices:
            gsm_group = gsm_joined[
                (gsm_joined["direction_type"] == direction_type)
                & (gsm_joined["cache_index"] == cache_index)
            ]
            pop_group = pop_joined[
                (pop_joined["direction_type"] == direction_type)
                & (pop_joined["cache_index"] == cache_index)
            ]
            if (
                len(gsm_group) != EXPECTED_GSM8K
                or gsm_group["sample_id"].nunique() != EXPECTED_GSM8K
                or len(pop_group) != EXPECTED_POPQA
                or pop_group["sample_id"].nunique() != EXPECTED_POPQA
            ):
                raise RuntimeError(
                    f"Join count/uniqueness failure for {model_name}/{direction_type}/{cache_index}"
                )
    return gsm_joined, pop_joined, cache_indices


def analyze_model(
    model_name: str,
    cross_dir: Path,
    gsm_metadata: pd.DataFrame,
    pop_metadata: pd.DataFrame,
    metadata_summary: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    print(f"\n[START] {model_name}", flush=True)
    source_path = cross_dir / model_name / "sample_metrics.csv.gz"
    print(f"  source: {source_path}", flush=True)
    gsm, pop, cache_indices = load_model_scores(
        cross_dir, model_name, gsm_metadata, pop_metadata
    )
    print(
        f"  joined rows: GSM8K={len(gsm)}, PopQA={len(pop)}; "
        f"cache={cache_indices[0]}..{cache_indices[-1]}",
        flush=True,
    )

    gsm_group_rows: list[dict[str, Any]] = []
    gsm_corr_rows: list[dict[str, Any]] = []
    pop_group_rows: list[dict[str, Any]] = []
    pop_corr_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    relation_corr_rows: list[dict[str, Any]] = []
    eligible_relations = set(metadata_summary["eligible_relations"])

    for direction_type in DIRECTION_TYPES:
        for position, cache_index in enumerate(cache_indices):
            gsm_layer = gsm[
                (gsm["direction_type"] == direction_type)
                & (gsm["cache_index"] == cache_index)
            ]
            pop_layer = pop[
                (pop["direction_type"] == direction_type)
                & (pop["cache_index"] == cache_index)
            ]
            relative_depth = float(gsm_layer["relative_layer_depth"].iloc[0])
            if not np.allclose(
                gsm_layer["relative_layer_depth"], relative_depth, rtol=0, atol=1e-15
            ) or not np.allclose(
                pop_layer["relative_layer_depth"], relative_depth, rtol=0, atol=1e-15
            ):
                raise RuntimeError(
                    f"Relative depth mismatch at {model_name}/{direction_type}/{cache_index}"
                )

            for step_group, group in gsm_layer.groupby("step_group", sort=False):
                gsm_group_rows.append(
                    describe_group(
                        group,
                        model_name=model_name,
                        direction_type=direction_type,
                        dataset="gsm8k",
                        characteristic="solution_calculation_steps",
                        group_name=str(step_group),
                        cache_index=cache_index,
                        relative_depth=relative_depth,
                    )
                )
            step_values = gsm_layer["solution_calculation_steps"].to_numpy(dtype=np.float64)
            for metric in ("cosine_similarity", "projection"):
                gsm_corr_rows.append(
                    spearman_row(
                        model_name=model_name,
                        direction_type=direction_type,
                        dataset="gsm8k",
                        analysis="gsm8k_solution_steps",
                        characteristic="solution_calculation_steps",
                        metric=metric,
                        cache_index=cache_index,
                        relative_depth=relative_depth,
                        x=step_values,
                        y=gsm_layer[metric].to_numpy(dtype=np.float64),
                    )
                )

            for popularity_group, group in pop_layer.groupby(
                "popularity_group", sort=False
            ):
                pop_group_rows.append(
                    describe_group(
                        group,
                        model_name=model_name,
                        direction_type=direction_type,
                        dataset="popqa",
                        characteristic="subject_popularity",
                        group_name=str(popularity_group),
                        cache_index=cache_index,
                        relative_depth=relative_depth,
                    )
                )
            popularity_values = pop_layer["s_pop"].to_numpy(dtype=np.float64)
            for metric in ("cosine_similarity", "projection"):
                pop_corr_rows.append(
                    spearman_row(
                        model_name=model_name,
                        direction_type=direction_type,
                        dataset="popqa",
                        analysis="popqa_subject_popularity",
                        characteristic="s_pop",
                        metric=metric,
                        cache_index=cache_index,
                        relative_depth=relative_depth,
                        x=popularity_values,
                        y=pop_layer[metric].to_numpy(dtype=np.float64),
                    )
                )

            for relation, group in pop_layer.groupby("prop", sort=False):
                relation_row = describe_group(
                    group,
                    model_name=model_name,
                    direction_type=direction_type,
                    dataset="popqa",
                    characteristic="relation",
                    group_name=str(relation),
                    cache_index=cache_index,
                    relative_depth=relative_depth,
                )
                relation_row["relation"] = relation_row.pop("group")
                relation_rows.append(relation_row)
                if relation in eligible_relations:
                    relation_popularity = group["s_pop"].to_numpy(dtype=np.float64)
                    for metric in ("cosine_similarity", "projection"):
                        relation_corr_rows.append(
                            spearman_row(
                                model_name=model_name,
                                direction_type=direction_type,
                                dataset="popqa",
                                analysis="popqa_subject_popularity_within_relation",
                                characteristic="s_pop",
                                metric=metric,
                                cache_index=cache_index,
                                relative_depth=relative_depth,
                                x=relation_popularity,
                                y=group[metric].to_numpy(dtype=np.float64),
                                relation=str(relation),
                            )
                        )

            if (
                position == 0
                or (position + 1) % 10 == 0
                or position == len(cache_indices) - 1
            ):
                latest_gsm = gsm_corr_rows[-2]
                latest_pop = pop_corr_rows[-2]
                print(
                    f"  [{direction_type} {position + 1:02d}/{len(cache_indices):02d}] "
                    f"cache={cache_index} step-cos rho={latest_gsm['spearman_rho']:.4f} "
                    f"pop-cos rho={latest_pop['spearman_rho']:.4f}",
                    flush=True,
                )

    gsm_group_frame = pd.DataFrame(gsm_group_rows)
    gsm_corr_frame = apply_fdr(
        pd.DataFrame(gsm_corr_rows), ["model", "direction_type", "analysis", "metric"]
    )
    pop_group_frame = pd.DataFrame(pop_group_rows)
    pop_corr_frame = apply_fdr(
        pd.DataFrame(pop_corr_rows), ["model", "direction_type", "analysis", "metric"]
    )
    relation_frame = pd.DataFrame(relation_rows)
    relation_corr_frame = apply_fdr(
        pd.DataFrame(relation_corr_rows),
        ["model", "direction_type", "analysis", "relation", "metric"],
    )

    expected_layers = len(cache_indices)
    expected_counts = {
        "gsm_step_group": expected_layers * len(DIRECTION_TYPES) * 4,
        "gsm_step_correlation": expected_layers * len(DIRECTION_TYPES) * 2,
        "pop_popularity_group": expected_layers * len(DIRECTION_TYPES) * 3,
        "pop_popularity_correlation": expected_layers * len(DIRECTION_TYPES) * 2,
        "pop_relation": expected_layers * len(DIRECTION_TYPES) * 16,
        "pop_relation_popularity_correlation": (
            expected_layers * len(DIRECTION_TYPES) * 15 * 2
        ),
    }
    outputs = {
        "gsm_step_group": gsm_group_frame,
        "gsm_step_correlation": gsm_corr_frame,
        "pop_popularity_group": pop_group_frame,
        "pop_popularity_correlation": pop_corr_frame,
        "pop_relation": relation_frame,
        "pop_relation_popularity_correlation": relation_corr_frame,
    }
    for name, frame in outputs.items():
        if len(frame) != expected_counts[name]:
            raise RuntimeError(
                f"Unexpected row count for {model_name}/{name}: "
                f"{len(frame)} != {expected_counts[name]}"
            )
    del gsm, pop
    gc.collect()
    print(f"[DONE] {model_name}", flush=True)
    return outputs


def load_existing_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, relative_path in TABLE_PATHS.items():
        path = output_dir / relative_path
        tables[name] = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    return tables


def model_is_complete(
    model_name: str,
    tables: dict[str, pd.DataFrame],
) -> bool:
    return all(not frame.empty and model_name in set(frame["model"]) for frame in tables.values())


def merge_model_outputs(
    output_dir: Path,
    existing: dict[str, pd.DataFrame],
    model_outputs: dict[str, pd.DataFrame],
    model_name: str,
) -> dict[str, pd.DataFrame]:
    merged: dict[str, pd.DataFrame] = {}
    for name, new_frame in model_outputs.items():
        old_frame = existing[name]
        if not old_frame.empty:
            old_frame = old_frame[old_frame["model"] != model_name]
            combined = pd.concat([old_frame, new_frame], ignore_index=True)
        else:
            combined = new_frame.copy()
        sort_columns = [
            column
            for column in [
                "model",
                "direction_type",
                "metric",
                "relation",
                "group",
                "cache_index",
            ]
            if column in combined.columns
        ]
        combined.sort_values(sort_columns, inplace=True, ignore_index=True)
        atomic_write_csv(output_dir / TABLE_PATHS[name], combined)
        merged[name] = combined
    return merged


def select_abs_peak(frame: pd.DataFrame) -> tuple[pd.Series, int]:
    target = float(frame["abs_rho"].max())
    tied = frame[np.abs(frame["abs_rho"].to_numpy() - target) <= PEAK_TOLERANCE]
    if tied.empty:
        raise RuntimeError("Could not select absolute rho peak.")
    return tied.sort_values("cache_index").iloc[0], len(tied)


def select_value_peak(frame: pd.DataFrame, metric: str) -> pd.Series:
    target = float(frame[metric].max())
    tied = frame[np.abs(frame[metric].to_numpy() - target) <= PEAK_TOLERANCE]
    return tied.sort_values("cache_index").iloc[0]


def build_peak_summary(
    tables: dict[str, pd.DataFrame],
    layerwise_dir: Path,
    cross_dir: Path,
) -> pd.DataFrame:
    layerwise = pd.read_csv(layerwise_dir / "all_models_layer_metrics.csv")
    cross = pd.read_csv(cross_dir / "all_models_cross_dataset_metrics.csv")
    rows: list[dict[str, Any]] = []
    analyses = {
        "gsm8k_solution_steps": tables["gsm_step_correlation"],
        "popqa_subject_popularity": tables["pop_popularity_correlation"],
    }
    for analysis, correlation_frame in analyses.items():
        for (model_name, direction_type, metric), frame in correlation_frame.groupby(
            ["model", "direction_type", "metric"], sort=True
        ):
            peak, tie_count = select_abs_peak(frame)
            mmlu_analysis = DIRECTION_TO_MMLU_ANALYSIS[direction_type]
            mmlu_frame = layerwise[
                (layerwise["model"] == model_name)
                & (layerwise["analysis_type"] == mmlu_analysis)
            ]
            cross_frame = cross[
                (cross["model"] == model_name)
                & (cross["direction_type"] == direction_type)
            ]
            if mmlu_frame.empty or cross_frame.empty:
                raise RuntimeError(
                    f"Missing matched peak reference for {model_name}/{direction_type}"
                )
            mmlu_peak = select_value_peak(mmlu_frame, "cosine_gap")
            cross_peak = select_value_peak(cross_frame, "cosine_gap")
            rows.append(
                {
                    "model": model_name,
                    "direction_type": direction_type,
                    "analysis": analysis,
                    "metric": f"abs_spearman_{'cosine' if metric == 'cosine_similarity' else metric}",
                    "peak_cache_index": int(peak["cache_index"]),
                    "peak_relative_depth": float(peak["relative_layer_depth"]),
                    "value": float(peak["abs_rho"]),
                    "signed_value_if_applicable": float(peak["spearman_rho"]),
                    "spearman_pvalue": float(peak["spearman_pvalue"]),
                    "spearman_qvalue_fdr_bh": float(
                        peak["spearman_qvalue_fdr_bh"]
                    ),
                    "peak_tie_count": tie_count,
                    "mmlu_reference_analysis_type": mmlu_analysis,
                    "mmlu_peak_cache_index": int(mmlu_peak["cache_index"]),
                    "mmlu_peak_relative_depth": float(
                        mmlu_peak["relative_layer_depth"]
                    ),
                    "cross_dataset_peak_cache_index": int(cross_peak["cache_index"]),
                    "cross_dataset_peak_relative_depth": float(
                        cross_peak["relative_layer_depth"]
                    ),
                }
            )
    summary = pd.DataFrame(rows).sort_values(
        ["analysis", "model", "direction_type", "metric"], ignore_index=True
    )
    return summary


def plot_model_outputs(
    model_name: str,
    outputs: dict[str, pd.DataFrame],
    metadata_summary: dict[str, Any],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gsm_figure_dir = output_dir / "gsm8k" / "figures"
    pop_figure_dir = output_dir / "popqa" / "figures"
    gsm_figure_dir.mkdir(parents=True, exist_ok=True)
    pop_figure_dir.mkdir(parents=True, exist_ok=True)

    def save(path: Path) -> None:
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        plt.tight_layout()
        plt.savefig(temporary, dpi=180, bbox_inches="tight")
        plt.close()
        os.replace(temporary, path)

    step_order = ["1-step", "2-step", "3-step", "4-step 이상"]
    popularity_order = ["Low", "Medium", "High"]
    for direction_type in DIRECTION_TYPES:
        prefix = f"{model_name}__{direction_type}"
        gsm_group = outputs["gsm_step_group"]
        gsm_group = gsm_group[gsm_group["direction_type"] == direction_type]
        plt.figure(figsize=(8, 5))
        for group_name in step_order:
            frame = gsm_group[gsm_group["group"] == group_name].sort_values(
                "relative_layer_depth"
            )
            plot_group_name = "4+ steps" if group_name == "4-step 이상" else group_name
            plt.plot(
                frame["relative_layer_depth"],
                frame["cosine_mean"],
                marker="o",
                ms=2.5,
                label=f"{plot_group_name} (n={int(frame['n'].iloc[0])})",
            )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Relative representation depth")
        plt.ylabel("Mean cosine similarity")
        plt.title(f"{model_name} — GSM8K Solution-step Groups ({direction_type})")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        save(gsm_figure_dir / f"{prefix}__step_group_mean_cosine.png")

        gsm_corr = outputs["gsm_step_correlation"]
        gsm_corr = gsm_corr[gsm_corr["direction_type"] == direction_type]
        for metric, label, filename in (
            ("cosine_similarity", "Step ↔ cosine Spearman rho", "step_cosine_rho"),
            ("projection", "Step ↔ projection Spearman rho", "step_projection_rho"),
        ):
            frame = gsm_corr[gsm_corr["metric"] == metric].sort_values(
                "relative_layer_depth"
            )
            plt.figure(figsize=(8, 5))
            plt.plot(frame["relative_layer_depth"], frame["spearman_rho"], marker="o", ms=3)
            plt.axhline(0.0, color="black", linewidth=0.8)
            plt.xlabel("Relative representation depth")
            plt.ylabel("Spearman rho")
            plt.title(f"{model_name} — {label} ({direction_type})")
            plt.grid(alpha=0.3)
            save(gsm_figure_dir / f"{prefix}__{filename}.png")

        pop_group = outputs["pop_popularity_group"]
        pop_group = pop_group[pop_group["direction_type"] == direction_type]
        plt.figure(figsize=(8, 5))
        for group_name in popularity_order:
            frame = pop_group[pop_group["group"] == group_name].sort_values(
                "relative_layer_depth"
            )
            plt.plot(
                frame["relative_layer_depth"],
                frame["cosine_mean"],
                marker="o",
                ms=2.5,
                label=f"{group_name} (n={int(frame['n'].iloc[0])})",
            )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Relative representation depth")
        plt.ylabel("Mean cosine similarity")
        plt.title(f"{model_name} — PopQA Subject Popularity Groups ({direction_type})")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        save(pop_figure_dir / f"{prefix}__popularity_group_mean_cosine.png")

        pop_corr = outputs["pop_popularity_correlation"]
        pop_corr = pop_corr[
            (pop_corr["direction_type"] == direction_type)
            & (pop_corr["metric"] == "cosine_similarity")
        ].sort_values("relative_layer_depth")
        plt.figure(figsize=(8, 5))
        plt.plot(
            pop_corr["relative_layer_depth"], pop_corr["spearman_rho"], marker="o", ms=3
        )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Relative representation depth")
        plt.ylabel("Spearman rho")
        plt.title(f"{model_name} — s_pop ↔ Cosine ({direction_type})")
        plt.grid(alpha=0.3)
        save(pop_figure_dir / f"{prefix}__popularity_cosine_rho.png")

        relation = outputs["pop_relation"]
        relation = relation[
            (relation["direction_type"] == direction_type)
            & (relation["relation"].isin(metadata_summary["figure_relations"]))
        ]
        plt.figure(figsize=(9, 6))
        for relation_name in metadata_summary["figure_relations"]:
            frame = relation[relation["relation"] == relation_name].sort_values(
                "relative_layer_depth"
            )
            plt.plot(
                frame["relative_layer_depth"],
                frame["cosine_mean"],
                linewidth=1.3,
                label=f"{relation_name} (n={int(frame['n'].iloc[0])})",
            )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Relative representation depth")
        plt.ylabel("Mean cosine similarity")
        plt.title(
            f"{model_name} — Top {RELATION_FIGURE_TOP_K} PopQA Relations ({direction_type})"
        )
        plt.legend(fontsize=7, ncol=2)
        plt.grid(alpha=0.3)
        save(pop_figure_dir / f"{prefix}__major_relation_mean_cosine.png")


def plot_summary(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "summary" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    def save(path: Path) -> None:
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        plt.tight_layout()
        plt.savefig(temporary, dpi=180, bbox_inches="tight")
        plt.close()
        os.replace(temporary, path)

    for name, frame, title in (
        (
            "gsm8k_step_cosine_rho_primary",
            tables["gsm_step_correlation"],
            "GSM8K solution-step heuristic ↔ cosine",
        ),
        (
            "popqa_popularity_cosine_rho_primary",
            tables["pop_popularity_correlation"],
            "PopQA subject popularity ↔ cosine",
        ),
    ):
        subset = frame[
            (frame["direction_type"] == "mmlu3000_full")
            & (frame["metric"] == "cosine_similarity")
        ]
        plt.figure(figsize=(10, 6))
        for model_name, model_frame in subset.groupby("model", sort=True):
            model_frame = model_frame.sort_values("relative_layer_depth")
            plt.plot(
                model_frame["relative_layer_depth"],
                model_frame["spearman_rho"],
                marker="o",
                ms=2,
                linewidth=1.2,
                label=model_name,
            )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Relative representation depth")
        plt.ylabel("Spearman rho")
        plt.title(f"All models — {title} (mmlu3000_full)")
        plt.legend(fontsize=7, ncol=2)
        plt.grid(alpha=0.3)
        save(figure_dir / f"{name}.png")


def write_readme(
    output_dir: Path,
    cross_dir: Path,
    dataset_analysis_dir: Path,
    metadata_summary: dict[str, Any],
    completed_models: list[str],
) -> None:
    readme = f"""# Problem Characteristic × LiReF Alignment Analysis

이 분석은 기존 `cross_dataset_projection/<model>/sample_metrics.csv.gz`의 cosine/projection을 source of truth로 사용한다. 기존 score를 다시 계산하지 않았으며 모델, hidden-state cache, LiReF vector도 불러오지 않았다.

## 분석 범위

- Models: {', '.join(completed_models)}
- PRIMARY direction: `mmlu3000_full`
- SECONDARY direction: `mmlu2400_train`
- GSM8K: solution calculation step heuristic
- PopQA: relative subject popularity indicator (`s_pop`)와 relation (`prop`)

## GSM8K heuristic

기존 `{metadata_summary['documented_step_stats']}`와 관련 문서에 기록된 규칙을 재현했다. `####` 앞 solution을 line별로 읽어 `<<식=결과>>` block 수를 세고, block이 없는 line은 `=`, 명시적 산술식 또는 계산 동사가 있으면 1 step으로 계산한다. 계산 흔적이 없으면 최소 1 step으로 둔다.

Test 1,319개의 검증된 그룹 수:

- 1-step: 3
- 2-step: 333
- 3-step: 381
- 4-step 이상: 602

이는 실제 모델 reasoning depth나 ground-truth 난이도가 아니라 정답 solution의 명시적 계산 구조를 요약한 heuristic이다. 1-step은 3개뿐이므로 그 그룹 평균을 일반화하지 않는다. PRIMARY characteristic 분석은 exact step 수와 score의 Spearman rho다.

## PopQA popularity와 relation

`s_pop`의 생성 의미는 dataset metadata만으로 확정할 수 없어 **relative subject popularity indicator**로만 해석한다. 기존 tertile 기준을 그대로 사용했다.

- Low: `s_pop <= {POPULARITY_LOW_MAX}` (4,760개)
- Medium: `{POPULARITY_LOW_MAX} < s_pop <= {POPULARITY_MEDIUM_MAX}` (4,752개)
- High: `s_pop > {POPULARITY_MEDIUM_MAX}` (4,755개)

전체 16개 relation은 CSV에 보존한다. Relation figure는 `n >= {RELATION_FIGURE_MIN_N}` 중 빈도 상위 {RELATION_FIGURE_TOP_K}개만 표시한다: {', '.join(metadata_summary['figure_relations'])}. `color` relation은 n=34인 소표본이라 figure 및 within-relation popularity 상관에서 제외하지만 relation layer 통계 CSV에는 포함한다. `n >= {RELATION_CORRELATION_MIN_N}`인 15개 relation 안에서도 `s_pop` Spearman을 별도로 계산해 relation composition confounding을 보조 점검한다.

## 정렬과 통계

- Join: `dataset + row_index + sample_id`; GSM8K는 `gsm8k_test_<row>`, PopQA는 `popqa_<실제 id>`
- Cosine은 primary alignment metric, projection은 supplementary metric
- Spearman rho와 raw p-value를 모든 cached depth에서 계산
- Benjamini–Hochberg FDR은 `model × direction × analysis × metric`별 depth family 안에서 적용
- Within-relation 분석은 위 family에 relation을 추가
- Characteristic peak는 `abs(rho)` 최대이며 signed rho를 함께 보존
- `abs(rho)` 차이가 `{PEAK_TOLERANCE}` 이하면 동률로 보고 가장 이른 cache index를 대표로 선택

Sample 수가 크므로 p/q-value보다 effect direction, rho 크기, depth/model-family 일관성, PRIMARY/SECONDARY robustness를 우선 해석한다.

## Bootstrap

기본 분석에는 bootstrap CI를 넣지 않았다. 모든 depth를 탐색한 뒤 선택된 peak에서만 CI를 계산하면 selection bias가 생길 수 있기 때문이다. 대신 모든 depth의 signed rho, absolute rho, p-value와 FDR q-value를 저장했다.

## 출력

- `gsm8k/sample_metadata.csv`: sample별 기존 step heuristic 재현 결과
- `gsm8k/step_group_layer_metrics.csv`: 1/2/3/4+ 그룹별 score 통계
- `gsm8k/step_correlation_by_layer.csv`: exact step과 cosine/projection Spearman
- `popqa/sample_metadata.csv`: s_pop tertile 및 relation metadata
- `popqa/popularity_group_layer_metrics.csv`: tertile별 score 통계
- `popqa/popularity_correlation_by_layer.csv`: 전체 s_pop Spearman
- `popqa/relation_layer_metrics.csv`: 16개 relation별 score 통계
- `popqa/popularity_correlation_by_relation.csv`: n>=100 relation 내부 s_pop Spearman
- `summary/characteristic_peak_summary.csv`: matched MMLU/cross-dataset peak 포함 characteristic peak

새로운 sample-depth score 파일은 만들지 않았다.

## 해석 제한

이 결과는 dataset-level GSM8K/PopQA 비교의 confound를 일부 줄이는 within-dataset association 분석이다. 그러나 step heuristic은 solution length·연산 수와, popularity는 relation과 연관될 수 있다. Relation 내부 분석도 관찰적 상관이다. 따라서 task-invariant reasoning/memory 축 또는 causal importance를 증명하지 않는다.

## 실행

```bash
/home/jinhyun/.conda/envs/torch/bin/python {SCRIPT_PATH} --models Meta-Llama-3-8B
/home/jinhyun/.conda/envs/torch/bin/python {SCRIPT_PATH} --models all --skip-existing
```

Source root: `{cross_dir.resolve()}`  
Dataset analysis reference: `{dataset_analysis_dir.resolve()}`  
Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}
"""
    atomic_write_text(output_dir / "README.md", readme)


def print_model_peaks(model_name: str, outputs: dict[str, pd.DataFrame]) -> None:
    print(f"\n[{model_name}] characteristic peaks", flush=True)
    for analysis, frame in (
        ("GSM8K steps", outputs["gsm_step_correlation"]),
        ("PopQA s_pop", outputs["pop_popularity_correlation"]),
    ):
        for direction_type in DIRECTION_TYPES:
            cosine = frame[
                (frame["direction_type"] == direction_type)
                & (frame["metric"] == "cosine_similarity")
            ]
            peak, tie_count = select_abs_peak(cosine)
            print(
                f"  {analysis} [{direction_type}]: cache={int(peak['cache_index'])}, "
                f"rho={peak['spearman_rho']:.6f}, abs={peak['abs_rho']:.6f}, "
                f"q={peak['spearman_qvalue_fdr_bh']:.3e}, ties={tie_count}",
                flush=True,
            )


def validate_all_tables(
    tables: dict[str, pd.DataFrame],
    models: list[str],
) -> None:
    if not models:
        raise RuntimeError("No completed models to validate.")
    for name, frame in tables.items():
        if frame.empty or set(frame["model"]) != set(models):
            raise RuntimeError(
                f"Output table {name} model coverage mismatch: {sorted(frame.get('model', []))}"
            )
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"Output table {name} contains NaN or Inf.")
    if not tables["gsm_step_correlation"]["spearman_qvalue_fdr_bh"].between(0, 1).all():
        raise RuntimeError("Invalid GSM8K FDR q-value.")
    if not tables["pop_popularity_correlation"]["spearman_qvalue_fdr_bh"].between(0, 1).all():
        raise RuntimeError("Invalid PopQA FDR q-value.")


def main() -> int:
    args = parse_args()
    dataset_analysis_dir = args.dataset_analysis_dir.resolve()
    layerwise_dir = args.layerwise_dir.resolve()
    cross_dir = args.cross_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    gsm_metadata, pop_metadata, metadata_summary = load_and_validate_metadata(
        cross_dir, dataset_analysis_dir
    )
    atomic_write_csv(output_dir / "gsm8k" / "sample_metadata.csv", gsm_metadata)
    atomic_write_csv(output_dir / "popqa" / "sample_metadata.csv", pop_metadata)
    available_models = discover_models(cross_dir)
    selected_models = select_models(args.models, available_models)
    tables = load_existing_tables(output_dir)

    print("Problem Characteristic × LiReF Alignment Analysis", flush=True)
    print(f"  output: {output_dir}", flush=True)
    print(f"  models: {selected_models}", flush=True)
    print(f"  GSM8K step groups: {metadata_summary['gsm_step_counts']}", flush=True)
    print(f"  PopQA popularity groups: {metadata_summary['popularity_counts']}", flush=True)
    print("  source scores only; no cosine/projection recomputation", flush=True)

    for model_name in selected_models:
        complete = model_is_complete(model_name, tables)
        if complete and args.skip_existing:
            print(f"[SKIP] Complete outputs already exist for {model_name}", flush=True)
            continue
        if complete and not args.overwrite:
            raise FileExistsError(
                f"Outputs already contain {model_name}; use --overwrite or --skip-existing."
            )
        outputs = analyze_model(
            model_name, cross_dir, gsm_metadata, pop_metadata, metadata_summary
        )
        tables = merge_model_outputs(output_dir, tables, outputs, model_name)
        if not args.no_figures:
            plot_model_outputs(
                model_name, outputs, metadata_summary, output_dir
            )
        print_model_peaks(model_name, outputs)

    completed_models = sorted(
        set.intersection(
            *(set(frame["model"]) for frame in tables.values() if not frame.empty)
        )
    )
    validate_all_tables(tables, completed_models)
    peak_summary = build_peak_summary(tables, layerwise_dir, cross_dir)
    atomic_write_csv(
        output_dir / "summary" / "characteristic_peak_summary.csv", peak_summary
    )
    if not args.no_figures:
        plot_summary(tables, output_dir)
    write_readme(
        output_dir,
        cross_dir,
        dataset_analysis_dir,
        metadata_summary,
        completed_models,
    )
    print(f"\nCompleted models: {completed_models}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; existing source results were not modified.", file=sys.stderr)
        raise
