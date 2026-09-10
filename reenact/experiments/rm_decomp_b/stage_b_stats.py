"""Frozen paired statistics for LiReF Stage B."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=np.float64)
    q = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return q
    order = np.argsort(p[valid], kind="mergesort")
    ranked = p[valid][order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    q[valid] = restored
    return q


def template_effects(
    template_ids: Iterable[str],
    differences: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(list(template_ids), dtype=object)
    values = np.asarray(list(differences), dtype=np.float64)
    mask = np.asarray([bool(value) for value in ids]) & np.isfinite(values)
    ids, values = ids[mask], values[mask]
    unique = np.asarray(sorted(set(ids.tolist())), dtype=object)
    effects = np.asarray([values[ids == item].mean() for item in unique], dtype=np.float64)
    return unique, effects


def sign_flip_pvalue(values: np.ndarray, iterations: int, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = int(iterations)
    while remaining:
        size = min(remaining, 4096)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(values)))
        permuted = np.abs((signs * values[None, :]).mean(axis=1))
        exceed += int((permuted >= observed).sum())
        remaining -= size
    return float((exceed + 1) / (iterations + 1))


def bootstrap_percentile_ci(
    values: np.ndarray,
    iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    remaining = iterations
    offset = 0
    while remaining:
        size = min(remaining, 4096)
        samples = rng.choice(values, size=(size, len(values)), replace=True)
        means[offset : offset + size] = samples.mean(axis=1)
        offset += size
        remaining -= size
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def paired_summary(
    template_ids: Iterable[str],
    differences: Iterable[float],
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> dict[str, float | int | str | None]:
    unique, effects = template_effects(template_ids, differences)
    result: dict[str, float | int | str | None] = {
        "n_template": int(len(unique)),
        "mean_template_effect": None,
        "template_sd": None,
        "cohen_dz": None,
        "ci95_low": None,
        "ci95_high": None,
        "sign_flip_p": None,
        "stat_na_reason": "",
    }
    if len(effects) < 2:
        result["stat_na_reason"] = "fewer_than_two_template_families"
        return result
    mean = float(effects.mean())
    sd = float(effects.std(ddof=1))
    result["mean_template_effect"] = mean
    result["template_sd"] = sd
    if not math.isfinite(sd) or sd == 0.0:
        result["stat_na_reason"] = "zero_or_nonfinite_template_sd"
    else:
        result["cohen_dz"] = mean / sd
    low, high = bootstrap_percentile_ci(effects, bootstrap_iterations, seed)
    result["ci95_low"] = low
    result["ci95_high"] = high
    result["sign_flip_p"] = sign_flip_pvalue(effects, permutation_iterations, seed)
    return result


def specificity_summary(
    template_ids: Iterable[str],
    candidate_differences: Iterable[float],
    control_differences: Iterable[float],
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> dict[str, float | int | str | None]:
    candidate = np.asarray(list(candidate_differences), dtype=np.float64)
    control = np.asarray(list(control_differences), dtype=np.float64)
    if candidate.shape != control.shape:
        raise ValueError("Candidate and control paired differences must have identical shapes")
    return paired_summary(
        template_ids,
        candidate - control,
        bootstrap_iterations,
        permutation_iterations,
        seed,
    )
