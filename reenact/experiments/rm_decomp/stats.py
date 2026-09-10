"""Streaming statistics and fixed candidate validation for Stage A."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class GroupMoments:
    """Per-group streaming moments for arrays with a fixed trailing shape."""

    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        self.count = np.zeros(2, dtype=np.int64)
        self.sum = np.zeros((2, *self.shape), dtype=np.float64)
        self.sumsq = np.zeros((2, *self.shape), dtype=np.float64)

    def update_reduced(
        self,
        group: int,
        count: int,
        value_sum: np.ndarray,
        value_sumsq: np.ndarray,
    ) -> None:
        if count == 0:
            return
        self.count[group] += int(count)
        self.sum[group] += np.asarray(value_sum, dtype=np.float64)
        self.sumsq[group] += np.asarray(value_sumsq, dtype=np.float64)

    def means(self) -> np.ndarray:
        result = np.full_like(self.sum, np.nan)
        for group in range(2):
            if self.count[group] > 0:
                result[group] = self.sum[group] / self.count[group]
        return result

    def variances(self) -> np.ndarray:
        result = np.full_like(self.sum, np.nan)
        for group in range(2):
            n = int(self.count[group])
            if n < 2:
                continue
            centered = self.sumsq[group] - (self.sum[group] ** 2) / n
            result[group] = np.maximum(centered / (n - 1), 0.0)
        return result

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"count": self.count, "sum": self.sum, "sumsq": self.sumsq}

    @classmethod
    def from_state_dict(cls, payload: dict[str, np.ndarray]) -> "GroupMoments":
        obj = cls(tuple(np.asarray(payload["sum"]).shape[1:]))
        obj.count = np.asarray(payload["count"], dtype=np.int64)
        obj.sum = np.asarray(payload["sum"], dtype=np.float64)
        obj.sumsq = np.asarray(payload["sumsq"], dtype=np.float64)
        return obj


def summarize_moments(moments: GroupMoments, epsilon: float) -> dict[str, np.ndarray]:
    """Compute Welch inference and pooled-SD Cohen's d elementwise."""

    means = moments.means()
    variances = moments.variances()
    n_m, n_r = (int(moments.count[0]), int(moments.count[1]))
    mean_m, mean_r = means[0], means[1]
    var_m, var_r = variances[0], variances[1]
    delta = mean_r - mean_m

    valid = (
        (n_m >= 2)
        & (n_r >= 2)
        & np.isfinite(var_m)
        & np.isfinite(var_r)
        & np.isfinite(delta)
    )
    se2 = var_r / max(n_r, 1) + var_m / max(n_m, 1)
    se = np.sqrt(np.maximum(se2, 0.0))
    valid_welch = valid & (se > epsilon)

    numerator = se2**2
    denominator = np.zeros_like(se2)
    if n_r >= 2:
        denominator += (var_r / n_r) ** 2 / (n_r - 1)
    if n_m >= 2:
        denominator += (var_m / n_m) ** 2 / (n_m - 1)
    df = np.full_like(delta, np.nan)
    # The Welch denominator contains squared variance-of-the-mean terms and is
    # naturally much smaller than the score-scale epsilon (often 1e-15 or
    # below).  It only needs to be finite and strictly positive; comparing it
    # with `epsilon` incorrectly discards otherwise valid tests.
    df_mask = valid_welch & np.isfinite(denominator) & (denominator > 0.0)
    df[df_mask] = numerator[df_mask] / denominator[df_mask]

    t_value = np.full_like(delta, np.nan)
    t_value[valid_welch] = delta[valid_welch] / se[valid_welch]
    p_value = np.full_like(delta, np.nan)
    ci_low = np.full_like(delta, np.nan)
    ci_high = np.full_like(delta, np.nan)
    infer_mask = valid_welch & np.isfinite(df) & (df > 0)
    if np.any(infer_mask):
        critical = scipy_stats.t.ppf(0.975, df[infer_mask])
        p_value[infer_mask] = 2.0 * scipy_stats.t.sf(
            np.abs(t_value[infer_mask]), df[infer_mask]
        )
        ci_low[infer_mask] = delta[infer_mask] - critical * se[infer_mask]
        ci_high[infer_mask] = delta[infer_mask] + critical * se[infer_mask]

    pooled_den = n_r + n_m - 2
    pooled_var = np.full_like(delta, np.nan)
    if pooled_den > 0:
        pooled_var = ((n_r - 1) * var_r + (n_m - 1) * var_m) / pooled_den
    pooled_sd = np.sqrt(np.maximum(pooled_var, 0.0))
    cohen_d = np.full_like(delta, np.nan)
    effect_mask = valid & np.isfinite(pooled_sd) & (pooled_sd > epsilon)
    cohen_d[effect_mask] = delta[effect_mask] / pooled_sd[effect_mask]

    reason = np.full(delta.shape, "", dtype=object)
    reason[~valid] = "insufficient_n_or_nonfinite_variance"
    reason[valid & ~valid_welch] = "zero_or_near_zero_standard_error"
    reason[valid_welch & ~infer_mask] = "welch_df_unavailable"

    return {
        "n_memory": np.full(delta.shape, n_m, dtype=np.int64),
        "n_reasoning": np.full(delta.shape, n_r, dtype=np.int64),
        "memory_mean": mean_m,
        "reasoning_mean": mean_r,
        "memory_variance": var_m,
        "reasoning_variance": var_r,
        "Delta": delta,
        "abs_Delta": np.abs(delta),
        "standard_error": se,
        "welch_df": df,
        "welch_p": p_value,
        "welch_ci_low": ci_low,
        "welch_ci_high": ci_high,
        "Cohen_d": cohen_d,
        "stat_na_reason": reason,
    }


def benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    """BH-adjust finite p-values; preserve invalid entries as NaN."""

    p = np.asarray(list(values), dtype=np.float64)
    q = np.full_like(p, np.nan)
    valid_indices = np.flatnonzero(np.isfinite(p))
    if len(valid_indices) == 0:
        return q
    valid_p = p[valid_indices]
    order = np.argsort(valid_p, kind="mergesort")
    ranked = valid_p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[valid_indices] = restored
    return q


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, str]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return math.nan, math.nan, "fewer_than_two_finite_pairs"
    if np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
        return math.nan, math.nan, "constant_input"
    result = scipy_stats.spearmanr(x[mask], y[mask])
    return float(result.statistic), float(result.pvalue), ""


def assign_signed_ranks(frame: pd.DataFrame, value_column: str = "Delta") -> pd.DataFrame:
    """Deterministic signed global and layer ranks within one component type."""

    result = frame.copy()
    result["sign_group"] = np.where(result[value_column] > 0, "positive", np.where(result[value_column] < 0, "negative", "zero"))
    result["rank_global"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["rank_layer"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    for sign in ("positive", "negative"):
        sign_mask = result["sign_group"] == sign
        ascending = sign == "negative"
        ordered = result.loc[sign_mask].sort_values(
            [value_column, "module_index", "component_id"],
            ascending=[ascending, True, True],
            kind="mergesort",
        )
        result.loc[ordered.index, "rank_global"] = np.arange(1, len(ordered) + 1)
        for _, layer_frame in result.loc[sign_mask].groupby("module_index", sort=True):
            layer_ordered = layer_frame.sort_values(
                [value_column, "component_id"],
                ascending=[ascending, True],
                kind="mergesort",
            )
            result.loc[layer_ordered.index, "rank_layer"] = np.arange(1, len(layer_ordered) + 1)
    return result
