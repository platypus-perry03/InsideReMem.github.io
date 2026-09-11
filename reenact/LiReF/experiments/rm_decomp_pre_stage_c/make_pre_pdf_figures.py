#!/usr/bin/env python3
"""Create manuscript-ready PNG figures without rendering the final PDF."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "liref_outputs" / "rm_decomp" / "v2"
OUTPUT = V2 / "pre_stage_c_v2_p01" / "stage_c_pre_pdf" / "figures"


def style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })


def stage_b_relation_scatter() -> None:
    path = V2 / "b_extension_v2_b04_corrected" / "tables" / "candidate_effects.csv"
    frame = pd.read_csv(path)
    frame = frame[(frame["component_role"] == "candidate") & frame["analysis_family"].isin([
        "relation_lexical_robustness", "relation_relevance_interaction"
    ])]
    pivot = frame.pivot(index=["candidate_id", "component_type"], columns="analysis_family", values="cohen_dz").reset_index()
    causal = {"L25H00021", "L31N13336", "L29H00030", "L30H00006", "L29H00031"}
    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    for component_type, marker, color in (("head", "o", "#2468A2"), ("neuron", "s", "#D1495B")):
        part = pivot[pivot["component_type"] == component_type]
        ax.scatter(part["relation_lexical_robustness"], part["relation_relevance_interaction"], marker=marker, s=48, color=color, alpha=.8, label=component_type.title())
    selected = pivot[pivot["candidate_id"].isin(causal)]
    ax.scatter(selected["relation_lexical_robustness"], selected["relation_relevance_interaction"], s=125, facecolors="none", edgecolors="#111111", linewidths=1.4, label="Frozen causal target")
    label_offsets = {
        "L25H00021": (5, 5),
        "L31N13336": (5, 5),
        "L29H00031": (5, 5),
        "L29H00030": (5, 7),
        "L30H00006": (5, -18),
    }
    for _, row in selected.iterrows():
        ax.annotate(
            row["candidate_id"],
            (row["relation_lexical_robustness"], row["relation_relevance_interaction"]),
            xytext=label_offsets[row["candidate_id"]], textcoords="offset points", fontsize=8,
        )
    ax.axhline(0, color="#999999", linewidth=.8)
    ax.axvline(0, color="#999999", linewidth=.8)
    ax.axhline(.5, color="#bbbbbb", linewidth=.8, linestyle="--")
    ax.axvline(.5, color="#bbbbbb", linewidth=.8, linestyle="--")
    ax.set_xlabel("Relation lexical robustness (Cohen's dz)")
    ax.set_ylabel("Relation task-relevance interaction (Cohen's dz)")
    ax.set_title("Frozen Stage A candidates: controlled relation sensitivity")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_stage_b_relation_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def causal_forest() -> None:
    summary = json.loads((V2 / "c_causal_v2_c01" / "causal_summary.json").read_text(encoding="utf-8"))
    cards = pd.DataFrame(summary["candidate_cards"])
    cards = cards.sort_values("gap_reduction")
    y = np.arange(len(cards))
    low = cards["gap_reduction"] - cards["gap_reduction_95ci"].str[0]
    high = cards["gap_reduction_95ci"].str[1] - cards["gap_reduction"]
    colors = np.where(cards["causal_gap_criterion_pass"], "#2A9D8F", "#D1495B")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for index, (_, row) in enumerate(cards.iterrows()):
        ax.errorbar(row["gap_reduction"], index, xerr=[[low.iloc[index]], [high.iloc[index]]], fmt="o", color=colors[index], capsize=3, markersize=7)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, cards["candidate_id"])
    ax.set_xlabel("Reduction in |R/M LiReF gap| under 100% intervention")
    ax.set_title("Causal validation of five frozen targets")
    ax.text(.99, .03, "Green: prespecified causal criterion PASS", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#2A9D8F")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_causal_gap_forest.png", bbox_inches="tight")
    plt.close(fig)


def memory_forest() -> None:
    path = V2 / "pre_stage_c_v2_p01" / "memory_side" / "tables" / "memory_validation_final.csv"
    frame = pd.read_csv(path).sort_values("beta")
    labels = frame["candidate_id"] + " × " + frame["feature_name"]
    y = np.arange(len(frame))
    colors = np.where(frame["replicated_with_specificity"], "#2A9D8F", np.where(frame["replicated_basic"], "#E9C46A", "#888888"))
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    for index, (_, row) in enumerate(frame.iterrows()):
        ax.errorbar(row["beta"], index, xerr=[[row["beta"] - row["ci_low"]], [row["ci_high"] - row["beta"]]], fmt="o", color=colors[index], capsize=3, markersize=7)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Adjusted standardized beta (Validation, 95% CI)")
    ax.set_title("Frozen Memory-side proxy hypotheses")
    ax.text(.99, .03, "Green: replicated with matched/random specificity", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#2A9D8F")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_memory_proxy_validation.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    style()
    stage_b_relation_scatter()
    causal_forest()
    memory_forest()
    print(OUTPUT)


if __name__ == "__main__":
    main()
