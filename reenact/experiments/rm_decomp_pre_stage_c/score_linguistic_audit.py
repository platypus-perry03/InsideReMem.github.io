#!/usr/bin/env python3
"""Score two blinded linguistic-audit forms and enforce the pre-PDF gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BOOLEAN_COLUMNS = [
    "grammar_a_pass",
    "grammar_b_pass",
    "only_intended_feature_changed",
    "relation_polarity_valid",
    "relevance_label_valid",
    "expected_answers_valid",
    "changed_spans_valid",
    "unintended_cue_absent",
    "overall_pass",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    return parser.parse_args()


def parse_bool(value: object, field: str, audit_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{audit_id}: {field} must be true/false, got {value!r}")
    return normalized == "true"


def cohen_kappa(left: pd.Series, right: pd.Series) -> float:
    a = left.astype(bool).to_numpy()
    b = right.astype(bool).to_numpy()
    observed = float(np.mean(a == b))
    p_left = float(np.mean(a))
    p_right = float(np.mean(b))
    expected = p_left * p_right + (1.0 - p_left) * (1.0 - p_right)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else (1.0 if observed == 1.0 else math.nan)


def load_completed(path: Path, expected_ids: set[str], reviewer: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing completed form: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    if set(frame["audit_id"]) != expected_ids or len(frame) != len(expected_ids):
        raise RuntimeError(f"{reviewer}: audit IDs differ from the frozen master form")
    for column in BOOLEAN_COLUMNS:
        frame[column] = [parse_bool(value, column, audit_id) for value, audit_id in zip(frame[column], frame["audit_id"])]
    for column in ("naturalness_a_1to5", "naturalness_b_1to5"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].between(1, 5).all():
            raise ValueError(f"{reviewer}: {column} must be an integer from 1 to 5")
        if not np.allclose(frame[column], frame[column].round()):
            raise ValueError(f"{reviewer}: {column} must contain integers")
    derived_core = frame[[column for column in BOOLEAN_COLUMNS if column != "overall_pass"]].all(axis=1)
    if not (derived_core == frame["overall_pass"]).all():
        bad = frame.loc[derived_core != frame["overall_pass"], "audit_id"].tolist()
        raise RuntimeError(f"{reviewer}: overall_pass disagrees with core fields for {bad[:10]}")
    return frame.sort_values("audit_id").reset_index(drop=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    config = json.loads(parse_args().config.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    audit = root / "linguistic_audit"
    completed = audit / "completed"
    master = pd.read_csv(audit / "audit_items_master_blank.csv", keep_default_na=False)
    expected_ids = set(master["audit_id"])
    reviewer_1 = load_completed(completed / "reviewer_1.csv", expected_ids, "reviewer_1")
    reviewer_2 = load_completed(completed / "reviewer_2.csv", expected_ids, "reviewer_2")
    merged = reviewer_1.merge(reviewer_2, on="audit_id", suffixes=("_r1", "_r2"), validate="one_to_one")
    disagreement_mask = np.zeros(len(merged), dtype=bool)
    for column in BOOLEAN_COLUMNS:
        disagreement_mask |= merged[f"{column}_r1"].to_numpy() != merged[f"{column}_r2"].to_numpy()
    disagreements = merged.loc[disagreement_mask, [
        "audit_id", "source_stage_r1", "claimed_relevance_r1", "text_a_r1", "text_b_r1",
        *[name for column in BOOLEAN_COLUMNS for name in (f"{column}_r1", f"{column}_r2")],
    ]].copy()
    disagreements["final_overall_pass"] = ""
    disagreements["final_expected_answers_valid"] = ""
    disagreements["adjudicator_notes"] = ""
    adjudication_path = completed / "audit_adjudication.csv"
    if len(disagreements) and not adjudication_path.exists():
        atomic_csv(completed / "audit_adjudication_REQUIRED.csv", disagreements)
        status = {
            "status": "NEEDS_ADJUDICATION",
            "pdf_ready": False,
            "reviewed_items": len(expected_ids),
            "disagreement_items": len(disagreements),
            "next_action": "Fill audit_adjudication_REQUIRED.csv and save it as audit_adjudication.csv, then rerun scorer.",
        }
        write_json(audit / "audit_final_status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    if len(disagreements):
        adjudication = pd.read_csv(adjudication_path, keep_default_na=False)
        if set(adjudication["audit_id"]) != set(disagreements["audit_id"]):
            raise RuntimeError("Adjudication IDs differ from reviewer disagreements")
        adjudication["final_overall_pass"] = [parse_bool(v, "final_overall_pass", i) for v, i in zip(adjudication["final_overall_pass"], adjudication["audit_id"])]
        adjudication["final_expected_answers_valid"] = [parse_bool(v, "final_expected_answers_valid", i) for v, i in zip(adjudication["final_expected_answers_valid"], adjudication["audit_id"])]
        adjudication = adjudication.set_index("audit_id")
    else:
        adjudication = pd.DataFrame()

    consensus_overall = {}
    consensus_answers = {}
    for _, row in merged.iterrows():
        audit_id = row["audit_id"]
        if row["overall_pass_r1"] == row["overall_pass_r2"]:
            consensus_overall[audit_id] = bool(row["overall_pass_r1"])
        else:
            consensus_overall[audit_id] = bool(adjudication.loc[audit_id, "final_overall_pass"])
        if row["expected_answers_valid_r1"] == row["expected_answers_valid_r2"]:
            consensus_answers[audit_id] = bool(row["expected_answers_valid_r1"])
        else:
            consensus_answers[audit_id] = bool(adjudication.loc[audit_id, "final_expected_answers_valid"])

    kappa = cohen_kappa(merged["overall_pass_r1"], merged["overall_pass_r2"])
    naturalness_mean = float(pd.concat([
        reviewer_1["naturalness_a_1to5"], reviewer_1["naturalness_b_1to5"],
        reviewer_2["naturalness_a_1to5"], reviewer_2["naturalness_b_1to5"],
    ]).mean())
    core_pass_rate = float(np.mean(list(consensus_overall.values())))
    expected_accuracy = float(np.mean(list(consensus_answers.values())))
    threshold = config["audit_acceptance"]
    checks = {
        "expected_answer_accuracy": expected_accuracy >= float(threshold["expected_answer_accuracy_min"]),
        "core_item_pass_rate": core_pass_rate >= float(threshold["core_item_pass_rate_min"]),
        "naturalness_mean": naturalness_mean >= float(threshold["naturalness_mean_min"]),
        "inter_rater_kappa": kappa >= float(threshold["inter_rater_kappa_min"]),
    }
    passed = all(checks.values())
    status = {
        "status": "PASS" if passed else "FAIL",
        "pdf_ready": passed,
        "reviewed_items": len(expected_ids),
        "disagreement_items": len(disagreements),
        "expected_answer_accuracy": expected_accuracy,
        "core_item_pass_rate": core_pass_rate,
        "naturalness_mean": naturalness_mean,
        "overall_pass_cohen_kappa": kappa,
        "threshold_checks": checks,
        "acceptance": threshold,
    }
    write_json(audit / "audit_final_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
