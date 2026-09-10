#!/usr/bin/env python3
"""Full AI linguistic audit for Stage D v2_d06; does not impersonate human reviewers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("/home/jinhyun/prj_ws/jiho/AI/reenact/liref_outputs/rm_decomp/v2/d_feature_causal_v2_d06")


def main() -> None:
    items = pd.read_csv(ROOT / "manifests" / "confirmatory_items.csv")
    source_audit = pd.read_csv(ROOT / "human_audit" / "ai_full_audit.csv").set_index("chain_id")
    rows = []
    for chain_id, frame in items.groupby("chain_id", sort=True):
        first = frame.iloc[0]
        options = json.loads(first["options_json"])
        source_pass = source_audit.loc[chain_id, "verdict"] == "AI_PASS_TO_HUMAN"
        ill_formed_relation = bool(
            frame["question"].str.contains("the country that contains or originates", regex=False).any()
        )
        distractor_option_overlap = first["distractor_capital"] in options
        four_conditions = set(frame["condition"]) == {"A", "B", "C", "D"}
        same_answer = (
            frame["capital"].nunique() == 1
            and frame["correct_label"].nunique() == 1
            and frame["options_json"].nunique() == 1
            and options[ord(first["correct_label"]) - ord("A")] == first["capital"]
        )
        grammar_ok = not ill_formed_relation
        single_feature_validity = not ill_formed_relation
        distractors_valid = not distractor_option_overlap
        approve = all(
            (source_pass, four_conditions, same_answer, grammar_ok, single_feature_validity, distractors_valid)
        )
        notes = []
        if ill_formed_relation:
            notes.append(
                "Template family 04 uses the ill-formed relation phrase "
                "'the country that contains or originates X'."
            )
        if distractor_option_overlap:
            notes.append(
                f"Distractor capital {first['distractor_capital']!r} is also an answer option and can cue a wrong answer."
            )
        rows.append(
            {
                "chain_id": chain_id,
                "template_family": first["template_family"],
                "fact_accuracy": source_pass,
                "grammar_ok": grammar_ok,
                "single_feature_validity": single_feature_validity,
                "same_answer_valid": same_answer,
                "distractors_valid": distractors_valid,
                "distractor_capital_in_options": distractor_option_overlap,
                "ai_approve": approve,
                "ai_verdict": "AI_PASS_TO_HUMAN" if approve else "AI_REVISION_REQUIRED",
                "ai_comment": " ".join(notes),
            }
        )

    result = pd.DataFrame(rows)
    audit_dir = ROOT / "human_audit"
    result.to_csv(audit_dir / "ai_linguistic_audit.csv", index=False)
    summary = {
        "audit_kind": "AI_FULL_LINGUISTIC_AUDIT_NOT_HUMAN",
        "chain_count": int(len(result)),
        "ai_pass_count": int(result["ai_approve"].sum()),
        "revision_required_count": int((~result["ai_approve"]).sum()),
        "ill_formed_relation_count": int((~result["grammar_ok"]).sum()),
        "distractor_option_overlap_count": int(result["distractor_capital_in_options"].sum()),
        "human_audit_substituted": False,
        "overall_status": (
            "PASS_TO_BLIND_HUMAN_AUDIT"
            if bool(result["ai_approve"].all())
            else "REVISION_REQUIRED_BEFORE_CONFIRMATORY"
        ),
    }
    (audit_dir / "ai_linguistic_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
