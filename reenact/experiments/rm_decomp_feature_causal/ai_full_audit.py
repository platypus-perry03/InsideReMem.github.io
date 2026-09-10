#!/usr/bin/env python3
"""Source-grounded AI pre-audit for frozen Stage D items; never substitutes for human review."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path("/home/jinhyun/prj_ws/jiho/AI/reenact/liref_outputs/rm_decomp/v2/d_feature_causal_v2_d06")
POPQA = Path("/home/jinhyun/prj_ws/jiho/AI/reenact/liref_outputs/dataset_analysis/popQa/popqa_test.json")


def main() -> None:
    items = pd.read_csv(ROOT / "manifests" / "confirmatory_items.csv")
    chains = items.groupby("chain_id", sort=True).first().reset_index()
    source = json.loads(POPQA.read_text(encoding="utf-8"))
    country_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    surface_countries: dict[str, set[str]] = defaultdict(set)
    for row in source:
        if row["prop"] != "country":
            continue
        country_rows[(str(row["subj"]), str(row["obj"]))].append(row)
        country_rows[(str(row.get("s_wiki_title") or row["subj"]), str(row["obj"]))].append(row)
        surface_countries[str(row["subj"]).casefold()].add(str(row["obj"]))
        surface_countries[str(row.get("s_wiki_title") or row["subj"]).casefold()].add(str(row["obj"]))

    output = []
    for _, chain in chains.iterrows():
        target_source = country_rows[(str(chain.entity), str(chain.country))]
        distractor_source = country_rows[(str(chain.distractor_entity), str(chain.distractor_country))]
        target_titles = sorted({str(row["s_wiki_title"]) for row in target_source})
        distractor_titles = sorted({str(row["s_wiki_title"]) for row in distractor_source})
        target_disambiguation = any(title.casefold() != str(chain.entity).casefold() for title in target_titles)
        distractor_disambiguation = any(title.casefold() != str(chain.distractor_entity).casefold() for title in distractor_titles)
        target_conflict = len(surface_countries[str(chain.entity).casefold()]) != 1
        distractor_conflict = len(surface_countries[str(chain.distractor_entity).casefold()]) != 1
        frame = items[items["chain_id"] == chain.chain_id]
        structural = bool(
            set(frame["condition"]) == {"A", "B", "C", "D"}
            and frame["capital"].nunique() == 1
            and frame["correct_label"].nunique() == 1
            and frame["options_json"].nunique() == 1
            and not frame["question"].str.contains(r"[.!?]{2,}", regex=True).any()
        )
        if not structural or not target_source or not distractor_source or target_conflict or distractor_conflict:
            verdict = "REJECT"
        elif target_disambiguation or distractor_disambiguation:
            verdict = "NEEDS_DISAMBIGUATION_REVIEW"
        else:
            verdict = "AI_PASS_TO_HUMAN"
        output.append({
            "chain_id": chain.chain_id,
            "verdict": verdict,
            "structural_pass": structural,
            "target_source_match": bool(target_source),
            "distractor_source_match": bool(distractor_source),
            "target_surface_country_conflict": target_conflict,
            "distractor_surface_country_conflict": distractor_conflict,
            "target_surface": chain.entity,
            "target_source_titles": " | ".join(target_titles),
            "target_disambiguation_needed": target_disambiguation,
            "distractor_surface": chain.distractor_entity,
            "distractor_source_titles": " | ".join(distractor_titles),
            "distractor_disambiguation_needed": distractor_disambiguation,
            "ai_note": "Source-relative facts are valid, but a different source title can make the prompt surface ambiguous; blind human review remains required.",
        })
    result = pd.DataFrame(output)
    target = ROOT / "human_audit" / "ai_full_audit.csv"
    result.to_csv(target, index=False)
    counts = result["verdict"].value_counts().to_dict()
    summary = {
        "audit_kind": "AI_SOURCE_GROUNDED_PRE_AUDIT_NOT_HUMAN",
        "chain_count": len(result),
        "verdict_counts": counts,
        "target_disambiguation_count": int(result["target_disambiguation_needed"].sum()),
        "distractor_disambiguation_count": int(result["distractor_disambiguation_needed"].sum()),
        "human_audit_substituted": False,
        "overall_status": "NEEDS_REVISION_OR_BLIND_HUMAN_ADJUDICATION" if counts.get("NEEDS_DISAMBIGUATION_REVIEW", 0) else "PASS_TO_BLIND_HUMAN_AUDIT",
    }
    (ROOT / "human_audit" / "ai_full_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
