#!/usr/bin/env python3
"""Run the model-blind, non-human full linguistic/semantic audit for v3.

The script reads only frozen design and dataset artifacts.  It never loads a
tokenizer or model, performs a forward pass, uses a GPU, or accesses LiReF,
candidate activations, hooks, or interventions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


STAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = STAGE_DIR / "calibration_v3_assets"
OUTPUT_DIR = ASSET_DIR / "ai_audit"
DATASET_PATH = ASSET_DIR / "calibration_v3_dataset_draft.json"
MANIFEST_PATH = ASSET_DIR / "calibration_v3_dataset_manifest.json"
AUTOMATIC_AUDIT_PATH = ASSET_DIR / "calibration_v3_automatic_audit.json"
DESIGN_PATH = STAGE_DIR / "calibration_v3_design_frozen.json"

EXPECTED_DATASET_SHA256 = "d2187c0623ba9752776cf0251dee3dabf9d80ac04e339cf3eb4bd1d1b42761a1"
EXPECTED_MANIFEST_SHA256 = "a157ec7bd463c739ee046f9e3f85a08d2e3ebb7dc6a794a63757653c93fad822"
EXPECTED_AUTOMATIC_AUDIT_SHA256 = "e904fcec13b97bf9d09afc707aedd2b37c100c911b560f2b88f0ed270e654e26"
EXPECTED_DESIGN_SHA256 = "c60a579729376d391582dbc03af9cfd3ba0a1e1743a9e9a884967aacc177adfc"
REVIEWER_ID = "codex_ai_audit_nonhuman_v3_20260830"


# These notes are the explicit template-level reading pass.  Pair-level checks
# below separately validate all 8 numeric/role/order realizations per template.
TEMPLATE_REVIEWS: dict[str, str] = {
    "v3_obj_lantern_dossier": "Case routing, ADD/SUB operands, selector entries, ledger, and current-count question are explicit and natural.",
    "v3_obj_scarf_registry": "Registry routing and the original-total/adjustment distinction are clear; selector choice is unambiguous.",
    "v3_obj_magnet_portfolio": "Baseline/change computation and seal-row selection are explicit and semantically parallel.",
    "v3_obj_postcard_journal": "Start/change pointers and marker-selected pointer retrieval are clear in both operation directions.",
    "v3_obj_button_compendium": "START/CHANGE and tagged line selection are explicit; the singular 'number ... belongs' question is grammatical.",
    "v3_obj_gear_inventory": "First/second operand order and code-selected key lookup are stated directly and naturally.",
    "v3_obj_mug_directory": "Initial/update computation and badge-linked key retrieval are clear and target-matched.",
    "v3_obj_flag_workbook": "Source/shift action and symbol-addressed retrieval have an unambiguous case-to-ledger path.",
    "v3_pts_quiz_scorecard": "Opening score and change are explicit; selector options resolve cleanly to ledger values.",
    "v3_pts_arcade_tally": "Base/adjustment arithmetic and active-token selection are clear and symmetric enough for the contrast.",
    "v3_pts_reward_sheet": "Beginning value, adjustment, and selector stamp have unambiguous balance semantics.",
    "v3_pts_league_record": "Prior points and point change are explicit; branch-selected direct retrieval is clear.",
    "v3_pts_loyalty_folio": "Origin/movement arithmetic and seal-selected lookup are natural and unambiguous.",
    "v3_pts_contest_docket": "Starting score/change operation and active-sign lookup are clearly distinguished.",
    "v3_pts_merit_index": "Baseline/revision arithmetic and emblem-resolved lookup form clear matched paths.",
    "v3_pts_challenge_book": "INPUT/CHANGE operation and selector routing are explicit; the point-line question is natural.",
    "v3_tmp_terrarium_monitor": "Initial reading plus WARM/COOL degree change and selector entry retrieval are explicit.",
    "v3_tmp_kiln_tracker": "Starting temperature, shift, and WARM/COOL directive are clear; active-mark lookup is unambiguous.",
    "v3_tmp_cooler_station": "Original reading/variation and physical cooler target make WARM/COOL direction clear.",
    "v3_tmp_reactor_panel": "Source/change temperature rule and selector rune both resolve through explicit keys.",
    "v3_tmp_greenhouse_sensor": "Baseline/adjustment temperature command and selected-glyph lookup are natural and clear.",
    "v3_tmp_vault_gauge": "Initial/change keys and active-crest retrieval are explicit with an unambiguous temperature question.",
    "v3_tmp_basin_console": "Earlier temperature/change update and flag-based selection are clear in both directions.",
    "v3_tmp_module_thermometer": "Reference/degree-shift operation and selector-badge retrieval are explicit and grammatical.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def numeric_mentions(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"(?<![A-Za-z0-9])-?\d+(?![A-Za-z0-9])", text)]


def question_skeleton(question: str, target: str) -> str:
    if question.count(target) != 1:
        return "INVALID_TARGET_COUNT"
    return question.replace(target, "<TARGET>")


def expected_value_keys(pair: dict[str, Any], candidate_pair: str) -> list[str]:
    return pair[f"value_key_pair_{candidate_pair}"]


def audit_pair(
    pair: dict[str, Any], prompt_template: str, frozen_matrix: list[list[int]]
) -> tuple[dict[str, str], list[str]]:
    arithmetic = pair["conditions"]["arithmetic"]
    selector = pair["conditions"]["selector"]
    block = pair["numeric_block"]
    arithmetic_values = block[pair["arithmetic_candidate_pair"]]
    selector_values = block[pair["selector_candidate_pair"]]
    arithmetic_keys = expected_value_keys(pair, pair["arithmetic_candidate_pair"])
    selector_keys = expected_value_keys(pair, pair["selector_candidate_pair"])
    start = int(pair["start"])
    delta = int(pair["delta"])
    if pair["operation"] == "add":
        expected_arithmetic = start + delta
        expected_foil = start - delta
        expected_word = "WARM" if pair["lexical_family"] == "temperature" else "ADD"
    else:
        expected_arithmetic = start - delta
        expected_foil = start + delta
        expected_word = "COOL" if pair["lexical_family"] == "temperature" else "SUB"
    expected_selector = selector_values["low"] if pair["selector_active_entry"] == 1 else selector_values["high"]
    expected_selector_foil = selector_values["high"] if pair["selector_active_entry"] == 1 else selector_values["low"]
    active_tag = pair["tag_pair"][pair["selector_active_entry"] - 1]
    expected_context = " ".join(
        [pair["arithmetic_block"], pair["selector_block"], pair["value_ledger_sentence"]]
        if pair["channel_block_order"] == "arithmetic_first"
        else [pair["selector_block"], pair["arithmetic_block"], pair["value_ledger_sentence"]]
    )
    context_numbers = numeric_mentions(pair["context"])
    ledger_numbers = numeric_mentions(pair["value_ledger_sentence"])
    arithmetic_correct = int(pair["arithmetic_answer"])
    arithmetic_foil = int(pair["arithmetic_primary_foil"])
    selector_correct = int(pair["selector_answer"])
    selector_foil = int(pair["selector_primary_foil"])
    factors = pair["factors"]
    factor_row = [
        factors["A_candidate_pair_assignment"],
        factors["B_arithmetic_operation"],
        factors["C_selector_active_entry"],
        factors["D_label_case_role"],
        factors["E_channel_block_order"],
    ]

    checks = {
        "arithmetic_answer_and_wrong_operation_foil_correct": (
            arithmetic_correct == expected_arithmetic
            and arithmetic_foil == expected_foil
            and int(arithmetic["canonical_answer"]) == expected_arithmetic
            and int(arithmetic["primary_alternative_answer"]) == expected_foil
        ),
        "selector_answer_and_competing_entry_foil_correct": (
            selector_correct == expected_selector
            and selector_foil == expected_selector_foil
            and int(selector["canonical_answer"]) == expected_selector
            and int(selector["primary_alternative_answer"]) == expected_selector_foil
        ),
        "operation_word_and_direction_unambiguous": (
            pair["operation_word"] == expected_word
            and pair["arithmetic_case_sentence"].count(expected_word) == 1
            and pair["arithmetic_case_sentence"].count(arithmetic_keys[0]) == 1
            and pair["arithmetic_case_sentence"].count(arithmetic_keys[1]) == 1
        ),
        "selector_tag_entry_binding_unambiguous": (
            pair["selector_case_sentence"].count(active_tag) >= 2
            and all(pair["selector_case_sentence"].count(tag) >= 1 for tag in pair["tag_pair"])
            and all(pair["selector_case_sentence"].count(key) == 1 for key in selector_keys)
        ),
        "matched_label_case_key_ledger_paths_valid": (
            pair["arithmetic_mapping_sentence"].count(pair["arithmetic_attribute"]) == 1
            and pair["arithmetic_mapping_sentence"].count(pair["arithmetic_case_key"]) == 1
            and pair["selector_mapping_sentence"].count(pair["selector_attribute"]) == 1
            and pair["selector_mapping_sentence"].count(pair["selector_case_key"]) == 1
            and pair["arithmetic_case_sentence"].count(pair["arithmetic_case_key"]) == 1
            and pair["selector_case_sentence"].count(pair["selector_case_key"]) == 1
            and all(pair["value_ledger_sentence"].count(key) == 1 for key in arithmetic_keys + selector_keys)
        ),
        "identical_context_and_prompt_contract": (
            pair["context"] == expected_context
            and arithmetic["full_question_text"] == f'{pair["context"]} {arithmetic["question"]}'
            and selector["full_question_text"] == f'{pair["context"]} {selector["question"]}'
            and arithmetic["full_prompt"] == prompt_template.format(question=arithmetic["full_question_text"])
            and selector["full_prompt"] == prompt_template.format(question=selector["full_question_text"])
        ),
        "question_target_only_change": (
            question_skeleton(arithmetic["question"], pair["arithmetic_attribute"])
            == question_skeleton(selector["question"], pair["selector_attribute"])
        ),
        "within_condition_exposure_contract": (
            context_numbers.count(arithmetic_correct) == 0
            and context_numbers.count(arithmetic_foil) == 0
            and ledger_numbers.count(selector_correct) == 1
            and ledger_numbers.count(selector_foil) == 1
        ),
        "arithmetic_transformation_required": (
            arithmetic["target_attribute"] == pair["arithmetic_attribute"]
            and ledger_numbers.count(start) == 1
            and ledger_numbers.count(delta) == 1
            and arithmetic_correct not in context_numbers
            and arithmetic_foil not in context_numbers
        ),
        "selector_matched_retrieval_required": (
            selector["target_attribute"] == pair["selector_attribute"]
            and active_tag in pair["selector_case_sentence"]
            and selector_correct in ledger_numbers
            and selector_foil in ledger_numbers
        ),
        "numeric_collision_and_shortcut_absence": (
            len({arithmetic_correct, arithmetic_foil, start, delta, selector_correct, selector_foil}) == 6
            and arithmetic_correct != selector_correct
        ),
        "one_numeral_output_instruction_clear": (
            arithmetic["full_prompt"].endswith("Answer with one Arabic numeral only.\nA: ")
            and selector["full_prompt"].endswith("Answer with one Arabic numeral only.\nA: ")
            and arithmetic["accepted_answers"] == [str(arithmetic_correct)]
            and selector["accepted_answers"] == [str(selector_correct)]
            and arithmetic["canonical_answer_continuation_token_count"] == 1
            and arithmetic["alternative_answer_continuation_token_count"] == 1
            and selector["canonical_answer_continuation_token_count"] == 1
            and selector["alternative_answer_continuation_token_count"] == 1
        ),
        "frozen_oa_row_and_metadata_consistent": (
            pair["oa_row"] == frozen_matrix[pair["frame_index"] - 1]
            and factor_row == pair["oa_row"]
            and ((factor_row[0] == 0 and pair["arithmetic_candidate_pair"] == "P") or (factor_row[0] == 1 and pair["arithmetic_candidate_pair"] == "Q"))
            and ((factor_row[1] == 0 and pair["operation"] == "add") or (factor_row[1] == 1 and pair["operation"] == "subtract"))
            and pair["selector_active_entry"] == factor_row[2] + 1
        ),
        "automatic_pair_checks_all_pass": all(pair["automatic_audit"].values()),
        "natural_grammar_and_semantic_clarity": pair["template_family_id"] in TEMPLATE_REVIEWS,
    }
    failures = [name for name, passed in checks.items() if not passed]
    status = "AI_AUDIT_PASS" if not failures else "AI_AUDIT_FAIL"
    row = {
        "pair_id": pair["pair_id"],
        "lexical_family": pair["lexical_family"],
        "template_family_id": pair["template_family_id"],
        "frame_index": str(pair["frame_index"]),
        "arithmetic_operation": pair["operation"],
        "selector_active_entry": str(pair["selector_active_entry"]),
        **{name: "YES" if passed else "NO" for name, passed in checks.items()},
        "reviewer_id": REVIEWER_ID,
        "review_status": status,
        "comments": ";".join(failures),
        "satisfies_independent_human_gate": "NO",
    }
    return row, failures


def audit_template_counterbalance(
    pairs: list[dict[str, Any]], frozen_matrix: list[list[int]]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["template_family_id"]].append(pair)
    reports: dict[str, Any] = {}
    expected_rows = {tuple(row) for row in frozen_matrix}
    for template_id, selected in sorted(grouped.items()):
        rows = [tuple(pair["oa_row"]) for pair in selected]
        factor_counts = {
            index: Counter(row[index] for row in rows) for index in range(5)
        }
        pairwise_counts = {
            f"{left}{right}": Counter((row[left], row[right]) for row in rows)
            for left, right in combinations(range(5), 2)
        }
        candidate_role_counts = Counter(pair["arithmetic_candidate_pair"] for pair in selected)
        operation_counts = Counter(pair["operation"] for pair in selected)
        selector_entry_counts = Counter(pair["selector_active_entry"] for pair in selected)
        arithmetic_label_counts = Counter(pair["arithmetic_attribute"] for pair in selected)
        selector_label_counts = Counter(pair["selector_attribute"] for pair in selected)
        order_counts = Counter(pair["channel_block_order"] for pair in selected)
        checks = {
            "eight_pairs": len(selected) == 8,
            "exact_frozen_oa_rows": set(rows) == expected_rows and len(rows) == len(set(rows)),
            "each_factor_4_to_4": all(counts == Counter({0: 4, 1: 4}) for counts in factor_counts.values()),
            "every_factor_pair_has_each_cell_twice": all(
                counts == Counter({(0, 0): 2, (0, 1): 2, (1, 0): 2, (1, 1): 2})
                for counts in pairwise_counts.values()
            ),
            "candidate_pair_assignment_4_to_4": candidate_role_counts == Counter({"P": 4, "Q": 4}),
            "add_sub_4_to_4": operation_counts == Counter({"add": 4, "subtract": 4}),
            "selector_entry_4_to_4": selector_entry_counts == Counter({1: 4, 2: 4}),
            "arithmetic_label_role_4_to_4": sorted(arithmetic_label_counts.values()) == [4, 4],
            "selector_label_role_4_to_4": sorted(selector_label_counts.values()) == [4, 4],
            "channel_order_4_to_4": order_counts == Counter({"arithmetic_first": 4, "selector_first": 4}),
        }
        reports[template_id] = {
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "linguistic_review_status": "PASS",
            "linguistic_review_note": TEMPLATE_REVIEWS[template_id],
        }
    return reports


def main() -> None:
    locked_paths = {
        DATASET_PATH: EXPECTED_DATASET_SHA256,
        MANIFEST_PATH: EXPECTED_MANIFEST_SHA256,
        AUTOMATIC_AUDIT_PATH: EXPECTED_AUTOMATIC_AUDIT_SHA256,
        DESIGN_PATH: EXPECTED_DESIGN_SHA256,
    }
    for path, expected_hash in locked_paths.items():
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Artifact hash mismatch: {path.name}: {actual_hash}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    automatic_audit = json.loads(AUTOMATIC_AUDIT_PATH.read_text(encoding="utf-8"))
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if dataset["pair_count"] != 192 or len(dataset["pairs"]) != 192 or dataset["prompt_count"] != 384:
        raise RuntimeError("Expected exactly 192 pairs and 384 prompts")
    if manifest["baseline_calibration_execution_allowed"] is not False:
        raise RuntimeError("Baseline Calibration execution unexpectedly allowed")
    if manifest["stage_e_pilot_allowed"] is not False:
        raise RuntimeError("Stage E Pilot unexpectedly allowed")
    if not automatic_audit["all_automatic_checks_pass"]:
        raise RuntimeError("Locked automatic audit is not fully PASS")
    template_ids = {pair["template_family_id"] for pair in dataset["pairs"]}
    if set(TEMPLATE_REVIEWS) != template_ids:
        raise RuntimeError("Template-family set differs from the explicitly reviewed set")

    frozen_matrix = design["counterbalance"]["matrix"]
    prompt_template = design["prompt_output_contract"]["template"]
    rows: list[dict[str, str]] = []
    failures: list[dict[str, Any]] = []
    for pair in dataset["pairs"]:
        row, pair_failures = audit_pair(pair, prompt_template, frozen_matrix)
        rows.append(row)
        if pair_failures:
            failures.append({"pair_id": pair["pair_id"], "failures": pair_failures})

    template_reports = audit_template_counterbalance(dataset["pairs"], frozen_matrix)
    template_failures = [
        template_id for template_id, report in template_reports.items()
        if not report["all_checks_pass"]
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "calibration_v3_ai_linguistic_audit.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)

    counts = Counter(row["review_status"] for row in rows)
    dataset_pass = (
        counts["AI_AUDIT_PASS"] == 192
        and not failures
        and not template_failures
    )
    summary = {
        "schema_version": "3.0",
        "status": (
            "ai_audit_pass_independent_human_audit_required"
            if dataset_pass else "ai_audit_fail_dataset_revision_required"
        ),
        "reviewer_id": REVIEWER_ID,
        "reviewer_type": "nonhuman_ai",
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "dataset_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "automatic_audit_sha256": EXPECTED_AUTOMATIC_AUDIT_SHA256,
        "pair_count": len(rows),
        "reviewed_pair_count": len(rows),
        "ai_audit_pass_count": counts["AI_AUDIT_PASS"],
        "ai_audit_fail_count": counts["AI_AUDIT_FAIL"],
        "pair_failures": failures,
        "template_family_count": len(template_reports),
        "template_failures": template_failures,
        "template_reports": template_reports,
        "ai_audit_csv_sha256": sha256_file(csv_path),
        "dataset_pass": dataset_pass,
        "official_dataset_modified": False,
        "independent_human_audit_started": False,
        "independent_human_audit_complete": False,
        "human_audit_waiver_allowed": False,
        "baseline_calibration_execution_allowed": False,
        "stage_e_pilot_allowed": False,
        "model_or_tokenizer_loaded": False,
        "model_forward_performed": False,
        "gpu_used": False,
        "liref_or_candidate_state_accessed": False,
        "claim_scope_note": (
            "This validates the controlled arithmetic-transformation versus selector-guided matched-retrieval dataset. "
            "It does not establish a Reasoning-versus-Memorization contrast or satisfy the independent human-audit gate."
        ),
    }
    summary_path = OUTPUT_DIR / "calibration_v3_ai_linguistic_audit_summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
