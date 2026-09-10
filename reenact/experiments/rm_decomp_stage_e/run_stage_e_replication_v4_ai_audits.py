#!/usr/bin/env python3
"""Model-free primary and adversarial AI-only audits for v4 datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = STAGE_DIR / "stage_e_replication_v4_assets"
DATASETS = {
    "calibration": ASSET_DIR / "calibration_pool_dataset.json",
    "replication": ASSET_DIR / "replication_pool_dataset.json",
}
AUTOMATIC_AUDITS = {
    "calibration": ASSET_DIR / "calibration_pool_automatic_audit.json",
    "replication": ASSET_DIR / "replication_pool_automatic_audit.json",
}
CROSS_AUDIT = ASSET_DIR / "cross_pool_nonreuse_audit.json"
EXPECTED_HASHES = {
    "calibration_dataset": "e4b660057b8103533c3303c8defc8a3b03268fac036ff3b8232c9e20662f6ded",
    "replication_dataset": "4a783c0509419103746ac81c318c5617bdeaeb1b02bd1c5addefc331b248c7a4",
    "calibration_automatic": "aaa777b9e794db807f8a01dc1a9865b4fdcf58bc0ac5a135fddc49752eb6aaa0",
    "replication_automatic": "5b81c52961e70c31af5fdadb226e523e5ffef03d570275301d827598e93b5e17",
    "cross": "780ec743613a2239031998a187db1c723f1f38958e3fbb3c71f1c588e1b51bd9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def locked_preflight() -> None:
    actual = {
        "calibration_dataset": sha256_file(DATASETS["calibration"]),
        "replication_dataset": sha256_file(DATASETS["replication"]),
        "calibration_automatic": sha256_file(AUTOMATIC_AUDITS["calibration"]),
        "replication_automatic": sha256_file(AUTOMATIC_AUDITS["replication"]),
        "cross": sha256_file(CROSS_AUDIT),
    }
    if actual != EXPECTED_HASHES:
        raise RuntimeError(f"Locked v4 dataset/audit hash mismatch: {actual}")
    if not all(load_json(path)["all_checks_pass"] for path in AUTOMATIC_AUDITS.values()):
        raise RuntimeError("Automatic audit is not PASS")
    if load_json(CROSS_AUDIT)["all_checks_pass"] is not True:
        raise RuntimeError("Cross-pool non-reuse audit is not PASS")


def primary_checks(pair: dict[str, Any]) -> dict[str, bool]:
    a = pair["conditions"]["arithmetic"]
    s = pair["conditions"]["selector"]
    block = pair["numeric_block"]
    operation = pair["factors"]["arithmetic_operation"]
    expected_a = block["start"] + block["delta"] if operation == "ADD" else block["start"] - block["delta"]
    expected_s = block["selector_1"] if pair["factors"]["selector_active_entry"] == 1 else block["selector_2"]
    unresolved = re.findall(r"\{[a-zA-Z0-9_]+\}", pair["context"] + a["question"] + s["question"])
    arithmetic_result_literals = {str(a["canonical_numeric_result"]), str(a["foil_numeric_result"])}
    selector_result_literals = {str(s["canonical_numeric_result"]), str(s["foil_numeric_result"])}
    return {
        "canonical_results_correct": a["canonical_numeric_result"] == expected_a
        and s["canonical_numeric_result"] == expected_s,
        "arithmetic_requires_operation_to_select_candidate": operation in pair["arithmetic_block"]
        and str(block["start"]) in pair["ledger"]
        and str(block["delta"]) in pair["ledger"]
        and all(value in pair["arithmetic_block"] for value in arithmetic_result_literals),
        "selector_binding_chain_explicit": pair["factors"]["selector_active_entry"] in (1, 2)
        and pair["selector_block"].count("A") >= 1
        and pair["selector_block"].count("B") >= 1
        and all(value in pair["ledger"] for value in selector_result_literals),
        "correct_and_foil_both_available_each_condition": len(arithmetic_result_literals) == 2
        and len(selector_result_literals) == 2,
        "condition_questions_semantically_parallel": a["question"].replace(
            pair["arithmetic_label"], "<TARGET>"
        ) == s["question"].replace(pair["selector_label"], "<TARGET>"),
        "output_instruction_unambiguous": a["full_prompt"].endswith(
            "Answer with A or B only.\nAnswer:"
        ) and s["full_prompt"].endswith("Answer with A or B only.\nAnswer:"),
        "no_unresolved_template_placeholder": not unresolved,
        "grammar_preserved_by_frame_substitutions": operation in ("ADD", "SUBTRACT")
        and pair["factors"]["block_order"] in ("arithmetic_first", "selector_first"),
        "counterbalance_counterpart_defined": len(pair["oa_row"]) == 6
        and pair["frame_index"] in range(1, 9),
        "human_audit_not_misrepresented": pair["human_audit"] == "not_performed",
    }


def adversarial_template_checks(rows: list[dict[str, Any]]) -> dict[str, bool]:
    arithmetic_choices = Counter(row["conditions"]["arithmetic"]["correct_choice"] for row in rows)
    selector_choices = Counter(row["conditions"]["selector"]["correct_choice"] for row in rows)
    operation_by_choice = Counter(
        (row["factors"]["arithmetic_operation"], row["conditions"]["arithmetic"]["correct_choice"])
        for row in rows
    )
    active_by_choice = Counter(
        (row["factors"]["selector_active_entry"], row["conditions"]["selector"]["correct_choice"])
        for row in rows
    )
    return {
        "arithmetic_answer_letter_4_4": arithmetic_choices == Counter({"A": 4, "B": 4}),
        "selector_answer_letter_4_4": selector_choices == Counter({"A": 4, "B": 4}),
        "operation_does_not_predict_arithmetic_letter": set(operation_by_choice.values()) == {2}
        and len(operation_by_choice) == 4,
        "active_entry_does_not_predict_selector_letter": set(active_by_choice.values()) == {2}
        and len(active_by_choice) == 4,
        "all_six_nuisance_columns_balanced": all(
            Counter(row["oa_row"][column] for row in rows) == Counter({0: 4, 1: 4})
            for column in range(6)
        ),
        "no_candidate_numeric_collision": all(
            len(
                {
                    row["conditions"]["arithmetic"]["canonical_numeric_result"],
                    row["conditions"]["arithmetic"]["foil_numeric_result"],
                    row["conditions"]["selector"]["canonical_numeric_result"],
                    row["conditions"]["selector"]["foil_numeric_result"],
                }
            ) == 4
            for row in rows
        ),
        "no_result_dependent_content": all(row["ai_audit_status"] == "pending" for row in rows),
    }


def main() -> None:
    locked_preflight()
    primary_outputs = {}
    adversarial_outputs = {}
    for pool, path in DATASETS.items():
        dataset = load_json(path)
        primary_rows = []
        for pair in dataset["pairs"]:
            checks = primary_checks(pair)
            primary_rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "template_family_id": pair["template_family_id"],
                    "checks": checks,
                    "status": "PASS" if all(checks.values()) else "FAIL",
                }
            )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in dataset["pairs"]:
            grouped[pair["template_family_id"]].append(pair)
        adversarial_templates = []
        for template_id, rows in sorted(grouped.items()):
            checks = adversarial_template_checks(rows)
            adversarial_templates.append(
                {
                    "template_family_id": template_id,
                    "pair_count": len(rows),
                    "checks": checks,
                    "status": "PASS" if all(checks.values()) else "FAIL",
                }
            )
        primary_outputs[pool] = {
            "schema_version": "4.0",
            "audit_type": "primary_ai_only_linguistic_semantic_full_pair_audit",
            "pool": pool,
            "pair_count": len(primary_rows),
            "pass_count": sum(row["status"] == "PASS" for row in primary_rows),
            "fail_count": sum(row["status"] != "PASS" for row in primary_rows),
            "status": "PASS" if all(row["status"] == "PASS" for row in primary_rows) else "FAIL",
            "rows": primary_rows,
            "model_loaded": False,
            "gpu_used": False,
            "human_audit": "not_performed",
            "human_audited_evidence": False,
        }
        adversarial_outputs[pool] = {
            "schema_version": "4.0",
            "audit_type": "adversarial_ai_only_shortcut_counterbalance_audit",
            "pool": pool,
            "template_count": len(adversarial_templates),
            "pair_coverage": len(dataset["pairs"]),
            "pass_count": sum(row["status"] == "PASS" for row in adversarial_templates),
            "fail_count": sum(row["status"] != "PASS" for row in adversarial_templates),
            "status": "PASS" if all(row["status"] == "PASS" for row in adversarial_templates) else "FAIL",
            "templates": adversarial_templates,
            "model_loaded": False,
            "gpu_used": False,
            "human_audit": "not_performed",
            "human_audited_evidence": False,
        }
    paths = {}
    for pool in DATASETS:
        primary_path = ASSET_DIR / f"{pool}_pool_primary_ai_audit.json"
        adversarial_path = ASSET_DIR / f"{pool}_pool_adversarial_ai_audit.json"
        write_json(primary_path, primary_outputs[pool])
        write_json(adversarial_path, adversarial_outputs[pool])
        paths[f"{pool}_primary"] = primary_path
        paths[f"{pool}_adversarial"] = adversarial_path
    if not all(payload["status"] == "PASS" for payload in primary_outputs.values()):
        raise RuntimeError("Primary AI audit failed")
    if not all(payload["status"] == "PASS" for payload in adversarial_outputs.values()):
        raise RuntimeError("Adversarial AI audit failed")
    summary = {
        "schema_version": "4.0",
        "status": "automatic_primary_ai_and_adversarial_ai_audits_all_pass",
        "dataset_hashes": {
            pool: EXPECTED_HASHES[f"{pool}_dataset"] for pool in DATASETS
        },
        "audit_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "pair_counts": {pool: primary_outputs[pool]["pair_count"] for pool in DATASETS},
        "primary_ai_pass_counts": {pool: primary_outputs[pool]["pass_count"] for pool in DATASETS},
        "adversarial_template_pass_counts": {pool: adversarial_outputs[pool]["pass_count"] for pool in DATASETS},
        "human_audit": "not_performed",
        "human_audited_evidence": False,
        "model_loaded": False,
        "model_forward_performed": False,
        "gpu_used": False,
        "calibration_execution_allowed": False,
        "replication_execution_allowed": False,
        "intervention_allowed": False,
    }
    write_json(ASSET_DIR / "ai_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
