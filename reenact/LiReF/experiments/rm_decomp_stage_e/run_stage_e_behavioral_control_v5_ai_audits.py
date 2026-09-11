#!/usr/bin/env python3
"""Exhaustive model-free AI-only audits for Stage E v5 datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = STAGE_DIR / "stage_e_behavioral_control_v5_assets"
PATHS = {
    "calibration_dataset": ASSET_DIR / "calibration_pool_dataset.json",
    "replication_dataset": ASSET_DIR / "replication_pool_dataset.json",
    "calibration_automatic": ASSET_DIR / "calibration_pool_automatic_audit.json",
    "replication_automatic": ASSET_DIR / "replication_pool_automatic_audit.json",
    "cross_pool": ASSET_DIR / "cross_pool_nonreuse_audit.json",
}
EXPECTED = {
    "calibration_dataset": "18005ac5c5733ab389f1b9f8d4850a671f4bdfa0f0c4755ddd62a43e4348922a",
    "replication_dataset": "b5bd9b083ebe7319bef02ebd30c064e9b9fe2efe85539b668ea8d1946f719dd3",
    "calibration_automatic": "ea2bc26a9a090167f63fdba65c4bb1c90d630f692c2cda3f9a6950f2a4e3fee4",
    "replication_automatic": "24f1538a7938c5aaddd677e17139cc26c07d5a0d896a8fac79203a0059674677",
    "cross_pool": "32efddb44897921ac06a21a9760cb7a2f64cf1f6d7af0de23631e1a6efca8a3f",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def primary_pair_audit(pair: dict[str, Any]) -> dict[str, bool]:
    b = pair["numeric_block"]
    f = pair["factors"]
    arithmetic = pair["conditions"]["arithmetic"]
    selector = pair["conditions"]["selector"]
    arithmetic_expected = b["high"] if f["arithmetic_operation"] == "ADD" else b["low"]
    selector_expected = b[f["selector_active_value"]]
    mapping = {int(value): choice for choice, value in pair["choice_mapping"].items()}
    return {
        "arithmetic_result_correct": arithmetic["canonical_numeric_result"] == arithmetic_expected,
        "arithmetic_requires_operation": all(marker in pair["arithmetic_block"] for marker in (str(b["start"]), str(b["delta"]), f["arithmetic_operation"])),
        "wrong_operation_foil_correct": arithmetic["foil_numeric_result"] == (b["low"] if arithmetic_expected == b["high"] else b["high"]),
        "selector_result_correct": selector["canonical_numeric_result"] == selector_expected,
        "selector_active_binding_explicit": f["selector_active_tag"] in pair["selector_block"] and str(selector_expected) in pair["selector_block"],
        "selector_inactive_foil_correct": selector["foil_numeric_result"] == (b["low"] if selector_expected == b["high"] else b["high"]),
        "arithmetic_choice_mapping_correct": arithmetic["correct_choice"] == mapping[arithmetic_expected],
        "selector_choice_mapping_correct": selector["correct_choice"] == mapping[selector_expected],
        "shared_candidate_mapping": pair["choice_mapping"] in ({"A":b["high"],"B":b["low"]},{"A":b["low"],"B":b["high"]}),
        "same_context": arithmetic["full_prompt"].split("\nQuestion:",1)[0] == selector["full_prompt"].split("\nQuestion:",1)[0],
        "question_targets_correct_labels": pair["arithmetic_label"] in arithmetic["question"] and pair["selector_label"] in selector["question"],
        "question_form_parallel": arithmetic["question"].replace(pair["arithmetic_label"],"<target>") == selector["question"].replace(pair["selector_label"],"<target>"),
        "answer_instruction_unambiguous": arithmetic["full_prompt"].endswith("Answer with A or B only.\nAnswer:") and selector["full_prompt"].endswith("Answer with A or B only.\nAnswer:"),
        "one_token_contract": all(len(c["correct_choice_token_ids"]) == len(c["alternative_choice_token_ids"]) == 1 for c in (arithmetic,selector)),
        "numeric_exposure_matched": pair["automatic_audit"]["candidate_numeric_exposure_equal"],
        "no_multihop_case_key_ledger": not any(word in pair["context"].lower() for word in ("ledger","record","docket","case file","lookup table","key ")),
        "language_complete": pair["context"].count(".") >= 2 and arithmetic["question"].endswith("?") and selector["question"].endswith("?"),
        "human_audit_disclosed": pair["human_audit"] == "not_performed",
    }


def audit_pool(pool: str, dataset: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    pair_rows = []
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in dataset["pairs"]:
        checks = primary_pair_audit(pair)
        status = "PASS" if all(checks.values()) else "FAIL"
        pair_rows.append({"pair_id":pair["pair_id"],"template_family_id":pair["template_family_id"],
                          "lexical_family":pair["lexical_family"],"status":status,"checks":checks})
        by_template[pair["template_family_id"]].append(pair)
    primary = {
        "schema_version":"5.0","audit_type":"primary_ai_only_semantic_linguistic_audit",
        "pool":pool,"status":"PASS" if all(r["status"]=="PASS" for r in pair_rows) else "FAIL",
        "pair_count":len(pair_rows),"pass_count":sum(r["status"]=="PASS" for r in pair_rows),
        "fail_count":sum(r["status"]!="PASS" for r in pair_rows),"pairs":pair_rows,
        "human_audit":"not_performed","human_audited_evidence":False,
        "model_loaded":False,"model_forward_performed":False,"gpu_used":False,
    }
    template_rows=[]
    for template, rows in sorted(by_template.items()):
        ar=Counter(p["conditions"]["arithmetic"]["correct_choice"] for p in rows)
        se=Counter(p["conditions"]["selector"]["correct_choice"] for p in rows)
        op=Counter(p["factors"]["arithmetic_operation"] for p in rows)
        active=Counter(p["factors"]["selector_active_value"] for p in rows)
        assignment=Counter(p["factors"]["candidate_assignment"] for p in rows)
        checks={
            "eight_frames":len(rows)==8,
            "arithmetic_A_B_balance":ar==Counter({"A":4,"B":4}),
            "selector_A_B_balance":se==Counter({"A":4,"B":4}),
            "operation_balance":op==Counter({"ADD":4,"SUBTRACT":4}),
            "active_high_low_balance":active==Counter({"high":4,"low":4}),
            "candidate_assignment_balance":assignment==Counter({"high_to_A":4,"high_to_B":4}),
            "no_fixed_condition_answer_coupling":len({(p["conditions"]["arithmetic"]["correct_choice"],p["conditions"]["selector"]["correct_choice"]) for p in rows})==4,
            "no_answer_copy_as_terminal_rule":all(p["conditions"]["arithmetic"]["canonical_numeric_result"] not in (p["numeric_block"]["start"],p["numeric_block"]["delta"]) for p in rows),
            "all_primary_pair_checks_pass":all(all(primary_pair_audit(p).values()) for p in rows),
        }
        template_rows.append({"template_family_id":template,"status":"PASS" if all(checks.values()) else "FAIL","checks":checks})
    adversarial={
        "schema_version":"5.0","audit_type":"adversarial_ai_only_shortcut_and_bias_audit",
        "pool":pool,"status":"PASS" if all(r["status"]=="PASS" for r in template_rows) else "FAIL",
        "template_count":len(template_rows),"pass_count":sum(r["status"]=="PASS" for r in template_rows),
        "fail_count":sum(r["status"]!="PASS" for r in template_rows),"pair_coverage":len(dataset["pairs"]),
        "templates":template_rows,"human_audit":"not_performed","human_audited_evidence":False,
        "model_loaded":False,"model_forward_performed":False,"gpu_used":False,
    }
    return primary,adversarial


def main() -> None:
    actual={name:sha(path) for name,path in PATHS.items()}
    if actual!=EXPECTED: raise RuntimeError(f"Locked audit input mismatch: {actual}")
    for key in ("calibration_automatic","replication_automatic","cross_pool"):
        if load(PATHS[key]).get("all_checks_pass") is not True:
            raise RuntimeError(f"Automatic prerequisite is not PASS: {key}")
    outputs={}
    counts={}
    for pool in ("calibration","replication"):
        dataset=load(PATHS[f"{pool}_dataset"])
        primary,adversarial=audit_pool(pool,dataset)
        primary_path=ASSET_DIR/f"{pool}_pool_primary_ai_audit.json"
        adversarial_path=ASSET_DIR/f"{pool}_pool_adversarial_ai_audit.json"
        write(primary_path,primary); write(adversarial_path,adversarial)
        outputs[f"{pool}_primary"]=sha(primary_path); outputs[f"{pool}_adversarial"]=sha(adversarial_path)
        counts[pool]={"pairs":primary["pair_count"],"primary_pass":primary["pass_count"],
                      "templates":adversarial["template_count"],"adversarial_pass":adversarial["pass_count"]}
        if primary["status"]!="PASS" or adversarial["status"]!="PASS":
            raise RuntimeError(f"AI audit failed: {pool}")
    summary={"schema_version":"5.0","status":"automatic_primary_ai_and_adversarial_ai_audits_all_pass",
             "counts":counts,"locked_input_hashes":actual,"audit_sha256":outputs,
             "human_audit":"not_performed","human_audited_evidence":False,
             "model_loaded":False,"model_forward_performed":False,"gpu_used":False,
             "calibration_execution_allowed":False,"replication_execution_allowed":False,
             "liref_loaded":False,"intervention_allowed":False}
    write(ASSET_DIR/"ai_audit_summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
