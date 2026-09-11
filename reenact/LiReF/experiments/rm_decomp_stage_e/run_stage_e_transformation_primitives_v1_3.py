#!/usr/bin/env python3
"""Forced-choice local annotation for Transformation primitive features."""

from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any


STAGE_DIR = Path(__file__).resolve().parent
BASE_RUNNER = STAGE_DIR / "run_stage_e_transformation_primitives_v1_2.py"


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("transformation_primitives_v1_2_base", BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_base()
BASE.MANIFEST_PATH = STAGE_DIR / "stage_e_transformation_primitives_v1_3_design_frozen.json"
BASE.AUTHORIZATION_PATH = STAGE_DIR / "stage_e_transformation_primitives_v1_3_local_execution_authorization_frozen.json"
BASE.ASSET_DIR = STAGE_DIR / "stage_e_transformation_primitives_v1_3_assets"
BASE.OUTPUT_DIR = BASE.ROOT / "liref_outputs" / "rm_decomp" / "v2" / "e_transformation_primitives_v1_3"

PRIMITIVE_QUESTIONS = {
    "NUM": "Does solving require actual numerical calculation, numerical comparison, mathematical expression manipulation, ratio computation, or unit conversion?",
    "RULE": "Does solving require applying a formula, scientific or domain law, grammar or legal rule, classification rule, or explicit procedure?",
    "REL": "Does solving require combining two or more given facts, quantities, or relations, beyond reading or directly comparing one fact with the options?",
    "COND": "Does solving require conditional, propositional, negation, quantifier, case-split, ordering, or constraint logic?",
    "CAUS": "Does solving require tracing a cause, mechanism, intervention, or counterfactual to an effect?",
    "INTER": "Does solving require creating a derived intermediate result and then using it in a later distinct judgment or operation?",
}

SYSTEM = """You perform one blind binary annotation of a multiple-choice item. The item was previously validated as requiring some transformation. Do not reconsider that parent label, solve the item, identify the correct option, or explain. Answer only the single token Y or N for the asked primitive. Use Y only when the operation is required to solve the item, not merely mentioned."""


def prompt_text(tokenizer: Any, item: dict[str, Any], primitive: str) -> str:
    user = f"{BASE.render_item(item)}\n\n{PRIMITIVE_QUESTIONS[primitive]}\nAnswer only Y or N."
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"System: {SYSTEM}\nUser: {user}\nAssistant:"


def forced_choice(model: Any, tokenizer: Any, items: list[dict[str, Any]], batch_size: int, device: str) -> list[dict[str, Any]]:
    import torch
    y_tokens = tokenizer.encode("Y", add_special_tokens=False)
    n_tokens = tokenizer.encode("N", add_special_tokens=False)
    if len(y_tokens) != 1 or len(n_tokens) != 1 or y_tokens[0] == n_tokens[0]:
        raise RuntimeError("Y/N candidates must be distinct single tokens")
    tasks = [(item, primitive) for item in items for primitive in BASE.PRIMITIVES]
    results = {item["annotation_id"]: {"annotation_id": item["annotation_id"], "parse_valid": True, "tie_count": 0} for item in items}
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        prompts = [prompt_text(tokenizer, item, primitive) for item, primitive in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, :]
        y_values = logits[:, y_tokens[0]].float().cpu().tolist()
        n_values = logits[:, n_tokens[0]].float().cpu().tolist()
        for (item, primitive), y_value, n_value in zip(batch, y_values, n_values):
            margin = float(y_value - n_value)
            choice = "Y" if margin > 0 else "N"
            row = results[item["annotation_id"]]
            row[primitive] = choice
            row[f"{primitive}_yn_logit_margin"] = margin
            if margin == 0:
                row["tie_count"] += 1
        del encoded, logits
    return [results[key] for key in sorted(results)]


def phase_prepare(manifest: dict[str, Any]) -> None:
    blind_all = {row["annotation_id"]: row for row in BASE.read_jsonl(BASE.resolve(manifest["inputs"]["blind_items"]["path"]))}
    key = {row["annotation_id"]: row for row in BASE.read_jsonl(BASE.resolve(manifest["inputs"]["annotation_key"]["path"]))}
    parent = {row["annotation_id"]: row for row in BASE.read_jsonl(BASE.resolve(manifest["inputs"]["parent_consensus"]["path"]))}
    dataset = BASE.read_json(BASE.resolve(manifest["inputs"]["dataset"]["path"]))
    if not (set(blind_all) == set(key) == set(parent)) or len(parent) != 3000:
        raise RuntimeError("Parent coverage mismatch")
    parent_y = sorted(item for item, row in parent.items() if row["transformation_required"] == "Y")
    parent_n = sorted(item for item, row in parent.items() if row["transformation_required"] == "N")
    parent_unc = sorted(item for item, row in parent.items() if row["transformation_required"] == "UNC")
    if (len(parent_y), len(parent_n), len(parent_unc)) != (895, 1739, 366):
        raise RuntimeError("Unexpected parent counts")
    private = []
    for item in parent_y:
        index = int(key[item]["row_index"]); record = dataset[index]
        private.append({"annotation_id": item, "row_index": index, "category": str(record["category"]), "src": str(record["src"])})
    prior_ids = set()
    for name in ("v1_preflight_key", "v1_1_preflight_key", "v1_2_preflight_key"):
        prior_ids.update(row["annotation_id"] for row in BASE.read_jsonl(BASE.resolve(manifest["inputs"][name]["path"])))
    eligible = [row for row in private if row["annotation_id"] not in prior_ids]
    selected = BASE.category_diverse_sample(eligible, 96, BASE.SEED + 33)
    if {row["annotation_id"] for row in selected} & prior_ids:
        raise RuntimeError("Preflight overlap detected")
    blind = [blind_all[item] for item in parent_y]
    BASE.atomic_jsonl(BASE.ASSET_DIR / "blind_parent_y_items.jsonl", blind)
    BASE.atomic_jsonl(BASE.ASSET_DIR / "private_key_DO_NOT_USE_DURING_ANNOTATION.jsonl", private)
    BASE.atomic_jsonl(BASE.ASSET_DIR / "preflight_blind_items.jsonl", [blind_all[row["annotation_id"]] for row in selected])
    BASE.atomic_jsonl(BASE.ASSET_DIR / "preflight_private_key.jsonl", selected)
    BASE.atomic_json(BASE.ASSET_DIR / "preparation_audit.json", {
        "status": "PASS", "parent_y_rows": 895, "parent_n_rows": 1739, "parent_unc_rows": 366,
        "preflight_rows": 96, "preflight_disjoint_from_v1_v1_1_v1_2": True,
        "preflight_category_counts": dict(Counter(row["category"] for row in selected)),
        "blind_fields": ["annotation_id", "question", "options"], "response_mode": "teacher_forced_y_vs_n",
        "external_api_used": False,
        "blind_parent_y_sha256": BASE.sha256_file(BASE.ASSET_DIR / "blind_parent_y_items.jsonl"),
        "preflight_blind_sha256": BASE.sha256_file(BASE.ASSET_DIR / "preflight_blind_items.jsonl"),
    })


def phase_final_audit() -> None:
    artifacts = {
        "manifest": BASE.MANIFEST_PATH, "authorization": BASE.AUTHORIZATION_PATH,
        "preflight": BASE.ASSET_DIR / "preflight_result.json", "reliability": BASE.ASSET_DIR / "full_reliability_result.json",
        "consensus": BASE.ASSET_DIR / "consensus_annotations.jsonl",
        "discovery": BASE.OUTPUT_DIR / "tables" / "discovery_within_transformation_associations.csv",
        "baseline": BASE.OUTPUT_DIR / "tables" / "discovery_parent_n_baseline_descriptive.csv",
        "cooccurrence": BASE.OUTPUT_DIR / "tables" / "discovery_primitive_cooccurrence.csv",
        "selection": BASE.OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json",
        "heldout": BASE.OUTPUT_DIR / "tables" / "heldout_selected_checks.csv",
        "summary": BASE.OUTPUT_DIR / "transformation_primitives_summary.json",
    }
    missing = [name for name, path in artifacts.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing artifacts: {missing}")
    blind = BASE.read_jsonl(BASE.ASSET_DIR / "blind_parent_y_items.jsonl")
    if len(blind) != 895 or any(set(row) != {"annotation_id", "question", "options"} for row in blind):
        raise RuntimeError("Blind annotation audit failed")
    BASE.atomic_json(BASE.OUTPUT_DIR / "final_audit.json", {
        "status": "PASS", "response_mode": "teacher_forced_y_vs_n", "blind_annotation_leakage": False,
        "external_api_used": False, "new_study_model_forward": False, "intervention_performed": False,
        "artifact_sha256": {name: BASE.sha256_file(path) for name, path in artifacts.items()},
        "implementation_sha256": BASE.sha256_file(Path(__file__)),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "preflight_a", "preflight_b", "audit_preflight", "annotate_a", "annotate_b", "consensus", "analyze_discovery", "analyze_heldout", "final_audit"))
    args = parser.parse_args(); manifest, authorization = BASE.verify()
    BASE.generate = forced_choice
    if args.phase == "prepare": phase_prepare(manifest)
    elif args.phase == "preflight_a": BASE.phase_annotate(manifest, authorization, "annotator_a", True)
    elif args.phase == "preflight_b": BASE.phase_annotate(manifest, authorization, "annotator_b", True)
    elif args.phase == "audit_preflight": BASE.phase_audit_preflight()
    elif args.phase == "annotate_a": BASE.phase_annotate(manifest, authorization, "annotator_a", False)
    elif args.phase == "annotate_b": BASE.phase_annotate(manifest, authorization, "annotator_b", False)
    elif args.phase == "consensus": BASE.phase_consensus(manifest)
    elif args.phase == "analyze_discovery": BASE.phase_analyze_discovery(manifest)
    elif args.phase == "analyze_heldout": BASE.phase_analyze_heldout(manifest)
    else: phase_final_audit()


if __name__ == "__main__":
    main()
