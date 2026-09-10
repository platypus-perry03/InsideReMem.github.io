#!/usr/bin/env python3
"""Disjoint v2.1 annotation-instrument preflight after v2 format failure."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = STAGE_DIR / "stage_e_transformation_replication_v2_assets"
AUTHORIZATION = STAGE_DIR / "stage_e_transformation_replication_v2_1_preflight_authorization_frozen.json"
MANIFEST = STAGE_DIR / "stage_e_transformation_replication_v2_design_frozen.json"
AMENDMENT = STAGE_DIR / "stage_e_transformation_replication_v2_1_annotation_instrument_amendment_frozen.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_module("trv2_base", STAGE_DIR / "run_stage_e_transformation_replication_v2_annotations.py")
nfv1 = load_module("nfv1", STAGE_DIR / "run_stage_e_natural_feature_discovery_v1.py")


SCORE_SYSTEM = """Analyze how much solving the multiple-choice question relies on reasoning versus memory recall using this fixed scale adapted from the published LiReF Appendix A prompt. 1.0 means it strictly requires multi-step reasoning, calculations, formulas, or deductions. 0.0 means purely factual recall or direct inference of humanities knowledge. Intermediate values indicate hybrid characteristics. Numerical values, required calculation steps, and unit conversion support a higher score; direct textbook recall supports a lower score. Do not provide analysis or explanation. Return exactly one line: SCORE=<number from 0.00 to 1.00>."""

SCORE_PATTERN = re.compile(r"SCORE\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", re.IGNORECASE)
T_PATTERN = re.compile(r"(?:^|;)\s*T\s*=\s*(Y|N|UNC)(?:\s*;|\s*$)", re.IGNORECASE)
SEED = 20260831


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def parse_score(text: str) -> float | None:
    matches = SCORE_PATTERN.findall(text)
    if not matches:
        return None
    value = float(matches[-1])
    return value if 0.0 <= value <= 1.0 else None


def parse_transformation(text: str) -> str | None:
    match = T_PATTERN.search(text.strip())
    return match.group(1).upper() if match else None


def verify() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = base.read_json(MANIFEST)
    authorization = base.read_json(AUTHORIZATION)
    if manifest.get("study_model_execution_allowed") is not False:
        raise RuntimeError("Study-model execution must remain forbidden")
    if authorization.get("execution_scope") != "v2_1_annotation_preflight_only":
        raise RuntimeError("Wrong authorization scope")
    for lock in authorization["locked_files"]:
        path = STAGE_DIR / lock["path"]
        if sha256_file(path) != lock["sha256"]:
            raise RuntimeError(f"Locked file mismatch: {path}")
    for spec in manifest["models"].values():
        config = ROOT / spec["path"] / "config.json"
        if sha256_file(config) != spec["config_sha256"]:
            raise RuntimeError(f"Model config mismatch: {config}")
    return manifest, authorization


def phase_prepare() -> None:
    records = base.read_json(base.PREVIOUS_DATASET)
    split = base.read_json(base.PREVIOUS_SPLIT)
    used = {row["question_id"] for row in base.read_jsonl(ASSET_DIR / "preflight_key_private.jsonl")}
    bins: dict[str, list[dict[str, Any]]] = {name: [] for name in ("zero", "low_nonzero", "high_subpointnine", "pointnine_or_one")}
    for index in split["heldout"]["row_indices"]:
        row = records[int(index)]
        if str(row["question_id"]) not in used:
            bins[base.score_bin(float(row["memory_reason_score"]))].append(row)
    selected = []
    for offset, name in enumerate(bins):
        candidates = sorted(bins[name], key=lambda row: (str(row["src"]), str(row["question_id"])))
        random.Random(SEED + 100 + offset).shuffle(candidates)
        if len(candidates) < 40:
            raise RuntimeError(f"Disjoint preflight bin {name} has fewer than 40 rows")
        selected.extend(candidates[:40])
    random.Random(SEED + 100).shuffle(selected)
    blind = [{"preflight_id": f"TRP21-{i:03d}", "question": row["question"], "options": row["options"]}
             for i, row in enumerate(selected)]
    key = [{"preflight_id": f"TRP21-{i:03d}", "question_id": str(row["question_id"]),
            "original_memory_reason_score": float(row["memory_reason_score"]),
            "original_binary_reasoning": int(float(row["memory_reason_score"]) > 0.5)}
           for i, row in enumerate(selected)]
    base.atomic_jsonl(ASSET_DIR / "preflight_v2_1_blind_items.jsonl", blind)
    base.atomic_jsonl(ASSET_DIR / "preflight_v2_1_key_private.jsonl", key)
    base.atomic_json(ASSET_DIR / "preflight_v2_1_preparation_audit.json", {
        "status": "PASS", "rows": 160, "score_bins": {name: 40 for name in bins},
        "disjoint_from_attempt01": True,
        "blind_items_sha256": sha256_file(ASSET_DIR / "preflight_v2_1_blind_items.jsonl"),
        "private_key_sha256": sha256_file(ASSET_DIR / "preflight_v2_1_key_private.jsonl"),
    })


def phase_annotate(name: str, manifest: dict[str, Any], authorization: dict[str, Any]) -> None:
    items = base.read_jsonl(ASSET_DIR / "preflight_v2_1_blind_items.jsonl")
    model, tokenizer = base.load_model(ROOT / manifest["models"][name]["path"], authorization["device"])
    score_raw = base.generate(model, tokenizer, items, SCORE_SYSTEM, 12, authorization["batch_size"], authorization["device"])
    transform_raw = base.generate(model, tokenizer, items, nfv1.ANNOTATION_SYSTEM, 48, authorization["batch_size"], authorization["device"])
    rows = [{"preflight_id": item["preflight_id"], "annotator": name,
             "score": parse_score(score_text), "transformation": parse_transformation(transform_text),
             "score_raw": score_text, "transformation_raw": transform_text}
            for item, score_text, transform_text in zip(items, score_raw, transform_raw)]
    base.atomic_jsonl(ASSET_DIR / f"preflight_v2_1_{name}_annotations.jsonl", rows)
    del model, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def safe_spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    value = base.spearman(a, b)
    return value if math.isfinite(value) else None


def safe_kappa(a: list[int], b: list[int]) -> float | None:
    if not a or len(a) != len(b):
        return None
    value = base.cohen_kappa(a, b)
    return value if math.isfinite(value) else None


def safe_balanced_accuracy(truth: list[int], prediction: list[int]) -> float | None:
    if not truth or set(truth) != {0, 1}:
        return None
    return base.balanced_accuracy(truth, prediction)


def phase_audit() -> None:
    key = {row["preflight_id"]: row for row in base.read_jsonl(ASSET_DIR / "preflight_v2_1_key_private.jsonl")}
    ann = {name: {row["preflight_id"]: row for row in base.read_jsonl(ASSET_DIR / f"preflight_v2_1_{name}_annotations.jsonl")}
           for name in ("annotator_a", "annotator_b")}
    ids = sorted(key)
    original = [float(key[item]["original_memory_reason_score"]) for item in ids]
    original_binary = [int(key[item]["original_binary_reasoning"]) for item in ids]
    scores = {name: [ann[name][item]["score"] for item in ids] for name in ann}
    parse_rate = {name: sum(value is not None for value in values) / len(ids) for name, values in scores.items()}
    per_annotator_spearman = {}
    for name, values in scores.items():
        keep = [value is not None for value in values]
        per_annotator_spearman[name] = safe_spearman([v for v, k in zip(original, keep) if k], [float(v) for v, k in zip(values, keep) if k])
    joint = [scores["annotator_a"][i] is not None and scores["annotator_b"][i] is not None for i in range(len(ids))]
    a_score = [float(scores["annotator_a"][i]) for i, keep in enumerate(joint) if keep]
    b_score = [float(scores["annotator_b"][i]) for i, keep in enumerate(joint) if keep]
    ref_score = [original[i] for i, keep in enumerate(joint) if keep]
    ref_binary = [original_binary[i] for i, keep in enumerate(joint) if keep]
    ensemble = [(a + b) / 2 for a, b in zip(a_score, b_score)]
    a_binary = [int(value > 0.5) for value in a_score]
    b_binary = [int(value > 0.5) for value in b_score]
    ensemble_binary = [int(value > 0.5) for value in ensemble]

    transforms = {name: [ann[name][item]["transformation"] for item in ids] for name in ann}
    t_parse = {name: sum(value is not None for value in values) / len(ids) for name, values in transforms.items()}
    t_joint = [transforms["annotator_a"][i] in {"Y", "N"} and transforms["annotator_b"][i] in {"Y", "N"} for i in range(len(ids))]
    ta = [int(transforms["annotator_a"][i] == "Y") for i, keep in enumerate(t_joint) if keep]
    tb = [int(transforms["annotator_b"][i] == "Y") for i, keep in enumerate(t_joint) if keep]
    t_agreement = sum(x == y for x, y in zip(ta, tb)) / len(ta) if ta else None
    t_kappa = safe_kappa(ta, tb)

    metrics = {
        "score_parse_rate_a": parse_rate["annotator_a"], "score_parse_rate_b": parse_rate["annotator_b"],
        "original_score_spearman_a": per_annotator_spearman["annotator_a"],
        "original_score_spearman_b": per_annotator_spearman["annotator_b"],
        "ensemble_original_spearman": safe_spearman(ref_score, ensemble),
        "ensemble_balanced_accuracy": safe_balanced_accuracy(ref_binary, ensemble_binary),
        "interannotator_score_spearman": safe_spearman(a_score, b_score),
        "interannotator_binary_kappa": safe_kappa(a_binary, b_binary),
        "transformation_parse_rate_a": t_parse["annotator_a"],
        "transformation_parse_rate_b": t_parse["annotator_b"],
        "transformation_joint_valid_coverage": len(ta) / len(ids),
        "transformation_raw_agreement": t_agreement,
        "transformation_kappa": t_kappa,
    }
    thresholds = {
        "score_parse_rate_a": 0.99, "score_parse_rate_b": 0.99,
        "original_score_spearman_a": 0.60, "original_score_spearman_b": 0.60,
        "ensemble_original_spearman": 0.70, "ensemble_balanced_accuracy": 0.70,
        "interannotator_score_spearman": 0.70, "interannotator_binary_kappa": 0.60,
        "transformation_parse_rate_a": 0.99, "transformation_parse_rate_b": 0.99,
        "transformation_joint_valid_coverage": 0.98, "transformation_raw_agreement": 0.80,
        "transformation_kappa": 0.60,
    }
    gates = {name: metrics[name] is not None and metrics[name] >= threshold for name, threshold in thresholds.items()}
    passed = all(gates.values())
    base.atomic_json(ASSET_DIR / "annotation_preflight_v2_1_result.json", {
        "schema_id": "stage_e_transformation_replication_v2_1_preflight",
        "status": "PASS" if passed else "FAIL", "rows": len(ids), "metrics": metrics,
        "thresholds": thresholds, "gates": gates,
        "full_candidate_annotation_allowed": passed,
        "study_model_execution_allowed": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "annotate_a", "annotate_b", "audit"))
    args = parser.parse_args()
    manifest, authorization = verify()
    if args.phase == "prepare": phase_prepare()
    elif args.phase == "annotate_a": phase_annotate("annotator_a", manifest, authorization)
    elif args.phase == "annotate_b": phase_annotate("annotator_b", manifest, authorization)
    else: phase_audit()


if __name__ == "__main__":
    main()
