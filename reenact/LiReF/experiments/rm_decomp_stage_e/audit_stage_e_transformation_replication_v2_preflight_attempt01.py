#!/usr/bin/env python3
"""Fail-safe audit of the immutable v2 annotation-preflight attempt 01."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


STAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = STAGE_DIR / "stage_e_transformation_replication_v2_assets"
LOCKS = {
    "preflight_blind_items.jsonl": "f3a2d381bf4eb15609acbbf5a0a2c8d57ee3412df87b04a67ee5a2bc17e805ac",
    "preflight_key_private.jsonl": "67be54cf43f760a10dba6cbbe5cac05b6cb9473c4c1cafb9ce93a2c91a893d84",
    "preflight_annotator_a_annotations.jsonl": "6caf00d0476c5955eefde173c5b7f433eebdfe082b4aae8a1447d70ef0b0bd42",
    "preflight_annotator_b_annotations.jsonl": "e2a099f47f4d43262dc6a79408d6de015ad61938677849e6e7cde814abd2e4fb",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def kappa(a: list[int], b: list[int]) -> float | None:
    if not a or len(a) != len(b):
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def main() -> None:
    for name, expected in LOCKS.items():
        path = ASSET_DIR / name
        if sha256_file(path) != expected:
            raise RuntimeError(f"Immutable preflight artifact changed: {name}")
    key = {row["preflight_id"]: row for row in read_jsonl(ASSET_DIR / "preflight_key_private.jsonl")}
    a = {row["preflight_id"]: row for row in read_jsonl(ASSET_DIR / "preflight_annotator_a_annotations.jsonl")}
    b = {row["preflight_id"]: row for row in read_jsonl(ASSET_DIR / "preflight_annotator_b_annotations.jsonl")}
    ids = sorted(key)
    if set(a) != set(ids) or set(b) != set(ids) or len(ids) != 192:
        raise RuntimeError("Preflight coverage mismatch")

    score_valid_a = [a[item]["score"] is not None for item in ids]
    score_valid_b = [b[item]["score"] is not None for item in ids]
    score_joint = [x and y for x, y in zip(score_valid_a, score_valid_b)]
    ta_raw = [a[item]["transformation"] for item in ids]
    tb_raw = [b[item]["transformation"] for item in ids]
    ta_valid = [value in {"Y", "N"} for value in ta_raw]
    tb_valid = [value in {"Y", "N"} for value in tb_raw]
    t_joint = [x and y for x, y in zip(ta_valid, tb_valid)]
    ta = [int(ta_raw[index] == "Y") for index, keep in enumerate(t_joint) if keep]
    tb = [int(tb_raw[index] == "Y") for index, keep in enumerate(t_joint) if keep]
    agreement = sum(x == y for x, y in zip(ta, tb)) / len(ta) if ta else None
    t_kappa = kappa(ta, tb)

    metrics = {
        "score_parse_rate_a": sum(score_valid_a) / len(ids),
        "score_parse_rate_b": sum(score_valid_b) / len(ids),
        "score_joint_valid_coverage": sum(score_joint) / len(ids),
        "transformation_parse_rate_a": sum(value is not None for value in ta_raw) / len(ids),
        "transformation_parse_rate_b": sum(value is not None for value in tb_raw) / len(ids),
        "transformation_joint_binary_coverage": sum(t_joint) / len(ids),
        "transformation_raw_agreement_on_joint_binary": agreement,
        "transformation_kappa_on_joint_binary": t_kappa,
    }
    gates = {
        "score_parse_rate_a": metrics["score_parse_rate_a"] >= 0.99,
        "score_parse_rate_b": metrics["score_parse_rate_b"] >= 0.99,
        "score_joint_valid_for_correlation": metrics["score_joint_valid_coverage"] > 0.0,
        "transformation_parse_rate_a": metrics["transformation_parse_rate_a"] >= 0.99,
        "transformation_parse_rate_b": metrics["transformation_parse_rate_b"] >= 0.99,
        "transformation_joint_valid_coverage": metrics["transformation_joint_binary_coverage"] >= 0.98,
        "transformation_raw_agreement": agreement is not None and agreement >= 0.80,
        "transformation_kappa": t_kappa is not None and t_kappa >= 0.60,
    }
    result = {
        "schema_id": "stage_e_transformation_replication_v2_annotation_preflight_attempt01",
        "status": "FAIL_INSTRUMENT_OUTPUT_CONTRACT",
        "rows": len(ids),
        "metrics": metrics,
        "gates": gates,
        "all_frozen_gates_pass": False,
        "failure_reasons": [
            "Annotator A used SCORE: syntax rather than frozen SCORE= syntax, yielding 0/192 parsed scores.",
            "Annotator B produced 126/192 parsed scores; many outputs did not reach a final score within the frozen generation limit.",
            "The two annotators had no jointly parseable score rows, so score correlation and binary agreement could not be evaluated.",
            "Annotator B transformation format coverage was below the frozen parse/coverage gates."
        ],
        "internal_outcomes_loaded": False,
        "study_model_executed": False,
        "full_candidate_annotation_allowed": False,
        "study_model_execution_allowed": False,
        "next_action": "Preserve v2 attempt and create a new annotation-instrument version before any full candidate annotation."
    }
    atomic_json(ASSET_DIR / "annotation_preflight_attempt01_result.json", result)


if __name__ == "__main__":
    main()
