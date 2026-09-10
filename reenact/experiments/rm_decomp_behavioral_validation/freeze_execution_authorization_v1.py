#!/usr/bin/env python3
"""Create the one-run execution authorization after model-free review."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / "run_meta_llama_behavioral_validation_v1.py"
REVIEW_PATH = SCRIPT_DIR / "STATIC_REVIEW_V1.md"
OUTPUT_PATH = SCRIPT_DIR / "execution_authorization_v1_frozen.json"

SPEC = importlib.util.spec_from_file_location("behavioral_v1", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"Refusing to overwrite frozen authorization: {OUTPUT_PATH}")
    design = runner.load_design()
    runner.validate_candidate_manifest(design)
    review = REVIEW_PATH.read_text(encoding="utf-8")
    if "Status: **PASS" not in review or "10/10 PASS" not in review:
        raise RuntimeError("Static review is not PASS")
    payload = {
        "schema_version": "1.0",
        "status": "FROZEN_EXECUTION_AUTHORIZED",
        "authorized_at": "2026-09-01",
        "study_id": design["study_id"],
        "run_id": design["run_id"],
        "physical_gpu": design["physical_gpu"],
        "logical_device": design["logical_device"],
        "batch_size": design["batch_size"],
        "dtype": design["dtype"],
        "implementation_sha256": runner.sha256_file(RUNNER_PATH),
        "test_sha256": runner.sha256_file(SCRIPT_DIR / "test_behavioral_validation_v1.py"),
        "static_review_sha256": runner.sha256_file(REVIEW_PATH),
        "locked_inputs": runner.locked_inputs(design),
        "baseline_gate_required_before_intervention": True,
        "automatic_pdf_update_authorized": False,
        "claim_boundary": design["interpretation"],
    }
    temporary = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT_PATH)
    print(OUTPUT_PATH)
    print(runner.sha256_file(OUTPUT_PATH))


if __name__ == "__main__":
    main()
