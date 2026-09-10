#!/usr/bin/env python3
"""Model-free AST/static review for the Stage E v4 Calibration runner."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parent
RUNNER_PATH = STAGE_DIR / "run_stage_e_replication_v4_calibration.py"
TEST_PATH = STAGE_DIR / "tests" / "test_stage_e_replication_v4_calibration.py"
OUTPUT_PATH = STAGE_DIR / "stage_e_replication_v4_assets" / "stage_e_replication_v4_calibration_static_review.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts = [node.func.attr]
        value = node.func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def main() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    module_imports = [
        alias.name
        for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    spec = importlib.util.spec_from_file_location("stage_e_v4_calibration_review_target", RUNNER_PATH)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    preflight = runner.validate_locked_inputs()

    forbidden_hook_calls = {
        "register_forward_hook", "register_forward_pre_hook", "register_full_backward_hook",
    }
    checks = {
        "runner_parses": True,
        "model_runtime_not_imported_at_module_load": not any(name in {"torch", "transformers"} for name in module_imports),
        "runtime_imports_contained_in_load_runtime": "def load_runtime" in source,
        "no_hook_registration_calls": not any(name.split(".")[-1] in forbidden_hook_calls for name in calls),
        "no_tensor_serialization": not any(name in {"torch.save", "numpy.save", "np.save", "save_file"} for name in calls),
        "no_replication_dataset_path": "replication_pool_dataset.json" not in source,
        "no_liref_artifact_path": "liref_directions" not in source and "LiReF_direction" not in source,
        "no_candidate_identifiers": not any(value in source for value in ("L31N13336", "L29H00030", "L30H00006", "L29H00031")),
        "hidden_state_outputs_disabled": "output_hidden_states=False" in source and "model.config.output_hidden_states = False" in source,
        "attention_outputs_disabled": "output_attentions=False" in source and "model.config.output_attentions = False" in source,
        "trust_remote_code_false": source.count("trust_remote_code=False") >= 2,
        "local_files_only_true": source.count("local_files_only=True") >= 2,
        "float32_locked": "torch_dtype=torch.float32" in source and runner.MODEL_DTYPE == "float32",
        "calibration_dataset_pair_count_locked": runner.EXPECTED_PAIR_COUNT == 128,
        "calibration_prompt_count_locked": runner.EXPECTED_PROMPT_COUNT == 256,
        "family_set_locked": runner.EXPECTED_FAMILIES == ("points_balance", "temperature"),
        "bootstrap_locked": runner.BOOTSTRAP_REPETITIONS == 10000 and runner.BOOTSTRAP_SEED == 20260901,
        "preflight_pass": all(preflight["checks"].values()),
        "preflight_model_free": not preflight["model_loaded"] and not preflight["gpu_used"],
        "replication_pool_preflight_closed": preflight["replication_pool_accessed"] is False,
        "human_audit_disclosure": preflight["human_audit"] == "not_performed" and preflight["human_audited_evidence"] is False,
        "separate_authorization_required": "validate_authorization" in source and "--authorization" in source,
        "overwrite_refusal_present": "Refusing to overwrite run directory" in source,
        "unit_test_file_present": TEST_PATH.exists(),
    }
    payload = {
        "schema_version": "4.0",
        "review_type": "model_free_static_safety_and_schema_review",
        "runner_sha256": sha256_file(RUNNER_PATH),
        "review_script_sha256": sha256_file(Path(__file__).resolve()),
        "unit_test_sha256": sha256_file(TEST_PATH),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "model_loaded": False,
        "model_forward_performed": False,
        "gpu_used": False,
        "liref_loaded": False,
        "candidate_components_accessed": False,
        "hidden_states_captured": False,
        "hooks_registered": False,
        "intervention_performed": False,
        "replication_pool_accessed": False,
        "execution_authorized": False,
        "human_audit": "not_performed",
        "human_audited_evidence": False,
        "status": "PASS_execution_not_authorized" if all(checks.values()) else "FAIL",
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
