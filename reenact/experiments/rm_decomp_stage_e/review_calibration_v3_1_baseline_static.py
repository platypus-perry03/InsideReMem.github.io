#!/usr/bin/env python3
"""Model-free static safety/schema review for Baseline Calibration v3.1.

This reviewer parses and imports the runner without importing ML runtimes.  It
does not load a tokenizer, model, CUDA runtime, LiReF artifact, hidden state, or
candidate component.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_DIR = Path(__file__).resolve().parent
RUNNER_PATH = STAGE_DIR / "run_calibration_v3_1_baseline.py"
DESIGN_PATH = STAGE_DIR / "calibration_v3_design_frozen.json"
REPORT_PATH = (
    STAGE_DIR
    / "calibration_v3_assets"
    / "calibration_v3_1_baseline_static_safety_review.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner() -> Any:
    specification = importlib.util.spec_from_file_location(
        "calibration_v3_1_baseline_under_review", RUNNER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load runner for static review")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"Missing required function: {name}")


def call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        parts = [call.func.attr]
        value = call.func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def keyword_literal(call: ast.Call, keyword: str) -> Any:
    for item in call.keywords:
        if item.arg == keyword:
            try:
                return ast.literal_eval(item.value)
            except (ValueError, TypeError):
                return "<nonliteral>"
    return "<missing>"


def synthetic_results(runner: Any, *, selector_ceiling: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in runner.EXPECTED_FAMILIES:
        for template_index in range(8):
            for frame_index in range(8):
                arithmetic_correct = frame_index < 6
                selector_correct = True if selector_ceiling else arithmetic_correct
                generation_correct = frame_index < 4

                def condition(margin: float, forced_choice: bool) -> dict[str, Any]:
                    return {
                        "margin_nats": margin,
                        "forced_choice_correct": forced_choice,
                        "generation_correct": generation_correct,
                        "generation_valid_format": True,
                        "correct_log_probability": -1.0,
                        "correct_geometric_probability": math.exp(-1.0),
                    }

                rows.append(
                    {
                        "pair_id": f"{family}_{template_index}_{frame_index}",
                        "lexical_family": family,
                        "template_family_id": f"{family}_template_{template_index}",
                        "conditions": {
                            "arithmetic": condition(
                                1.0 if arithmetic_correct else -1.0,
                                arithmetic_correct,
                            ),
                            "selector": condition(
                                1.0 if selector_correct else -1.0,
                                selector_correct,
                            ),
                        },
                    }
                )
    return rows


def main() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER_PATH))
    runner = load_runner()
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    preflight = runner.validate_locked_inputs()

    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")

    load_runtime = function_node(tree, "load_runtime")
    runtime_imports = []
    for node in ast.walk(load_runtime):
        if isinstance(node, ast.Import):
            runtime_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            runtime_imports.append(node.module or "")

    execute_node = function_node(tree, "execute")
    execute_calls = [node for node in ast.walk(execute_node) if isinstance(node, ast.Call)]
    named_execute_calls = [(call_name(call), call) for call in execute_calls]
    call_lines = {name: call.lineno for name, call in named_execute_calls if name}
    from_pretrained_calls = [
        call for name, call in named_execute_calls if name.endswith("from_pretrained")
    ]
    all_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    generate_calls = [call for call in all_calls if call_name(call).endswith("generate")]
    model_forward_calls = [
        call
        for call in all_calls
        if isinstance(call.func, ast.Name) and call.func.id == "model"
    ]

    forbidden_call_suffixes = {
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
        "register_backward_hook",
    }
    forbidden_calls = sorted(
        {
            call_name(call)
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and call_name(call).split(".")[-1] in forbidden_call_suffixes
        }
    )
    suspicious_imports = sorted(
        name
        for name in top_level_imports + runtime_imports
        if any(fragment in name.lower() for fragment in ("liref", "baukit", "nnsight", "transformer_lens"))
    )
    attribute_names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    criteria = design["acceptance_criteria"]
    fc = criteria["candidate_forced_choice"]
    generation = criteria["one_token_generation"]
    balanced_summary = runner.summarize_results(
        synthetic_results(runner, selector_ceiling=False), design
    )
    ceiling_summary = runner.summarize_results(
        synthetic_results(runner, selector_ceiling=True), design
    )

    output_schema_fields = {
        "pair_results": all(
            token in source
            for token in (
                '"correct_log_probability"',
                '"alternative_log_probability"',
                '"margin_nats"',
                '"forced_choice_correct"',
                '"generation_correct"',
            )
        ),
        "family_summary": all(
            token in source
            for token in (
                '"template_contrasts"',
                '"mean_template_contrast_nats"',
                '"cluster_dz"',
                '"passed_families"',
                '"failed_families"',
            )
        ),
        "provenance": all(
            token in source
            for token in (
                '"implementation_sha256"',
                '"static_safety_review_sha256"',
                '"execution_authorization_sha256"',
                '"locked_input_hashes"',
                '"model_hashes"',
            )
        ),
    }

    checks = {
        "runner_parses": True,
        "locked_input_preflight_pass": all(preflight["checks"].values()),
        "preflight_keeps_execution_closed": preflight["execution_allowed"] is False,
        "no_top_level_torch_or_transformers_import": not any(
            name == "torch" or name.startswith("transformers") for name in top_level_imports
        ),
        "runtime_imports_are_lazy": "torch" in runtime_imports and "transformers" in runtime_imports,
        "authorization_precedes_model_hash_and_runtime_load": (
            call_lines.get("validate_execution_authorization", sys.maxsize)
            < call_lines.get("validate_model_file_hashes", -1)
            < call_lines.get("load_runtime", -1)
        ),
        "exactly_two_local_from_pretrained_calls": len(from_pretrained_calls) == 2
        and all(keyword_literal(call, "local_files_only") is True for call in from_pretrained_calls)
        and all(keyword_literal(call, "trust_remote_code") is False for call in from_pretrained_calls),
        "generation_contract_frozen": len(generate_calls) == 1
        and keyword_literal(generate_calls[0], "do_sample") is False
        and keyword_literal(generate_calls[0], "num_beams") == 1
        and keyword_literal(generate_calls[0], "max_new_tokens") == 1,
        "model_forward_explicitly_disables_state_outputs": len(model_forward_calls) == 1
        and keyword_literal(model_forward_calls[0], "output_hidden_states") is False
        and keyword_literal(model_forward_calls[0], "output_attentions") is False,
        "generation_explicitly_disables_state_outputs": len(generate_calls) == 1
        and keyword_literal(generate_calls[0], "output_hidden_states") is False
        and keyword_literal(generate_calls[0], "output_attentions") is False,
        "no_hook_registration_calls": not forbidden_calls,
        "no_liref_or_intervention_runtime_import": not suspicious_imports,
        "no_hidden_state_or_attention_attribute_read": "hidden_states" not in attribute_names
        and "attentions" not in attribute_names,
        "model_file_hash_lock_includes_all_shards_and_index": (
            len(runner.MODEL_PARAMETER_FILE_HASHES) == 4
            and "model.safetensors.index.json" in runner.MODEL_SMALL_FILE_HASHES
        ),
        "dataset_contract_192_pairs_384_prompts": runner.EXPECTED_PAIR_COUNT == 192
        and runner.EXPECTED_PROMPT_COUNT == 384,
        "family_contract_exact": tuple(runner.EXPECTED_FAMILIES)
        == ("object_count", "points_balance", "temperature"),
        "primary_scoring_uses_frozen_condition_alternatives": (
            'payload["primary_alternative_answer"]' in source
            and "margin = correct_logp - alternative_logp" in source
        ),
        "template_contrast_uses_arithmetic_minus_selector": (
            'item["conditions"]["arithmetic"]["margin_nats"]\n                - item["conditions"]["selector"]["margin_nats"]'
            in source
        ),
        "forced_choice_tie_is_incorrect": '"forced_choice_correct": margin > 0.0' in source,
        "frozen_integer_thresholds_exact": (
            fc["minimum_correct_count"] == 40
            and fc["maximum_correct_count"] == 56
            and fc["maximum_absolute_condition_count_gap"] == 6
            and generation["minimum_correct_count"] == 16
            and generation["maximum_correct_count"] == 56
            and generation["maximum_absolute_condition_count_gap"] == 6
            and criteria["maximum_absolute_mean_template_contrast_nats"] == 0.4
            and criteria["maximum_absolute_cluster_dz"] == 0.3
        ),
        "cluster_statistics_exact": (
            runner.BOOTSTRAP_REPLICATES == 10000
            and runner.BOOTSTRAP_SEED == 20260831
            and design["cluster_statistics"]["standard_deviation_ddof"] == 1
        ),
        "balanced_synthetic_case_passes": balanced_summary["status"] == "PASS"
        and len(balanced_summary["passed_families"]) == 3,
        "selector_ceiling_synthetic_case_fails": ceiling_summary["status"] == "FAIL"
        and len(ceiling_summary["failed_families"]) == 3,
        "normalization_rejects_units": runner.normalize_generated_token("42 items") == "",
        "normalization_accepts_nfkc_digits": runner.normalize_generated_token(" ４２ ") == "42",
        "required_output_schema_present": all(output_schema_fields.values()),
        "pilot_remains_closed_in_run_manifest": '"pilot_allowed_from_this_manifest": False' in source,
        "human_audited_claim_remains_false": '"human_audited_evidence": False' in source,
    }

    report = {
        "schema_version": "3.1",
        "review_type": "static_model_free_safety_and_schema_review",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_checks_pass": all(checks.values()),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "locked_input_hashes": preflight["locked_hashes"],
        "checks": checks,
        "details": {
            "top_level_imports": sorted(top_level_imports),
            "lazy_runtime_imports": sorted(runtime_imports),
            "forbidden_hook_calls": forbidden_calls,
            "suspicious_runtime_imports": suspicious_imports,
            "output_schema_fields": output_schema_fields,
            "synthetic_balanced_passed_families": balanced_summary["passed_families"],
            "synthetic_selector_ceiling_failed_families": ceiling_summary["failed_families"],
        },
        "execution": {
            "model_runtime_imported": False,
            "tokenizer_loaded": False,
            "model_loaded": False,
            "model_forward_performed": False,
            "gpu_used": False,
            "liref_loaded": False,
            "candidate_components_accessed": False,
            "hidden_states_captured": False,
            "hooks_or_interventions_used": False,
        },
        "authorization_state": {
            "baseline_calibration_execution_allowed": False,
            "separate_hash_locked_authorization_required": True,
            "stage_e_pilot_allowed": False,
        },
        "result_label_if_later_authorized": (
            "protocol-authorized AI-only-audited Baseline Calibration v3.1"
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
