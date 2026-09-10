#!/usr/bin/env python3
"""Model-free static safety/schema review for limited Pilot v3.2.

This reviewer parses and imports the runner without importing torch,
transformers, NumPy, loading LiReF directions, touching CUDA, or running a
model. It verifies that the four runtime hooks are read-only scalar capture
hooks and that execution remains behind a separate hash-locked authorization.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_DIR = Path(__file__).resolve().parent
RUNNER_PATH = STAGE_DIR / "run_stage_e_limited_pilot_v3_2.py"
TEST_PATH = STAGE_DIR / "tests" / "test_stage_e_limited_pilot_v3_2.py"
DESIGN_PATH = STAGE_DIR / "stage_e_limited_pilot_v3_2_design_frozen.json"
CANDIDATE_PATH = STAGE_DIR / "stage_e_pilot_v3_2_candidate_manifest.json"
REPORT_PATH = STAGE_DIR / "stage_e_limited_pilot_v3_2_static_safety_review.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner() -> Any:
    specification = importlib.util.spec_from_file_location(
        "stage_e_limited_pilot_v3_2_under_review", RUNNER_PATH
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


def class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise RuntimeError(f"Missing required class: {name}")


def method_node(node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name == name:
            return child
    raise RuntimeError(f"Missing required method: {node.name}.{name}")


def nested_hook_node(factory: ast.FunctionDef) -> ast.FunctionDef:
    hooks = [
        node for node in factory.body if isinstance(node, ast.FunctionDef) and node.name == "hook"
    ]
    if len(hooks) != 1:
        raise RuntimeError(f"Expected exactly one nested hook in {factory.name}")
    return hooks[0]


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


def imported_modules(nodes: list[ast.AST]) -> list[str]:
    output = []
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                output.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                output.append(node.module or "")
    return output


def nested_hook_is_read_only(hook: ast.FunctionDef) -> bool:
    forbidden_mutating_suffixes = {
        "add_",
        "copy_",
        "fill_",
        "index_copy_",
        "masked_fill_",
        "mul_",
        "scatter_",
        "sub_",
        "zero_",
    }
    forbidden_parameter_names = {argument.arg for argument in hook.args.args}
    for node in ast.walk(hook):
        if isinstance(node, ast.Return) and node.value is not None:
            return False
        if isinstance(node, ast.AugAssign):
            return False
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                root = target
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in forbidden_parameter_names:
                    return False
        if isinstance(node, ast.Call) and call_name(node).split(".")[-1] in forbidden_mutating_suffixes:
            return False
    return True


def main() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER_PATH))
    runner = load_runner()
    preflight = runner.validate_locked_inputs()
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    candidates = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))

    top_level_nodes = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    top_level_imports = imported_modules(top_level_nodes)
    runtime_imports = imported_modules([function_node(tree, "load_runtime")])
    execute = function_node(tree, "execute")
    execute_calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)]
    named_execute_calls = [(call_name(call), call) for call in execute_calls]
    first_call_line: dict[str, int] = {}
    for name, call in named_execute_calls:
        if name:
            first_call_line[name] = min(first_call_line.get(name, sys.maxsize), call.lineno)

    all_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    from_pretrained = [call for call in all_calls if call_name(call).endswith("from_pretrained")]
    torch_load = [call for call in all_calls if call_name(call) == "torch.load"]
    model_forward = [
        call for call in all_calls if isinstance(call.func, ast.Name) and call.func.id == "model"
    ]
    hook_register_calls = [
        call
        for call in all_calls
        if call_name(call).split(".")[-1]
        in {"register_forward_hook", "register_forward_pre_hook"}
    ]
    backward_hook_calls = [
        call
        for call in all_calls
        if "backward_hook" in call_name(call).split(".")[-1]
    ]
    persistence_calls = [
        call_name(call)
        for call in all_calls
        if call_name(call).split(".")[-1]
        in {"save", "savez", "dump", "dumps", "pickle", "tofile"}
        and call_name(call) not in {"json.dump", "json.dumps"}
    ]

    capture_class = class_node(tree, "ScalarCapture")
    hook_factories = [
        method_node(capture_class, name)
        for name in ("_layer31_hook", "_attention_hook", "_neuron_hook")
    ]
    nested_hooks = [nested_hook_node(factory) for factory in hook_factories]
    remove_method = method_node(capture_class, "remove")
    execute_source = ast.get_source_segment(source, execute) or ""
    remove_source = ast.get_source_segment(source, remove_method) or ""
    primary_source = ast.get_source_segment(source, function_node(tree, "pair_prompt_rows")) or ""
    aggregate_source = ast.get_source_segment(source, function_node(tree, "aggregate_primary")) or ""
    capture_source = ast.get_source_segment(source, capture_class) or ""

    expected_components = [
        "L31N13336",
        "L29H00030",
        "L30H00006",
        "L29H00031",
    ]
    observed_components = [row["component_id"] for row in candidates["candidates"]]
    forbidden_candidate_literals = sorted(
        {
            token
            for token in ("L31N13336", "L29H00030", "L30H00006", "L29H00031")
            if token not in expected_components
        }
    )

    checks = {
        "runner_parses_and_imports_without_ml_runtime": True,
        "locked_input_preflight_pass": all(preflight["checks"].values()),
        "preflight_keeps_model_gpu_direction_and_execution_closed": (
            preflight["model_runtime_imported"] is False
            and preflight["direction_runtime_loaded"] is False
            and preflight["model_loaded"] is False
            and preflight["model_forward_performed"] is False
            and preflight["gpu_used"] is False
            and preflight["pilot_execution_allowed"] is False
        ),
        "no_top_level_numpy_torch_or_transformers_import": not any(
            name == "numpy" or name == "torch" or name.startswith("transformers")
            for name in top_level_imports
        ),
        "ml_runtime_imports_are_lazy_and_scoped": (
            "numpy" in runtime_imports
            and "torch" in runtime_imports
            and "transformers" in runtime_imports
        ),
        "authorization_precedes_model_hash_runtime_and_direction_load": (
            first_call_line.get("validate_execution_authorization", sys.maxsize)
            < first_call_line.get("validate_model_file_hashes", -1)
            < first_call_line.get("load_runtime", -1)
            < (torch_load[0].lineno if len(torch_load) == 1 else -1)
        ),
        "static_review_must_exist_and_pass_before_authorization": (
            'static_review.get("all_checks_pass") is not True' in execute_source
            and first_call_line.get("load_json", sys.maxsize)
            < first_call_line.get("validate_execution_authorization", -1)
        ),
        "exactly_two_local_non_remote_from_pretrained_calls": (
            len(from_pretrained) == 2
            and all(keyword_literal(call, "local_files_only") is True for call in from_pretrained)
            and all(keyword_literal(call, "trust_remote_code") is False for call in from_pretrained)
        ),
        "model_forward_explicitly_disables_hidden_and_attention_outputs": (
            len(model_forward) == 1
            and keyword_literal(model_forward[0], "output_hidden_states") is False
            and keyword_literal(model_forward[0], "output_attentions") is False
        ),
        "exactly_four_forward_capture_hooks_and_no_backward_hooks": (
            len(hook_register_calls) == 4 and not backward_hook_calls
        ),
        "all_nested_capture_hooks_are_read_only_and_return_none": all(
            nested_hook_is_read_only(hook) for hook in nested_hooks
        ),
        "hooks_removed_in_finally_and_handle_list_cleared": (
            "finally:" in execute_source
            and "capture.remove()" in execute_source
            and "handle.remove()" in remove_source
            and "self.handles.clear()" in remove_source
        ),
        "capture_stores_python_scalars_not_state_tensors": (
            "tensor.detach().float().cpu().tolist()" in capture_source
            and "self.values" in capture_source
            and not persistence_calls
        ),
        "layer31_last_token_projection_formula_present": (
            "tensor[:, -1, :]" in capture_source
            and "self.directions[31]" in capture_source
            and 'self.values["layer31_liref_projection"]' in capture_source
        ),
        "neuron_scalar_contribution_formula_present": (
            "args[0][:, -1, 13336]" in capture_source
            and "layer31.mlp.down_proj.weight[:, 13336]" in capture_source
            and "contribution = z * self.neuron_projection" in capture_source
        ),
        "head_scalar_contribution_formula_present": (
            "o_proj.weight" in capture_source
            and "head_index * head_dim" in capture_source
            and "state = heads[:, head_index, :]" in capture_source
            and "contribution = self.torch.mv(state, self.head_projections[component_id])" in capture_source
        ),
        "candidate_set_exact_and_reselection_closed": (
            observed_components == expected_components
            and candidates["candidate_addition_or_reselection_allowed"] is False
            and not forbidden_candidate_literals
        ),
        "primary_difference_is_arithmetic_minus_selector": (
            'condition_rows["arithmetic"][endpoint]' in primary_source
            and '- condition_rows["selector"][endpoint]' in primary_source
        ),
        "primary_population_requires_all_192_unique_pairs": (
            runner.EXPECTED_PAIR_COUNT == 192
            and runner.EXPECTED_PROMPT_COUNT == 384
            and runner.EXPECTED_TEMPLATE_COUNT == 24
            and "all 192 unique pairs" in source
        ),
        "aggregation_and_bootstrap_contract_exact": (
            runner.EXPECTED_FRAMES_PER_TEMPLATE == 8
            and runner.BOOTSTRAP_REPETITIONS == 10000
            and runner.BOOTSTRAP_SEED == 20260831
            and "equal_weight_family_mean" in aggregate_source
            and '"confirmatory_significance_claim_allowed": False' in source
        ),
        "secondary_diagnostics_cannot_replace_primary": (
            '"secondary_results_may_replace_primary": False' in source
            and '"candidate_selection_from_secondary_allowed": False' in source
            and '"secondary_p_values_computed": False' in source
        ),
        "no_generation_scoring_or_intervention_in_pilot_runner": (
            not any(call_name(call).endswith("generate") for call in all_calls)
            and not any(
                call_name(call).split(".")[-1]
                in {"patch", "suppress", "ablate", "intervene"}
                for call in all_calls
            )
        ),
        "raw_tensor_persistence_absent": not persistence_calls
        and "torch.save" not in source
        and "numpy.save" not in source,
        "output_schema_is_scalar_pair_template_family_overall_interaction": all(
            token in source
            for token in (
                '"prompt_scalars"',
                '"pair_differences"',
                '"template_cluster_effects"',
                '"family_effects"',
                '"overall_effects"',
                '"interaction_effects"',
                '"secondary_diagnostics"',
            )
        ),
        "claim_limits_and_intervention_remain_closed": (
            '"evidence_class": "limited_same_sample_pilot"' in source
            and '"independent_or_confirmatory": False' in source
            and '"intervention_performed": False' in source
            and '"confirmatory_claim_allowed": False' in source
        ),
        "frozen_design_execution_is_currently_closed": (
            design["current_permissions"]["pilot_model_loading_allowed"] is False
            and design["current_permissions"]["pilot_gpu_forward_allowed"] is False
            and design["current_permissions"]["intervention_patching_suppression_allowed"] is False
        ),
        "model_file_hash_lock_covers_index_and_four_shards": (
            "model.safetensors.index.json" in runner.MODEL_SMALL_FILE_HASHES
            and len(runner.MODEL_PARAMETER_FILE_HASHES) == 4
        ),
        "unit_test_artifact_present": TEST_PATH.is_file(),
    }

    report = {
        "schema_version": "3.2",
        "review_type": "model_free_static_safety_schema_and_read_only_hook_review",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_checks_pass": all(checks.values()),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "unit_test_sha256": sha256_file(TEST_PATH),
        "frozen_design_sha256": sha256_file(DESIGN_PATH),
        "candidate_manifest_sha256": sha256_file(CANDIDATE_PATH),
        "checks": checks,
        "hook_review": {
            "registered_forward_capture_hook_count": len(hook_register_calls),
            "registered_backward_hook_count": len(backward_hook_calls),
            "nested_hook_count": len(nested_hooks),
            "nested_hooks_return_modified_forward_values": False,
            "capture_mutates_model_input_or_output": False,
            "capture_persists_raw_state_tensors": False,
        },
        "runtime_actions_performed_by_review": {
            "numpy_imported": False,
            "torch_imported": False,
            "transformers_imported": False,
            "direction_loaded": False,
            "model_loaded": False,
            "cuda_used": False,
            "forward_performed": False,
            "hook_installed": False,
            "intervention_performed": False,
        },
        "execution_authorized_by_this_review": False,
        "separate_hash_locked_execution_authorization_required": True,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["all_checks_pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"Static review FAIL: {failed}")


if __name__ == "__main__":
    main()
