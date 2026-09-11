#!/usr/bin/env python3
from __future__ import annotations
import ast,hashlib,importlib.util,json
from pathlib import Path

STAGE=Path(__file__).resolve().parent
RUNNER=STAGE/"run_stage_e_behavioral_control_v5_calibration.py"
HELPER=STAGE/"run_stage_e_replication_v4_calibration.py"
TEST=STAGE/"tests"/"test_stage_e_behavioral_control_v5_calibration.py"
OUTPUT=STAGE/"stage_e_behavioral_control_v5_assets"/"stage_e_behavioral_control_v5_calibration_static_review.json"

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def call_name(n):
    if isinstance(n.func,ast.Name):return n.func.id
    if isinstance(n.func,ast.Attribute):return n.func.attr
    return ""

source=RUNNER.read_text(); tree=ast.parse(source); calls=[call_name(n) for n in ast.walk(tree) if isinstance(n,ast.Call)]
spec=importlib.util.spec_from_file_location("v5_review",RUNNER); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
preflight=module.preflight()
checks={
 "runner_parses":True,"preflight_pass":all(preflight["checks"].values()),"preflight_model_free":not preflight["model_loaded"] and not preflight["gpu_used"],
 "runtime_helper_hash_locked":sha(HELPER)==module.LOCKED_INPUT_HASHES["runtime_helper"],
 "no_replication_dataset_path":"replication_pool_dataset.json" not in source,
 "no_hook_registration":not any(x in calls for x in ("register_forward_hook","register_forward_pre_hook","register_full_backward_hook")),
 "no_tensor_serialization":not any(x in calls for x in ("save","save_file")),
 "no_liref_path":"liref_directions" not in source,
 "no_candidate_ids":not any(x in source for x in ("L31N13336","L29H00030","L30H00006","L29H00031")),
 "hidden_and_attention_disabled":"output_hidden_states=False" in source and "output_attentions=False" in source,
 "local_only_and_no_remote_code":"local_files_only=True" in source and "trust_remote_code=False" in source,
 "float32_locked":"torch_dtype=torch.float32" in source and module.MODEL_DTYPE=="float32",
 "counts_locked":module.EXPECTED_PAIRS==64 and module.EXPECTED_PROMPTS==128,
 "dz_descriptive_only":"cluster_dz_used_for_pass_fail\":False" in source,
 "bootstrap_locked":module.BOOTSTRAP_REPETITIONS==10000 and module.BOOTSTRAP_SEED==20260902,
 "authorization_required":"validate_authorization" in source and "--authorization" in source,
 "overwrite_refusal":"Refusing to overwrite run directory" in source,
 "test_present":TEST.exists(),"human_disclosure":preflight["human_audit"]=="not_performed" and not preflight["human_audited_evidence"],
}
payload={"schema_version":"5.0","status":"PASS_execution_not_authorized" if all(checks.values()) else "FAIL",
 "review_type":"model_free_static_safety_and_schema_review","runner_sha256":sha(RUNNER),"runtime_helper_sha256":sha(HELPER),
 "review_script_sha256":sha(Path(__file__).resolve()),"unit_test_sha256":sha(TEST),"checks":checks,"all_checks_pass":all(checks.values()),
 "model_loaded":False,"model_forward_performed":False,"gpu_used":False,"replication_pool_accessed":False,"liref_loaded":False,
 "candidate_components_accessed":False,"hidden_states_captured":False,"hooks_registered":False,"intervention_performed":False,
 "execution_authorized":False,"human_audit":"not_performed","human_audited_evidence":False}
OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps(payload,indent=2,sort_keys=True))
if not payload["all_checks_pass"]:raise SystemExit(1)
