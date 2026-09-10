#!/usr/bin/env python3
"""Behavior-only Stage E v5 Calibration; independent replication remains sealed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
STAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = STAGE_DIR / "stage_e_behavioral_control_v5_assets"
OUTPUT_ROOT = ROOT / "AI" / "reenact" / "liref_outputs" / "rm_decomp" / "v5" / "calibration"
DESIGN_PATH = STAGE_DIR / "stage_e_behavioral_control_v5_design_frozen.json"
AMENDMENT_PATH = STAGE_DIR / "stage_e_behavioral_control_v5_calibration_implementation_amendment_frozen.json"
DATASET_PATH = ASSET_DIR / "calibration_pool_dataset.json"
AUTOMATIC_PATH = ASSET_DIR / "calibration_pool_automatic_audit.json"
PRIMARY_PATH = ASSET_DIR / "calibration_pool_primary_ai_audit.json"
ADVERSARIAL_PATH = ASSET_DIR / "calibration_pool_adversarial_ai_audit.json"
AI_SUMMARY_PATH = ASSET_DIR / "ai_audit_summary.json"
STATIC_REVIEW_PATH = ASSET_DIR / "stage_e_behavioral_control_v5_calibration_static_review.json"
RUNTIME_HELPER_PATH = STAGE_DIR / "run_stage_e_replication_v4_calibration.py"

LOCKED_INPUT_HASHES = {
    "design":"033969f85f9f982ff686c22f4bdd3977baae1dcc76a4f91b537e758a8fd98982",
    "implementation_amendment":"8de748ecfd1b889b5c88f46774cf6bd0b50bf37ae3a3b1bec03d595ae4476813",
    "calibration_dataset":"18005ac5c5733ab389f1b9f8d4850a671f4bdfa0f0c4755ddd62a43e4348922a",
    "calibration_automatic":"ea2bc26a9a090167f63fdba65c4bb1c90d630f692c2cda3f9a6950f2a4e3fee4",
    "calibration_primary_ai":"5c68f3c14adab276a0a226359adb96484533bd730b1989c5d5a91191ffe99e24",
    "calibration_adversarial_ai":"3ab9b070e8c96af1b38ae8dbfc22c7051c29b5711fdaa7408664b4434d4e9fb8",
    "ai_audit_summary":"d07b7adc284cd4110fecf38b9187e01946622b5604ce1ece893e5a7347af0163",
    "runtime_helper":"299d8531c6ec31eb1f09890456c0b1b3ae77da3915f71dfdf75527325ec7f28f",
}
LOCKED_PATHS = {
    "design":DESIGN_PATH,"implementation_amendment":AMENDMENT_PATH,
    "calibration_dataset":DATASET_PATH,"calibration_automatic":AUTOMATIC_PATH,
    "calibration_primary_ai":PRIMARY_PATH,"calibration_adversarial_ai":ADVERSARIAL_PATH,
    "ai_audit_summary":AI_SUMMARY_PATH,"runtime_helper":RUNTIME_HELPER_PATH,
}
FAMILIES=("points_balance","temperature")
EXPECTED_PAIRS=64
EXPECTED_PROMPTS=128
BOOTSTRAP_REPETITIONS=10000
BOOTSTRAP_SEED=20260902
MODEL_DTYPE="float32"


def load_helper() -> Any:
    spec=importlib.util.spec_from_file_location("v5_locked_behavior_runtime_helper",RUNTIME_HELPER_PATH)
    module=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str,Any]: return json.loads(path.read_text())


def write(path: Path,payload: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    os.replace(temporary,path)


def percentile(values:list[float],p:float)->float:
    ordered=sorted(values); position=(len(ordered)-1)*p; low=math.floor(position); high=math.ceil(position)
    if low==high:return float(ordered[low])
    weight=position-low; return float(ordered[low]*(1-weight)+ordered[high]*weight)


def bootstrap(values:list[float])->list[float]:
    rng=random.Random(BOOTSTRAP_SEED)
    means=[statistics.fmean(rng.choice(values) for _ in values) for _ in range(BOOTSTRAP_REPETITIONS)]
    return [percentile(means,.025),percentile(means,.975)]


def dz_descriptive(values:list[float])->tuple[float|None,str]:
    if len(values)!=4 or any(not math.isfinite(v) for v in values):return None,"missing_or_nonfinite"
    mean=statistics.fmean(values)
    if all(v==0 for v in values):return 0.0,"all_zero"
    sd=statistics.stdev(values)
    if sd<=sys.float_info.epsilon:return (0.0,"zero_sd_zero_mean") if mean==0 else (None,"zero_sd_nonzero_mean")
    return mean/sd,"descriptive_only"


def preflight()->dict[str,Any]:
    actual={name:sha(path) for name,path in LOCKED_PATHS.items()}
    if actual!=LOCKED_INPUT_HASHES:raise RuntimeError(f"Locked input mismatch: {actual}")
    design,dataset,automatic,primary,adversarial,ai_summary,amendment=(
        load(DESIGN_PATH),load(DATASET_PATH),load(AUTOMATIC_PATH),load(PRIMARY_PATH),
        load(ADVERSARIAL_PATH),load(AI_SUMMARY_PATH),load(AMENDMENT_PATH))
    pairs=dataset["pairs"]
    checks={
        "calibration_pool_only":dataset["pool"]=="calibration",
        "counts":len(pairs)==dataset["pair_count"]==EXPECTED_PAIRS and dataset["prompt_count"]==EXPECTED_PROMPTS,
        "families":tuple(sorted({p["lexical_family"] for p in pairs}))==tuple(sorted(FAMILIES)),
        "clusters":len({p["template_family_id"] for p in pairs})==8,
        "automatic":automatic["all_checks_pass"] is True,
        "primary_ai":primary["status"]=="PASS" and primary["pass_count"]==64,
        "adversarial_ai":adversarial["status"]=="PASS" and adversarial["pass_count"]==8,
        "all_ai":ai_summary["status"]=="automatic_primary_ai_and_adversarial_ai_audits_all_pass",
        "implementation_allowed":amendment["permissions"]["calibration_runner_implementation_allowed"] is True,
        "execution_closed":amendment["permissions"]["model_loading_allowed"] is False,
        "dz_descriptive":design["calibration"]["cluster_dz"]["pass_fail_use"] is False,
        "threshold_changes_forbidden":design["calibration"]["post_result_item_template_threshold_change_allowed"] is False,
        "human_disclosure":ai_summary["human_audit"]=="not_performed" and ai_summary["human_audited_evidence"] is False,
    }
    if not all(checks.values()):raise RuntimeError(f"Preflight failure: {checks}")
    return {"status":"preflight_pass_execution_requires_separate_authorization","checks":checks,
            "locked_input_hashes":actual,"model_loaded":False,"model_forward_performed":False,
            "gpu_used":False,"replication_pool_accessed":False,"liref_loaded":False,
            "candidate_components_accessed":False,"hidden_states_captured":False,
            "hooks_registered":False,"intervention_performed":False,
            "human_audit":"not_performed","human_audited_evidence":False,"execution_allowed":False}


def validate_authorization(path:Path,run_id:str,implementation_hash:str,review_hash:str)->dict[str,Any]:
    auth=load(path.resolve())
    checks={
        "status":auth.get("status")=="execution_authorized","scope":auth.get("scope")=="stage_e_v5_behavioral_calibration_only",
        "execution":auth.get("execution_allowed") is True,"run_id":auth.get("run_id")==run_id,
        "device":str(auth.get("device","")).startswith("cuda:"),"batch":isinstance(auth.get("batch_size"),int) and auth["batch_size"]>0,
        "dtype":auth.get("dtype")==MODEL_DTYPE,"implementation":auth.get("implementation_sha256")==implementation_hash,
        "review":auth.get("static_review_sha256")==review_hash,"inputs":auth.get("locked_input_hashes")==LOCKED_INPUT_HASHES,
        "model":auth.get("model_loading_allowed") is True,"gpu":auth.get("gpu_forward_allowed") is True,
        "replication_closed":auth.get("replication_pool_access_allowed") is False,
        "liref_closed":auth.get("liref_loading_allowed") is False,"candidate_closed":auth.get("candidate_component_access_allowed") is False,
        "hidden_closed":auth.get("hidden_state_capture_allowed") is False,"hooks_closed":auth.get("hooks_allowed") is False,
        "intervention_closed":auth.get("intervention_allowed") is False,"human_claim_closed":auth.get("human_audited_evidence") is False,
    }
    if not all(checks.values()):raise RuntimeError(f"Authorization rejected: {checks}")
    return auth


def summarize(rows:list[dict[str,Any]],design:dict[str,Any])->dict[str,Any]:
    criteria=design["calibration"]; by_family:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for row in rows:by_family[row["lexical_family"]].append(row)
    summaries={}; passed=[]; failed=[]
    for family in FAMILIES:
        selected=by_family[family]
        if len(selected)!=32:raise RuntimeError(f"Expected 32 {family} pairs")
        by_template:dict[str,list[dict[str,Any]]]=defaultdict(list)
        for row in selected:by_template[row["template_family_id"]].append(row)
        if len(by_template)!=4 or any(len(v)!=8 for v in by_template.values()):raise RuntimeError("Cluster contract failure")
        contrasts={template:statistics.fmean(r["conditions"]["arithmetic"]["margin_nats"]-r["conditions"]["selector"]["margin_nats"] for r in items)
                   for template,items in sorted(by_template.items())}
        values=list(contrasts.values()); mean_d=statistics.fmean(values); dz,dz_reason=dz_descriptive(values)
        metrics={}
        for condition in ("arithmetic","selector"):
            cr=[r["conditions"][condition] for r in selected]
            metrics[condition]={"denominator":32,
                "forced_choice_correct_count":sum(bool(x["forced_choice_correct"]) for x in cr),
                "generation_correct_count":sum(bool(x["generation_correct"]) for x in cr),
                "generation_valid_format_count":sum(bool(x["generation_valid_format"]) for x in cr),
                "mean_margin_nats":statistics.fmean(x["margin_nats"] for x in cr),
                "mean_correct_probability":statistics.fmean(x["correct_probability"] for x in cr)}
        afc,sfc=metrics["arithmetic"]["forced_choice_correct_count"],metrics["selector"]["forced_choice_correct_count"]
        agen,sgen=metrics["arithmetic"]["generation_correct_count"],metrics["selector"]["generation_correct_count"]
        flo,fhi=criteria["forced_choice_correct_count_range_inclusive"]; glo,ghi=criteria["generation_correct_count_range_inclusive"]
        checks={"arithmetic_forced_choice":flo<=afc<=fhi,"selector_forced_choice":flo<=sfc<=fhi,
                "forced_choice_gap":abs(afc-sfc)<=criteria["forced_choice_max_condition_count_gap"],
                "arithmetic_generation":glo<=agen<=ghi,"selector_generation":glo<=sgen<=ghi,
                "generation_gap":abs(agen-sgen)<=criteria["generation_max_condition_count_gap"],
                "mean_template_contrast":abs(mean_d)<=criteria["maximum_absolute_mean_template_contrast_nats"]}
        status="PASS" if all(checks.values()) else "FAIL"; (passed if status=="PASS" else failed).append(family)
        summaries[family]={"status":status,"checks":checks,"condition_metrics":metrics,
                           "forced_choice_count_gap":abs(afc-sfc),"generation_count_gap":abs(agen-sgen),
                           "template_contrasts_nats":contrasts,"mean_template_contrast_nats":mean_d,
                           "cluster_dz_descriptive":dz,"cluster_dz_reason":dz_reason,
                           "cluster_dz_used_for_pass_fail":False,"descriptive_cluster_bootstrap_mean_95ci":bootstrap(values)}
    return {"schema_version":"5.0","result_label":"stage_e_v5_behavioral_calibration",
            "status":"PASS" if passed else "FAIL","passed_families":passed,"failed_families":failed,
            "primary_replication_gate_open":"points_balance" in passed,
            "interaction_replication_gate_open":all(f in passed for f in FAMILIES),
            "cluster_dz_used_for_pass_fail":False,"all_frozen_hard_criteria_required":True,
            "result_dependent_change_performed":False,"human_audit":"not_performed",
            "human_audited_evidence":False,"family_summaries":summaries}


def execute(args:argparse.Namespace)->None:
    helper=load_helper(); implementation_hash=sha(Path(__file__).resolve())
    if not STATIC_REVIEW_PATH.exists() or load(STATIC_REVIEW_PATH).get("all_checks_pass") is not True:raise RuntimeError("Static review missing/not PASS")
    review_hash=sha(STATIC_REVIEW_PATH); auth=validate_authorization(args.authorization,args.run_id,implementation_hash,review_hash)
    run_dir=args.output_root.resolve()/args.run_id
    if run_dir.exists():raise RuntimeError(f"Refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True); write(run_dir/"status.json",{"status":"running","run_id":args.run_id})
    try:
        model_hashes=helper.validate_model_hashes(); torch,transformers,classes=helper.load_runtime(); AutoModelForCausalLM,AutoTokenizer=classes
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        device=auth["device"]; device_index=int(device.split(":")[1]); torch.cuda.set_device(device_index)
        random.seed(BOOTSTRAP_SEED); torch.manual_seed(BOOTSTRAP_SEED); torch.cuda.manual_seed_all(BOOTSTRAP_SEED); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        tokenizer=AutoTokenizer.from_pretrained(str(helper.MODEL_DIR),local_files_only=True,trust_remote_code=False,use_fast=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(str(helper.MODEL_DIR),local_files_only=True,trust_remote_code=False,
                                                   torch_dtype=torch.float32,low_cpu_mem_usage=True)
        model.to(device); model.eval(); model.config.output_hidden_states=False; model.config.output_attentions=False
        config_checks={"llama":model.config.model_type=="llama","layers":model.config.num_hidden_layers==32,
                       "hidden":model.config.hidden_size==4096,"heads":model.config.num_attention_heads==32,
                       "hooks_zero":helper.registered_hook_count(model)==0}
        if not all(config_checks.values()):raise RuntimeError(f"Model contract failure: {config_checks}")
        dataset=load(DATASET_PATH); records=[]
        for pair in dataset["pairs"]:
            for condition in ("arithmetic","selector"):
                c=pair["conditions"][condition]
                records.append({"record_id":f'{pair["pair_id"]}:{condition}',"prompt":c["full_prompt"],
                                "correct_choice":c["correct_choice"],"alternative_choice":c["alternative_choice"],
                                "correct_token_id":int(c["correct_choice_token_ids"][0]),
                                "alternative_token_id":int(c["alternative_choice_token_ids"][0])})
        scored=helper.evaluate_prompts(model,tokenizer,torch,device,auth["batch_size"],records)
        if helper.registered_hook_count(model)!=0:raise RuntimeError("Unexpected hooks")
        rows=[{"pair_id":p["pair_id"],"lexical_family":p["lexical_family"],
               "template_family_id":p["template_family_id"],"frame_index":p["frame_index"],"factors":p["factors"],
               "conditions":{c:scored[f'{p["pair_id"]}:{c}'] for c in ("arithmetic","selector")}} for p in dataset["pairs"]]
        summary=summarize(rows,load(DESIGN_PATH))
        environment={"run_id":args.run_id,"timestamp_utc":datetime.now(timezone.utc).isoformat(),"platform":platform.platform(),
                     "python":sys.version,"torch":torch.__version__,"transformers":transformers.__version__,"device":device,
                     "device_name":torch.cuda.get_device_name(device_index),"batch_size":auth["batch_size"],"dtype":MODEL_DTYPE,
                     "tf32":False,"model_hashes":model_hashes,"model_config_checks":config_checks,
                     "liref_loaded":False,"candidate_components_accessed":False,"hidden_states_captured":False,
                     "hooks_registered":False,"intervention_performed":False,"replication_pool_accessed":False,
                     "human_audit":"not_performed","human_audited_evidence":False}
        write(run_dir/"pair_results.json",{"pair_count":len(rows),"pairs":rows}); write(run_dir/"summary.json",summary); write(run_dir/"environment.json",environment)
        output_hashes={name:sha(run_dir/name) for name in ("pair_results.json","summary.json","environment.json")}
        manifest={"schema_version":"5.0","run_id":args.run_id,"status":"complete","scope":"stage_e_v5_behavioral_calibration_only",
                  "implementation_sha256":implementation_hash,"static_review_sha256":review_hash,
                  "authorization_sha256":sha(args.authorization.resolve()),"locked_input_hashes":LOCKED_INPUT_HASHES,
                  "output_hashes":output_hashes,"passed_families":summary["passed_families"],"failed_families":summary["failed_families"],
                  "primary_replication_gate_open":summary["primary_replication_gate_open"],
                  "interaction_replication_gate_open":summary["interaction_replication_gate_open"],
                  "independent_replication_automatically_executed":False,"human_audited_evidence":False}
        write(run_dir/"run_manifest.json",manifest); write(run_dir/"status.json",{"status":"complete","run_id":args.run_id})
        print(json.dumps({"run_dir":str(run_dir),**summary},indent=2))
    except Exception as error:
        write(run_dir/"status.json",{"status":"failed","run_id":args.run_id,"error":repr(error)}); raise


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(); parser.add_argument("--preflight-only",action="store_true"); parser.add_argument("--execute",action="store_true")
    parser.add_argument("--authorization",type=Path); parser.add_argument("--run-id"); parser.add_argument("--output-root",type=Path,default=OUTPUT_ROOT)
    args=parser.parse_args()
    if args.preflight_only==args.execute:parser.error("Choose exactly one mode")
    if args.execute and (not args.authorization or not args.run_id):parser.error("Execution requires authorization and run ID")
    return args


def main()->None:
    args=parse_args(); result=preflight()
    if args.preflight_only:print(json.dumps(result,indent=2)); return
    execute(args)


if __name__=="__main__":main()
