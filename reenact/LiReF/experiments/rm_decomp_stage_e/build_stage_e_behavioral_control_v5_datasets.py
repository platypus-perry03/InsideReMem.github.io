#!/usr/bin/env python3
"""Deterministically build disjoint Stage E v5 Calibration/replication pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
STAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "AI" / "reenact" / "models" / "Meta-Llama-3-8B"
DESIGN_PATH = STAGE_DIR / "stage_e_behavioral_control_v5_design_frozen.json"
CATALOG_PATH = STAGE_DIR / "stage_e_behavioral_control_v5_template_catalog_frozen.json"
CATALOG_AMENDMENT_PATH = STAGE_DIR / "stage_e_behavioral_control_v5_0_1_catalog_amendment_frozen.json"
DEFAULT_OUTPUT_DIR = STAGE_DIR / "stage_e_behavioral_control_v5_assets"
PRIOR_DATASET_PATHS = (
    STAGE_DIR / "calibration_v2_assets" / "calibration_v2_dataset_draft.json",
    STAGE_DIR / "calibration_v2_1_assets" / "calibration_v2_1_dataset_draft.json",
    STAGE_DIR / "calibration_v2_1_1_assets" / "calibration_v2_1_1_dataset_draft.json",
    STAGE_DIR / "calibration_v3_assets" / "calibration_v3_dataset_draft.json",
    STAGE_DIR / "stage_e_replication_v4_assets" / "calibration_pool_dataset.json",
    STAGE_DIR / "stage_e_replication_v4_assets" / "replication_pool_dataset.json",
)

EXPECTED_HASHES = {
    "design": "033969f85f9f982ff686c22f4bdd3977baae1dcc76a4f91b537e758a8fd98982",
    "catalog": "995d2086164a1ccf51b458b11ad70c2b328742c16b51c4ab6f34ad3bba010de2",
    "catalog_amendment": "56cd132085180db25322d2a1563a53fb835e01a541a5848324fa34bfb47fed19",
    "tokenizer.json": "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
    "tokenizer_config.json": "690727b4fed286383df1c7ca5e805124cb70c6eb4529f807c7b2e60ff741da7e",
    "special_tokens_map.json": "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
}
OA_MATRIX = (
    (0,0,0,0,0), (0,0,1,1,1), (0,1,0,0,1), (0,1,1,1,0),
    (1,0,0,1,0), (1,0,1,0,1), (1,1,0,1,1), (1,1,1,0,0),
)
FAMILIES = ("points_balance", "temperature")
POOLS = ("calibration", "replication")
CONDITIONS = ("arithmetic", "selector")
SEED = 20260902


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_locked_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    actual = {
        "design": sha256_file(DESIGN_PATH),
        "catalog": sha256_file(CATALOG_PATH),
        "catalog_amendment": sha256_file(CATALOG_AMENDMENT_PATH),
        **{name: sha256_file(MODEL_DIR / name) for name in (
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"
        )},
    }
    if actual != EXPECTED_HASHES:
        raise RuntimeError(f"Locked input mismatch: {actual}")
    design, catalog = load_json(DESIGN_PATH), load_json(CATALOG_PATH)
    amendment = load_json(CATALOG_AMENDMENT_PATH)
    if amendment["base_catalog_sha256"] != EXPECTED_HASHES["catalog"]:
        raise RuntimeError("Catalog amendment base hash mismatch")
    replacement = amendment["replacement"]
    current = catalog["pools"][replacement["pool"]]["label_pairs"][replacement["pair_index_zero_based"]]
    if current != replacement["old"]:
        raise RuntimeError("Catalog amendment old value mismatch")
    catalog["pools"][replacement["pool"]]["label_pairs"][replacement["pair_index_zero_based"]] = replacement["new"]
    checks = {
        "design_frozen": design["status"] == "design_frozen_dataset_build_allowed_model_execution_not_authorized",
        "catalog_frozen": catalog["status"] == "template_catalog_frozen_model_execution_not_authorized",
        "seed": design["numeric_design"]["seed"] == catalog["random_seed"] == SEED,
        "calibration_count": design["pools"]["calibration"]["pair_count"] == 64,
        "replication_count": design["pools"]["replication"]["pair_count"] == 128,
        "style_counts": len(catalog["calibration_styles"]) == 4 and len(catalog["replication_styles"]) == 8,
        "execution_closed": design["permissions"]["model_loading_allowed"] is False and design["permissions"]["gpu_forward_allowed"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen semantic check failed: {checks}")
    return design, catalog


def validate_oa() -> dict[str, Any]:
    if len(OA_MATRIX) != 8 or any(len(row) != 5 for row in OA_MATRIX):
        raise RuntimeError("OA must be 8x5")
    factor_counts = []
    for column in range(5):
        counts = Counter(row[column] for row in OA_MATRIX)
        if counts != Counter({0:4, 1:4}):
            raise RuntimeError(f"OA factor imbalance: {column} {counts}")
        factor_counts.append(dict(counts))
    pair_counts = {}
    expected = Counter({(0,0):2, (0,1):2, (1,0):2, (1,1):2})
    for left, right in combinations(range(5), 2):
        counts = Counter((row[left], row[right]) for row in OA_MATRIX)
        if counts != expected:
            raise RuntimeError(f"OA pair imbalance: {left},{right} {counts}")
        pair_counts[f"{left}_{right}"] = {f"{a}{b}": counts[(a,b)] for a in (0,1) for b in (0,1)}
    return {"pass": True, "factor_counts": factor_counts, "pair_counts": pair_counts}


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False, use_fast=True)


def suffix_token_ids(tokenizer: Any, prompt: str, continuation: str) -> list[int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    joint_ids = tokenizer(prompt + continuation, add_special_tokens=True)["input_ids"]
    if joint_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Continuation changes prompt token prefix")
    return [int(value) for value in joint_ids[len(prompt_ids):]]


def numeric_block(template_key: str, used: set[tuple[int, int, int, int]]) -> dict[str, int]:
    rng = random.Random(stable_int(f"{SEED}|{template_key}"))
    for _ in range(10000):
        start, delta = rng.randint(40,69), rng.randint(2,9)
        low, high = start-delta, start+delta
        signature = (start, delta, low, high)
        if len(set(signature)) != 4 or min(low,high) < 10 or max(low,high) > 99:
            continue
        if abs(high-low) < 4 or signature in used:
            continue
        used.add(signature)
        return {"start":start, "delta":delta, "low":low, "high":high}
    raise RuntimeError(f"Unable to create numeric block: {template_key}")


def numeric_mentions(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"(?<![A-Za-z0-9])-?\d+(?![A-Za-z0-9])", text)]


def normalized_words(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"(?<![a-z0-9])-?\d+(?![a-z0-9])", " <num> ", text)
    return re.findall(r"[a-z]+|<num>", text)


def ngrams(words: list[str], n: int = 5) -> set[tuple[str, ...]]:
    return {tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1))}


def jaccard(left: set[Any], right: set[Any]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def collect_prior_prompts() -> list[str]:
    prompts = []
    for path in PRIOR_DATASET_PATHS:
        if not path.exists():
            continue
        payload = load_json(path)
        for pair in payload.get("pairs", payload.get("items", [])):
            conditions = pair.get("conditions", {})
            for condition in conditions.values():
                prompt = condition.get("full_prompt") or condition.get("prompt")
                if prompt:
                    prompts.append(prompt)
    return prompts


def render_pair(
    *, pool: str, family: str, style: dict[str,str], style_index: int,
    frame_index: int, row: tuple[int,...], name: str, labels: tuple[str,str],
    tags: tuple[str,str], item: str, block: dict[str,int], tokenizer: Any,
) -> dict[str, Any]:
    operation_bit, active_bit, assignment_bit, label_bit, order_bit = row
    operation = "ADD" if operation_bit == 0 else "SUBTRACT"
    arithmetic_result = block["high"] if operation_bit == 0 else block["low"]
    arithmetic_foil = block["low"] if operation_bit == 0 else block["high"]
    tag_values = {tags[0]: block["high"], tags[1]: block["low"]}
    active_tag = tags[0] if active_bit == 0 else tags[1]
    selector_result = tag_values[active_tag]
    selector_foil = block["low"] if selector_result == block["high"] else block["high"]
    choices = {"A": block["high"], "B": block["low"]} if assignment_bit == 0 else {"A": block["low"], "B": block["high"]}
    numeric_to_choice = {value: choice for choice, value in choices.items()}
    arithmetic_choice = numeric_to_choice[arithmetic_result]
    selector_choice = numeric_to_choice[selector_result]
    arithmetic_label, selector_label = labels if label_bit == 0 else (labels[1], labels[0])
    fmt = {"name":name, "item":item, "start":block["start"], "delta":block["delta"],
           "operation":operation, "choice_a":choices["A"], "choice_b":choices["B"],
           "tag_1":tags[0], "tag_2":tags[1], "tag_1_value":tag_values[tags[0]],
           "tag_2_value":tag_values[tags[1]], "active_tag":active_tag}
    arithmetic_block = style["arithmetic"].format(**fmt, label=arithmetic_label)
    selector_block = style["selector"].format(**fmt, label=selector_label)
    context = "\n".join((arithmetic_block, selector_block) if order_bit == 0 else (selector_block, arithmetic_block))
    prompts, conditions = {}, {}
    for condition, target_label, correct_choice, result, foil in (
        ("arithmetic", arithmetic_label, arithmetic_choice, arithmetic_result, arithmetic_foil),
        ("selector", selector_label, selector_choice, selector_result, selector_foil),
    ):
        question = style["question"].format(name=name, item=item, target_label=target_label)
        prompt = context + "\nQuestion: " + question + "\nAnswer with A or B only.\nAnswer:"
        alternative = "B" if correct_choice == "A" else "A"
        correct_ids = suffix_token_ids(tokenizer, prompt, " " + correct_choice)
        alternative_ids = suffix_token_ids(tokenizer, prompt, " " + alternative)
        conditions[condition] = {
            "question": question,
            "full_prompt": prompt,
            "correct_choice": correct_choice,
            "alternative_choice": alternative,
            "correct_continuation": " " + correct_choice,
            "alternative_continuation": " " + alternative,
            "correct_choice_token_ids": correct_ids,
            "alternative_choice_token_ids": alternative_ids,
            "canonical_numeric_result": result,
            "foil_numeric_result": foil,
        }
        prompts[condition] = prompt
    mentions = numeric_mentions(context)
    question_skeletons = {
        condition: re.sub(re.escape(label), "<target>", payload["question"], flags=re.I)
        for condition, label, payload in (
            ("arithmetic", arithmetic_label, conditions["arithmetic"]),
            ("selector", selector_label, conditions["selector"]),
        )
    }
    pair_id = f"v5_{pool}_{family}_{style_index:02d}_f{frame_index:02d}"
    automatic = {
        "shared_context": True,
        "question_differs_only_target": question_skeletons["arithmetic"] == question_skeletons["selector"],
        "prompt_token_length_exact_match": len(tokenizer(prompts["arithmetic"])["input_ids"]) == len(tokenizer(prompts["selector"])["input_ids"]),
        "choice_tokens_one_token": all(len(c["correct_choice_token_ids"]) == len(c["alternative_choice_token_ids"]) == 1 for c in conditions.values()),
        "choice_token_ids_exact": {conditions[c]["correct_choice"]: conditions[c]["correct_choice_token_ids"] for c in CONDITIONS}.get("A", [362]) == [362] or True,
        "arithmetic_answer_correct": arithmetic_result == (block["start"] + block["delta"] if operation == "ADD" else block["start"] - block["delta"]),
        "wrong_operation_foil_correct": arithmetic_foil == (block["start"] - block["delta"] if operation == "ADD" else block["start"] + block["delta"]),
        "selector_answer_correct": selector_result == tag_values[active_tag],
        "shared_candidate_mapping": choices == ({"A":block["high"],"B":block["low"]} if assignment_bit == 0 else {"A":block["low"],"B":block["high"]}),
        "candidate_numeric_exposure_equal": mentions.count(block["high"]) == mentions.count(block["low"]) == 3,
        "operands_exposed_once": mentions.count(block["start"]) == mentions.count(block["delta"]) == 1,
        "numeric_values_pairwise_distinct": len(set(block.values())) == 4,
        "answer_instruction_exact": all("Answer with A or B only.\nAnswer:" in p for p in prompts.values()),
        "human_audit_disclosed": True,
        "no_prior_result_fields": True,
    }
    # Enforce the exact global A/B token IDs independently of which answer is correct.
    token_a = suffix_token_ids(tokenizer, prompts["arithmetic"], " A")
    token_b = suffix_token_ids(tokenizer, prompts["arithmetic"], " B")
    automatic["choice_token_ids_exact"] = token_a == [362] and token_b == [426]
    if not all(automatic.values()):
        raise RuntimeError(f"Pair automatic audit failed {pair_id}: {automatic}")
    return {
        "pair_id": pair_id, "pool":pool, "lexical_family":family,
        "template_family_id":f"v5_{pool}_{family}_{style['style_id']}",
        "matched_style_index":style_index, "frame_index":frame_index, "oa_row":list(row),
        "name":name, "labels":list(labels), "tags":list(tags), "item":item,
        "arithmetic_label":arithmetic_label, "selector_label":selector_label,
        "context":context, "arithmetic_block":arithmetic_block, "selector_block":selector_block,
        "numeric_block":block, "choice_mapping":choices,
        "factors":{"arithmetic_operation":operation, "selector_active_tag":active_tag,
                   "selector_active_value":"high" if active_bit==0 else "low",
                   "candidate_assignment":"high_to_A" if assignment_bit==0 else "high_to_B",
                   "label_role_orientation":label_bit,
                   "block_order":"arithmetic_first" if order_bit==0 else "selector_first"},
        "conditions":conditions,
        "prompt_token_lengths":{c:len(tokenizer(prompts[c])["input_ids"]) for c in CONDITIONS},
        "automatic_audit":automatic, "ai_audit_status":"pending",
        "human_audit":"not_performed",
    }


def audit_pool(pool: str, pairs: list[dict[str,Any]], design: dict[str,Any]) -> dict[str,Any]:
    expected_pairs = design["pools"][pool]["pair_count"]
    expected_templates = design["pools"][pool]["templates_per_family"] * 2
    by_template: dict[str,list[dict[str,Any]]] = defaultdict(list)
    for pair in pairs:
        by_template[pair["template_family_id"]].append(pair)
    template_checks = {}
    for template, rows in sorted(by_template.items()):
        factor_counts = {
            key: dict(Counter(row["factors"][key] for row in rows))
            for key in ("arithmetic_operation","selector_active_value","candidate_assignment","label_role_orientation","block_order")
        }
        ar_choices = Counter(row["conditions"]["arithmetic"]["correct_choice"] for row in rows)
        se_choices = Counter(row["conditions"]["selector"]["correct_choice"] for row in rows)
        check = {
            "eight_frames":len(rows)==8,
            "oa_rows_exact":sorted(tuple(row["oa_row"]) for row in rows)==sorted(OA_MATRIX),
            "factor_balance":all(sorted(count.values())==[4,4] for count in factor_counts.values()),
            "arithmetic_choice_balance":ar_choices==Counter({"A":4,"B":4}),
            "selector_choice_balance":se_choices==Counter({"A":4,"B":4}),
            "all_pair_audits_pass":all(all(row["automatic_audit"].values()) for row in rows),
        }
        if not all(check.values()):
            raise RuntimeError(f"Template audit failed: {template} {check}")
        template_checks[template] = {"checks":check,"factor_counts":factor_counts,
                                     "arithmetic_choice_counts":dict(ar_choices),"selector_choice_counts":dict(se_choices)}
    checks = {
        "pair_count":len(pairs)==expected_pairs,
        "prompt_count":len(pairs)*2==design["pools"][pool]["prompt_count"],
        "template_count":len(by_template)==expected_templates,
        "pair_ids_unique":len({p["pair_id"] for p in pairs})==len(pairs),
        "full_prompts_unique":len({p["conditions"][c]["full_prompt"] for p in pairs for c in CONDITIONS})==len(pairs)*2,
        "all_template_checks_pass":len(template_checks)==expected_templates,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Pool audit failed {pool}: {checks}")
    return {"schema_version":"5.0","pool":pool,"status":"PASS","all_checks_pass":True,
            "pair_count":len(pairs),"prompt_count":len(pairs)*2,"template_count":len(by_template),
            "checks":checks,"templates":template_checks,"model_loaded":False,"gpu_used":False,
            "human_audit":"not_performed","human_audited_evidence":False}


def cross_nonreuse(calibration: list[dict[str,Any]], replication: list[dict[str,Any]]) -> dict[str,Any]:
    def values(rows, fn): return {fn(row) for row in rows}
    cal_prompts=values(calibration,lambda p:p["conditions"]["arithmetic"]["full_prompt"]) | values(calibration,lambda p:p["conditions"]["selector"]["full_prompt"])
    rep_prompts=values(replication,lambda p:p["conditions"]["arithmetic"]["full_prompt"]) | values(replication,lambda p:p["conditions"]["selector"]["full_prompt"])
    prior_prompts=collect_prior_prompts()
    prior_exact=set(prior_prompts)
    prior_grams=[ngrams(normalized_words(p)) for p in prior_prompts]
    current_prompts=sorted(cal_prompts|rep_prompts)
    max_j=0.0
    for prompt in current_prompts:
        grams=ngrams(normalized_words(prompt))
        for old in prior_grams:
            max_j=max(max_j,jaccard(grams,old))
    checks = {
        "cross_pool_prompt_overlap_zero":not (cal_prompts & rep_prompts),
        "cross_pool_name_overlap_zero":not (values(calibration,lambda p:p["name"]) & values(replication,lambda p:p["name"])),
        "cross_pool_label_overlap_zero":not ({x for p in calibration for x in p["labels"]} & {x for p in replication for x in p["labels"]}),
        "cross_pool_item_overlap_zero":not (values(calibration,lambda p:p["item"]) & values(replication,lambda p:p["item"])),
        "cross_pool_numeric_block_overlap_zero":not (values(calibration,lambda p:tuple(p["numeric_block"].values())) & values(replication,lambda p:tuple(p["numeric_block"].values()))),
        "prior_exact_prompt_overlap_zero":not ((cal_prompts|rep_prompts) & prior_exact),
        "prior_normalized_5gram_jaccard_below_0_80":max_j < 0.80,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Cross/nonreuse audit failed: {checks}, max={max_j}")
    return {"schema_version":"5.0","status":"PASS","all_checks_pass":True,"checks":checks,
            "prior_prompt_count":len(prior_prompts),"maximum_prior_normalized_5gram_jaccard":max_j,
            "model_loaded":False,"gpu_used":False}


def build() -> tuple[dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any]]:
    design,catalog=validate_locked_inputs()
    oa=validate_oa()
    tokenizer=load_tokenizer()
    # Validate answer token contract before any item generation.
    if tokenizer.encode(" A",add_special_tokens=False)!=[362] or tokenizer.encode(" B",add_special_tokens=False)!=[426]:
        raise RuntimeError("Frozen A/B token contract mismatch")
    used_blocks:set[tuple[int,int,int,int]]=set()
    datasets={}
    audits={}
    for pool in POOLS:
        styles=catalog[f"{pool}_styles"]
        pairs=[]
        for family in FAMILIES:
            items=catalog["family_items"][pool][family]
            if len(items)!=len(styles): raise RuntimeError("Item/style count mismatch")
            for index,(style,item) in enumerate(zip(styles,items),start=1):
                name=catalog["pools"][pool]["names"][index-1]
                labels=tuple(catalog["pools"][pool]["label_pairs"][index-1])
                tags=tuple(catalog["pools"][pool]["tag_pairs"][index-1])
                if len(tokenizer.encode(" "+labels[0],add_special_tokens=False)) != len(tokenizer.encode(" "+labels[1],add_special_tokens=False)):
                    raise RuntimeError(f"Label token length mismatch: {pool} {labels}")
                key=f"{pool}|{family}|{style['style_id']}"
                block=numeric_block(key,used_blocks)
                for frame,row in enumerate(OA_MATRIX,start=1):
                    pairs.append(render_pair(pool=pool,family=family,style=style,style_index=index,
                                             frame_index=frame,row=row,name=name,labels=labels,tags=tags,
                                             item=item,block=block,tokenizer=tokenizer))
        dataset={"schema_version":"5.0","dataset_id":f"stage_e_v5_{pool}_pool_20260830",
                 "status":"dataset_created_automatic_audit_pass_ai_audits_pending","pool":pool,
                 "design_sha256":EXPECTED_HASHES["design"],"catalog_sha256":EXPECTED_HASHES["catalog"],
                 "catalog_amendment_sha256":EXPECTED_HASHES["catalog_amendment"],
                 "random_seed":SEED,"pair_count":len(pairs),"prompt_count":len(pairs)*2,
                 "template_count":len({p["template_family_id"] for p in pairs}),
                 "human_audit":"not_performed","human_audited_evidence":False,"pairs":pairs}
        datasets[pool]=dataset
        audits[pool]=audit_pool(pool,pairs,design)
    cross=cross_nonreuse(datasets["calibration"]["pairs"],datasets["replication"]["pairs"])
    return datasets["calibration"],datasets["replication"],audits["calibration"],audits["replication"],{"oa":oa,"cross":cross}


def main() -> None:
    args=parse_args()
    cal,rep,cal_audit,rep_audit,extra=build()
    summary={"status":"validation_pass_model_execution_not_authorized","calibration_pairs":cal["pair_count"],
             "replication_pairs":rep["pair_count"],"oa":extra["oa"],"cross_nonreuse":extra["cross"],
             "model_loaded":False,"model_forward_performed":False,"gpu_used":False}
    if args.validate_only:
        print(json.dumps(summary,indent=2)); return
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True,exist_ok=True)
    paths={
        "calibration_dataset":out/"calibration_pool_dataset.json",
        "replication_dataset":out/"replication_pool_dataset.json",
        "calibration_audit":out/"calibration_pool_automatic_audit.json",
        "replication_audit":out/"replication_pool_automatic_audit.json",
        "cross_pool_audit":out/"cross_pool_nonreuse_audit.json",
    }
    for key,payload in (("calibration_dataset",cal),("replication_dataset",rep),
                        ("calibration_audit",cal_audit),("replication_audit",rep_audit),
                        ("cross_pool_audit",extra["cross"])):
        atomic_json(paths[key],payload)
    hashes={key:sha256_file(path) for key,path in paths.items()}
    manifest={"schema_version":"5.0","status":"datasets_created_automatic_audits_pass_ai_audits_pending",
              "design_sha256":EXPECTED_HASHES["design"],"catalog_sha256":EXPECTED_HASHES["catalog"],
              "catalog_amendment_sha256":EXPECTED_HASHES["catalog_amendment"],
              "builder_sha256":sha256_file(Path(__file__).resolve()),"random_seed":SEED,
              "output_sha256":hashes,"oa_validation":extra["oa"],
              "baseline_calibration_execution_allowed":False,"replication_execution_allowed":False,
              "model_loaded":False,"model_forward_performed":False,"gpu_used":False,
              "liref_loaded":False,"intervention_allowed":False,"human_audit":"not_performed"}
    atomic_json(out/"dataset_manifest.json",manifest)
    print(json.dumps({**summary,"output_dir":str(out),"output_sha256":hashes,
                      "builder_sha256":manifest["builder_sha256"]},indent=2))


if __name__=="__main__":
    main()
