#!/usr/bin/env python3
"""Local-only multi-label decomposition of validated Transformation-positive items."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = STAGE_DIR / "stage_e_transformation_primitives_v1_2_design_frozen.json"
AUTHORIZATION_PATH = STAGE_DIR / "stage_e_transformation_primitives_v1_2_local_execution_authorization_frozen.json"
ASSET_DIR = STAGE_DIR / "stage_e_transformation_primitives_v1_2_assets"
OUTPUT_DIR = ROOT / "liref_outputs" / "rm_decomp" / "v2" / "e_transformation_primitives_v1_2"
PARENT_RUNNER = STAGE_DIR / "run_stage_e_natural_feature_discovery_v1.py"

SEED = 20260831
PRIMITIVES = ("NUM", "RULE", "REL", "COND", "CAUS", "INTER")
PRIMARY_ENDPOINTS = ("layer31_liref", "component_L29H00030", "component_L30H00006")
SECONDARY_ENDPOINTS = ("component_L31N13336", "component_L29H00031")

ANNOTATION_SYSTEM = """You annotate independent primitive operations in a previously validated transformation-required multiple-choice item. Every supplied item already has authoritative Transformation=Y. Do not reconsider that parent label, do not solve the item, and do not identify the correct option. Multiple fields may be Y.

- NUM=Y only if actual numerical calculation, numerical comparison, mathematical expression manipulation, ratio, or unit conversion is required.
- RULE=Y only if a formula, scientific/domain law, grammar/legal/classification rule, or explicit procedure must be applied. A numeric formula may make both NUM=Y and RULE=Y.
- REL=Y only if two or more given facts, quantities, or relations must be combined. Reading one fact or comparing one fact directly with options is N.
- COND=Y only if conditional, propositional, negation, quantifier, case-split, ordering, or constraint logic must be applied.
- CAUS=Y only if cause, mechanism, intervention, or counterfactual must be followed to an effect.
- INTER=Y only if a derived intermediate result must be created and then used in a later distinct judgment or operation. Completing arithmetic inside one formula is not automatically an intermediate state.

Return exactly one line and no explanation:
NUM=Y|N;RULE=Y|N;REL=Y|N;COND=Y|N;CAUS=Y|N;INTER=Y|N"""

PATTERN = re.compile(
    r"NUM\s*=\s*(Y|N)\s*;\s*RULE\s*=\s*(Y|N)\s*;\s*REL\s*=\s*(Y|N)\s*;\s*"
    r"COND\s*=\s*(Y|N)\s*;\s*CAUS\s*=\s*(Y|N)\s*;\s*INTER\s*=\s*(Y|N)",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def verify() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(MANIFEST_PATH); authorization = read_json(AUTHORIZATION_PATH)
    if manifest.get("status") != "design_frozen_local_preflight_allowed":
        raise RuntimeError("v1.2 design is not frozen")
    if authorization.get("execution_allowed") is not True:
        raise RuntimeError("v1.2 local execution is not authorized")
    for document in (manifest, authorization):
        if document.get("external_api_allowed") is not False:
            raise RuntimeError("External API must remain forbidden")
        if document.get("new_study_model_forward_allowed") is not False:
            raise RuntimeError("New study-model forward must remain forbidden")
        if document.get("intervention_allowed") is not False:
            raise RuntimeError("Intervention must remain forbidden")
    for lock in authorization["locked_files"]:
        path = STAGE_DIR / lock["path"]
        if sha256_file(path) != lock["sha256"]:
            raise RuntimeError(f"Locked file hash mismatch: {path}")
    for spec in manifest["inputs"].values():
        path = resolve(spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"Input hash mismatch: {path}")
    for spec in manifest["models"].values():
        config = resolve(spec["path"]) / "config.json"
        if sha256_file(config) != spec["config_sha256"]:
            raise RuntimeError(f"Model config hash mismatch: {config}")
    return manifest, authorization


def parse_annotation(text: str) -> dict[str, str] | None:
    match = PATTERN.search(text.strip())
    if not match:
        return None
    return dict(zip(PRIMITIVES, (value.upper() for value in match.groups())))


def render_item(item: dict[str, Any]) -> str:
    options = "\n".join(f"{chr(65 + index)}. {value}" for index, value in enumerate(item["options"]))
    return f"Question:\n{item['question']}\nOptions:\n{options}"


def encode_prompts(tokenizer: Any, items: list[dict[str, Any]], retry: bool) -> list[str]:
    suffix = "\nThe last format was invalid. Return the exact six-field one-line grammar only." if retry else ""
    prompts = []
    for item in items:
        messages = [{"role": "system", "content": ANNOTATION_SYSTEM}, {"role": "user", "content": render_item(item) + suffix}]
        if getattr(tokenizer, "chat_template", None):
            try:
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                continue
            except Exception:
                pass
        prompts.append(f"System: {ANNOTATION_SYSTEM}\nUser: {render_item(item) + suffix}\nAssistant:")
    return prompts


def load_model(model_path: Path, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    return model, tokenizer


def generate(model: Any, tokenizer: Any, items: list[dict[str, Any]], batch_size: int, device: str) -> list[dict[str, Any]]:
    import torch
    pending = [(item, 1) for item in items]; complete: dict[str, dict[str, Any]] = {}
    while pending:
        batch, pending = pending[:batch_size], pending[batch_size:]
        retry = any(attempt > 1 for _item, attempt in batch)
        encoded = tokenizer(
            encode_prompts(tokenizer, [item for item, _attempt in batch], retry), return_tensors="pt",
            padding=True, truncation=True, max_length=2048,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=48,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        start = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(output[:, start:], skip_special_tokens=True)
        for (item, attempt), text in zip(batch, texts):
            parsed = parse_annotation(text)
            if parsed is None and attempt < 3:
                pending.append((item, attempt + 1)); continue
            valid = parsed is not None
            if parsed is None:
                parsed = {name: "UNC" for name in PRIMITIVES}
            complete[item["annotation_id"]] = {
                "annotation_id": item["annotation_id"], **parsed, "raw_output": text.strip(),
                "attempts": attempt, "parse_valid": valid,
            }
        del encoded, output
    return [complete[key] for key in sorted(complete)]


def category_diverse_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    rng = random.Random(seed)
    for values in grouped.values():
        values.sort(key=lambda row: (row["src"], row["annotation_id"])); rng.shuffle(values)
    categories = sorted(grouped); rng.shuffle(categories); chosen = []
    while len(chosen) < count:
        progressed = False
        for category in categories:
            if grouped[category] and len(chosen) < count:
                chosen.append(grouped[category].pop()); progressed = True
        if not progressed:
            raise RuntimeError("Unable to construct preflight")
    return chosen


def phase_prepare(manifest: dict[str, Any]) -> None:
    blind_all = {row["annotation_id"]: row for row in read_jsonl(resolve(manifest["inputs"]["blind_items"]["path"]))}
    key = {row["annotation_id"]: row for row in read_jsonl(resolve(manifest["inputs"]["annotation_key"]["path"]))}
    parent = {row["annotation_id"]: row for row in read_jsonl(resolve(manifest["inputs"]["parent_consensus"]["path"]))}
    dataset = read_json(resolve(manifest["inputs"]["dataset"]["path"]))
    if not (set(blind_all) == set(key) == set(parent)) or len(parent) != 3000:
        raise RuntimeError("Parent coverage mismatch")
    parent_y = sorted(item for item, row in parent.items() if row["transformation_required"] == "Y")
    parent_n = sorted(item for item, row in parent.items() if row["transformation_required"] == "N")
    parent_unc = sorted(item for item, row in parent.items() if row["transformation_required"] == "UNC")
    if (len(parent_y), len(parent_n), len(parent_unc)) != (895, 1739, 366):
        raise RuntimeError("Unexpected parent counts")
    blind = [blind_all[item] for item in parent_y]
    private = []
    for item in parent_y:
        index = int(key[item]["row_index"]); record = dataset[index]
        private.append({"annotation_id": item, "row_index": index, "category": str(record["category"]), "src": str(record["src"])})
    prior_ids = set()
    for input_name in ("v1_preflight_key", "v1_1_preflight_key"):
        prior_ids.update(row["annotation_id"] for row in read_jsonl(resolve(manifest["inputs"][input_name]["path"])))
    eligible = [row for row in private if row["annotation_id"] not in prior_ids]
    selected = category_diverse_sample(eligible, 96, SEED + 22)
    if {row["annotation_id"] for row in selected} & prior_ids:
        raise RuntimeError("Preflight overlap detected")
    atomic_jsonl(ASSET_DIR / "blind_parent_y_items.jsonl", blind)
    atomic_jsonl(ASSET_DIR / "private_key_DO_NOT_USE_DURING_ANNOTATION.jsonl", private)
    atomic_jsonl(ASSET_DIR / "preflight_blind_items.jsonl", [blind_all[row["annotation_id"]] for row in selected])
    atomic_jsonl(ASSET_DIR / "preflight_private_key.jsonl", selected)
    atomic_json(ASSET_DIR / "preparation_audit.json", {
        "status": "PASS", "parent_y_rows": 895, "parent_n_rows": 1739, "parent_unc_rows": 366,
        "preflight_rows": 96, "preflight_disjoint_from_v1_and_v1_1": True,
        "preflight_category_counts": dict(Counter(row["category"] for row in selected)),
        "blind_fields": ["annotation_id", "question", "options"], "external_api_used": False,
        "blind_parent_y_sha256": sha256_file(ASSET_DIR / "blind_parent_y_items.jsonl"),
        "preflight_blind_sha256": sha256_file(ASSET_DIR / "preflight_blind_items.jsonl"),
    })


def phase_annotate(manifest: dict[str, Any], authorization: dict[str, Any], slot: str, preflight: bool) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not preflight:
        audit = read_json(ASSET_DIR / "preflight_result.json")
        if audit.get("status") != "PASS" or audit.get("full_annotation_allowed") is not True:
            raise RuntimeError("Full annotation blocked by preflight")
    source = ASSET_DIR / ("preflight_blind_items.jsonl" if preflight else "blind_parent_y_items.jsonl")
    model, tokenizer = load_model(resolve(manifest["models"][slot]["path"]), authorization["device"])
    try:
        rows = generate(model, tokenizer, read_jsonl(source), int(authorization["batch_size"]), authorization["device"])
        atomic_jsonl(ASSET_DIR / f"{'preflight_' if preflight else ''}{slot}_annotations.jsonl", rows)
    finally:
        del model, tokenizer; torch.cuda.empty_cache()


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    labels = sorted(set(a) | set(b)); n = len(a); observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); expected = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    if expected >= 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def primitive_metrics(ids: list[str], a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    joint = [item for item in ids if a[item][name] in {"Y", "N"} and b[item][name] in {"Y", "N"}]
    av = [a[item][name] for item in joint]; bv = [b[item][name] for item in joint]
    prevalence = (sum(value == "Y" for value in av) + sum(value == "Y" for value in bv)) / (2 * len(joint)) if joint else None
    agreement = sum(x == y for x, y in zip(av, bv)) / len(joint) if joint else None
    return {"joint_valid_coverage": len(joint) / len(ids), "raw_agreement": agreement,
            "cohen_kappa": cohen_kappa(av, bv), "pooled_positive_prevalence": prevalence,
            "a_positive": sum(value == "Y" for value in av), "b_positive": sum(value == "Y" for value in bv)}


def reliable_preflight_status(metrics: dict[str, Any]) -> str:
    prevalence = metrics["pooled_positive_prevalence"]
    if prevalence is None or prevalence < 0.05 or prevalence > 0.95:
        return "INSUFFICIENT_PREVALENCE"
    if metrics["raw_agreement"] is None or metrics["raw_agreement"] < 0.80:
        return "UNRELIABLE"
    if metrics["cohen_kappa"] is None or metrics["cohen_kappa"] < 0.50:
        return "UNRELIABLE"
    return "USABLE"


def phase_audit_preflight() -> None:
    ids = sorted(row["annotation_id"] for row in read_jsonl(ASSET_DIR / "preflight_private_key.jsonl"))
    annotations = {slot: {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / f"preflight_{slot}_annotations.jsonl")} for slot in ("annotator_a", "annotator_b")}
    if any(set(rows) != set(ids) for rows in annotations.values()):
        raise RuntimeError("Preflight coverage mismatch")
    parse = {slot: sum(bool(rows[item]["parse_valid"]) for item in ids) / len(ids) for slot, rows in annotations.items()}
    joint_parse = sum(annotations["annotator_a"][item]["parse_valid"] and annotations["annotator_b"][item]["parse_valid"] for item in ids) / len(ids)
    metrics = {name: primitive_metrics(ids, annotations["annotator_a"], annotations["annotator_b"], name) for name in PRIMITIVES}
    status = {name: reliable_preflight_status(value) for name, value in metrics.items()}
    usable = [name for name in PRIMITIVES if status[name] == "USABLE"]
    format_gates = {"parse_a": parse["annotator_a"] >= 0.99, "parse_b": parse["annotator_b"] >= 0.99, "joint_valid_coverage": joint_parse >= 0.98}
    passed = all(format_gates.values()) and len(usable) >= 2
    atomic_json(ASSET_DIR / "preflight_result.json", {
        "status": "PASS" if passed else "FAIL", "rows": len(ids), "parse_rates": parse,
        "joint_parse_coverage": joint_parse, "format_gates": format_gates,
        "primitive_metrics": metrics, "primitive_status": status, "usable_primitives": usable,
        "minimum_usable_primitives": 2, "full_annotation_allowed": passed,
        "external_api_used": False, "internal_outcomes_merged": False,
    })


def phase_consensus(manifest: dict[str, Any]) -> None:
    preflight = read_json(ASSET_DIR / "preflight_result.json"); preflight_usable = preflight["usable_primitives"]
    parent = {row["annotation_id"]: row for row in read_jsonl(resolve(manifest["inputs"]["parent_consensus"]["path"]))}
    parent_y = sorted(item for item, row in parent.items() if row["transformation_required"] == "Y")
    annotations = {slot: {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / f"{slot}_annotations.jsonl")} for slot in ("annotator_a", "annotator_b")}
    if any(set(rows) != set(parent_y) for rows in annotations.values()):
        raise RuntimeError("Full annotation coverage mismatch")
    metrics = {name: primitive_metrics(parent_y, annotations["annotator_a"], annotations["annotator_b"], name) for name in preflight_usable}
    full_status = {name: "USABLE" if value["joint_valid_coverage"] >= 0.90 and value["cohen_kappa"] is not None and value["cohen_kappa"] >= 0.50 else "UNRELIABLE" for name, value in metrics.items()}
    usable = [name for name in preflight_usable if full_status[name] == "USABLE"]
    passed = len(usable) >= 2
    a, b = annotations["annotator_a"], annotations["annotator_b"]
    rows = []
    for item in sorted(parent):
        parent_t = parent[item]["transformation_required"]
        row: dict[str, Any] = {"annotation_id": item, "parent_transformation": parent_t}
        for name in PRIMITIVES:
            if parent_t == "N":
                row[name] = "N"
            elif parent_t == "Y" and name in usable and a[item][name] == b[item][name] and a[item][name] in {"Y", "N"}:
                row[name] = a[item][name]
            else:
                row[name] = "UNC"
        rows.append(row)
    atomic_jsonl(ASSET_DIR / "consensus_annotations.jsonl", rows)
    atomic_json(ASSET_DIR / "full_reliability_result.json", {
        "status": "PASS" if passed else "FAIL", "parent_y_rows": len(parent_y),
        "preflight_usable_primitives": preflight_usable, "primitive_metrics": metrics,
        "primitive_status": full_status, "analysis_usable_primitives": usable,
        "analysis_allowed": passed, "external_api_used": False,
        "consensus_sha256": sha256_file(ASSET_DIR / "consensus_annotations.jsonl"),
    })


def bh_adjust(values: list[float]) -> list[float]:
    import numpy as np
    p = np.asarray(values, dtype=float); output = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p)); order = valid[np.argsort(p[valid])]; running = 1.0
    for reverse in range(len(order) - 1, -1, -1):
        index = order[reverse]; rank = reverse + 1
        running = min(running, float(p[index]) * len(order) / rank); output[index] = running
    return output.tolist()


def fit_hc3(frame: Any, feature: str, endpoint: str, adjust_label: bool) -> dict[str, Any]:
    import numpy as np
    from scipy import stats
    needed = [feature, endpoint, "token_length", "option_count", "has_numeric", "src"]
    if adjust_label:
        needed.append("label_reasoning")
    work = frame[needed].dropna().copy(); base = {"n": len(work)}
    empty = {key: math.nan for key in ("beta", "se_hc3", "p", "ci_low", "ci_high", "rank")}
    if len(work) < 10 or work[feature].nunique() < 2:
        return base | empty
    columns = [work[feature].to_numpy(float), np.log1p(work["token_length"].to_numpy(float)), work["option_count"].to_numpy(float), work["has_numeric"].to_numpy(float)]
    if adjust_label:
        columns.append(work["label_reasoning"].to_numpy(float))
    source = work["src"].astype(str).to_numpy(); levels = sorted(set(source))
    x = np.column_stack([np.ones(len(work)), *columns, *[(source == value).astype(float) for value in levels[1:]]])
    x = x[:, np.r_[True, np.ptp(x[:, 1:], axis=0) > 0]]; y = work[endpoint].to_numpy(float)
    if endpoint != "label_reasoning":
        scale = y.std(ddof=0)
        if not np.isfinite(scale) or scale <= 0:
            return base | empty
        y = (y - y.mean()) / scale
    inverse = np.linalg.pinv(x.T @ x, rcond=1e-10); coefficients = inverse @ x.T @ y
    residual = y - x @ coefficients; leverage = np.sum((x @ inverse) * x, axis=1)
    adjusted = residual / np.clip(1.0 - leverage, 1e-8, None)
    covariance = inverse @ ((x.T * adjusted**2) @ x) @ inverse
    se = float(math.sqrt(max(float(covariance[1, 1]), 0.0))); beta = float(coefficients[1])
    rank = int(np.linalg.matrix_rank(x)); df = max(len(work) - rank, 1); t_value = beta / se if se > 0 else math.nan
    p = float(2 * stats.t.sf(abs(t_value), df)) if math.isfinite(t_value) else math.nan; critical = float(stats.t.ppf(0.975, df))
    return base | {"beta": beta, "se_hc3": se, "p": p, "ci_low": beta - critical * se, "ci_high": beta + critical * se, "rank": rank}


def load_analysis_frame(manifest: dict[str, Any]) -> tuple[Any, list[str]]:
    import pandas as pd
    reliability = read_json(ASSET_DIR / "full_reliability_result.json")
    if reliability.get("status") != "PASS" or reliability.get("analysis_allowed") is not True:
        raise RuntimeError("Analysis blocked by full reliability")
    parent_runner = load_module("natural_feature_v1_for_primitives", PARENT_RUNNER)
    parent_manifest = parent_runner.load_and_verify_manifest(); frame = parent_runner.load_analysis_frame(parent_manifest)
    key = pd.DataFrame(read_jsonl(resolve(manifest["inputs"]["annotation_key"]["path"])))
    annotations = pd.DataFrame(read_jsonl(ASSET_DIR / "consensus_annotations.jsonl"))
    annotations = key[["annotation_id", "row_index"]].merge(annotations, on="annotation_id", validate="one_to_one")
    usable = reliability["analysis_usable_primitives"]
    frame = frame.merge(annotations[["row_index", "parent_transformation", *usable]], on="row_index", validate="one_to_one")
    for name in usable:
        frame[f"primitive_{name}"] = frame[name].map({"Y": 1.0, "N": 0.0})
    return frame, usable


def association_table(split: Any, usable: list[str], within_parent_y: bool) -> Any:
    import pandas as pd
    rows = []
    for name in usable:
        if within_parent_y:
            work = split[split["parent_transformation"] == "Y"].copy(); analysis = "within_parent_y"
        else:
            work = split[(split["parent_transformation"] == "N") | ((split["parent_transformation"] == "Y") & (split[name] == "Y"))].copy(); analysis = "primitive_y_vs_parent_n"
        feature = f"primitive_{name}"; positive = int((work[feature] == 1).sum()); negative = int((work[feature] == 0).sum())
        for endpoint in (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS, "label_reasoning"):
            result = fit_hc3(work, feature, endpoint, adjust_label=endpoint != "label_reasoning")
            rows.append({"analysis": analysis, "primitive": name, "endpoint": endpoint,
                         "positive_n": positive, "negative_n": negative, **result})
    return pd.DataFrame(rows)


def phase_analyze_discovery(manifest: dict[str, Any]) -> None:
    import numpy as np
    frame, usable = load_analysis_frame(manifest); split = frame[frame["analysis_split"] == "discovery"].copy()
    primary = association_table(split, usable, True); baseline = association_table(split, usable, False)
    primary["count_gate"] = (primary["positive_n"] >= 50) & (primary["negative_n"] >= 50)
    primary["primary_test"] = primary["endpoint"].isin(PRIMARY_ENDPOINTS) & primary["count_gate"]
    primary["q_discovery_primary_bh"] = np.nan; mask = primary["primary_test"]
    primary.loc[mask, "q_discovery_primary_bh"] = bh_adjust(primary.loc[mask, "p"].tolist())
    primary["discovery_supported"] = primary["primary_test"] & (primary["q_discovery_primary_bh"] < 0.05) & ((primary["ci_low"] > 0) | (primary["ci_high"] < 0))
    primary_path = OUTPUT_DIR / "tables" / "discovery_within_transformation_associations.csv"
    baseline_path = OUTPUT_DIR / "tables" / "discovery_parent_n_baseline_descriptive.csv"
    atomic_csv(primary_path, primary); atomic_csv(baseline_path, baseline)
    cooccurrence = split[split["parent_transformation"] == "Y"][[f"primitive_{name}" for name in usable]].corr()
    cooccurrence_path = OUTPUT_DIR / "tables" / "discovery_primitive_cooccurrence.csv"; atomic_csv(cooccurrence_path, cooccurrence.reset_index())
    selected = [{"primitive": row.primitive, "endpoint": row.endpoint, "discovery_beta": float(row.beta)} for row in primary[primary["discovery_supported"]].itertuples(index=False)]
    atomic_json(OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json", {
        "status": "DISCOVERY_COMPLETE_HELDOUT_NOT_ANALYZED", "usable_primitives": usable,
        "selected_pairs": selected, "primary_table_sha256": sha256_file(primary_path),
        "baseline_table_sha256": sha256_file(baseline_path), "cooccurrence_sha256": sha256_file(cooccurrence_path),
        "heldout_used_for_selection": False,
    })


def phase_analyze_heldout(manifest: dict[str, Any]) -> None:
    import pandas as pd
    selection_path = OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json"; selection = read_json(selection_path)
    primary_path = OUTPUT_DIR / "tables" / "discovery_within_transformation_associations.csv"
    if selection.get("status") != "DISCOVERY_COMPLETE_HELDOUT_NOT_ANALYZED" or sha256_file(primary_path) != selection["primary_table_sha256"]:
        raise RuntimeError("Discovery selection/table is not frozen")
    frame, usable = load_analysis_frame(manifest); split = frame[frame["analysis_split"] == "validation"].copy()
    all_rows = association_table(split, usable, True); selected_rows = []
    for spec in selection["selected_pairs"]:
        row = all_rows[(all_rows["primitive"] == spec["primitive"]) & (all_rows["endpoint"] == spec["endpoint"])].iloc[0].to_dict()
        row["discovery_beta"] = spec["discovery_beta"]; selected_rows.append(row)
    heldout = pd.DataFrame(selected_rows)
    if len(heldout):
        heldout["count_gate"] = (heldout["positive_n"] >= 15) & (heldout["negative_n"] >= 15)
        heldout["q_heldout_selected_bh"] = bh_adjust(heldout["p"].tolist())
        heldout["same_sign"] = heldout["beta"] * heldout["discovery_beta"] > 0
        heldout["heldout_supported"] = heldout["count_gate"] & heldout["same_sign"] & (heldout["q_heldout_selected_bh"] < 0.05) & ((heldout["ci_low"] > 0) | (heldout["ci_high"] < 0))
    heldout_path = OUTPUT_DIR / "tables" / "heldout_selected_checks.csv"; atomic_csv(heldout_path, heldout)
    atomic_json(OUTPUT_DIR / "transformation_primitives_summary.json", {
        "status": "COMPLETE_POST_DISCOVERY_MULTI_LABEL_ASSOCIATION_ONLY", "usable_primitives": usable,
        "selected_pair_count": len(selection["selected_pairs"]),
        "heldout_supported_pairs": [] if not len(heldout) else [{"primitive": row.primitive, "endpoint": row.endpoint, "beta": float(row.beta)} for row in heldout[heldout["heldout_supported"]].itertuples(index=False)],
        "heldout_is_independent_confirmatory": False, "external_api_used": False,
        "causal_claim_allowed": False, "mediation_claim_allowed": False,
        "selection_sha256": sha256_file(selection_path), "heldout_sha256": sha256_file(heldout_path),
    })


def phase_final_audit() -> None:
    artifacts = {
        "manifest": MANIFEST_PATH, "authorization": AUTHORIZATION_PATH,
        "preflight": ASSET_DIR / "preflight_result.json", "reliability": ASSET_DIR / "full_reliability_result.json",
        "consensus": ASSET_DIR / "consensus_annotations.jsonl",
        "discovery": OUTPUT_DIR / "tables" / "discovery_within_transformation_associations.csv",
        "baseline": OUTPUT_DIR / "tables" / "discovery_parent_n_baseline_descriptive.csv",
        "cooccurrence": OUTPUT_DIR / "tables" / "discovery_primitive_cooccurrence.csv",
        "selection": OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json",
        "heldout": OUTPUT_DIR / "tables" / "heldout_selected_checks.csv",
        "summary": OUTPUT_DIR / "transformation_primitives_summary.json",
    }
    missing = [name for name, path in artifacts.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing artifacts: {missing}")
    blind = read_jsonl(ASSET_DIR / "blind_parent_y_items.jsonl")
    if len(blind) != 895 or any(set(row) != {"annotation_id", "question", "options"} for row in blind):
        raise RuntimeError("Blind annotation audit failed")
    atomic_json(OUTPUT_DIR / "final_audit.json", {
        "status": "PASS", "blind_annotation_leakage": False, "external_api_used": False,
        "new_study_model_forward": False, "intervention_performed": False,
        "artifact_sha256": {name: sha256_file(path) for name, path in artifacts.items()},
        "implementation_sha256": sha256_file(Path(__file__)),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "preflight_a", "preflight_b", "audit_preflight", "annotate_a", "annotate_b", "consensus", "analyze_discovery", "analyze_heldout", "final_audit"))
    args = parser.parse_args(); manifest, authorization = verify()
    if args.phase == "prepare": phase_prepare(manifest)
    elif args.phase == "preflight_a": phase_annotate(manifest, authorization, "annotator_a", True)
    elif args.phase == "preflight_b": phase_annotate(manifest, authorization, "annotator_b", True)
    elif args.phase == "audit_preflight": phase_audit_preflight()
    elif args.phase == "annotate_a": phase_annotate(manifest, authorization, "annotator_a", False)
    elif args.phase == "annotate_b": phase_annotate(manifest, authorization, "annotator_b", False)
    elif args.phase == "consensus": phase_consensus(manifest)
    elif args.phase == "analyze_discovery": phase_analyze_discovery(manifest)
    elif args.phase == "analyze_heldout": phase_analyze_heldout(manifest)
    else: phase_final_audit()


if __name__ == "__main__":
    main()
