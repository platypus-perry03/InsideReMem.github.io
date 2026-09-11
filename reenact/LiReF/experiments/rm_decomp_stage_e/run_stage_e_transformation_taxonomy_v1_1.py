#!/usr/bin/env python3
"""Local-only subtype/step annotation for validated Transformation-positive items.

The authoritative parent ``transformation_required`` annotation is never rejudged.
The two local annotators see only anonymous question/options for parent-Y items.
No external API, study-model forward pass, hidden-state extraction, or intervention
is implemented in this runner.
"""

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
MANIFEST_PATH = STAGE_DIR / "stage_e_transformation_taxonomy_v1_1_instrument_amendment_frozen.json"
AUTHORIZATION_PATH = STAGE_DIR / "stage_e_transformation_taxonomy_v1_1_local_execution_authorization_frozen.json"
ASSET_DIR = STAGE_DIR / "stage_e_transformation_taxonomy_v1_1_assets"
OUTPUT_DIR = ROOT / "liref_outputs" / "rm_decomp" / "v2" / "e_transformation_taxonomy_v1_1"
PARENT_ANALYSIS_RUNNER = STAGE_DIR / "run_stage_e_natural_feature_discovery_v1.py"
V1_PREFLIGHT_KEY = STAGE_DIR / "stage_e_transformation_taxonomy_v1_assets" / "preflight_private_key.jsonl"

SEED = 20260831
SUBTYPES = ("ARITH", "LOGIC", "FORMAL", "CAUSAL", "MIXED")
STEPS = ("1", "2", "3P")
PRIMARY_ENDPOINTS = ("layer31_liref", "component_L29H00030", "component_L30H00006")
SECONDARY_ENDPOINTS = ("component_L31N13336", "component_L29H00031")

ANNOTATION_SYSTEM = """You annotate only the subtype and conceptual depth of a previously validated transformation-required multiple-choice item. Every item supplied to you already has authoritative Transformation=Y. Do not reconsider that parent label, do not solve the item, and do not identify the correct option.

Choose one dominant TYPE:
- ARITH: solving requires actually calculating, numerically comparing, converting units, or transforming mathematical expressions. Use ARITH even when a named formula or scientific law supplies the calculation.
- CAUSAL: solving primarily traces a cause, mechanism, intervention, or counterfactual to an effect.
- FORMAL: solving applies a non-numeric domain rule, law, grammar rule, legal rule, classification rule, or procedure. Do not use FORMAL merely because a numeric formula is named; numeric calculation is ARITH.
- LOGIC: solving primarily uses qualitative conditions, propositions, negation, quantifiers, cases, ordering, or relations, and is not better classified above.
- MIXED: two or more types are independently indispensable and no single type is dominant. Use sparingly.
- UNC: genuinely cannot classify.

Count dependent conceptual transformation STEPS:
- 1: one conceptual operation. Substituting values into one formula and completing its internal arithmetic is one step.
- 2: an intermediate result from one conceptual operation must feed a second distinct operation or rule.
- 3P: three or more dependent conceptual operations.
- UNC: genuinely cannot count.

Do not count reading, fact recall, option checking, or each multiplication/addition inside one formula as separate steps.
Return exactly one line and no explanation:
TYPE=ARITH|LOGIC|FORMAL|CAUSAL|MIXED|UNC;STEPS=1|2|3P|UNC"""

PATTERN = re.compile(
    r"TYPE\s*=\s*(ARITH|LOGIC|FORMAL|CAUSAL|MIXED|UNC)\s*;\s*"
    r"STEPS\s*=\s*(1|2|3P|UNC)", re.IGNORECASE,
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
    manifest = read_json(MANIFEST_PATH)
    authorization = read_json(AUTHORIZATION_PATH)
    if manifest.get("status") != "v1_1_instrument_amendment_frozen":
        raise RuntimeError("v1.1 instrument amendment is not frozen")
    if authorization.get("execution_allowed") is not True:
        raise RuntimeError("Local v1.1 execution is not authorized")
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
    subtype, steps = (value.upper() for value in match.groups())
    if (subtype == "UNC") != (steps == "UNC"):
        return None
    return {"subtype": subtype, "steps": steps}


def render_item(item: dict[str, Any]) -> str:
    options = "\n".join(f"{chr(65 + index)}. {value}" for index, value in enumerate(item["options"]))
    return f"Question:\n{item['question']}\nOptions:\n{options}"


def encode_prompts(tokenizer: Any, items: list[dict[str, Any]], retry: bool) -> list[str]:
    suffix = "\nYour last output format was invalid. Return the exact one-line grammar only." if retry else ""
    prompts = []
    for item in items:
        messages = [
            {"role": "system", "content": ANNOTATION_SYSTEM},
            {"role": "user", "content": render_item(item) + suffix},
        ]
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    return model, tokenizer


def generate(model: Any, tokenizer: Any, items: list[dict[str, Any]], batch_size: int, device: str) -> list[dict[str, Any]]:
    import torch
    pending = [(item, 1) for item in items]
    complete: dict[str, dict[str, Any]] = {}
    while pending:
        batch, pending = pending[:batch_size], pending[batch_size:]
        batch_items = [item for item, _attempt in batch]
        retry = any(attempt > 1 for _item, attempt in batch)
        encoded = tokenizer(
            encode_prompts(tokenizer, batch_items, retry), return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded, do_sample=False, num_beams=1, max_new_tokens=24,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        start = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(output[:, start:], skip_special_tokens=True)
        for (item, attempt), text in zip(batch, texts):
            parsed = parse_annotation(text)
            if parsed is None and attempt < 3:
                pending.append((item, attempt + 1))
                continue
            parse_valid = parsed is not None
            if parsed is None:
                parsed = {"subtype": "UNC", "steps": "UNC"}
            complete[item["annotation_id"]] = {
                "annotation_id": item["annotation_id"], **parsed,
                "raw_output": text.strip(), "attempts": attempt, "parse_valid": parse_valid,
            }
        del encoded, output
    return [complete[key] for key in sorted(complete)]


def category_diverse_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    rng = random.Random(seed)
    for values in grouped.values():
        values.sort(key=lambda row: (row["src"], row["annotation_id"]))
        rng.shuffle(values)
    categories = sorted(grouped)
    rng.shuffle(categories)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count:
        progressed = False
        for category in categories:
            if grouped[category] and len(chosen) < count:
                chosen.append(grouped[category].pop())
                progressed = True
        if not progressed:
            raise RuntimeError("Unable to construct category-diverse preflight")
    return chosen


def phase_prepare(manifest: dict[str, Any]) -> None:
    blind_all = {row["annotation_id"]: row for row in read_jsonl(resolve(manifest["inputs"]["blind_items"]["path"]))}
    key = {row["annotation_id"]: row for row in read_jsonl(resolve(manifest["inputs"]["annotation_key"]["path"]))}
    parent = {row["annotation_id"]: row for row in read_jsonl(resolve(manifest["inputs"]["parent_consensus"]["path"]))}
    dataset = read_json(resolve(manifest["inputs"]["dataset"]["path"]))
    if not (set(blind_all) == set(key) == set(parent)) or len(parent) != 3000:
        raise RuntimeError("Parent annotation coverage mismatch")
    parent_y_ids = sorted(item for item, row in parent.items() if row["transformation_required"] == "Y")
    parent_n_ids = sorted(item for item, row in parent.items() if row["transformation_required"] == "N")
    parent_unc_ids = sorted(item for item, row in parent.items() if row["transformation_required"] == "UNC")
    if (len(parent_y_ids), len(parent_n_ids), len(parent_unc_ids)) != (895, 1739, 366):
        raise RuntimeError("Unexpected parent Transformation counts")
    blind = [blind_all[item] for item in parent_y_ids]
    if any(set(row) != {"annotation_id", "question", "options"} for row in blind):
        raise RuntimeError("Blind fields are invalid")
    private = []
    for item in parent_y_ids:
        row_index = int(key[item]["row_index"])
        record = dataset[row_index]
        private.append({
            "annotation_id": item, "row_index": row_index,
            "category": str(record["category"]), "src": str(record["src"]),
        })
    prior_preflight_y = {
        row["annotation_id"] for row in read_jsonl(V1_PREFLIGHT_KEY)
        if row["parent_transformation"] == "Y"
    }
    eligible = [row for row in private if row["annotation_id"] not in prior_preflight_y]
    selected = category_diverse_sample(eligible, 96, SEED + 11)
    selected_ids = {row["annotation_id"] for row in selected}
    if selected_ids & prior_preflight_y:
        raise RuntimeError("v1.1 preflight is not disjoint from v1 parent-Y preflight")
    atomic_jsonl(ASSET_DIR / "blind_parent_y_items.jsonl", blind)
    atomic_jsonl(ASSET_DIR / "private_key_DO_NOT_USE_DURING_ANNOTATION.jsonl", private)
    atomic_jsonl(ASSET_DIR / "preflight_blind_items.jsonl", [blind_all[row["annotation_id"]] for row in selected])
    atomic_jsonl(ASSET_DIR / "preflight_private_key.jsonl", selected)
    atomic_json(ASSET_DIR / "preparation_audit.json", {
        "status": "PASS", "parent_y_rows": len(parent_y_ids), "parent_n_rows": len(parent_n_ids),
        "parent_unc_rows": len(parent_unc_ids), "preflight_rows": len(selected),
        "preflight_disjoint_from_v1_parent_y": True,
        "preflight_category_counts": dict(Counter(row["category"] for row in selected)),
        "blind_fields": ["annotation_id", "question", "options"],
        "external_api_used": False,
        "blind_parent_y_sha256": sha256_file(ASSET_DIR / "blind_parent_y_items.jsonl"),
        "preflight_blind_sha256": sha256_file(ASSET_DIR / "preflight_blind_items.jsonl"),
    })


def phase_annotate(manifest: dict[str, Any], authorization: dict[str, Any], slot: str, preflight: bool) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local annotation")
    if not preflight:
        audit = read_json(ASSET_DIR / "preflight_result.json")
        if audit.get("status") != "PASS" or audit.get("full_annotation_allowed") is not True:
            raise RuntimeError("Full annotation blocked by v1.1 preflight")
    source = ASSET_DIR / ("preflight_blind_items.jsonl" if preflight else "blind_parent_y_items.jsonl")
    model, tokenizer = load_model(resolve(manifest["models"][slot]["path"]), authorization["device"])
    try:
        rows = generate(model, tokenizer, read_jsonl(source), int(authorization["batch_size"]), authorization["device"])
        atomic_jsonl(ASSET_DIR / f"{'preflight_' if preflight else ''}{slot}_annotations.jsonl", rows)
    finally:
        del model, tokenizer
        torch.cuda.empty_cache()


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    labels = sorted(set(a) | set(b)); n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    if expected >= 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def weighted_kappa(a: list[str], b: list[str]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    mapping = {"1": 1, "2": 2, "3P": 3}; n = len(a)
    if any(value not in mapping for value in a + b):
        return None
    observed = sum(((mapping[x] - mapping[y]) ** 2) / 4.0 for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); expected = 0.0
    for x in mapping:
        for y in mapping:
            expected += (ca[x] / n) * (cb[y] / n) * (((mapping[x] - mapping[y]) ** 2) / 4.0)
    return 1.0 - observed / expected if expected > 0 else (1.0 if observed == 0 else None)


def safe_ge(value: float | None, threshold: float) -> bool:
    return value is not None and math.isfinite(value) and value >= threshold


def agreement_metrics(ids: list[str], a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, Any]:
    joint = [
        item for item in ids
        if a[item]["parse_valid"] and b[item]["parse_valid"]
        and a[item]["subtype"] in SUBTYPES and b[item]["subtype"] in SUBTYPES
        and a[item]["steps"] in STEPS and b[item]["steps"] in STEPS
    ]
    type_a = [a[item]["subtype"] for item in joint]; type_b = [b[item]["subtype"] for item in joint]
    step_a = [a[item]["steps"] for item in joint]; step_b = [b[item]["steps"] for item in joint]
    return {
        "joint_valid_rows": len(joint), "joint_valid_coverage": len(joint) / len(ids),
        "subtype_raw_agreement": sum(x == y for x, y in zip(type_a, type_b)) / len(joint) if joint else None,
        "subtype_kappa": cohen_kappa(type_a, type_b),
        "step_raw_agreement": sum(x == y for x, y in zip(step_a, step_b)) / len(joint) if joint else None,
        "step_weighted_kappa": weighted_kappa(step_a, step_b),
    }


def concentration(ids: list[str], rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    types = [rows[item]["subtype"] for item in ids if rows[item]["subtype"] in SUBTYPES]
    steps = [rows[item]["steps"] for item in ids if rows[item]["steps"] in STEPS]
    return {
        "subtype_max_fraction": max(Counter(types).values()) / len(types) if types else 1.0,
        "step_max_fraction": max(Counter(steps).values()) / len(steps) if steps else 1.0,
    }


def phase_audit_preflight() -> None:
    ids = sorted(row["annotation_id"] for row in read_jsonl(ASSET_DIR / "preflight_private_key.jsonl"))
    annotations = {
        slot: {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / f"preflight_{slot}_annotations.jsonl")}
        for slot in ("annotator_a", "annotator_b")
    }
    if any(set(rows) != set(ids) for rows in annotations.values()):
        raise RuntimeError("Preflight annotation coverage mismatch")
    metrics = agreement_metrics(ids, annotations["annotator_a"], annotations["annotator_b"])
    parse_rates = {
        slot: sum(bool(annotations[slot][item]["parse_valid"]) for item in ids) / len(ids)
        for slot in annotations
    }
    concentrations = {slot: concentration(ids, rows) for slot, rows in annotations.items()}
    metrics.update({"parse_rate_a": parse_rates["annotator_a"], "parse_rate_b": parse_rates["annotator_b"]})
    gates = {
        "parse_a": safe_ge(metrics["parse_rate_a"], 0.99),
        "parse_b": safe_ge(metrics["parse_rate_b"], 0.99),
        "joint_valid_coverage": safe_ge(metrics["joint_valid_coverage"], 0.98),
        "subtype_raw_agreement": safe_ge(metrics["subtype_raw_agreement"], 0.65),
        "subtype_kappa": safe_ge(metrics["subtype_kappa"], 0.50),
        "step_raw_agreement": safe_ge(metrics["step_raw_agreement"], 0.65),
        "step_weighted_kappa": safe_ge(metrics["step_weighted_kappa"], 0.55),
        "nondegenerate": all(
            value["subtype_max_fraction"] <= 0.95 and value["step_max_fraction"] <= 0.95
            for value in concentrations.values()
        ),
    }
    passed = all(gates.values())
    atomic_json(ASSET_DIR / "preflight_result.json", {
        "status": "PASS" if passed else "FAIL", "rows": len(ids), "metrics": metrics,
        "concentrations": concentrations, "gates": gates, "full_annotation_allowed": passed,
        "external_api_used": False, "internal_outcomes_merged": False,
    })


def phase_consensus(manifest: dict[str, Any]) -> None:
    parent = {row["annotation_id"]: row for row in read_jsonl(resolve(manifest["inputs"]["parent_consensus"]["path"]))}
    annotations = {
        slot: {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / f"{slot}_annotations.jsonl")}
        for slot in ("annotator_a", "annotator_b")
    }
    parent_y = sorted(item for item, row in parent.items() if row["transformation_required"] == "Y")
    if any(set(rows) != set(parent_y) for rows in annotations.values()):
        raise RuntimeError("Full parent-Y annotation coverage mismatch")
    a, b = annotations["annotator_a"], annotations["annotator_b"]
    metrics = agreement_metrics(parent_y, a, b)
    gates = {
        "joint_valid_coverage": safe_ge(metrics["joint_valid_coverage"], 0.90),
        "subtype_kappa": safe_ge(metrics["subtype_kappa"], 0.50),
        "step_weighted_kappa": safe_ge(metrics["step_weighted_kappa"], 0.55),
    }
    passed = all(gates.values())
    rows = []
    for item in sorted(parent):
        parent_t = parent[item]["transformation_required"]
        if parent_t == "N":
            subtype, steps, usable_none, usable_subtype, usable_strength = "NONE", "0", True, False, False
        elif parent_t == "Y":
            subtype = a[item]["subtype"] if a[item]["subtype"] == b[item]["subtype"] and a[item]["subtype"] in SUBTYPES else "UNC"
            steps = a[item]["steps"] if a[item]["steps"] == b[item]["steps"] and a[item]["steps"] in STEPS else "UNC"
            usable_none = False; usable_subtype = subtype in SUBTYPES; usable_strength = steps in STEPS
        else:
            subtype, steps, usable_none, usable_subtype, usable_strength = "UNC", "UNC", False, False, False
        rows.append({
            "annotation_id": item, "parent_transformation": parent_t, "subtype": subtype, "steps": steps,
            "usable_none": usable_none, "usable_subtype": usable_subtype, "usable_strength": usable_strength,
        })
    atomic_jsonl(ASSET_DIR / "consensus_annotations.jsonl", rows)
    atomic_json(ASSET_DIR / "full_reliability_result.json", {
        "status": "PASS" if passed else "FAIL", "parent_y_rows": len(parent_y),
        "metrics": metrics, "gates": gates, "analysis_allowed": passed,
        "external_api_used": False, "consensus_sha256": sha256_file(ASSET_DIR / "consensus_annotations.jsonl"),
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
    columns = [
        work[feature].to_numpy(float), np.log1p(work["token_length"].to_numpy(float)),
        work["option_count"].to_numpy(float), work["has_numeric"].to_numpy(float),
    ]
    if adjust_label:
        columns.append(work["label_reasoning"].to_numpy(float))
    source = work["src"].astype(str).to_numpy(); levels = sorted(set(source))
    x = np.column_stack([np.ones(len(work)), *columns, *[(source == value).astype(float) for value in levels[1:]]])
    x = x[:, np.r_[True, np.ptp(x[:, 1:], axis=0) > 0]]
    y = work[endpoint].to_numpy(float)
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
    rank = int(np.linalg.matrix_rank(x)); df = max(len(work) - rank, 1)
    t_value = beta / se if se > 0 else math.nan
    p = float(2 * stats.t.sf(abs(t_value), df)) if math.isfinite(t_value) else math.nan
    critical = float(stats.t.ppf(0.975, df))
    return base | {"beta": beta, "se_hc3": se, "p": p, "ci_low": beta - critical * se,
                   "ci_high": beta + critical * se, "rank": rank}


def load_analysis_frame(manifest: dict[str, Any]) -> Any:
    import pandas as pd
    reliability = read_json(ASSET_DIR / "full_reliability_result.json")
    if reliability.get("status") != "PASS" or reliability.get("analysis_allowed") is not True:
        raise RuntimeError("Analysis blocked by full annotation reliability")
    parent_runner = load_module("natural_feature_v1_for_taxonomy_v1_1", PARENT_ANALYSIS_RUNNER)
    parent_manifest = parent_runner.load_and_verify_manifest()
    frame = parent_runner.load_analysis_frame(parent_manifest)
    key = pd.DataFrame(read_jsonl(resolve(manifest["inputs"]["annotation_key"]["path"])))
    taxonomy = pd.DataFrame(read_jsonl(ASSET_DIR / "consensus_annotations.jsonl"))
    taxonomy = key[["annotation_id", "row_index"]].merge(taxonomy, on="annotation_id", validate="one_to_one")
    fields = ["row_index", "parent_transformation", "subtype", "steps", "usable_subtype", "usable_none", "usable_strength"]
    frame = frame.merge(taxonomy[fields], on="row_index", validate="one_to_one")
    frame["step_ordinal"] = frame["steps"].map({"1": 1.0, "2": 2.0, "3P": 3.0})
    return frame


def subtype_table(split: Any) -> Any:
    import pandas as pd
    rows = []
    for subtype in SUBTYPES:
        work = split[(split["usable_none"]) | ((split["usable_subtype"]) & (split["subtype"] == subtype))].copy()
        work["feature"] = (work["subtype"] == subtype).astype(float)
        n_type = int((work["feature"] == 1).sum()); n_none = int((work["feature"] == 0).sum())
        for endpoint in ("label_reasoning", *PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS):
            result = fit_hc3(work, "feature", endpoint, adjust_label=endpoint != "label_reasoning")
            rows.append({"analysis": "subtype_vs_none", "subtype": subtype, "endpoint": endpoint,
                         "n_subtype": n_type, "n_none": n_none, **result})
    return pd.DataFrame(rows)


def strength_table(split: Any) -> Any:
    import pandas as pd
    work = split[split["usable_strength"]].copy(); rows = []
    counts = {str(level): int((work["step_ordinal"] == level).sum()) for level in (1, 2, 3)}
    for endpoint in (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS, "label_reasoning"):
        result = fit_hc3(work, "step_ordinal", endpoint, adjust_label=endpoint != "label_reasoning")
        rows.append({"analysis": "within_transformation_step_trend", "endpoint": endpoint,
                     "step1_n": counts["1"], "step2_n": counts["2"], "step3p_n": counts["3"], **result})
    return pd.DataFrame(rows)


def phase_analyze_discovery(manifest: dict[str, Any]) -> None:
    import numpy as np
    frame = load_analysis_frame(manifest); split = frame[frame["analysis_split"] == "discovery"].copy()
    subtype = subtype_table(split); strength = strength_table(split)
    subtype["count_gate"] = (subtype["n_subtype"] >= 50) & (subtype["n_none"] >= 100)
    subtype["primary_test"] = subtype["endpoint"].isin(PRIMARY_ENDPOINTS) & subtype["count_gate"]
    subtype["q_discovery_primary_bh"] = np.nan; mask = subtype["primary_test"]
    subtype.loc[mask, "q_discovery_primary_bh"] = bh_adjust(subtype.loc[mask, "p"].tolist())
    subtype["discovery_supported"] = subtype["primary_test"] & (subtype["q_discovery_primary_bh"] < 0.05) & ((subtype["ci_low"] > 0) | (subtype["ci_high"] < 0))
    strength["count_gate"] = strength[["step1_n", "step2_n", "step3p_n"]].min(axis=1) >= 30
    strength["primary_test"] = strength["endpoint"].isin(PRIMARY_ENDPOINTS) & strength["count_gate"]
    strength["q_discovery_primary_bh"] = np.nan; smask = strength["primary_test"]
    strength.loc[smask, "q_discovery_primary_bh"] = bh_adjust(strength.loc[smask, "p"].tolist())
    strength["discovery_supported"] = strength["primary_test"] & (strength["q_discovery_primary_bh"] < 0.05) & ((strength["ci_low"] > 0) | (strength["ci_high"] < 0))
    subtype_path = OUTPUT_DIR / "tables" / "discovery_subtype_associations.csv"
    strength_path = OUTPUT_DIR / "tables" / "discovery_step_trend.csv"
    atomic_csv(subtype_path, subtype); atomic_csv(strength_path, strength)
    selected_subtype = [{"subtype": row.subtype, "endpoint": row.endpoint, "discovery_beta": float(row.beta)} for row in subtype[subtype["discovery_supported"]].itertuples(index=False)]
    selected_strength = [{"endpoint": row.endpoint, "discovery_beta": float(row.beta)} for row in strength[strength["discovery_supported"]].itertuples(index=False)]
    atomic_json(OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json", {
        "status": "DISCOVERY_COMPLETE_HELDOUT_NOT_ANALYZED", "selected_subtype_pairs": selected_subtype,
        "selected_step_pairs": selected_strength, "discovery_subtype_sha256": sha256_file(subtype_path),
        "discovery_step_sha256": sha256_file(strength_path), "heldout_used_for_selection": False,
    })


def phase_analyze_heldout(manifest: dict[str, Any]) -> None:
    import pandas as pd
    selection_path = OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json"
    selection = read_json(selection_path)
    if selection.get("status") != "DISCOVERY_COMPLETE_HELDOUT_NOT_ANALYZED":
        raise RuntimeError("Discovery selection is not frozen")
    for name, expected in (("discovery_subtype_associations.csv", selection["discovery_subtype_sha256"]), ("discovery_step_trend.csv", selection["discovery_step_sha256"])):
        if sha256_file(OUTPUT_DIR / "tables" / name) != expected:
            raise RuntimeError(f"Frozen discovery table changed: {name}")
    frame = load_analysis_frame(manifest); split = frame[frame["analysis_split"] == "validation"].copy()
    all_subtype = subtype_table(split); all_strength = strength_table(split)
    subtype = pd.DataFrame([
        all_subtype[(all_subtype["subtype"] == spec["subtype"]) & (all_subtype["endpoint"] == spec["endpoint"])].iloc[0].to_dict() | {"discovery_beta": spec["discovery_beta"]}
        for spec in selection["selected_subtype_pairs"]
    ])
    if len(subtype):
        subtype["count_gate"] = (subtype["n_subtype"] >= 15) & (subtype["n_none"] >= 30)
        subtype["q_heldout_selected_bh"] = bh_adjust(subtype["p"].tolist())
        subtype["same_sign"] = subtype["beta"] * subtype["discovery_beta"] > 0
        subtype["heldout_supported"] = subtype["count_gate"] & subtype["same_sign"] & (subtype["q_heldout_selected_bh"] < 0.05) & ((subtype["ci_low"] > 0) | (subtype["ci_high"] < 0))
    strength = pd.DataFrame([
        all_strength[all_strength["endpoint"] == spec["endpoint"]].iloc[0].to_dict() | {"discovery_beta": spec["discovery_beta"]}
        for spec in selection["selected_step_pairs"]
    ])
    if len(strength):
        strength["count_gate"] = strength[["step1_n", "step2_n", "step3p_n"]].min(axis=1) >= 10
        strength["q_heldout_selected_bh"] = bh_adjust(strength["p"].tolist())
        strength["same_sign"] = strength["beta"] * strength["discovery_beta"] > 0
        strength["heldout_supported"] = strength["count_gate"] & strength["same_sign"] & (strength["q_heldout_selected_bh"] < 0.05) & ((strength["ci_low"] > 0) | (strength["ci_high"] < 0))
    subtype_path = OUTPUT_DIR / "tables" / "heldout_subtype_checks.csv"; strength_path = OUTPUT_DIR / "tables" / "heldout_step_checks.csv"
    atomic_csv(subtype_path, subtype); atomic_csv(strength_path, strength)
    atomic_json(OUTPUT_DIR / "transformation_taxonomy_summary.json", {
        "status": "COMPLETE_POST_DISCOVERY_TAXONOMY_ASSOCIATION_ONLY", "external_api_used": False,
        "selected_subtype_pair_count": len(selection["selected_subtype_pairs"]),
        "heldout_supported_subtype_pairs": [] if not len(subtype) else [{"subtype": row.subtype, "endpoint": row.endpoint, "beta": float(row.beta)} for row in subtype[subtype["heldout_supported"]].itertuples(index=False)],
        "selected_step_pair_count": len(selection["selected_step_pairs"]),
        "heldout_supported_step_pairs": [] if not len(strength) else [{"endpoint": row.endpoint, "beta": float(row.beta)} for row in strength[strength["heldout_supported"]].itertuples(index=False)],
        "heldout_is_independent_confirmatory": False, "causal_claim_allowed": False,
        "mediation_claim_allowed": False, "selection_sha256": sha256_file(selection_path),
        "heldout_subtype_sha256": sha256_file(subtype_path), "heldout_step_sha256": sha256_file(strength_path),
    })


def phase_final_audit() -> None:
    artifacts = {
        "manifest": MANIFEST_PATH, "authorization": AUTHORIZATION_PATH,
        "preflight": ASSET_DIR / "preflight_result.json", "reliability": ASSET_DIR / "full_reliability_result.json",
        "consensus": ASSET_DIR / "consensus_annotations.jsonl",
        "discovery_subtype": OUTPUT_DIR / "tables" / "discovery_subtype_associations.csv",
        "discovery_step": OUTPUT_DIR / "tables" / "discovery_step_trend.csv",
        "selection": OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json",
        "heldout_subtype": OUTPUT_DIR / "tables" / "heldout_subtype_checks.csv",
        "heldout_step": OUTPUT_DIR / "tables" / "heldout_step_checks.csv",
        "summary": OUTPUT_DIR / "transformation_taxonomy_summary.json",
    }
    missing = [name for name, path in artifacts.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing final artifacts: {missing}")
    blind = read_jsonl(ASSET_DIR / "blind_parent_y_items.jsonl")
    consensus = read_jsonl(ASSET_DIR / "consensus_annotations.jsonl")
    if len(blind) != 895 or len(consensus) != 3000:
        raise RuntimeError("Final row count mismatch")
    if any(set(row) != {"annotation_id", "question", "options"} for row in blind):
        raise RuntimeError("Blind annotation leakage detected")
    atomic_json(OUTPUT_DIR / "final_audit.json", {
        "status": "PASS", "parent_y_annotation_rows": 895, "consensus_rows": 3000,
        "blind_annotation_leakage": False, "external_api_used": False,
        "new_study_model_forward": False, "intervention_performed": False,
        "artifact_sha256": {name: sha256_file(path) for name, path in artifacts.items()},
        "implementation_sha256": sha256_file(Path(__file__)),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=(
        "prepare", "preflight_a", "preflight_b", "audit_preflight", "annotate_a", "annotate_b",
        "consensus", "analyze_discovery", "analyze_heldout", "final_audit",
    ))
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
