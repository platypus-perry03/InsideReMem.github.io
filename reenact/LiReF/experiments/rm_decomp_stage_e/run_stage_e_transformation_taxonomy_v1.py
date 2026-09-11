#!/usr/bin/env python3
"""Blind local annotation and post-discovery Transformation taxonomy analysis.

No external API is used. Annotation phases only read question/options and anonymous
IDs. Analysis phases reuse frozen scalar artifacts from Natural Feature Discovery v1.
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
MANIFEST_PATH = STAGE_DIR / "stage_e_transformation_taxonomy_v1_design_frozen.json"
AUTHORIZATION_PATH = STAGE_DIR / "stage_e_transformation_taxonomy_v1_local_execution_authorization_frozen.json"
ASSET_DIR = STAGE_DIR / "stage_e_transformation_taxonomy_v1_assets"
OUTPUT_DIR = ROOT / "liref_outputs" / "rm_decomp" / "v2" / "e_transformation_taxonomy_v1"
V1_RUNNER_PATH = STAGE_DIR / "run_stage_e_natural_feature_discovery_v1.py"

TYPE_VALUES = ("ARITH", "LOGIC", "FORMAL", "CAUSAL", "MIXED", "NONE", "UNC")
STEP_VALUES = ("0", "1", "2", "3P", "UNC")
PRIMARY_ENDPOINTS = ("layer31_liref", "component_L29H00030", "component_L30H00006")
SECONDARY_ENDPOINTS = ("component_L31N13336", "component_L29H00031")
SUBTYPES = ("ARITH", "LOGIC", "FORMAL", "CAUSAL", "MIXED")
SEED = 20260831

ANNOTATION_SYSTEM = """You classify the transformation operations required to solve a multiple-choice item. Do not solve the item and do not identify the correct option. You receive only the question and options.

T=Y only when information in the prompt must be changed or combined by an arithmetic, logical, formal/rule-based, or causal operation. T=N when one explicit fact or one recalled fact maps directly to an option. Use UNC only when genuinely unclear.

Choose one dominant TYPE:
- ARITH: the key terminal operation is numerical calculation, numerical comparison, ratio, or unit conversion.
- CAUSAL: the key operation traces cause, mechanism, intervention, or counterfactual to an effect.
- FORMAL: a domain-specific formula, law, grammar, legal rule, or algorithmic procedure is applied, and the key operation is not primarily arithmetic or causal.
- LOGIC: qualitative inference over propositions, conditions, negation, quantifiers, cases, or relations.
- MIXED: two or more types are independently indispensable and no dominant type can be selected using the priorities above.
- NONE: no transformation is required.
- UNC: cannot classify reliably.

Count dependent transformation STEPS only:
- 0: no transformation.
- 1: one operation.
- 2: two dependent operations, where the first result feeds the second.
- 3P: three or more dependent operations.
- UNC: cannot count reliably.
Fact recall, reading, checking options, or mechanically expanding one calculation are not extra steps.

Consistency: T=N implies TYPE=NONE and STEPS=0. A non-NONE TYPE implies T=Y and STEPS is 1, 2, or 3P.
Return exactly one line and no explanation:
T=Y|N|UNC;TYPE=ARITH|LOGIC|FORMAL|CAUSAL|MIXED|NONE|UNC;STEPS=0|1|2|3P|UNC"""

PATTERN = re.compile(
    r"T\s*=\s*(Y|N|UNC)\s*;\s*TYPE\s*=\s*"
    r"(ARITH|LOGIC|FORMAL|CAUSAL|MIXED|NONE|UNC)\s*;\s*"
    r"STEPS\s*=\s*(0|1|2|3P|UNC)", re.IGNORECASE,
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


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(MANIFEST_PATH)
    authorization = read_json(AUTHORIZATION_PATH)
    if manifest.get("status") != "design_frozen_local_blind_annotation_and_analysis_allowed":
        raise RuntimeError("Taxonomy design is not frozen")
    if manifest.get("external_api_allowed") is not False or manifest.get("intervention_allowed") is not False:
        raise RuntimeError("External API/intervention must remain forbidden")
    if authorization.get("execution_allowed") is not True:
        raise RuntimeError("Local execution is not authorized")
    if authorization.get("external_api_allowed") is not False:
        raise RuntimeError("Authorization unexpectedly permits an external API")
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
    t_value, type_value, step_value = (value.upper() for value in match.groups())
    if t_value == "N" and not (type_value == "NONE" and step_value == "0"):
        return None
    if type_value in SUBTYPES and not (t_value == "Y" and step_value in {"1", "2", "3P"}):
        return None
    if type_value == "NONE" and not (t_value == "N" and step_value == "0"):
        return None
    return {"transformation": t_value, "subtype": type_value, "steps": step_value}


def render_item(item: dict[str, Any]) -> str:
    options = "\n".join(f"{chr(65 + index)}. {value}" for index, value in enumerate(item["options"]))
    return f"Question:\n{item['question']}\nOptions:\n{options}"


def encode_prompts(tokenizer: Any, items: list[dict[str, Any]], retry: bool) -> list[str]:
    suffix = "\nYour last format was invalid. Return the exact one-line grammar only." if retry else ""
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
        prompts = encode_prompts(tokenizer, batch_items, retry)
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded, do_sample=False, num_beams=1, max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        start = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(output[:, start:], skip_special_tokens=True)
        for (item, attempt), text in zip(batch, texts):
            parsed = parse_annotation(text)
            if parsed is None and attempt < 3:
                pending.append((item, attempt + 1))
                continue
            if parsed is None:
                parsed = {"transformation": "UNC", "subtype": "UNC", "steps": "UNC"}
            complete[item["annotation_id"]] = {
                "annotation_id": item["annotation_id"], **parsed,
                "raw_output": text.strip(), "attempts": attempt,
                "parse_valid": parsed["transformation"] != "UNC" or bool(PATTERN.search(text.strip())),
            }
        del encoded, output
    return [complete[key] for key in sorted(complete)]


def stratified_preflight(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Choose a deterministic, category-diverse subset without any internal outcome."""
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
    cursor = 0
    while len(chosen) < count:
        category = categories[cursor % len(categories)]
        if grouped[category]:
            chosen.append(grouped[category].pop())
        cursor += 1
        if cursor > count * len(categories) * 4:
            raise RuntimeError("Unable to construct preflight sample")
    return chosen


def phase_prepare(manifest: dict[str, Any]) -> None:
    v1_blind = read_jsonl(resolve(manifest["inputs"]["blind_items"]["path"]))
    v1_key = {row["annotation_id"]: row for row in read_jsonl(resolve(manifest["inputs"]["annotation_key"]["path"]))}
    parent = {row["annotation_id"]: row for row in read_jsonl(resolve(manifest["inputs"]["parent_consensus"]["path"]))}
    dataset = read_json(resolve(manifest["inputs"]["dataset"]["path"]))
    if len(v1_blind) != 3000 or set(v1_key) != set(parent):
        raise RuntimeError("Parent annotation coverage mismatch")
    blind = []
    private = []
    for row in v1_blind:
        annotation_id = row["annotation_id"]
        index = int(v1_key[annotation_id]["row_index"])
        record = dataset[index]
        blind.append({"annotation_id": annotation_id, "question": row["question"], "options": row["options"]})
        private.append({
            "annotation_id": annotation_id, "row_index": index,
            "question_id": str(record["question_id"]), "category": str(record["category"]),
            "src": str(record["src"]), "parent_transformation": parent[annotation_id]["transformation_required"],
        })
    if any(set(row) != {"annotation_id", "question", "options"} for row in blind):
        raise RuntimeError("Blind fields are invalid")
    atomic_jsonl(ASSET_DIR / "blind_items.jsonl", blind)
    atomic_jsonl(ASSET_DIR / "private_key_DO_NOT_USE_DURING_ANNOTATION.jsonl", private)
    by_id = {row["annotation_id"]: row for row in blind}
    candidates = {value: [row for row in private if row["parent_transformation"] == value] for value in ("Y", "N")}
    selected_private = []
    for offset, value in enumerate(("Y", "N")):
        selected_private.extend(stratified_preflight(candidates[value], 48, SEED + offset))
    random.Random(SEED + 9).shuffle(selected_private)
    preflight_blind = [by_id[row["annotation_id"]] for row in selected_private]
    atomic_jsonl(ASSET_DIR / "preflight_blind_items.jsonl", preflight_blind)
    atomic_jsonl(ASSET_DIR / "preflight_private_key.jsonl", selected_private)
    atomic_json(ASSET_DIR / "preparation_audit.json", {
        "status": "PASS", "rows": 3000, "preflight_rows": 96,
        "preflight_parent_counts": dict(Counter(row["parent_transformation"] for row in selected_private)),
        "blind_fields": ["annotation_id", "question", "options"],
        "blind_items_sha256": sha256_file(ASSET_DIR / "blind_items.jsonl"),
        "preflight_blind_sha256": sha256_file(ASSET_DIR / "preflight_blind_items.jsonl"),
        "external_api_used": False,
    })


def phase_annotate(manifest: dict[str, Any], authorization: dict[str, Any], slot: str, preflight: bool) -> None:
    import torch
    device = authorization["device"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local annotation")
    source = ASSET_DIR / ("preflight_blind_items.jsonl" if preflight else "blind_items.jsonl")
    if not preflight:
        result = read_json(ASSET_DIR / "preflight_result.json")
        if result.get("status") != "PASS" or not result.get("full_annotation_allowed"):
            raise RuntimeError("Full annotation blocked by preflight")
    model, tokenizer = load_model(resolve(manifest["models"][slot]["path"]), device)
    try:
        rows = generate(model, tokenizer, read_jsonl(source), int(authorization["batch_size"]), device)
        name = f"{'preflight_' if preflight else ''}{slot}_annotations.jsonl"
        atomic_jsonl(ASSET_DIR / name, rows)
    finally:
        del model, tokenizer
        torch.cuda.empty_cache()


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    labels = sorted(set(a) | set(b))
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[label] / len(a)) * (cb[label] / len(a)) for label in labels)
    if expected >= 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def weighted_kappa(a: list[str], b: list[str]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    mapping = {"0": 0, "1": 1, "2": 2, "3P": 3}
    if any(value not in mapping for value in a + b):
        return None
    n = len(a); maximum = 9.0
    observed = sum(((mapping[x] - mapping[y]) ** 2) / maximum for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = 0.0
    for x in mapping:
        for y in mapping:
            expected += (ca[x] / n) * (cb[y] / n) * (((mapping[x] - mapping[y]) ** 2) / maximum)
    return 1.0 - observed / expected if expected > 0 else (1.0 if observed == 0 else None)


def balanced_accuracy(truth: list[str], prediction: list[str]) -> float | None:
    recalls = []
    for label in ("Y", "N"):
        indexes = [index for index, value in enumerate(truth) if value == label]
        if not indexes:
            return None
        recalls.append(sum(prediction[index] == label for index in indexes) / len(indexes))
    return sum(recalls) / 2


def safe_ge(value: float | None, threshold: float) -> bool:
    return value is not None and math.isfinite(value) and value >= threshold


def phase_audit_preflight() -> None:
    key = {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / "preflight_private_key.jsonl")}
    annotations = {
        slot: {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / f"preflight_{slot}_annotations.jsonl")}
        for slot in ("annotator_a", "annotator_b")
    }
    ids = sorted(key)
    if any(set(values) != set(ids) for values in annotations.values()):
        raise RuntimeError("Preflight annotation coverage mismatch")
    parse_rate = {
        slot: sum(bool(annotations[slot][item]["parse_valid"]) for item in ids) / len(ids)
        for slot in annotations
    }
    truth = [key[item]["parent_transformation"] for item in ids]
    t_values = {slot: [annotations[slot][item]["transformation"] for item in ids] for slot in annotations}
    ba = {slot: balanced_accuracy(truth, values) for slot, values in t_values.items()}
    t_agreement = sum(x == y for x, y in zip(t_values["annotator_a"], t_values["annotator_b"])) / len(ids)
    t_kappa = cohen_kappa(t_values["annotator_a"], t_values["annotator_b"])
    y_ids = [item for item in ids if key[item]["parent_transformation"] == "Y"]
    type_joint = [
        item for item in y_ids
        if annotations["annotator_a"][item]["transformation"] == "Y"
        and annotations["annotator_b"][item]["transformation"] == "Y"
        and annotations["annotator_a"][item]["subtype"] in SUBTYPES
        and annotations["annotator_b"][item]["subtype"] in SUBTYPES
    ]
    type_a = [annotations["annotator_a"][item]["subtype"] for item in type_joint]
    type_b = [annotations["annotator_b"][item]["subtype"] for item in type_joint]
    type_coverage = len(type_joint) / len(y_ids)
    type_agreement = sum(x == y for x, y in zip(type_a, type_b)) / len(type_a) if type_a else None
    type_kappa = cohen_kappa(type_a, type_b)
    step_joint = [
        item for item in ids
        if annotations["annotator_a"][item]["steps"] in {"0", "1", "2", "3P"}
        and annotations["annotator_b"][item]["steps"] in {"0", "1", "2", "3P"}
    ]
    step_a = [annotations["annotator_a"][item]["steps"] for item in step_joint]
    step_b = [annotations["annotator_b"][item]["steps"] for item in step_joint]
    step_agreement = sum(x == y for x, y in zip(step_a, step_b)) / len(step_a) if step_a else None
    step_kappa = weighted_kappa(step_a, step_b)
    concentrations = {}
    for slot in annotations:
        types = [annotations[slot][item]["subtype"] for item in y_ids if annotations[slot][item]["subtype"] in SUBTYPES]
        steps = [annotations[slot][item]["steps"] for item in ids if annotations[slot][item]["steps"] in {"0", "1", "2", "3P"}]
        concentrations[slot] = {
            "type_max_fraction": max(Counter(types).values()) / len(types) if types else 1.0,
            "step_max_fraction": max(Counter(steps).values()) / len(steps) if steps else 1.0,
        }
    metrics = {
        "parse_rate_a": parse_rate["annotator_a"], "parse_rate_b": parse_rate["annotator_b"],
        "parent_balanced_accuracy_a": ba["annotator_a"], "parent_balanced_accuracy_b": ba["annotator_b"],
        "t_raw_agreement": t_agreement, "t_kappa": t_kappa,
        "parent_y_subtype_joint_coverage": type_coverage,
        "subtype_raw_agreement": type_agreement, "subtype_kappa": type_kappa,
        "step_raw_agreement": step_agreement, "step_weighted_kappa": step_kappa,
    }
    gates = {
        "parse_a": safe_ge(metrics["parse_rate_a"], 0.99), "parse_b": safe_ge(metrics["parse_rate_b"], 0.99),
        "parent_ba_a": safe_ge(metrics["parent_balanced_accuracy_a"], 0.80),
        "parent_ba_b": safe_ge(metrics["parent_balanced_accuracy_b"], 0.80),
        "t_agreement": safe_ge(t_agreement, 0.80), "t_kappa": safe_ge(t_kappa, 0.60),
        "subtype_coverage": safe_ge(type_coverage, 0.80),
        "subtype_agreement": safe_ge(type_agreement, 0.55), "subtype_kappa": safe_ge(type_kappa, 0.40),
        "step_agreement": safe_ge(step_agreement, 0.55), "step_kappa": safe_ge(step_kappa, 0.45),
        "nondegenerate": all(value["type_max_fraction"] <= 0.95 and value["step_max_fraction"] <= 0.95 for value in concentrations.values()),
    }
    passed = all(gates.values())
    atomic_json(ASSET_DIR / "preflight_result.json", {
        "status": "PASS" if passed else "FAIL", "rows": len(ids), "metrics": metrics,
        "concentrations": concentrations, "gates": gates,
        "full_annotation_allowed": passed, "external_api_used": False,
    })


def phase_consensus() -> None:
    parent = {row["annotation_id"]: row for row in read_jsonl(resolve(read_json(MANIFEST_PATH)["inputs"]["parent_consensus"]["path"]))}
    annotations = {
        slot: {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / f"{slot}_annotations.jsonl")}
        for slot in ("annotator_a", "annotator_b")
    }
    ids = sorted(parent)
    if any(set(values) != set(ids) for values in annotations.values()):
        raise RuntimeError("Full annotation coverage mismatch")
    a, b = annotations["annotator_a"], annotations["annotator_b"]
    t_a = [a[item]["transformation"] for item in ids]
    t_b = [b[item]["transformation"] for item in ids]
    t_kappa = cohen_kappa(t_a, t_b)
    parent_y = [item for item in ids if parent[item]["transformation_required"] == "Y"]
    type_joint = [item for item in parent_y if a[item]["transformation"] == b[item]["transformation"] == "Y" and a[item]["subtype"] in SUBTYPES and b[item]["subtype"] in SUBTYPES]
    type_a = [a[item]["subtype"] for item in type_joint]; type_b = [b[item]["subtype"] for item in type_joint]
    type_kappa = cohen_kappa(type_a, type_b)
    type_coverage = len(type_joint) / len(parent_y)
    step_joint = [item for item in parent_y if a[item]["steps"] in {"1", "2", "3P"} and b[item]["steps"] in {"1", "2", "3P"}]
    step_a = [a[item]["steps"] for item in step_joint]; step_b = [b[item]["steps"] for item in step_joint]
    step_kappa = weighted_kappa(step_a, step_b)
    step_coverage = len(step_joint) / len(parent_y)
    gates = {
        "t_kappa": safe_ge(t_kappa, 0.60), "subtype_kappa": safe_ge(type_kappa, 0.40),
        "subtype_coverage": safe_ge(type_coverage, 0.70), "step_weighted_kappa": safe_ge(step_kappa, 0.45),
        "step_coverage": safe_ge(step_coverage, 0.70),
    }
    passed = all(gates.values())
    rows = []
    for item in ids:
        row = {"annotation_id": item, "parent_transformation": parent[item]["transformation_required"]}
        for output, field in (("transformation", "transformation"), ("subtype", "subtype"), ("steps", "steps")):
            row[output] = a[item][field] if a[item][field] == b[item][field] else "UNC"
        row["usable_subtype"] = bool(row["parent_transformation"] == "Y" and row["transformation"] == "Y" and row["subtype"] in SUBTYPES)
        row["usable_none"] = bool(row["parent_transformation"] == "N" and row["transformation"] == "N" and row["subtype"] == "NONE" and row["steps"] == "0")
        row["usable_strength"] = bool(row["usable_subtype"] and row["steps"] in {"1", "2", "3P"})
        rows.append(row)
    atomic_jsonl(ASSET_DIR / "consensus_annotations.jsonl", rows)
    atomic_json(ASSET_DIR / "full_reliability_result.json", {
        "status": "PASS" if passed else "FAIL", "rows": len(ids),
        "metrics": {"t_kappa": t_kappa, "subtype_kappa": type_kappa, "subtype_joint_coverage": type_coverage,
                    "step_weighted_kappa": step_kappa, "step_joint_coverage": step_coverage},
        "gates": gates, "analysis_allowed": passed, "external_api_used": False,
        "consensus_sha256": sha256_file(ASSET_DIR / "consensus_annotations.jsonl"),
    })


def bh_adjust(values: list[float]) -> list[float]:
    import numpy as np
    p = np.asarray(values, dtype=float); output = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 1.0
    for reverse in range(len(order) - 1, -1, -1):
        index = order[reverse]; rank = reverse + 1
        running = min(running, float(p[index]) * len(order) / rank)
        output[index] = running
    return output.tolist()


def fit_hc3(frame: Any, feature: str, endpoint: str, adjust_label: bool) -> dict[str, Any]:
    import numpy as np
    from scipy import stats
    needed = [feature, endpoint, "token_length", "option_count", "has_numeric", "src"]
    if adjust_label:
        needed.append("label_reasoning")
    work = frame[needed].dropna().copy()
    base = {"n": len(work)}
    if len(work) < 10 or work[feature].nunique() < 2:
        return base | {key: math.nan for key in ("beta", "se_hc3", "p", "ci_low", "ci_high", "rank")}
    columns = [
        work[feature].to_numpy(float), np.log1p(work["token_length"].to_numpy(float)),
        work["option_count"].to_numpy(float), work["has_numeric"].to_numpy(float),
    ]
    if adjust_label:
        columns.append(work["label_reasoning"].to_numpy(float))
    source_values = work["src"].astype(str).to_numpy(); sources = sorted(set(source_values))
    dummies = [(source_values == source).astype(float) for source in sources[1:]]
    x = np.column_stack([np.ones(len(work)), *columns, *dummies])
    varying = np.r_[True, np.ptp(x[:, 1:], axis=0) > 0]; x = x[:, varying]
    y = work[endpoint].to_numpy(float)
    if endpoint != "label_reasoning":
        scale = y.std(ddof=0)
        if not np.isfinite(scale) or scale <= 0:
            return base | {key: math.nan for key in ("beta", "se_hc3", "p", "ci_low", "ci_high", "rank")}
        y = (y - y.mean()) / scale
    inverse = np.linalg.pinv(x.T @ x, rcond=1e-10)
    coefficients = inverse @ x.T @ y; residual = y - x @ coefficients
    leverage = np.sum((x @ inverse) * x, axis=1)
    adjusted = residual / np.clip(1.0 - leverage, 1e-8, None)
    covariance = inverse @ ((x.T * adjusted**2) @ x) @ inverse
    se = float(math.sqrt(max(float(covariance[1, 1]), 0.0)))
    beta = float(coefficients[1]); rank = int(np.linalg.matrix_rank(x)); df = max(len(work) - rank, 1)
    t_value = beta / se if se > 0 else math.nan
    p = float(2 * stats.t.sf(abs(t_value), df)) if math.isfinite(t_value) else math.nan
    critical = float(stats.t.ppf(0.975, df))
    return base | {"beta": beta, "se_hc3": se, "p": p, "ci_low": beta - critical * se,
                   "ci_high": beta + critical * se, "rank": rank}


def load_analysis_frame() -> Any:
    import pandas as pd
    reliability = read_json(ASSET_DIR / "full_reliability_result.json")
    if reliability.get("status") != "PASS" or not reliability.get("analysis_allowed"):
        raise RuntimeError("Analysis blocked by full annotation reliability")
    v1 = load_module("natural_feature_v1_for_taxonomy", V1_RUNNER_PATH)
    v1_manifest = v1.load_and_verify_manifest()
    frame = v1.load_analysis_frame(v1_manifest)
    key = pd.DataFrame(read_jsonl(resolve(read_json(MANIFEST_PATH)["inputs"]["annotation_key"]["path"])))
    taxonomy = pd.DataFrame(read_jsonl(ASSET_DIR / "consensus_annotations.jsonl"))
    taxonomy = key[["annotation_id", "row_index"]].merge(taxonomy, on="annotation_id", validate="one_to_one")
    frame = frame.merge(taxonomy[["row_index", "parent_transformation", "transformation", "subtype", "steps", "usable_subtype", "usable_none", "usable_strength"]], on="row_index", validate="one_to_one")
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
    work = split[split["usable_strength"]].copy()
    rows = []
    counts = {str(level): int((work["step_ordinal"] == level).sum()) for level in (1, 2, 3)}
    for endpoint in (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS, "label_reasoning"):
        result = fit_hc3(work, "step_ordinal", endpoint, adjust_label=endpoint != "label_reasoning")
        rows.append({"analysis": "within_transformation_step_trend", "endpoint": endpoint,
                     "step1_n": counts["1"], "step2_n": counts["2"], "step3p_n": counts["3"], **result})
    return pd.DataFrame(rows)


def phase_analyze_discovery() -> None:
    import numpy as np
    frame = load_analysis_frame(); split = frame[frame["analysis_split"] == "discovery"].copy()
    subtype = subtype_table(split); strength = strength_table(split)
    subtype["count_gate"] = (subtype["n_subtype"] >= 50) & (subtype["n_none"] >= 100)
    subtype["primary_test"] = subtype["endpoint"].isin(PRIMARY_ENDPOINTS) & subtype["count_gate"]
    subtype["q_discovery_primary_bh"] = np.nan
    mask = subtype["primary_test"]
    subtype.loc[mask, "q_discovery_primary_bh"] = bh_adjust(subtype.loc[mask, "p"].tolist())
    subtype["discovery_supported"] = subtype["primary_test"] & (subtype["q_discovery_primary_bh"] < 0.05) & ((subtype["ci_low"] > 0) | (subtype["ci_high"] < 0))
    strength["count_gate"] = (strength[["step1_n", "step2_n", "step3p_n"]].min(axis=1) >= 30)
    strength["primary_test"] = strength["endpoint"].isin(PRIMARY_ENDPOINTS) & strength["count_gate"]
    strength["q_discovery_primary_bh"] = np.nan
    smask = strength["primary_test"]
    strength.loc[smask, "q_discovery_primary_bh"] = bh_adjust(strength.loc[smask, "p"].tolist())
    strength["discovery_supported"] = strength["primary_test"] & (strength["q_discovery_primary_bh"] < 0.05) & ((strength["ci_low"] > 0) | (strength["ci_high"] < 0))
    subtype_path = OUTPUT_DIR / "tables" / "discovery_subtype_associations.csv"
    strength_path = OUTPUT_DIR / "tables" / "discovery_step_trend.csv"
    atomic_csv(subtype_path, subtype); atomic_csv(strength_path, strength)
    selected_subtype = [{"subtype": row.subtype, "endpoint": row.endpoint, "discovery_beta": float(row.beta)} for row in subtype[subtype["discovery_supported"]].itertuples(index=False)]
    selected_strength = [{"endpoint": row.endpoint, "discovery_beta": float(row.beta)} for row in strength[strength["discovery_supported"]].itertuples(index=False)]
    atomic_json(OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json", {
        "status": "DISCOVERY_COMPLETE_HELDOUT_NOT_ANALYZED",
        "selected_subtype_pairs": selected_subtype, "selected_step_pairs": selected_strength,
        "discovery_subtype_sha256": sha256_file(subtype_path), "discovery_step_sha256": sha256_file(strength_path),
        "heldout_used_for_selection": False,
    })


def phase_analyze_heldout() -> None:
    import numpy as np
    import pandas as pd
    selection_path = OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json"
    selection = read_json(selection_path)
    if selection.get("status") != "DISCOVERY_COMPLETE_HELDOUT_NOT_ANALYZED":
        raise RuntimeError("Discovery selection is not frozen")
    if sha256_file(OUTPUT_DIR / "tables" / "discovery_subtype_associations.csv") != selection["discovery_subtype_sha256"]:
        raise RuntimeError("Discovery subtype table changed")
    if sha256_file(OUTPUT_DIR / "tables" / "discovery_step_trend.csv") != selection["discovery_step_sha256"]:
        raise RuntimeError("Discovery step table changed")
    frame = load_analysis_frame(); split = frame[frame["analysis_split"] == "validation"].copy()
    all_subtype = subtype_table(split); all_strength = strength_table(split)
    subtype_rows = []
    for spec in selection["selected_subtype_pairs"]:
        row = all_subtype[(all_subtype["subtype"] == spec["subtype"]) & (all_subtype["endpoint"] == spec["endpoint"])].iloc[0].to_dict()
        row["discovery_beta"] = spec["discovery_beta"]; subtype_rows.append(row)
    subtype = pd.DataFrame(subtype_rows)
    if len(subtype):
        subtype["count_gate"] = (subtype["n_subtype"] >= 15) & (subtype["n_none"] >= 30)
        subtype["q_heldout_selected_bh"] = bh_adjust(subtype["p"].tolist())
        subtype["same_sign"] = subtype["beta"] * subtype["discovery_beta"] > 0
        subtype["heldout_supported"] = subtype["count_gate"] & subtype["same_sign"] & (subtype["q_heldout_selected_bh"] < 0.05) & ((subtype["ci_low"] > 0) | (subtype["ci_high"] < 0))
    strength_rows = []
    for spec in selection["selected_step_pairs"]:
        row = all_strength[all_strength["endpoint"] == spec["endpoint"]].iloc[0].to_dict()
        row["discovery_beta"] = spec["discovery_beta"]; strength_rows.append(row)
    strength = pd.DataFrame(strength_rows)
    if len(strength):
        strength["count_gate"] = strength[["step1_n", "step2_n", "step3p_n"]].min(axis=1) >= 10
        strength["q_heldout_selected_bh"] = bh_adjust(strength["p"].tolist())
        strength["same_sign"] = strength["beta"] * strength["discovery_beta"] > 0
        strength["heldout_supported"] = strength["count_gate"] & strength["same_sign"] & (strength["q_heldout_selected_bh"] < 0.05) & ((strength["ci_low"] > 0) | (strength["ci_high"] < 0))
    subtype_path = OUTPUT_DIR / "tables" / "heldout_subtype_checks.csv"
    strength_path = OUTPUT_DIR / "tables" / "heldout_step_checks.csv"
    atomic_csv(subtype_path, subtype); atomic_csv(strength_path, strength)
    atomic_json(OUTPUT_DIR / "transformation_taxonomy_summary.json", {
        "status": "COMPLETE_POST_DISCOVERY_TAXONOMY_ASSOCIATION_ONLY",
        "human_annotation": False, "external_api_used": False,
        "selected_subtype_pair_count": len(selection["selected_subtype_pairs"]),
        "heldout_supported_subtype_pairs": [] if not len(subtype) else [
            {"subtype": row.subtype, "endpoint": row.endpoint, "beta": float(row.beta)}
            for row in subtype[subtype["heldout_supported"]].itertuples(index=False)
        ],
        "selected_step_pair_count": len(selection["selected_step_pairs"]),
        "heldout_supported_step_pairs": [] if not len(strength) else [
            {"endpoint": row.endpoint, "beta": float(row.beta)}
            for row in strength[strength["heldout_supported"]].itertuples(index=False)
        ],
        "heldout_is_independent_confirmatory": False,
        "causal_claim_allowed": False, "mediation_claim_allowed": False,
        "selection_sha256": sha256_file(selection_path),
        "heldout_subtype_sha256": sha256_file(subtype_path), "heldout_step_sha256": sha256_file(strength_path),
    })


def phase_final_audit() -> None:
    artifacts = {
        "design_manifest": MANIFEST_PATH, "authorization": AUTHORIZATION_PATH,
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
    blind = read_jsonl(ASSET_DIR / "blind_items.jsonl")
    consensus = read_jsonl(ASSET_DIR / "consensus_annotations.jsonl")
    if len(blind) != 3000 or len(consensus) != 3000:
        raise RuntimeError("Final row count mismatch")
    if any(set(row) != {"annotation_id", "question", "options"} for row in blind):
        raise RuntimeError("Blind annotation leakage detected")
    atomic_json(OUTPUT_DIR / "final_audit.json", {
        "status": "PASS", "rows": 3000, "blind_annotation_leakage": False,
        "external_api_used": False, "new_study_model_forward": False, "intervention_performed": False,
        "artifact_sha256": {name: sha256_file(path) for name, path in artifacts.items()},
        "implementation_sha256": sha256_file(Path(__file__)),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=(
        "prepare", "preflight_a", "preflight_b", "audit_preflight",
        "annotate_a", "annotate_b", "consensus", "analyze_discovery",
        "analyze_heldout", "final_audit",
    ))
    args = parser.parse_args()
    manifest, authorization = verify()
    if args.phase == "prepare": phase_prepare(manifest)
    elif args.phase == "preflight_a": phase_annotate(manifest, authorization, "annotator_a", True)
    elif args.phase == "preflight_b": phase_annotate(manifest, authorization, "annotator_b", True)
    elif args.phase == "audit_preflight": phase_audit_preflight()
    elif args.phase == "annotate_a": phase_annotate(manifest, authorization, "annotator_a", False)
    elif args.phase == "annotate_b": phase_annotate(manifest, authorization, "annotator_b", False)
    elif args.phase == "consensus": phase_consensus()
    elif args.phase == "analyze_discovery": phase_analyze_discovery()
    elif args.phase == "analyze_heldout": phase_analyze_heldout()
    else: phase_final_audit()


if __name__ == "__main__":
    main()
