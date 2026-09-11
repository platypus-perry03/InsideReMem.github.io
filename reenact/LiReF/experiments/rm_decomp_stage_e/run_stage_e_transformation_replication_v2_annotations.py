#!/usr/bin/env python3
"""Blind annotation preflight for transformation replication v2.

This runner never loads LiReF directions, component responses, or study-model
hidden states.  It labels only question/option text with two local instruct
models and evaluates the frozen annotation-instrument gates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = STAGE_DIR / "stage_e_transformation_replication_v2_assets"
MANIFEST_PATH = STAGE_DIR / "stage_e_transformation_replication_v2_design_frozen.json"
AUTHORIZATION_PATH = STAGE_DIR / "stage_e_transformation_replication_v2_annotation_preflight_authorization_frozen.json"
PREVIOUS_DATASET = ROOT / "liref" / "dataset" / "mmlu-pro-3000samples.json"
PREVIOUS_SPLIT = ROOT / "liref_outputs" / "layerwise_liref" / "split_ids.json"
SEED = 20260831


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


def verify_locked_files() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(MANIFEST_PATH)
    authorization = read_json(AUTHORIZATION_PATH)
    if manifest.get("status") != "design_frozen_annotation_preflight_allowed_study_model_execution_not_authorized":
        raise RuntimeError("Design manifest does not authorize annotation preflight")
    if authorization.get("execution_scope") != "annotation_preflight_only":
        raise RuntimeError("Authorization scope is not annotation preflight only")
    for item in authorization["locked_files"]:
        path = STAGE_DIR / item["path"] if not Path(item["path"]).is_absolute() else Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Locked file hash mismatch: {path}")
    for spec in manifest["models"].values():
        config = ROOT / spec["path"] / "config.json"
        if sha256_file(config) != spec["config_sha256"]:
            raise RuntimeError(f"Model config hash mismatch: {config}")
    return manifest, authorization


def render_item(item: dict[str, Any]) -> str:
    options = "\n".join(f"{chr(65 + index)}. {value}" for index, value in enumerate(item["options"]))
    return f"Question:\n{item['question']}\nOptions:\n{options}"


SCORE_SYSTEM = """Analyze how much solving the multiple-choice question relies on reasoning versus memory recall. Follow this fixed scale adapted from the published LiReF Appendix A prompt:
- 1.0 means it strictly requires multi-step reasoning, calculations, formulas, or deductions.
- 0.0 means purely factual recall or direct inference of humanities knowledge.
- Intermediate values indicate hybrid characteristics.
Guidelines: numerical values or percentages support a higher score; required calculation steps and unit conversion support a higher score; direct textbook recall supports a lower score. Do not infer or mention any dataset label. Give one brief analysis sentence, then a final line exactly SCORE=<number from 0.00 to 1.00>."""

TRANSFORMATION_SYSTEM = """Decide whether solving the multiple-choice item requires transforming given values or statements by an arithmetic, logical, formal, or causal rule. T=Y only when such a transformation is necessary. Direct recall, definition lookup, or locating an explicitly stated fact is T=N. Minor reading, option recognition, and remembering a relation are not transformations. Use T=UNC only if genuinely indeterminate. Return exactly T=Y, T=N, or T=UNC and no explanation."""

SCORE_PATTERN = re.compile(r"SCORE\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)", re.IGNORECASE)
TRANSFORMATION_PATTERN = re.compile(r"^\s*T\s*=\s*(Y|N|UNC)\s*$", re.IGNORECASE)


def parse_score(text: str) -> float | None:
    matches = SCORE_PATTERN.findall(text)
    if not matches:
        return None
    value = float(matches[-1])
    return value if 0.0 <= value <= 1.0 else None


def parse_transformation(text: str) -> str | None:
    match = TRANSFORMATION_PATTERN.match(text)
    return match.group(1).upper() if match else None


def chat_prompts(tokenizer: Any, system: str, items: list[dict[str, Any]]) -> list[str]:
    prompts = []
    for item in items:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": render_item(item)}]
        if getattr(tokenizer, "chat_template", None):
            try:
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            except Exception:
                combined = [{"role": "user", "content": system + "\n\n" + render_item(item)}]
                prompts.append(tokenizer.apply_chat_template(combined, tokenize=False, add_generation_prompt=True))
        else:
            prompts.append(f"System: {system}\nUser: {render_item(item)}\nAssistant:")
    return prompts


def load_model(path: Path, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device).eval()
    return model, tokenizer


def generate(model: Any, tokenizer: Any, items: list[dict[str, Any]], system: str,
             max_new_tokens: int, batch_size: int, device: str) -> list[str]:
    import torch
    outputs: list[str] = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = chat_prompts(tokenizer, system, batch)
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        input_width = encoded["input_ids"].shape[1]
        outputs.extend(tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True))
    return outputs


def score_bin(value: float) -> str:
    if value == 0.0:
        return "zero"
    if value <= 0.5:
        return "low_nonzero"
    if value < 0.9:
        return "high_subpointnine"
    return "pointnine_or_one"


def phase_prepare_preflight() -> None:
    records = read_json(PREVIOUS_DATASET)
    split = read_json(PREVIOUS_SPLIT)
    heldout = [records[int(index)] for index in split["heldout"]["row_indices"]]
    bins: dict[str, list[dict[str, Any]]] = {name: [] for name in ("zero", "low_nonzero", "high_subpointnine", "pointnine_or_one")}
    for row in heldout:
        bins[score_bin(float(row["memory_reason_score"]))].append(row)
    selected = []
    for offset, name in enumerate(bins):
        candidates = sorted(bins[name], key=lambda row: (str(row["src"]), str(row["question_id"])))
        random.Random(SEED + offset).shuffle(candidates)
        if len(candidates) < 48:
            raise RuntimeError(f"Preflight score bin {name} has fewer than 48 items")
        selected.extend(candidates[:48])
    random.Random(SEED).shuffle(selected)
    blind = [{"preflight_id": f"TRP-{index:03d}", "question": row["question"], "options": row["options"]}
             for index, row in enumerate(selected)]
    key = [{
        "preflight_id": f"TRP-{index:03d}", "question_id": str(row["question_id"]),
        "original_memory_reason_score": float(row["memory_reason_score"]),
        "original_binary_reasoning": int(float(row["memory_reason_score"]) > 0.5),
        "category": str(row["category"]), "src": str(row["src"]),
    } for index, row in enumerate(selected)]
    atomic_jsonl(ASSET_DIR / "preflight_blind_items.jsonl", blind)
    atomic_jsonl(ASSET_DIR / "preflight_key_private.jsonl", key)
    atomic_json(ASSET_DIR / "preflight_preparation_audit.json", {
        "status": "PASS", "rows": len(blind), "score_bins": {name: 48 for name in bins},
        "blind_fields": ["preflight_id", "question", "options"],
        "blind_items_sha256": sha256_file(ASSET_DIR / "preflight_blind_items.jsonl"),
        "private_key_sha256": sha256_file(ASSET_DIR / "preflight_key_private.jsonl"),
    })


def phase_annotate(annotator: str, manifest: dict[str, Any], authorization: dict[str, Any]) -> None:
    items = read_jsonl(ASSET_DIR / "preflight_blind_items.jsonl")
    spec = manifest["models"][annotator]
    device = authorization["device"]
    model, tokenizer = load_model(ROOT / spec["path"], device)
    score_text = generate(model, tokenizer, items, SCORE_SYSTEM, 96, authorization["batch_size"], device)
    transformation_text = generate(model, tokenizer, items, TRANSFORMATION_SYSTEM, 12, authorization["batch_size"], device)
    rows = []
    for item, score_raw, transformation_raw in zip(items, score_text, transformation_text):
        rows.append({
            "preflight_id": item["preflight_id"], "annotator": annotator,
            "score": parse_score(score_raw), "transformation": parse_transformation(transformation_raw),
            "score_raw": score_raw, "transformation_raw": transformation_raw,
        })
    atomic_jsonl(ASSET_DIR / f"preflight_{annotator}_annotations.jsonl", rows)
    del model, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def rank_values(values: list[float]) -> list[float]:
    import scipy.stats as stats
    return list(stats.rankdata(values, method="average"))


def spearman(a: list[float], b: list[float]) -> float:
    import scipy.stats as stats
    return float(stats.spearmanr(a, b).statistic)


def cohen_kappa(a: list[int], b: list[int]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def balanced_accuracy(truth: list[int], prediction: list[int]) -> float:
    recalls = []
    for label in (0, 1):
        indices = [index for index, value in enumerate(truth) if value == label]
        recalls.append(sum(prediction[index] == label for index in indices) / len(indices))
    return sum(recalls) / 2.0


def phase_audit_preflight() -> None:
    key = {row["preflight_id"]: row for row in read_jsonl(ASSET_DIR / "preflight_key_private.jsonl")}
    annotations = {name: {row["preflight_id"]: row for row in read_jsonl(ASSET_DIR / f"preflight_{name}_annotations.jsonl")}
                   for name in ("annotator_a", "annotator_b")}
    ids = sorted(key)
    if any(set(rows) != set(ids) for rows in annotations.values()):
        raise RuntimeError("Preflight annotation coverage mismatch")
    original = [float(key[item]["original_memory_reason_score"]) for item in ids]
    original_binary = [int(key[item]["original_binary_reasoning"]) for item in ids]
    metrics: dict[str, Any] = {}
    scores: dict[str, list[float]] = {}
    for name, rows in annotations.items():
        valid = [rows[item]["score"] is not None for item in ids]
        parsed = [float(rows[item]["score"]) for item in ids if rows[item]["score"] is not None]
        ref = [value for value, keep in zip(original, valid) if keep]
        scores[name] = [float(rows[item]["score"]) if rows[item]["score"] is not None else math.nan for item in ids]
        metrics[name] = {
            "score_parse_rate": sum(valid) / len(valid),
            "original_score_spearman": spearman(ref, parsed) if len(parsed) > 2 else math.nan,
        }
    both_score = [math.isfinite(scores["annotator_a"][i]) and math.isfinite(scores["annotator_b"][i]) for i in range(len(ids))]
    a_score = [scores["annotator_a"][i] for i, keep in enumerate(both_score) if keep]
    b_score = [scores["annotator_b"][i] for i, keep in enumerate(both_score) if keep]
    ref_score = [original[i] for i, keep in enumerate(both_score) if keep]
    ensemble = [(a + b) / 2.0 for a, b in zip(a_score, b_score)]
    ref_binary = [int(value > 0.5) for value in ref_score]
    ensemble_binary = [int(value > 0.5) for value in ensemble]
    a_binary = [int(value > 0.5) for value in a_score]
    b_binary = [int(value > 0.5) for value in b_score]

    transform_a = [annotations["annotator_a"][item]["transformation"] for item in ids]
    transform_b = [annotations["annotator_b"][item]["transformation"] for item in ids]
    transform_parse_rate_a = sum(value is not None for value in transform_a) / len(transform_a)
    transform_parse_rate_b = sum(value is not None for value in transform_b) / len(transform_b)
    transform_valid = [a in {"Y", "N"} and b in {"Y", "N"} for a, b in zip(transform_a, transform_b)]
    ta = [int(transform_a[i] == "Y") for i, keep in enumerate(transform_valid) if keep]
    tb = [int(transform_b[i] == "Y") for i, keep in enumerate(transform_valid) if keep]
    raw_transform_agreement = sum(a == b for a, b in zip(ta, tb)) / len(ta) if ta else math.nan

    gates = {
        "score_parse_rate_a": metrics["annotator_a"]["score_parse_rate"] >= 0.99,
        "score_parse_rate_b": metrics["annotator_b"]["score_parse_rate"] >= 0.99,
        "original_spearman_a": metrics["annotator_a"]["original_score_spearman"] >= 0.60,
        "original_spearman_b": metrics["annotator_b"]["original_score_spearman"] >= 0.60,
        "ensemble_original_spearman": spearman(ref_score, ensemble) >= 0.70,
        "ensemble_balanced_accuracy": balanced_accuracy(ref_binary, ensemble_binary) >= 0.70,
        "interannotator_score_spearman": spearman(a_score, b_score) >= 0.70,
        "interannotator_binary_kappa": cohen_kappa(a_binary, b_binary) >= 0.60,
        "transformation_parse_rate_a": transform_parse_rate_a >= 0.99,
        "transformation_parse_rate_b": transform_parse_rate_b >= 0.99,
        "transformation_joint_valid_coverage": (len(ta) / len(ids)) >= 0.98,
        "transformation_raw_agreement": raw_transform_agreement >= 0.80,
        "transformation_kappa": cohen_kappa(ta, tb) >= 0.60 if ta else False,
    }
    result = {
        "schema_id": "stage_e_transformation_replication_v2_annotation_preflight",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "rows": len(ids), "both_score_valid_rows": len(a_score), "both_transformation_valid_rows": len(ta),
        "annotators": metrics,
        "ensemble_original_spearman": spearman(ref_score, ensemble),
        "ensemble_balanced_accuracy": balanced_accuracy(ref_binary, ensemble_binary),
        "interannotator_score_spearman": spearman(a_score, b_score),
        "interannotator_binary_kappa": cohen_kappa(a_binary, b_binary),
        "transformation_parse_rate_a": transform_parse_rate_a,
        "transformation_parse_rate_b": transform_parse_rate_b,
        "transformation_joint_valid_coverage": len(ta) / len(ids),
        "transformation_raw_agreement": raw_transform_agreement,
        "transformation_kappa": cohen_kappa(ta, tb) if ta else math.nan,
        "gates": gates,
        "full_candidate_annotation_allowed": all(gates.values()),
        "study_model_execution_allowed": False,
    }
    atomic_json(ASSET_DIR / "annotation_preflight_result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare_preflight", "annotate_a", "annotate_b", "audit_preflight"))
    args = parser.parse_args()
    manifest, authorization = verify_locked_files()
    if args.phase == "prepare_preflight":
        phase_prepare_preflight()
    elif args.phase == "annotate_a":
        phase_annotate("annotator_a", manifest, authorization)
    elif args.phase == "annotate_b":
        phase_annotate("annotator_b", manifest, authorization)
    else:
        phase_audit_preflight()


if __name__ == "__main__":
    main()
