#!/usr/bin/env python3
"""Blind natural-feature annotation and association analysis for Stage E.

The annotation phases never load R/M labels or internal outcomes.  The scalar
extraction phase uses read-only hooks and stores projections only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = STAGE_DIR / "stage_e_natural_feature_discovery_v1_design_frozen.json"
ANNOTATION_AMENDMENT_PATH = STAGE_DIR / "stage_e_natural_feature_discovery_v1_1_annotation_instrument_amendment_frozen.json"
ANNOTATION_AMENDMENT_SHA256 = "2680fca0bbe841b73e3ede444b77c71d4e87fe85ea5a770754d3de69a28d7029"
CONSENSUS_AMENDMENT_PATH = STAGE_DIR / "stage_e_natural_feature_discovery_v1_2_consensus_amendment_frozen.json"
CONSENSUS_AMENDMENT_SHA256 = "f53778a56864ea9756c87cde884538b63ef62590a6b6cb6ef5aa660a3a9b140f"
ASSET_DIR = STAGE_DIR / "stage_e_natural_feature_discovery_v1_assets"
OUTPUT_DIR = ROOT / "liref_outputs" / "rm_decomp" / "v2" / "e_natural_feature_discovery_v1"

FORBIDDEN_ANNOTATION_FIELDS = {
    "answer", "answer_index", "cot_content", "category", "src",
    "memory_reason_score", "label", "analysis_split", "row_index",
    "liref", "projection", "activation", "total_contribution",
}
FEATURE_FIELDS = (
    "answer_mode", "transformation_required", "composition_required",
    "multi_step_required", "external_knowledge_required", "answer_directness",
)
VALID_VALUES = {
    "answer_mode": {"RET", "DER", "MIX", "UNC"},
    "transformation_required": {"Y", "N", "UNC"},
    "composition_required": {"Y", "N", "UNC"},
    "multi_step_required": {"Y", "N", "UNC"},
    "external_knowledge_required": {"Y", "N", "UNC"},
    "answer_directness": {"DIR", "IND", "UNC"},
}
COMPONENT_IDS = ("L31N13336", "L29H00030", "L30H00006", "L29H00031")
ANALYSIS_FEATURES = (
    "mode_derivation_vs_retrieval", "transformation_required",
    "composition_required", "multi_step_required",
    "external_knowledge_required", "answer_indirect",
)
INTERNAL_ENDPOINTS = ("layer31_liref",) + tuple(f"component_{x}" for x in COMPONENT_IDS)
ALL_ENDPOINTS = ("label_reasoning",) + INTERNAL_ENDPOINTS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_and_verify_manifest() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    if manifest["status"] != "design_frozen_ai_annotation_and_read_only_scalar_extraction_allowed":
        raise RuntimeError("Feature-discovery design is not frozen for execution")
    if not manifest.get("execution_allowed") or manifest.get("intervention_allowed"):
        raise RuntimeError("Execution/intervention flags violate the frozen design")
    if sha256_file(ANNOTATION_AMENDMENT_PATH) != ANNOTATION_AMENDMENT_SHA256:
        raise RuntimeError("Frozen v1.1 annotation-instrument amendment hash mismatch")
    amendment = read_json(ANNOTATION_AMENDMENT_PATH)
    if amendment.get("status") != "frozen_before_second_annotator_and_before_internal_outcome_analysis":
        raise RuntimeError("Annotation-instrument amendment status is invalid")
    if sha256_file(CONSENSUS_AMENDMENT_PATH) != CONSENSUS_AMENDMENT_SHA256:
        raise RuntimeError("Frozen v1.2 consensus amendment hash mismatch")
    consensus_amendment = read_json(CONSENSUS_AMENDMENT_PATH)
    if consensus_amendment.get("status") != "frozen_before_internal_outcome_analysis":
        raise RuntimeError("Consensus amendment status is invalid")
    design = STAGE_DIR / manifest["design_document"]
    if sha256_file(design) != manifest["design_document_sha256"]:
        raise RuntimeError("Frozen design document hash mismatch")
    for spec in manifest["inputs"].values():
        path = resolve(ROOT, spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"Locked input hash mismatch: {path}")
    for spec in manifest["models"].values():
        config = resolve(ROOT, spec["path"]) / "config.json"
        if sha256_file(config) != spec["config_sha256"]:
            raise RuntimeError(f"Locked model config mismatch: {config}")
    return manifest


def blind_item(record: dict[str, Any], annotation_id: str) -> dict[str, Any]:
    result = {
        "annotation_id": annotation_id,
        "question": str(record["question"]),
        "options": [str(value) for value in record["options"]],
    }
    leak = FORBIDDEN_ANNOTATION_FIELDS.intersection(result)
    if leak:
        raise RuntimeError(f"Blind item leaked forbidden fields: {sorted(leak)}")
    return result


def phase_prepare_blind(manifest: dict[str, Any]) -> None:
    dataset = read_json(resolve(ROOT, manifest["inputs"]["dataset"]["path"]))
    if len(dataset) != 3000:
        raise RuntimeError("Expected exactly 3,000 LiReF items")
    rows = [blind_item(record, f"NF{index:04d}") for index, record in enumerate(dataset)]
    if len({row["annotation_id"] for row in rows}) != 3000:
        raise RuntimeError("Blind annotation IDs are not unique")
    path = ASSET_DIR / "blind_annotation_items.jsonl"
    atomic_jsonl(path, rows)
    key = [{"annotation_id": f"NF{index:04d}", "row_index": index, "question_id": str(record["question_id"])}
           for index, record in enumerate(dataset)]
    atomic_jsonl(ASSET_DIR / "annotation_key_DO_NOT_USE_DURING_ANNOTATION.jsonl", key)
    atomic_json(ASSET_DIR / "blind_preparation_audit.json", {
        "status": "PASS", "rows": len(rows), "allowed_fields": ["annotation_id", "question", "options"],
        "forbidden_fields_absent": True, "blind_items_sha256": sha256_file(path),
        "annotation_key_sha256": sha256_file(ASSET_DIR / "annotation_key_DO_NOT_USE_DURING_ANNOTATION.jsonl"),
    })


ANNOTATION_SYSTEM = """You label the requirements of solving a multiple-choice item, not its correct answer. You receive only the question and options. Use this fixed codebook:
MODE: RET if selecting the answer mainly requires recalling one stored domain fact, definition, attribution, or association. DER if the needed premises are stated in the prompt and applying arithmetic, logical, formal, causal, or rule-based operations derives the answer. MIX only when BOTH external factual knowledge AND nontrivial derivation are indispensable. Do not use MIX as a safe/default choice. Use UNC if genuinely unclear.
T: Y only if values or statements must be transformed by arithmetic, logical, formal, or causal rules.
C: Y only if two or more distinct premises or constraints must be integrated.
S: Y only if at least two dependent inference operations are required.
K: Y if domain facts, definitions, laws, attributions, or associations absent from the prompt are required. Merely naming concepts in answer options does NOT provide the facts relating those concepts.
A: DIR if one explicit premise or one recalled fact maps directly to an option; IND if transformation/composition/multiple dependent steps are needed.
Examples outside the study data:
- 'What is the capital of France?' => MODE=RET;T=N;C=N;S=N;K=Y;A=DIR
- 'A box has 8 balls and gains 3. How many now?' => MODE=DER;T=Y;C=Y;S=N;K=N;A=IND
- 'Using Ohm's law, compute current from stated voltage/resistance' => MODE=MIX;T=Y;C=Y;S=N;K=Y;A=IND
Return exactly one line in this grammar and no explanation:
MODE=RET|DER|MIX|UNC;T=Y|N|UNC;C=Y|N|UNC;S=Y|N|UNC;K=Y|N|UNC;A=DIR|IND|UNC"""


ANNOTATION_PATTERN = re.compile(
    r"MODE\s*=\s*(RET|DER|MIX|UNC)\s*;\s*T\s*=\s*(Y|N|UNC)\s*;\s*"
    r"C\s*=\s*(Y|N|UNC)\s*;\s*S\s*=\s*(Y|N|UNC)\s*;\s*"
    r"K\s*=\s*(Y|N|UNC)\s*;\s*A\s*=\s*(DIR|IND|UNC)", re.IGNORECASE,
)


def parse_annotation(text: str) -> dict[str, str] | None:
    match = ANNOTATION_PATTERN.search(text.strip())
    if not match:
        return None
    values = [value.upper() for value in match.groups()]
    result = dict(zip(FEATURE_FIELDS, values))
    if any(value not in VALID_VALUES[key] for key, value in result.items()):
        return None
    return result


def render_question(item: dict[str, Any]) -> str:
    options = "\n".join(f"{chr(65 + index)}. {value}" for index, value in enumerate(item["options"]))
    extra = ""
    if "annotator_a" in item and "annotator_b" in item:
        extra = (
            "\nTwo blind annotations disagree. Re-evaluate the item using the codebook; "
            "do not choose by majority.\n"
            f"Annotation A: {json.dumps(item['annotator_a'], sort_keys=True)}\n"
            f"Annotation B: {json.dumps(item['annotator_b'], sort_keys=True)}\n"
            f"Disputed fields: {','.join(item.get('disagreement_fields', []))}\n"
        )
    return f"Question:\n{item['question']}\nOptions:\n{options}{extra}"


def encode_chat(tokenizer: Any, user_texts: list[str], retry: bool = False) -> list[str]:
    suffix = "\nYour previous format was invalid. Output the exact one-line grammar only." if retry else ""
    result = []
    for text in user_texts:
        messages = [{"role": "system", "content": ANNOTATION_SYSTEM}, {"role": "user", "content": text + suffix}]
        if getattr(tokenizer, "chat_template", None):
            try:
                result.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            except Exception:
                combined = [{"role": "user", "content": ANNOTATION_SYSTEM + "\n\n" + text + suffix}]
                result.append(tokenizer.apply_chat_template(combined, tokenize=False, add_generation_prompt=True))
        else:
            result.append(f"System: {ANNOTATION_SYSTEM}\nUser: {text + suffix}\nAssistant:")
    return result


def load_generation_model(model_path: Path, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device).eval()
    return model, tokenizer


def generate_annotations(model: Any, tokenizer: Any, items: list[dict[str, Any]], batch_size: int, device: str) -> list[dict[str, Any]]:
    import torch
    completed: list[dict[str, Any]] = []
    pending = [(item, False) for item in items]
    attempts: Counter[str] = Counter()
    while pending:
        current, pending = pending[:batch_size], pending[batch_size:]
        batch_items = [item for item, _retry in current]
        prompts = encode_chat(tokenizer, [render_question(item) for item in batch_items], retry=any(flag for _item, flag in current))
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=48,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        # With left padding, generated tokens begin after the common padded input width.
        start = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(output[:, start:], skip_special_tokens=True)
        for item, text in zip(batch_items, texts):
            annotation_id = item["annotation_id"]
            parsed = parse_annotation(text)
            attempts[annotation_id] += 1
            if parsed is None and attempts[annotation_id] < 3:
                pending.append((item, True))
                continue
            if parsed is None:
                parsed = {
                    "answer_mode": "UNC", "transformation_required": "UNC",
                    "composition_required": "UNC", "multi_step_required": "UNC",
                    "external_knowledge_required": "UNC", "answer_directness": "UNC",
                }
            completed.append({"annotation_id": annotation_id, **parsed, "raw_output": text.strip(), "attempts": attempts[annotation_id]})
        del encoded, output
    return sorted(completed, key=lambda row: row["annotation_id"])


def annotation_output_name(slot: str, preflight: bool = False) -> str:
    return f"{slot}_{'preflight_' if preflight else ''}annotations_v1_1.jsonl"


def audit_preflight(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    if len(rows) != expected:
        raise RuntimeError(f"Preflight row count mismatch: {len(rows)} != {expected}")
    parse_validity = sum(int(row["attempts"] < 3 or any(row[field] != "UNC" for field in FEATURE_FIELDS)) for row in rows) / len(rows)
    concentration = {}
    for field in FEATURE_FIELDS:
        counts = Counter(row[field] for row in rows)
        concentration[field] = {"counts": dict(counts), "maximum_fraction": max(counts.values()) / len(rows)}
    passed = parse_validity >= 0.95 and all(value["maximum_fraction"] <= 0.95 for value in concentration.values())
    return {"status": "PASS" if passed else "FAIL", "rows": len(rows), "parse_validity": parse_validity, "feature_concentration": concentration}


def phase_annotate(manifest: dict[str, Any], slot: str, preflight: bool = False) -> None:
    if slot not in {"annotator_a", "annotator_b", "adjudicator"}:
        raise ValueError(slot)
    import torch
    device = manifest["model_execution"]["device"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local AI annotation")
    model_path = resolve(ROOT, manifest["models"][slot]["path"])
    model, tokenizer = load_generation_model(model_path, device)
    try:
        if slot == "adjudicator":
            source = read_jsonl(ASSET_DIR / "blind_adjudication_items.jsonl")
        else:
            source = read_jsonl(ASSET_DIR / "blind_annotation_items.jsonl")
        if preflight:
            source = source[:48]
        output = generate_annotations(model, tokenizer, source, int(manifest["model_execution"]["annotation_batch_size"]), device)
        path = ASSET_DIR / annotation_output_name(slot, preflight)
        atomic_jsonl(path, output)
        if preflight:
            audit = audit_preflight(output, 48)
            audit["output_sha256"] = sha256_file(path)
            atomic_json(ASSET_DIR / f"{slot}_preflight_audit_v1_1.json", audit)
            if audit["status"] != "PASS":
                raise RuntimeError(f"{slot} failed frozen blind preflight")
    finally:
        del model
        torch.cuda.empty_cache()


def phase_prepare_adjudication(_manifest: dict[str, Any]) -> None:
    blind = {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / "blind_annotation_items.jsonl")}
    a = {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / annotation_output_name("annotator_a"))}
    b = {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / annotation_output_name("annotator_b"))}
    if set(blind) != set(a) or set(blind) != set(b):
        raise RuntimeError("Annotator coverage mismatch")
    rows = []
    for annotation_id in sorted(blind):
        disagreements = [field for field in FEATURE_FIELDS if a[annotation_id][field] != b[annotation_id][field]]
        if disagreements:
            item = dict(blind[annotation_id])
            item["annotator_a"] = {field: a[annotation_id][field] for field in FEATURE_FIELDS}
            item["annotator_b"] = {field: b[annotation_id][field] for field in FEATURE_FIELDS}
            item["disagreement_fields"] = disagreements
            rows.append(item)
    atomic_jsonl(ASSET_DIR / "blind_adjudication_items.jsonl", rows)
    atomic_json(ASSET_DIR / "adjudication_preparation_audit.json", {
        "status": "PASS", "total_items": len(blind), "items_requiring_adjudication": len(rows),
        "forbidden_outcome_fields_absent": all(not FORBIDDEN_ANNOTATION_FIELDS.intersection(row) for row in rows),
    })


def cohen_kappa(values_a: list[str], values_b: list[str]) -> float:
    labels = sorted(set(values_a) | set(values_b))
    if not labels:
        return math.nan
    n = len(values_a)
    observed = sum(x == y for x, y in zip(values_a, values_b)) / n
    ca, cb = Counter(values_a), Counter(values_b)
    expected = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else (1.0 if observed == 1 else math.nan)


def phase_consensus(_manifest: dict[str, Any]) -> None:
    a = {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / annotation_output_name("annotator_a"))}
    b = {row["annotation_id"]: row for row in read_jsonl(ASSET_DIR / annotation_output_name("annotator_b"))}
    rows = []
    reliability: dict[str, Any] = {}
    for field in FEATURE_FIELDS:
        va = [a[key][field] for key in sorted(a)]
        vb = [b[key][field] for key in sorted(b)]
        reliability[field] = {
            "raw_agreement": sum(x == y for x, y in zip(va, vb)) / len(va),
            "cohen_kappa": cohen_kappa(va, vb),
            "eligible_for_feature_selection": cohen_kappa(va, vb) >= 0.6,
            "annotator_a_counts": dict(Counter(va)), "annotator_b_counts": dict(Counter(vb)),
        }
    for annotation_id in sorted(a):
        row = {"annotation_id": annotation_id}
        sources = {}
        for field in FEATURE_FIELDS:
            if a[annotation_id][field] == b[annotation_id][field]:
                row[field] = a[annotation_id][field]; sources[field] = "agreement"
            else:
                row[field] = "UNC"; sources[field] = "unresolved"
        row["consensus_sources"] = sources
        rows.append(row)
    atomic_jsonl(ASSET_DIR / "consensus_annotations_v1_2.jsonl", rows)
    atomic_json(ASSET_DIR / "annotation_reliability_summary_v1_2.json", {
        "status": "COMPLETE_AI_ONLY", "human_annotation": False, "rows": len(rows),
        "consensus_rule": "exact_two_annotator_agreement_else_UNC",
        "feature_selection_kappa_minimum": 0.6,
        "features": reliability, "consensus_sha256": sha256_file(ASSET_DIR / "consensus_annotations_v1_2.jsonl"),
    })


def load_frozen_directions(manifest: dict[str, Any]) -> Any:
    """Load a locally produced, hash-verified Stage A checkpoint."""
    import numpy as np
    import torch
    path = resolve(ROOT, manifest["inputs"]["directions"]["path"])
    if sha256_file(path) != manifest["inputs"]["directions"]["sha256"]:
        raise RuntimeError("Direction checkpoint hash mismatch before trusted load")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    directions = np.asarray(payload["result"]["unit_directions"], dtype=np.float32)
    if directions.shape != (32, 4096) or not np.all(np.isfinite(directions)):
        raise RuntimeError(f"Unexpected frozen direction shape: {directions.shape}")
    return directions


def phase_extract_layer_scalars(manifest: dict[str, Any]) -> None:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_name = manifest["model_execution"]["device"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Layer scalar extraction")
    device = torch.device(device_name)
    dataset = read_json(resolve(ROOT, manifest["inputs"]["dataset"]["path"]))
    directions_np = load_frozen_directions(manifest)
    model_path = resolve(ROOT, manifest["models"]["study"]["path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).to(device).eval()
    directions = torch.as_tensor(directions_np, device=device, dtype=torch.float32)
    current: dict[int, list[float]] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3 or hidden.shape[-1] != 4096:
                raise RuntimeError(f"Unexpected block output at layer {layer_index}: {tuple(hidden.shape)}")
            scalar = torch.matmul(hidden[:, -1, :].float(), directions[layer_index])
            current[layer_index] = scalar.detach().cpu().tolist()
            return None
        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(layer_index)))

    batch_size = int(manifest["model_execution"]["study_batch_size"])
    output_rows: list[dict[str, Any]] = []
    try:
        for start in range(0, len(dataset), batch_size):
            records = dataset[start:start + batch_size]
            prompts = [f"Q: {record['question']}\nA: " for record in records]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            current.clear()
            with torch.inference_mode():
                _ = model(**encoded, use_cache=False, output_hidden_states=False, return_dict=True)
            if set(current) != set(range(32)):
                raise RuntimeError(f"Missing read-only hook outputs: {sorted(set(range(32)) - set(current))}")
            for offset, record in enumerate(records):
                row = {"row_index": start + offset, "question_id": str(record["question_id"])}
                row.update({f"layer_{layer}": float(current[layer][offset]) for layer in range(32)})
                output_rows.append(row)
            del encoded
    finally:
        for handle in handles:
            handle.remove()
        del model
        torch.cuda.empty_cache()

    frame = pd.DataFrame(output_rows)
    if len(frame) != 3000 or frame["row_index"].nunique() != 3000:
        raise RuntimeError("Layer scalar extraction did not cover all items exactly once")
    scalar_columns = [f"layer_{layer}" for layer in range(32)]
    if not np.isfinite(frame[scalar_columns].to_numpy()).all():
        raise RuntimeError("Non-finite Layer scalar detected")
    path = OUTPUT_DIR / "tables" / "layer_liref_scalars.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)
    atomic_json(OUTPUT_DIR / "manifests" / "layer_scalar_extraction.json", {
        "status": "PASS", "rows": 3000, "layers": 32,
        "read_only_hooks": True, "hidden_vectors_saved": False,
        "output_sha256": sha256_file(path),
    })


def bh_adjust(values: list[float]) -> list[float]:
    import numpy as np
    p = np.asarray(values, dtype=np.float64)
    output = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return output.tolist()
    order = valid[np.argsort(p[valid])]
    adjusted = np.empty(len(order), dtype=np.float64)
    running = 1.0
    for reverse_rank in range(len(order) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(p[index]) * len(order) / rank)
        adjusted[reverse_rank] = running
    for position, index in enumerate(order):
        output[index] = adjusted[position]
    return output.tolist()


def annotation_to_features(row: dict[str, Any]) -> dict[str, float]:
    def binary(value: str) -> float:
        return 1.0 if value == "Y" else 0.0 if value == "N" else math.nan
    return {
        "mode_derivation_vs_retrieval": 1.0 if row["answer_mode"] == "DER" else 0.0 if row["answer_mode"] == "RET" else math.nan,
        "transformation_required": binary(row["transformation_required"]),
        "composition_required": binary(row["composition_required"]),
        "multi_step_required": binary(row["multi_step_required"]),
        "external_knowledge_required": binary(row["external_knowledge_required"]),
        "answer_indirect": 1.0 if row["answer_directness"] == "IND" else 0.0 if row["answer_directness"] == "DIR" else math.nan,
    }


def ols_hc3(frame: Any, feature: str, endpoint: str, adjust_for_label: bool = False) -> dict[str, Any]:
    import numpy as np
    from scipy import stats

    needed = [feature, endpoint, "token_length", "option_count", "has_numeric", "src"]
    if adjust_for_label and endpoint != "label_reasoning" and "label_reasoning" not in needed:
        needed.append("label_reasoning")
    work = frame[needed].dropna().copy()
    n0 = int((work[feature] == 0).sum()); n1 = int((work[feature] == 1).sum())
    base = {"n": len(work), "n_absent": n0, "n_present": n1}
    if min(n0, n1) < 2:
        return base | {key: math.nan for key in ("beta", "se_hc3", "t", "p", "ci_low", "ci_high", "rank")}
    feature_values = work[feature].to_numpy(dtype=np.float64)
    continuous_columns = [
        feature_values,
        np.log1p(work["token_length"].to_numpy(dtype=np.float64)),
        work["option_count"].to_numpy(dtype=np.float64),
        work["has_numeric"].to_numpy(dtype=np.float64),
    ]
    if adjust_for_label and endpoint != "label_reasoning":
        continuous_columns.append(work["label_reasoning"].to_numpy(dtype=np.float64))
    continuous = np.column_stack(continuous_columns)
    sources = sorted(work["src"].astype(str).unique())
    dummies = np.column_stack([(work["src"].astype(str).to_numpy() == source).astype(float) for source in sources[1:]]) if len(sources) > 1 else np.empty((len(work), 0))
    x = np.column_stack([np.ones(len(work)), continuous, dummies])
    varying = np.r_[True, np.ptp(x[:, 1:], axis=0) > 0]
    x = x[:, varying]
    y = work[endpoint].to_numpy(dtype=np.float64)
    if endpoint != "label_reasoning":
        standard = y.std(ddof=0)
        if not np.isfinite(standard) or standard <= 0:
            return base | {key: math.nan for key in ("beta", "se_hc3", "t", "p", "ci_low", "ci_high", "rank")}
        y = (y - y.mean()) / standard
    inverse = np.linalg.pinv(x.T @ x, rcond=1e-10)
    coefficients = inverse @ x.T @ y
    residual = y - x @ coefficients
    leverage = np.sum((x @ inverse) * x, axis=1)
    adjusted = residual / np.clip(1.0 - leverage, 1e-8, None)
    covariance = inverse @ ((x.T * adjusted**2) @ x) @ inverse
    se = float(math.sqrt(max(float(covariance[1, 1]), 0.0)))
    beta = float(coefficients[1]); rank = int(np.linalg.matrix_rank(x)); df = max(len(work) - rank, 1)
    t_value = beta / se if se > 0 else math.nan
    p_value = float(2 * stats.t.sf(abs(t_value), df)) if math.isfinite(t_value) else math.nan
    critical = float(stats.t.ppf(0.975, df))
    return base | {"beta": beta, "se_hc3": se, "t": t_value, "p": p_value,
                   "ci_low": beta - critical * se, "ci_high": beta + critical * se, "rank": rank}


def odds_ratio(frame: Any, feature: str) -> float:
    work = frame[[feature, "label_reasoning"]].dropna()
    a = float(((work[feature] == 1) & (work["label_reasoning"] == 1)).sum()) + 0.5
    b = float(((work[feature] == 1) & (work["label_reasoning"] == 0)).sum()) + 0.5
    c = float(((work[feature] == 0) & (work["label_reasoning"] == 1)).sum()) + 0.5
    d = float(((work[feature] == 0) & (work["label_reasoning"] == 0)).sum()) + 0.5
    return (a * d) / (b * c)


def load_analysis_frame(manifest: dict[str, Any]) -> Any:
    import numpy as np
    import pandas as pd

    dataset = read_json(resolve(ROOT, manifest["inputs"]["dataset"]["path"]))
    split = read_json(resolve(ROOT, manifest["inputs"]["split"]["path"]))
    discovery = set(int(value) for value in split["train"]["row_indices"])
    validation = set(int(value) for value in split["heldout"]["row_indices"])
    rows = []
    for index, record in enumerate(dataset):
        rows.append({
            "row_index": index, "question_id": str(record["question_id"]),
            "analysis_split": "discovery" if index in discovery else "validation" if index in validation else "ERROR",
            "label_reasoning": int(float(record["memory_reason_score"]) > 0.5),
            "category": str(record["category"]), "src": str(record["src"]),
            "option_count": len(record["options"]),
            "has_numeric": int(bool(re.search(r"\d", str(record["question"])))),
        })
    frame = pd.DataFrame(rows)
    if (frame["analysis_split"] == "ERROR").any():
        raise RuntimeError("Split does not cover dataset")

    key = pd.DataFrame(read_jsonl(ASSET_DIR / "annotation_key_DO_NOT_USE_DURING_ANNOTATION.jsonl"))
    annotations = pd.DataFrame(read_jsonl(ASSET_DIR / "consensus_annotations_v1_2.jsonl"))
    annotation = key.merge(annotations, on="annotation_id", validate="one_to_one")
    for feature in ANALYSIS_FEATURES:
        annotation[feature] = [annotation_to_features(row)[feature] for row in annotation.to_dict("records")]
    frame = frame.merge(annotation[["row_index", *ANALYSIS_FEATURES]], on="row_index", validate="one_to_one")

    layer_path = OUTPUT_DIR / "tables" / "layer_liref_scalars.csv.gz"
    layer = pd.read_csv(layer_path, usecols=["row_index", "question_id", "layer_31"])
    layer["row_index"] = layer["row_index"].astype(int)
    layer["question_id"] = layer["question_id"].astype(str)
    layer = layer.rename(columns={"layer_31": "layer31_liref"})
    frame = frame.merge(layer, on=["row_index", "question_id"], validate="one_to_one")

    component_rows = []
    for split_name, input_name in (("discovery", "discovery_component_responses"), ("validation", "validation_component_responses")):
        source = pd.read_csv(resolve(ROOT, manifest["inputs"][input_name]["path"]), compression="gzip")
        source = source[(source["candidate_id"].isin(COMPONENT_IDS)) & (source["component_id"] == source["candidate_id"])].copy()
        if len(source) != (2400 if split_name == "discovery" else 600) * len(COMPONENT_IDS):
            raise RuntimeError(f"Unexpected component coverage for {split_name}: {len(source)}")
        source["analysis_split"] = split_name
        source["row_index"] = source["row_index"].astype(int)
        source["question_id"] = source["question_id"].astype(str)
        component_rows.append(source[["analysis_split", "row_index", "question_id", "token_length", "component_id", "total_contribution"]])
    long = pd.concat(component_rows, ignore_index=True)
    token_lengths = long.groupby(["analysis_split", "row_index", "question_id"], as_index=False)["token_length"].first()
    wide = long.pivot(index=["analysis_split", "row_index", "question_id"], columns="component_id", values="total_contribution").reset_index()
    wide = wide.rename(columns={component: f"component_{component}" for component in COMPONENT_IDS})
    wide = wide.merge(token_lengths, on=["analysis_split", "row_index", "question_id"], validate="one_to_one")
    frame = frame.merge(wide, on=["analysis_split", "row_index", "question_id"], validate="one_to_one")
    if len(frame) != 3000 or not np.isfinite(frame[["layer31_liref", *[f"component_{x}" for x in COMPONENT_IDS]]].to_numpy()).all():
        raise RuntimeError("Merged analysis frame is incomplete or non-finite")
    return frame


def write_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temporary, path)


def phase_analyze_discovery(manifest: dict[str, Any]) -> None:
    import pandas as pd
    frame = load_analysis_frame(manifest)
    discovery = frame[frame["analysis_split"] == "discovery"].copy()
    rows = []
    for feature in ANALYSIS_FEATURES:
        for endpoint in ALL_ENDPOINTS:
            result = ols_hc3(discovery, feature, endpoint)
            rows.append({
                "analysis_split": "discovery", "feature": feature, "endpoint": endpoint,
                **result, "unadjusted_odds_ratio": odds_ratio(discovery, feature) if endpoint == "label_reasoning" else math.nan,
            })
    table = pd.DataFrame(rows)
    table["q_discovery_global_bh"] = bh_adjust(table["p"].tolist())
    minimum = int(manifest["minimum_counts"]["discovery_per_level"])
    table["count_gate"] = (table["n_absent"] >= minimum) & (table["n_present"] >= minimum)
    table["discovery_supported"] = table["count_gate"] & (table["q_discovery_global_bh"] < float(manifest["fdr_alpha"])) & ((table["ci_low"] > 0) | (table["ci_high"] < 0))
    reliability = read_json(ASSET_DIR / "annotation_reliability_summary_v1_2.json")["features"]
    analysis_to_annotation = {
        "mode_derivation_vs_retrieval": "answer_mode", "transformation_required": "transformation_required",
        "composition_required": "composition_required", "multi_step_required": "multi_step_required",
        "external_knowledge_required": "external_knowledge_required", "answer_indirect": "answer_directness",
    }
    table["annotation_reliability_gate"] = [
        bool(reliability[analysis_to_annotation[feature]]["eligible_for_feature_selection"])
        for feature in table["feature"]
    ]
    path = OUTPUT_DIR / "tables" / "discovery_associations.csv"
    write_csv(path, table)

    selected = []
    for feature in ANALYSIS_FEATURES:
        group = table[table["feature"] == feature]
        label = group[group["endpoint"] == "label_reasoning"]
        internal = group[group["endpoint"].isin(INTERNAL_ENDPOINTS) & group["discovery_supported"]]
        reliability_ok = bool(group.iloc[0]["annotation_reliability_gate"])
        if reliability_ok and len(label) == 1 and bool(label.iloc[0]["discovery_supported"]) and len(internal):
            selected.append({
                "feature": feature,
                "label_discovery_beta": float(label.iloc[0]["beta"]),
                "internal_endpoints": [
                    {"endpoint": str(row.endpoint), "discovery_beta": float(row.beta), "discovery_q": float(row.q_discovery_global_bh)}
                    for row in internal.itertuples(index=False)
                ],
            })
    selection = {
        "schema_id": "stage_e_natural_feature_discovery_v1_selection",
        "status": "DISCOVERY_COMPLETE_VALIDATION_NOT_INSPECTED",
        "selection_rule": "label q<0.05 and at least one internal endpoint q<0.05 with frozen count gates",
        "selected_features": selected,
        "selected_feature_count": len(selected),
        "annotation_reliability_gate": "two-annotator Cohen kappa >= 0.6",
        "discovery_table_sha256": sha256_file(path),
        "validation_used_for_selection": False,
    }
    atomic_json(OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json", selection)


def phase_analyze_validation(manifest: dict[str, Any]) -> None:
    import pandas as pd
    selection_path = OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json"
    selection = read_json(selection_path)
    if selection.get("status") != "DISCOVERY_COMPLETE_VALIDATION_NOT_INSPECTED":
        raise RuntimeError("Discovery selection is not frozen before validation")
    discovery_path = OUTPUT_DIR / "tables" / "discovery_associations.csv"
    if sha256_file(discovery_path) != selection["discovery_table_sha256"]:
        raise RuntimeError("Discovery table changed after selection freeze")
    frame = load_analysis_frame(manifest)
    validation = frame[frame["analysis_split"] == "validation"].copy()
    rows = []
    for selected in selection["selected_features"]:
        feature = selected["feature"]
        endpoint_specs = [{"endpoint": "label_reasoning", "discovery_beta": selected["label_discovery_beta"]}] + selected["internal_endpoints"]
        for spec in endpoint_specs:
            result = ols_hc3(validation, feature, spec["endpoint"])
            rows.append({
                "analysis_split": "validation", "feature": feature, "endpoint": spec["endpoint"],
                "discovery_beta": float(spec["discovery_beta"]), **result,
                "unadjusted_odds_ratio": odds_ratio(validation, feature) if spec["endpoint"] == "label_reasoning" else math.nan,
            })
    base_columns = [
        "analysis_split", "feature", "endpoint", "discovery_beta", "n", "n_absent", "n_present",
        "beta", "se_hc3", "t", "p", "ci_low", "ci_high", "rank", "unadjusted_odds_ratio",
    ]
    table = pd.DataFrame(rows, columns=base_columns)
    if len(table):
        table["q_validation_selected_bh"] = bh_adjust(table["p"].tolist())
        minimum = int(manifest["minimum_counts"]["validation_per_level"])
        table["count_gate"] = (table["n_absent"] >= minimum) & (table["n_present"] >= minimum)
        table["same_sign"] = table["beta"] * table["discovery_beta"] > 0
        table["validation_supported"] = (
            table["count_gate"] & table["same_sign"] &
            (table["q_validation_selected_bh"] < float(manifest["fdr_alpha"])) &
            ((table["ci_low"] > 0) | (table["ci_high"] < 0))
        )
    path = OUTPUT_DIR / "tables" / "validation_associations.csv"
    write_csv(path, table)

    supported_features = []
    for selected in selection["selected_features"]:
        feature = selected["feature"]
        group = table[table["feature"] == feature]
        label_ok = len(group[group["endpoint"] == "label_reasoning"]) == 1 and bool(group[group["endpoint"] == "label_reasoning"].iloc[0]["validation_supported"])
        internal_ok = group[group["endpoint"].isin(INTERNAL_ENDPOINTS) & group["validation_supported"]] if len(group) else group
        if label_ok and len(internal_ok):
            supported_features.append({
                "feature": feature,
                "supported_internal_endpoints": [str(value) for value in internal_ok["endpoint"].tolist()],
            })
    result = {
        "status": "COMPLETE_EXPLORATORY_ASSOCIATION_ONLY",
        "human_annotation": False,
        "discovery_selected_features": [row["feature"] for row in selection["selected_features"]],
        "validation_supported_features": supported_features,
        "validation_supported_feature_count": len(supported_features),
        "causal_feature_claim_allowed": False,
        "component_mediation_claim_allowed": False,
        "independent_controlled_replication_required": bool(supported_features),
        "discovery_selection_sha256": sha256_file(selection_path),
        "validation_table_sha256": sha256_file(path),
    }
    atomic_json(OUTPUT_DIR / "natural_feature_discovery_summary.json", result)


def phase_posthoc_label_diagnostic(manifest: dict[str, Any]) -> None:
    """Diagnose whether the selected feature association is only inherited from R/M label.

    This phase is explicitly post-hoc and cannot alter the frozen discovery selection or
    validation-support result.
    """
    import pandas as pd
    frame = load_analysis_frame(manifest)
    feature = "transformation_required"
    rows = []
    for split_name in ("discovery", "validation"):
        split = frame[frame["analysis_split"] == split_name].copy()
        prevalence = split.dropna(subset=[feature]).groupby(feature)["label_reasoning"].agg(["count", "mean"]).reset_index()
        for row in prevalence.itertuples(index=False):
            rows.append({
                "analysis_split": split_name, "diagnostic": "label_prevalence", "label_stratum": "all",
                "feature_level": int(row[0]), "endpoint": "label_reasoning", "n": int(row.count),
                "mean": float(row.mean), "beta": math.nan, "ci_low": math.nan, "ci_high": math.nan, "p": math.nan,
            })
        for endpoint in INTERNAL_ENDPOINTS:
            adjusted = ols_hc3(split, feature, endpoint, adjust_for_label=True)
            rows.append({
                "analysis_split": split_name, "diagnostic": "label_adjusted", "label_stratum": "all",
                "feature_level": math.nan, "endpoint": endpoint, "mean": math.nan, **adjusted,
            })
            for label_value, label_name in ((0, "M"), (1, "R")):
                within = ols_hc3(split[split["label_reasoning"] == label_value], feature, endpoint, adjust_for_label=False)
                rows.append({
                    "analysis_split": split_name, "diagnostic": "within_label", "label_stratum": label_name,
                    "feature_level": math.nan, "endpoint": endpoint, "mean": math.nan, **within,
                })
    table = pd.DataFrame(rows)
    inferential = table["p"].notna()
    table["q_posthoc_bh"] = math.nan
    table.loc[inferential, "q_posthoc_bh"] = bh_adjust(table.loc[inferential, "p"].tolist())
    table["posthoc_not_primary"] = True
    path = OUTPUT_DIR / "tables" / "posthoc_label_adjusted_diagnostics.csv"
    write_csv(path, table)
    atomic_json(OUTPUT_DIR / "manifests" / "posthoc_label_adjusted_diagnostic.json", {
        "status": "COMPLETE_POSTHOC_DIAGNOSTIC_ONLY",
        "reason": "Check whether primary feature/internal associations are inherited solely from the frozen R/M label.",
        "changes_primary_selection": False,
        "changes_validation_support": False,
        "table_sha256": sha256_file(path),
    })


def phase_audit(manifest: dict[str, Any]) -> None:
    import pandas as pd
    artifacts = {
        "blind_items": ASSET_DIR / "blind_annotation_items.jsonl",
        "annotator_a": ASSET_DIR / annotation_output_name("annotator_a"),
        "annotator_b": ASSET_DIR / annotation_output_name("annotator_b"),
        "consensus": ASSET_DIR / "consensus_annotations_v1_2.jsonl",
        "reliability": ASSET_DIR / "annotation_reliability_summary_v1_2.json",
        "layer_scalars": OUTPUT_DIR / "tables" / "layer_liref_scalars.csv.gz",
        "discovery": OUTPUT_DIR / "tables" / "discovery_associations.csv",
        "selection": OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json",
        "validation": OUTPUT_DIR / "tables" / "validation_associations.csv",
        "summary": OUTPUT_DIR / "natural_feature_discovery_summary.json",
        "posthoc_label_diagnostic": OUTPUT_DIR / "tables" / "posthoc_label_adjusted_diagnostics.csv",
        "result_document": STAGE_DIR / "STAGE_E_NATURAL_FEATURE_DISCOVERY_V1_RESULT_KO.md",
    }
    missing = [name for name, path in artifacts.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing final artifacts: {missing}")
    blind = read_jsonl(artifacts["blind_items"])
    consensus = read_jsonl(artifacts["consensus"])
    layer = pd.read_csv(artifacts["layer_scalars"])
    if len(blind) != 3000 or len(consensus) != 3000 or len(layer) != 3000:
        raise RuntimeError("Final row-count audit failed")
    if any(FORBIDDEN_ANNOTATION_FIELDS.intersection(row) for row in blind):
        raise RuntimeError("Blind annotation input contains forbidden fields")
    summary = read_json(artifacts["summary"])
    audit = {
        "status": "PASS",
        "design_manifest_sha256": sha256_file(MANIFEST_PATH),
        "implementation_sha256": sha256_file(Path(__file__)),
        "annotation_instrument_amendment_sha256": sha256_file(ANNOTATION_AMENDMENT_PATH),
        "consensus_amendment_sha256": sha256_file(CONSENSUS_AMENDMENT_PATH),
        "row_counts": {"blind": len(blind), "consensus": len(consensus), "layer_scalars": len(layer)},
        "blind_outcome_leakage": False,
        "read_only_scalar_extraction": True,
        "intervention_performed": False,
        "validation_supported_feature_count": summary["validation_supported_feature_count"],
        "artifact_sha256": {name: sha256_file(path) for name, path in artifacts.items()},
    }
    atomic_json(OUTPUT_DIR / "final_audit.json", audit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=[
        "prepare_blind", "preflight_a", "preflight_b", "preflight_j", "annotate_a", "annotate_b", "prepare_adjudication",
        "adjudicate", "consensus", "extract_layer_scalars",
        "analyze_discovery", "analyze_validation", "posthoc_label_diagnostic", "audit",
    ])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_and_verify_manifest()
    phases = {
        "prepare_blind": lambda: phase_prepare_blind(manifest),
        "preflight_a": lambda: phase_annotate(manifest, "annotator_a", preflight=True),
        "preflight_b": lambda: phase_annotate(manifest, "annotator_b", preflight=True),
        "preflight_j": lambda: phase_annotate(manifest, "adjudicator", preflight=True),
        "annotate_a": lambda: phase_annotate(manifest, "annotator_a"),
        "annotate_b": lambda: phase_annotate(manifest, "annotator_b"),
        "prepare_adjudication": lambda: phase_prepare_adjudication(manifest),
        "adjudicate": lambda: phase_annotate(manifest, "adjudicator"),
        "consensus": lambda: phase_consensus(manifest),
        "extract_layer_scalars": lambda: phase_extract_layer_scalars(manifest),
        "analyze_discovery": lambda: phase_analyze_discovery(manifest),
        "analyze_validation": lambda: phase_analyze_validation(manifest),
        "posthoc_label_diagnostic": lambda: phase_posthoc_label_diagnostic(manifest),
        "audit": lambda: phase_audit(manifest),
    }
    phases[args.phase]()
    print(json.dumps({"status": "PASS", "phase": args.phase}, sort_keys=True))


if __name__ == "__main__":
    main()
