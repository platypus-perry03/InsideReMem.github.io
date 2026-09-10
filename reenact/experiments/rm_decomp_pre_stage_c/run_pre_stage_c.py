#!/usr/bin/env python3
"""Prepare linguistic audit and Memory-side natural/validation analyses before Stage C PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_B_CODE = SCRIPT_DIR.parent / "rm_decomp_b"
STAGE_A_CODE = SCRIPT_DIR.parent / "rm_decomp"
for source_dir in (BASE_B_CODE, STAGE_A_CODE):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
FEATURE_NAMES = [
    "has_capitalized_span",
    "has_year",
    "has_date_word",
    "has_person_cue",
    "has_location_cue",
    "has_temporal_cue",
    "has_factual_association_cue",
    "has_entity_attribute_proxy",
]
AUDIT_RESPONSE_COLUMNS = [
    "reviewer_id", "grammar_a_pass", "grammar_b_pass", "naturalness_a_1to5", "naturalness_b_1to5",
    "only_intended_feature_changed", "relation_polarity_valid", "relevance_label_valid",
    "expected_answers_valid", "changed_spans_valid", "unintended_cue_absent", "overall_pass",
    "issue_code", "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temporary, path)


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=np.float64)
    if len(values) == 0:
        return values
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(values)
    output[order] = np.clip(ranked, 0.0, 1.0)
    return output


def ols_hc3(y: np.ndarray, x: np.ndarray, feature_column: int) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    inverse = np.linalg.pinv(x.T @ x, rcond=1e-12)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, inverse, x)
    adjusted = residual / np.clip(1.0 - leverage, 1e-8, None)
    meat = x.T @ (x * adjusted[:, None] ** 2)
    covariance = inverse @ meat @ inverse
    variance = float(max(covariance[feature_column, feature_column], 0.0))
    standard_error = math.sqrt(variance)
    estimate = float(beta[feature_column])
    statistic = estimate / standard_error if standard_error > 0 else math.nan
    rank = int(np.linalg.matrix_rank(x))
    degrees = max(int(len(y) - rank), 1)
    p_value = float(2.0 * stats.t.sf(abs(statistic), degrees)) if math.isfinite(statistic) else math.nan
    return {
        "beta": estimate,
        "standard_error_hc3": standard_error,
        "t_statistic": statistic,
        "p_value": p_value,
        "ci_low": estimate - float(stats.t.ppf(0.975, degrees)) * standard_error,
        "ci_high": estimate + float(stats.t.ppf(0.975, degrees)) * standard_error,
        "n": int(len(y)),
        "design_rank": rank,
        "residual_df": degrees,
    }


def contains_any_word(text: str, words: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) for word in words)


def contains_any_phrase(text: str, phrases: list[str]) -> bool:
    lowered = text.casefold()
    return any(phrase.casefold() in lowered for phrase in phrases)


def extract_memory_features(question: str, schema: dict[str, Any]) -> dict[str, int]:
    exclusions = set(schema["capitalized_exclusions"])
    capitalized = [
        match.group(0)
        for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9'’-]*)(?:\s+[A-Z][A-Za-z0-9'’-]*)*\b", question)
        if match.group(0) not in exclusions
    ]
    has_capitalized = int(bool(capitalized))
    has_person = int(contains_any_word(question, schema["person_cues"]))
    has_location = int(contains_any_word(question, schema["location_cues"]))
    has_temporal = int(contains_any_word(question, schema["temporal_cues"]))
    has_association = int(contains_any_phrase(question, schema["factual_association_phrases"]))
    attribute = int(has_capitalized and (
        contains_any_word(question, schema["attribute_cues"])
        or contains_any_phrase(question, schema["attribute_cues"])
    ))
    return {
        "capitalized_span_count": len(capitalized),
        "has_capitalized_span": has_capitalized,
        "has_year": int(bool(re.search(r"(?<!\d)(?:1\d{3}|20\d{2})(?!\d)", question))),
        "has_date_word": int(contains_any_word(question, schema["date_words"])),
        "has_person_cue": has_person,
        "has_location_cue": has_location,
        "has_temporal_cue": has_temporal,
        "has_factual_association_cue": has_association,
        "has_entity_attribute_proxy": attribute,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--phase", choices=["analyze", "extract_validation"], default="analyze")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    config["config_path"] = str(path.resolve())
    config["config_hash"] = canonical_hash({key: value for key, value in config.items() if key not in {"config_path", "config_hash"}})
    return config


def paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["output_root"])
    return {
        "root": root,
        "audit": root / "linguistic_audit",
        "audit_forms": root / "linguistic_audit" / "reviewer_forms",
        "audit_completed": root / "linguistic_audit" / "completed",
        "memory": root / "memory_side",
        "tables": root / "memory_side" / "tables",
        "manifests": root / "manifests",
        "stage_c": root / "stage_c_pre_pdf",
    }


def ensure_dirs(p: dict[str, Path]) -> None:
    for value in p.values():
        value.mkdir(parents=True, exist_ok=True)


def source_hashes(config: dict[str, Any]) -> dict[str, str]:
    files = {
        "dataset": Path(config["dataset_path"]),
        "split": Path(config["split_path"]),
        "stage_a_candidates": Path(config["stage_a_root"]) / "manifests" / "candidate_manifest.json",
        "natural_responses": Path(config["stage_b_root"]) / "tables" / "natural_responses.csv.gz",
        "stage_b_extension_pairs": Path(config["stage_b_extension_root"]) / "manifests" / "confirmatory_pairs.csv",
        "causal_pairs": Path(config["causal_root"]) / "manifests" / "mediation_pairs.csv",
        "causal_summary": Path(config["causal_root"]) / "causal_summary.json",
    }
    return {key: sha256_file(path) for key, path in files.items()}


def audit_source_rows(config: dict[str, Any]) -> pd.DataFrame:
    causal = pd.read_csv(Path(config["causal_root"]) / "manifests" / "mediation_pairs.csv")
    causal_rows = pd.DataFrame({
        "source_stage": "causal_c01",
        "source_pair_id": causal["pair_id"].astype(str),
        "stratum": causal["relation_family"].astype(str) + "::" + causal["relevance"].astype(str),
        "claimed_relevance": causal["relevance"].astype(str),
        "text_original": causal["original_text"].astype(str),
        "text_modified": causal["modified_text"].astype(str),
        "answer_original": causal["expected_answer_original"].astype(str),
        "answer_modified": causal["expected_answer_modified"].astype(str),
        "span_original": causal["changed_span_original"].astype(str),
        "span_modified": causal["changed_span_modified"].astype(str),
        "sampling_role": "full_census",
    })

    extension = pd.read_csv(Path(config["stage_b_extension_root"]) / "manifests" / "confirmatory_pairs.csv")
    extension = extension[extension["feature_family"] == "relation_polarity"].copy()
    sampled = []
    count = int(config["stage_b_audit_per_stratum"])
    for number, ((_family, _condition), group) in enumerate(extension.groupby(["lexical_family", "condition"], sort=True)):
        if len(group) < count:
            raise RuntimeError("Insufficient Stage B audit stratum size")
        sampled.append(group.sample(n=count, random_state=int(config["seed"]) + number, replace=False))
    extension = pd.concat(sampled, ignore_index=True)
    extension_rows = pd.DataFrame({
        "source_stage": "stage_b_b04",
        "source_pair_id": extension["pair_id"].astype(str),
        "stratum": extension["lexical_family"].astype(str) + "::" + extension["condition"].astype(str),
        "claimed_relevance": extension["condition"].astype(str),
        "text_original": extension["original_text"].astype(str),
        "text_modified": extension["modified_text"].astype(str),
        "answer_original": extension["expected_answer_original"].astype(str),
        "answer_modified": extension["expected_answer_modified"].astype(str),
        "span_original": extension["changed_spans_original"].astype(str),
        "span_modified": extension["changed_spans_modified"].astype(str),
        "sampling_role": "stratified_random_sample",
    })
    return pd.concat([causal_rows, extension_rows], ignore_index=True)


def generate_audit(config: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    source = audit_source_rows(config)
    rng = np.random.default_rng(int(config["seed"]))
    rows = []
    keys = []
    for index, row in enumerate(source.to_dict("records")):
        swap = bool(rng.integers(0, 2))
        a_suffix, b_suffix = ("modified", "original") if swap else ("original", "modified")
        audit_id = f"AUD{index:04d}"
        output = {
            "audit_id": audit_id,
            "source_stage": row["source_stage"],
            "claimed_relevance": row["claimed_relevance"],
            "text_a": row[f"text_{a_suffix}"],
            "text_b": row[f"text_{b_suffix}"],
            "claimed_changed_span_a": row[f"span_{a_suffix}"],
            "claimed_changed_span_b": row[f"span_{b_suffix}"],
            "expected_answer_a": row[f"answer_{a_suffix}"],
            "expected_answer_b": row[f"answer_{b_suffix}"],
        }
        output.update({column: "" for column in AUDIT_RESPONSE_COLUMNS})
        rows.append(output)
        keys.append({
            "audit_id": audit_id,
            "source_stage": row["source_stage"],
            "source_pair_id": row["source_pair_id"],
            "stratum": row["stratum"],
            "sampling_role": row["sampling_role"],
            "a_is": a_suffix,
            "b_is": b_suffix,
        })
    form = pd.DataFrame(rows)
    key = pd.DataFrame(keys)
    for reviewer_number in (1, 2):
        reviewer_form = form.sample(frac=1.0, random_state=int(config["seed"]) + reviewer_number).reset_index(drop=True)
        reviewer_form["reviewer_slot"] = f"R{reviewer_number}"
        atomic_csv(p["audit_forms"] / f"reviewer_{reviewer_number}_blind.csv", reviewer_form)
    atomic_csv(p["audit"] / "audit_key_DO_NOT_SHARE_BEFORE_COMPLETION.csv", key)
    atomic_csv(p["audit"] / "audit_items_master_blank.csv", form)
    automated = pd.DataFrame({
        "audit_id": form["audit_id"],
        "texts_differ": form["text_a"] != form["text_b"],
        "span_a_found": [str(span).strip("[]\"'") in text for span, text in zip(form["claimed_changed_span_a"], form["text_a"])],
        "span_b_found": [str(span).strip("[]\"'") in text for span, text in zip(form["claimed_changed_span_b"], form["text_b"])],
        "answers_nonempty": form["expected_answer_a"].astype(str).str.len().gt(0) & form["expected_answer_b"].astype(str).str.len().gt(0),
    })
    automated["automated_structure_pass"] = automated[["texts_differ", "span_a_found", "span_b_found", "answers_nonempty"]].all(axis=1)
    atomic_csv(p["audit"] / "automated_structure_checks.csv", automated)
    acceptance = config["audit_acceptance"]
    instructions = f"""# Blind Linguistic Audit 지침

## 목적

모델 결과·후보 ID를 보지 않고 synthetic relation pair의 문법, 단일-feature 변화, 관계 polarity, task relevance, 정답과 changed span을 독립적으로 평가한다.

## 평가 방법

- Reviewer 1과 Reviewer 2는 서로 상의하지 않고 각자의 CSV를 작성한다.
- boolean 항목은 `true` 또는 `false`, 자연스러움은 1~5 정수로 기록한다.
- `claimed_relevance=relevant`이면 바뀐 관계가 정답 계산에 반드시 필요한지 확인한다.
- `claimed_relevance=irrelevant`이면 관계가 바뀌어도 정답이 유지되는지 확인한다.
- `overall_pass`는 핵심 항목이 모두 유효할 때만 true로 기록한다.
- 모델 반응, component ID, 기존 통계 결과는 검수 종료 전 공개하지 않는다.

## 사전 동결 acceptance 기준

- expected-answer 정확도: `{acceptance['expected_answer_accuracy_min']:.0%}` 이상
- core-item pass rate: `{acceptance['core_item_pass_rate_min']:.0%}` 이상
- 평균 자연스러움: `{acceptance['naturalness_mean_min']:.1f}/5 이상`
- 두 평가자 Cohen's kappa: `{acceptance['inter_rater_kappa_min']:.2f}` 이상
- 불일치는 별도 adjudication CSV에서 해결한다.

## 완료 파일

완료된 파일을 `completed/reviewer_1.csv`, `completed/reviewer_2.csv`로 저장한다. 기존 blank form을 덮어쓰지 않는다.

## 현재 상태

이 패키지는 audit 준비만 완료했다. 실제 독립 human audit은 아직 수행되지 않았으며 완료로 주장하면 안 된다.
"""
    (p["audit"] / "AUDIT_INSTRUCTIONS_KO.md").write_text(instructions, encoding="utf-8")
    (p["audit_completed"] / "README.md").write_text("두 명의 독립 검수 완료 CSV를 이 폴더에 넣습니다. 현재 human audit은 PENDING입니다.\n", encoding="utf-8")
    status = {
        "status": "PENDING_HUMAN_REVIEW",
        "causal_full_census_count": int((source["source_stage"] == "causal_c01").sum()),
        "stage_b_stratified_sample_count": int((source["source_stage"] == "stage_b_b04").sum()),
        "total_audit_items": len(source),
        "automated_structure_pass_count": int(automated["automated_structure_pass"].sum()),
        "reviewer_count_required": 2,
        "results_blinded": True,
        "acceptance": acceptance,
    }
    write_json(p["audit"] / "audit_status.json", status)
    return status


def build_feature_table(config: dict[str, Any], schema: dict[str, Any]) -> pd.DataFrame:
    records = read_json(Path(config["dataset_path"]))
    rows = []
    for row_index, record in enumerate(records):
        rows.append({
            "row_index": row_index,
            "question_id": str(record["question_id"]),
            "question": record["question"],
            "category": record["category"],
            "memory_reason_score": float(record["memory_reason_score"]),
            **extract_memory_features(record["question"], schema),
        })
    return pd.DataFrame(rows)


def design_matrix(frame: pd.DataFrame, feature_name: str) -> tuple[np.ndarray, int]:
    feature = frame[feature_name].to_numpy(dtype=np.float64)
    controls = pd.DataFrame({
        "log_token_length": np.log1p(frame["token_length"].to_numpy(dtype=np.float64)),
        "numeric_span_count": frame["numeric_span_count"].to_numpy(dtype=np.float64),
        "relation_span_count": frame["relation_span_count"].to_numpy(dtype=np.float64),
        "operator_span_count": frame["operator_span_count"].to_numpy(dtype=np.float64),
    })
    for column in controls:
        values = controls[column].to_numpy(dtype=np.float64)
        scale = values.std(ddof=0)
        controls[column] = (values - values.mean()) / scale if scale > 0 else 0.0
    categories = pd.get_dummies(frame["category"].astype(str), prefix="category", drop_first=True, dtype=float)
    x = np.column_stack([np.ones(len(frame)), feature, controls.to_numpy(dtype=float), categories.to_numpy(dtype=float)])
    varying = np.r_[True, True, np.std(x[:, 2:], axis=0) > 0]
    x = x[:, varying]
    return x, 1


def response_column(component_type: str) -> str:
    return "total_contribution" if component_type == "head" else "activation"


def regress_component(group: pd.DataFrame, feature_name: str, component_type: str) -> dict[str, float]:
    column = response_column(component_type)
    y = group[column].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(y)) or y.std(ddof=0) <= 0:
        return {key: math.nan for key in ("beta", "standard_error_hc3", "t_statistic", "p_value", "ci_low", "ci_high")} | {"n": len(y), "design_rank": 0, "residual_df": 0}
    y = (y - y.mean()) / y.std(ddof=0)
    x, feature_column = design_matrix(group, feature_name)
    return ols_hc3(y, x, feature_column)


def extract_validation_responses(config: dict[str, Any], p: dict[str, Path]) -> None:
    """Extract the missing per-question validation component responses without inspecting outcomes."""
    import torch

    from core import load_dataset_and_split, release_model
    from stage_b_core import (
        FeatureExtractor,
        StageBCapture,
        component_lookup,
        load_directions,
        load_model_and_tokenizer,
        model_parameter_checksum,
        projections_for_components,
        unique_components,
    )

    output_path = p["tables"] / "natural_validation_responses.csv.gz"
    frozen_b_config = read_json(Path(config["stage_b_root"]) / "manifests" / "frozen_config.json")
    candidates_payload = read_json(Path(config["stage_b_extension_root"]) / "manifests" / "frozen_stage_b_candidates.json")
    controls_payload = read_json(Path(config["stage_b_extension_root"]) / "manifests" / "frozen_control_components.json")
    lookup = component_lookup(candidates_payload, controls_payload)
    components = unique_components(lookup)
    extractor = FeatureExtractor(read_json(BASE_B_CODE / "feature_schema.json"))
    data = load_dataset_and_split(frozen_b_config)
    gpu_id = int(frozen_b_config["gpu_id"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for validation component extraction")
    device = torch.device(f"cuda:{gpu_id}")
    directions = load_directions(Path(config["stage_a_root"]) / "checkpoints" / "discovery_liref_directions.pt")
    indices = [int(value) for value in data["indices"]["validation"]]
    before, _ = model_parameter_checksum(Path(frozen_b_config["model_path"]))
    model = None
    capture = None
    rows: list[dict[str, Any]] = []
    try:
        model, tokenizer = load_model_and_tokenizer(frozen_b_config, device)
        token_length_by_index = {
            value: len(tokenizer(data["prompts"][value], add_special_tokens=True)["input_ids"])
            for value in indices
        }
        indices.sort(key=token_length_by_index.__getitem__)
        projections = projections_for_components(model, directions, components)
        capture = StageBCapture(model, components, capture_sources=False)
        capture.install()
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        max_batch_size = int(config["validation_max_batch_size"])
        max_padded_tokens = int(config["validation_max_padded_tokens"])
        batches: list[list[int]] = []
        current: list[int] = []
        for row_index in indices:
            proposed_size = len(current) + 1
            proposed_tokens = proposed_size * token_length_by_index[row_index]
            if current and (proposed_size > max_batch_size or proposed_tokens > max_padded_tokens):
                batches.append(current)
                current = []
            current.append(row_index)
        if current:
            batches.append(current)
        processed = 0
        for batch_number, batch_indices in enumerate(batches, start=1):
            prompts = [data["prompts"][value] for value in batch_indices]
            encoded = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=False, return_token_type_ids=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            capture.reset()
            model(**encoded, use_cache=False, return_dict=True)
            capture.validate()
            token_lengths = encoded["attention_mask"].sum(dim=-1).tolist()
            feature_rows = [extractor.summarize(data["records"][value]["question"]) for value in batch_indices]
            component_scores: dict[str, torch.Tensor] = {}
            component_activations: dict[str, torch.Tensor | None] = {}
            for component_id, component in components.items():
                if component["component_type"] == "neuron":
                    activation = capture.z[component["module_index"]][:, component["component_index"]].float()
                    component_activations[component_id] = activation
                    component_scores[component_id] = activation * projections[component_id].float()
                else:
                    pre = capture.pre_o[component["module_index"]].reshape(len(batch_indices), model.config.num_attention_heads, head_dim)
                    component_activations[component_id] = None
                    component_scores[component_id] = (pre[:, component["component_index"]].float() * projections[component_id].float()).sum(dim=-1)
            for association in lookup.values():
                component_id = association["component_id"]
                projection = projections[component_id]
                activation_values = component_activations[component_id]
                for offset, row_index in enumerate(batch_indices):
                    feature = feature_rows[offset]
                    rows.append({
                        "analysis_split": "validation", "row_index": row_index,
                        "question_id": data["question_ids"][row_index],
                        "label": "reasoning" if int(data["labels"][row_index]) == 1 else "memory",
                        "category": data["records"][row_index]["category"], "token_length": int(token_lengths[offset]),
                        **{key: feature[key] for key in ("numeric_span_count", "has_numeric", "relation_span_count", "has_relation", "operator_span_count", "has_operator")},
                        "candidate_id": association["candidate_id"], "component_id": component_id,
                        "component_type": association["component_type"], "component_role": association["role"],
                        "control_kind": association["control_kind"], "module_index": association["module_index"],
                        "component_index": association["component_index"],
                        "activation": float(activation_values[offset]) if activation_values is not None else math.nan,
                        "projection": float(projection.norm()) if projection.ndim else float(projection),
                        "total_contribution": float(component_scores[component_id][offset]),
                    })
            processed += len(batch_indices)
            if batch_number == 1 or batch_number == len(batches) or batch_number % 10 == 0:
                print(
                    f"validation extraction: {processed}/{len(indices)} "
                    f"(batch={len(batch_indices)}, max_tokens={max(token_length_by_index[value] for value in batch_indices)})",
                    flush=True,
                )
            del encoded
    finally:
        if capture is not None:
            capture.remove()
        release_model(model)
    after, _ = model_parameter_checksum(Path(frozen_b_config["model_path"]))
    if before != after:
        raise RuntimeError("Model parameter checksum changed during validation extraction")
    frame = pd.DataFrame(rows)
    expected = len(indices) * len(lookup)
    if len(frame) != expected or frame["row_index"].nunique() != len(indices):
        raise RuntimeError(f"Unexpected validation extraction size: {len(frame)} != {expected}")
    atomic_csv(output_path, frame)
    write_json(p["memory"] / "validation_extraction_status.json", {
        "status": "PASS", "timestamp": utc_now(), "validation_samples": len(indices),
        "association_rows": len(frame), "component_associations_per_sample": len(lookup),
        "validation_max_batch_size": int(config["validation_max_batch_size"]),
        "validation_max_padded_tokens": int(config["validation_max_padded_tokens"]),
        "output_sha256": sha256_file(output_path), "model_checksum_before": before, "model_checksum_after": after,
        "hypothesis_manifest_frozen_before_extraction": sha256_file(p["manifests"] / "memory_hypothesis_manifest.json"),
    })


def feature_prevalence(frame: pd.DataFrame, split: str, feature_name: str) -> dict[str, Any]:
    values = frame[feature_name].astype(int)
    return {
        "analysis_split": split,
        "feature_name": feature_name,
        "n": len(frame),
        "n_present": int(values.sum()),
        "n_absent": int((1 - values).sum()),
        "prevalence": float(values.mean()),
    }


def memory_analysis(config: dict[str, Any], p: dict[str, Path], schema: dict[str, Any]) -> dict[str, Any]:
    natural = pd.read_csv(Path(config["stage_b_root"]) / "tables" / "natural_responses.csv.gz")
    validation_response_path = p["tables"] / "natural_validation_responses.csv.gz"
    if validation_response_path.exists():
        validation_natural = pd.read_csv(validation_response_path)
        natural = pd.concat([natural, validation_natural], ignore_index=True)
    features = build_feature_table(config, schema)
    atomic_csv(p["tables"] / "memory_question_features.csv.gz", features)
    candidate = natural[(natural["label"] == config["memory_label"]) & (natural["component_role"] == "candidate")].copy()
    candidate = candidate.merge(features[["row_index", *FEATURE_NAMES]], on="row_index", how="left", validate="many_to_one")
    prevalence_rows = []
    for split in ("discovery", "validation"):
        unique = candidate[candidate["analysis_split"] == split].drop_duplicates("row_index")
        for feature_name in FEATURE_NAMES:
            prevalence_rows.append(feature_prevalence(unique, split, feature_name))
    prevalence = pd.DataFrame(prevalence_rows)
    atomic_csv(p["tables"] / "memory_feature_prevalence.csv", prevalence)

    discovery_rows = []
    for (candidate_id, component_type), group in candidate[candidate["analysis_split"] == "discovery"].groupby(["candidate_id", "component_type"], sort=True):
        for feature_name in FEATURE_NAMES:
            counts = feature_prevalence(group, "discovery", feature_name)
            eligible = counts["n_present"] >= int(config["minimum_discovery_present"]) and counts["n_absent"] >= int(config["minimum_discovery_absent"])
            result = regress_component(group, feature_name, component_type) if eligible else {key: math.nan for key in ("beta", "standard_error_hc3", "t_statistic", "p_value", "ci_low", "ci_high")} | {"n": len(group), "design_rank": 0, "residual_df": 0}
            discovery_rows.append({
                "candidate_id": candidate_id, "component_type": component_type,
                "response": response_column(component_type), "feature_name": feature_name,
                "eligible": eligible, **counts, **result,
            })
    discovery = pd.DataFrame(discovery_rows)
    valid = discovery["p_value"].notna()
    discovery["bh_q_discovery"] = np.nan
    discovery.loc[valid, "bh_q_discovery"] = bh_adjust(discovery.loc[valid, "p_value"])
    discovery["hypothesis_selected"] = (
        discovery["eligible"]
        & (discovery["bh_q_discovery"] < float(config["fdr_alpha"]))
        & (discovery["beta"].abs() >= float(config["discovery_effect_threshold"]))
    )
    atomic_csv(p["tables"] / "memory_discovery_associations.csv", discovery)

    selected = discovery[discovery["hypothesis_selected"]].copy()
    hypotheses = []
    for number, row in enumerate(selected.to_dict("records"), 1):
        hypotheses.append({
            "hypothesis_id": f"MH{number:03d}",
            "candidate_id": row["candidate_id"],
            "component_type": row["component_type"],
            "response": row["response"],
            "feature_name": row["feature_name"],
            "expected_sign": "positive" if row["beta"] > 0 else "negative",
            "discovery_beta": row["beta"],
            "discovery_q": row["bh_q_discovery"],
            "selection_rule": f"Discovery BH-q<{config['fdr_alpha']} and |adjusted standardized beta|>={config['discovery_effect_threshold']}",
            "frozen_before_validation": True,
        })
    manifest = {
        "manifest_id": "memory_feature_hypotheses_v1",
        "timestamp": utc_now(),
        "candidate_set": "all 20 frozen Stage A detailed candidates",
        "candidate_count": int(candidate["candidate_id"].nunique()),
        "feature_schema_hash": sha256_file(SCRIPT_DIR / "memory_feature_schema.json"),
        "discovery_table_hash": sha256_file(p["tables"] / "memory_discovery_associations.csv"),
        "validation_inspected_before_freeze": False,
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
    }
    hypothesis_path = p["manifests"] / "memory_hypothesis_manifest.json"
    if hypothesis_path.exists():
        frozen_manifest = read_json(hypothesis_path)
        if frozen_manifest["discovery_table_hash"] != sha256_file(p["tables"] / "memory_discovery_associations.csv"):
            raise RuntimeError("Discovery table changed after memory hypothesis freeze")
        hypotheses = frozen_manifest["hypotheses"]
        manifest = frozen_manifest
    else:
        write_json(hypothesis_path, manifest)

    validation_rows = []
    validation_candidate = candidate[candidate["analysis_split"] == "validation"]
    for hypothesis in hypotheses:
        group = validation_candidate[validation_candidate["candidate_id"] == hypothesis["candidate_id"]]
        counts = feature_prevalence(group, "validation", hypothesis["feature_name"])
        eligible = counts["n_present"] >= int(config["minimum_validation_present"]) and counts["n_absent"] >= int(config["minimum_validation_absent"])
        result = regress_component(group, hypothesis["feature_name"], hypothesis["component_type"]) if eligible else {key: math.nan for key in ("beta", "standard_error_hc3", "t_statistic", "p_value", "ci_low", "ci_high")} | {"n": len(group), "design_rank": 0, "residual_df": 0}
        validation_rows.append({**hypothesis, "eligible_validation": eligible, **counts, **result})
    validation = pd.DataFrame(validation_rows)
    if len(validation):
        finite = validation["p_value"].notna()
        validation["bh_q_validation"] = np.nan
        validation.loc[finite, "bh_q_validation"] = bh_adjust(validation.loc[finite, "p_value"])
        validation["same_sign"] = np.sign(validation["beta"]) == np.sign(validation["discovery_beta"])
        validation["replicated_basic"] = (
            validation["eligible_validation"]
            & validation["same_sign"]
            & (validation["bh_q_validation"] < float(config["fdr_alpha"]))
            & (validation["beta"].abs() >= float(config["validation_effect_threshold"]))
        )
    else:
        validation = pd.DataFrame(columns=[
            "hypothesis_id", "candidate_id", "component_type", "response", "feature_name", "expected_sign",
            "discovery_beta", "discovery_q", "selection_rule", "frozen_before_validation", "eligible_validation",
            "analysis_split", "n", "n_present", "n_absent", "prevalence", "beta", "standard_error_hc3",
            "t_statistic", "p_value", "ci_low", "ci_high", "design_rank", "residual_df", "bh_q_validation",
            "same_sign", "replicated_basic",
        ])
    atomic_csv(p["tables"] / "memory_validation_replication.csv", validation)

    controls = natural[(natural["label"] == config["memory_label"]) & (natural["component_role"] == "control") & (natural["analysis_split"] == "validation")].copy()
    controls = controls.merge(features[["row_index", *FEATURE_NAMES]], on="row_index", how="left", validate="many_to_one")
    specificity_rows = []
    for hypothesis in hypotheses:
        subset = controls[controls["candidate_id"] == hypothesis["candidate_id"]]
        effects = []
        for (component_id, control_kind), group in subset.groupby(["component_id", "control_kind"], sort=True):
            result = regress_component(group, hypothesis["feature_name"], hypothesis["component_type"])
            effects.append({"component_id": component_id, "control_kind": control_kind, "beta": result["beta"], "p_value": result["p_value"]})
        candidate_row = validation[validation["hypothesis_id"] == hypothesis["hypothesis_id"]]
        candidate_beta = float(candidate_row["beta"].iloc[0]) if len(candidate_row) else math.nan
        matched = [abs(row["beta"]) for row in effects if row["control_kind"] == "matched" and math.isfinite(row["beta"])]
        random = [abs(row["beta"]) for row in effects if row["control_kind"] == "random" and math.isfinite(row["beta"])]
        specificity_rows.append({
            "hypothesis_id": hypothesis["hypothesis_id"], "candidate_id": hypothesis["candidate_id"], "feature_name": hypothesis["feature_name"],
            "candidate_beta_validation": candidate_beta,
            "matched_abs_beta": matched[0] if matched else math.nan,
            "random_mean_abs_beta": float(np.mean(random)) if random else math.nan,
            "candidate_gt_matched": bool(matched and abs(candidate_beta) > matched[0]),
            "candidate_gt_random_mean": bool(random and abs(candidate_beta) > np.mean(random)),
            "control_effects": json.dumps(effects, ensure_ascii=False, sort_keys=True),
        })
    specificity = pd.DataFrame(specificity_rows, columns=[
        "hypothesis_id", "candidate_id", "feature_name", "candidate_beta_validation", "matched_abs_beta",
        "random_mean_abs_beta", "candidate_gt_matched", "candidate_gt_random_mean", "control_effects",
    ])
    atomic_csv(p["tables"] / "memory_control_specificity.csv", specificity)
    if len(validation):
        validation = validation.merge(specificity.drop(columns=["candidate_id", "feature_name", "candidate_beta_validation", "control_effects"]), on="hypothesis_id", how="left")
        validation["replicated_with_specificity"] = validation["replicated_basic"] & validation["candidate_gt_matched"] & validation["candidate_gt_random_mean"]
    else:
        validation["replicated_with_specificity"] = pd.Series(dtype=bool)
    atomic_csv(p["tables"] / "memory_validation_final.csv", validation)
    return {
        "candidate_count": int(candidate["candidate_id"].nunique()),
        "feature_count": len(FEATURE_NAMES),
        "eligible_discovery_tests": int(valid.sum()),
        "discovery_hypothesis_count": len(hypotheses),
        "basic_validation_replication_count": int(validation["replicated_basic"].sum()) if len(validation) else 0,
        "specific_validation_replication_count": int(validation["replicated_with_specificity"].sum()) if len(validation) else 0,
        "replicated_hypotheses": validation[validation["replicated_with_specificity"]][["hypothesis_id", "candidate_id", "feature_name", "beta", "bh_q_validation"]].to_dict("records") if len(validation) else [],
        "interpretation_boundary": schema["interpretation_boundary"],
    }


def synthesize(config: dict[str, Any], p: dict[str, Path], audit: dict[str, Any], memory: dict[str, Any]) -> None:
    causal = read_json(Path(config["causal_root"]) / "causal_summary.json")
    replicated = memory["replicated_hypotheses"]
    if replicated:
        memory_conclusion = f"사전 정의한 memory-side proxy 중 matched/random specificity까지 만족한 재현 가설은 {len(replicated)}개였다."
        memory_status = "SUPPORTED_FOR_SPECIFIC_PROXIES"
    else:
        memory_conclusion = "검사한 factual/surface proxy에서는 matched/random specificity까지 만족한 선택적 반응이 재현되지 않았다."
        memory_status = "NO_SPECIFIC_REPLICATION"
    claims = pd.DataFrame([
        {"claim_id": "C1", "claim": "R/M representation 차이는 frozen LiReF 방향에서 존재한다.", "evidence": "Stage A discovery/validation", "status": "SUPPORTED", "limitation": "모델·데이터·prompt 한정"},
        {"claim_id": "C2", "claim": "일부 frozen component는 여러 relation 표현과 task relevance에 민감하다.", "evidence": "Stage B b04", "status": "SUPPORTED", "limitation": "synthetic English template sensitivity"},
        {"claim_id": "C3", "claim": "4개 component는 측정된 R/M representation gap에 인과적으로 기여한다.", "evidence": "Causal c01: dose, matched/random, FDR", "status": "SUPPORTED", "limitation": "마지막 prompt token intervention"},
        {"claim_id": "C4", "claim": "Stage B relation sensitivity가 해당 component에 의해 직접 매개된다.", "evidence": "Causal relation hidden-score mediation", "status": "NOT_SUPPORTED", "limitation": "attenuation CI 0/5"},
        {"claim_id": "C5", "claim": "검사한 Memory-side proxy와 frozen component 반응의 선택적 연결이 재현된다.", "evidence": "Memory Discovery freeze + Validation + controls", "status": memory_status, "limitation": "deterministic proxy이며 NER/knowledge retrieval 직접 측정 아님"},
        {"claim_id": "C6", "claim": "Synthetic relation 문항의 언어적 타당성이 독립 검수됐다.", "evidence": "Two-reviewer blind linguistic audit", "status": "PENDING", "limitation": "사람 2명의 completed form 필요"},
    ])
    atomic_csv(p["stage_c"] / "claim_evidence_matrix.csv", claims)
    replicated_lines = [f"- `{row['candidate_id']}` × `{row['feature_name']}`: beta={row['beta']:.3f}, q={row['bh_q_validation']:.4g}" for row in replicated]
    lines = [
        "# Stage C PDF 작성 전 종합 상태", "", "## 현재 결론", "",
        f"- Causal primary gap 기준: **{causal['candidate_primary_criterion_pass_count']}/5개 PASS**",
        f"- 표현 gap과 synthetic 행동 증거 수렴: **{causal['gap_and_behavior_convergent_count']}개**",
        f"- Relation hidden-score mediation 지지: **{causal['relation_attenuation_supported_count']}/5개**",
        f"- Memory-side Discovery 동결 가설: **{memory['discovery_hypothesis_count']}개**",
        f"- Memory-side 기본 Validation 재현: **{memory['basic_validation_replication_count']}개**",
        f"- Memory-side control specificity 포함 재현: **{memory['specific_validation_replication_count']}개**", "",
        "## Memory-side 해석", "", memory_conclusion, "",
    ]
    lines.extend(replicated_lines or ["- 최종 specificity 기준을 통과한 memory-side feature는 없음."])
    lines.extend([
        "", "이 null/limited 결과는 Memorization feature가 존재하지 않는다는 뜻이 아니라, 사전 정의한 capitalization·year/date·person/location/temporal·factual-association·entity-attribute proxy에서 선택적 재현을 찾지 못했다는 뜻이다.",
        "", "## Linguistic audit 상태", "",
        f"- Causal 96쌍 전수 + Stage B 층화표본 {audit['stage_b_stratified_sample_count']}쌍 = 총 **{audit['total_audit_items']}쌍** blind 평가표 준비 완료",
        f"- 자동 구조 검사: **{audit['automated_structure_pass_count']}/{audit['total_audit_items']} PASS**",
        "- 독립 human reviewer 2명의 실제 평가는 **PENDING**",
        "- 따라서 PDF 최종화 전에 두 reviewer form과 adjudication 결과를 반영해야 한다.",
        "", "## PDF 작성 전 남은 유일한 필수 작업", "",
        "1. `linguistic_audit/reviewer_forms/`의 두 CSV를 독립 검수자가 작성",
        "2. 불일치 adjudication 및 acceptance 기준 확인",
        "3. audit PASS이면 Stage C PDF 작성; FAIL이면 기존 결과를 보존한 채 수정 manifest로 새 run 결정", "",
        "## 해석 경계", "",
        "- Reasoning mechanism 일반을 발견했다고 주장하지 않는다. task-relevant relation sensitivity와 측정된 gap 기여를 주장한다.",
        "- Memory proxy 분석은 question text의 표면/factual cue 분석이며 모델의 실제 retrieval 과정을 직접 측정하지 않는다.",
        "- Human linguistic audit이 완료되기 전에는 controlled 문항의 독립 언어 타당성 검증 완료를 주장하지 않는다.",
    ])
    (p["stage_c"] / "STAGE_C_PREPDF_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(p["stage_c"] / "pre_pdf_status.json", {
        "status": "WAITING_FOR_HUMAN_LINGUISTIC_AUDIT",
        "pdf_ready": False,
        "memory_analysis_complete": True,
        "memory_summary": memory,
        "audit_summary": audit,
        "next_gate": "two independent completed reviewer forms plus adjudication/acceptance evaluation",
    })


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    p = paths(config)
    ensure_dirs(p)
    if args.phase == "extract_validation":
        if not (p["manifests"] / "memory_hypothesis_manifest.json").exists():
            raise RuntimeError("Discovery hypothesis manifest must be frozen before validation extraction")
        extract_validation_responses(config, p)
        return
    schema = read_json(SCRIPT_DIR / "memory_feature_schema.json")
    audit = generate_audit(config, p)
    memory = memory_analysis(config, p, schema)
    synthesize(config, p, audit, memory)
    code_files = {path.name: sha256_file(path) for path in sorted(SCRIPT_DIR.iterdir()) if path.is_file() and path.suffix in {".py", ".json", ".sh", ".md"}}
    write_json(p["manifests"] / "run_manifest.json", {
        "run_id": config["run_id"], "timestamp": utc_now(), "config": config,
        "source_hashes": source_hashes(config), "feature_schema_hash": sha256_file(SCRIPT_DIR / "memory_feature_schema.json"),
        "code_hash": canonical_hash(code_files), "code_files": code_files,
        "audit_status": audit["status"], "memory_analysis_complete": True,
        "interpretation_boundary": schema["interpretation_boundary"],
    })
    checksums = {
        str(path.relative_to(p["root"])): sha256_file(path)
        for path in sorted(p["root"].rglob("*"))
        if path.is_file() and path.name != "artifact_checksums.json"
    }
    write_json(p["root"] / "artifact_checksums.json", checksums)
    print(json.dumps({"audit": audit, "memory": memory, "output_root": str(p["root"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
