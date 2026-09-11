#!/usr/bin/env python3
"""Build sealed external-task records and model-specific candidate manifests."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_from_disk


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DESIGN = HERE / "design_v2_frozen.json"
ASSETS = HERE / "v2_assets"
DATA = ROOT / "AI/reenact/liref/dataset"
R_ROOT = ROOT / "AI/reenact/liref_outputs/rm_decomp/cross_model_r_directed_v1_5"
M_ROOT = ROOT / "AI/reenact/liref_outputs/rm_decomp/cross_model_m_directed_v1_3"
META_M_ROOT = ROOT / "AI/reenact/liref_outputs/rm_decomp/meta_llama_m_directed_v1_4"
FINAL_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
COMP_RE = re.compile(r"^L(\d{2})([HN])(\d{5})$")
CEVAL_SUBJECTS = [
    "modern_chinese_history", "ideological_and_moral_cultivation", "logic", "law",
    "chinese_language_and_literature", "art_studies", "professional_tour_guide",
    "legal_professional", "high_school_chinese", "high_school_history", "middle_school_history",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}::{text}".encode()).hexdigest()


def final_number(text: str) -> str:
    hits = FINAL_RE.findall(str(text))
    if not hits:
        stripped = str(text).replace(",", "")
        hits = re.findall(r"-?\d+(?:\.\d+)?", stripped)
    if not hits:
        raise ValueError(f"No numeric answer: {text!r}")
    raw = hits[-1].replace(",", "")
    value = float(raw)
    return str(int(value)) if value.is_integer() else f"{value:.8f}".rstrip("0").rstrip(".")


def numeric_options(answer: str, key: str) -> tuple[list[str], int]:
    value = float(answer)
    step = 1.0 if abs(value) < 20 else max(1.0, round(abs(value) * 0.1))
    proposals = [value + step, value - step, value + 2 * step, value * 2, value / 2]
    values = [value]
    for proposal in proposals:
        if math.isfinite(proposal) and all(abs(proposal - old) > 1e-9 for old in values):
            values.append(proposal)
        if len(values) == 4:
            break
    if len(values) != 4:
        raise RuntimeError(f"Could not build numeric foils for {answer}")
    formatted = [str(int(v)) if float(v).is_integer() else f"{v:.8f}".rstrip("0").rstrip(".") for v in values]
    correct = int(stable_key(0, key)[:8], 16) % 4
    correct_text = formatted.pop(0)
    options = formatted[:]
    options.insert(correct, correct_text)
    return options, correct


def choose(rows: list[dict[str, Any]], n: int, seed: int, namespace: str) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: stable_key(seed, namespace + "::" + row["sample_id"]))
    if len(ranked) < n:
        raise RuntimeError(f"{namespace}: need {n}, found {len(ranked)}")
    return ranked[:n]


def ceval_rows() -> list[dict[str, Any]]:
    rows = []
    for subject in CEVAL_SUBJECTS:
        frame = pd.read_csv(DATA / "ceval-exam" / "val" / f"{subject}_val.csv")
        for record in frame.to_dict("records"):
            answer = str(record["answer"]).strip()
            rows.append({
                "sample_id": f"ceval_h::{subject}::{int(record['id'])}", "dataset": "ceval_h", "pole": "M",
                "question": str(record["question"]), "options": [str(record[x]) for x in "ABCD"],
                "answer_index": "ABCD".index(answer), "source_split": "val",
            })
    return rows


def gsm8k_rows(excluded: set[int]) -> list[dict[str, Any]]:
    test = load_from_disk(str(DATA / "gsm8k" / "main"))["test"]
    rows = []
    for index, record in enumerate(test):
        if index in excluded:
            continue
        answer = final_number(record["answer"])
        options, correct = numeric_options(answer, f"gsm8k::{index}")
        rows.append({"sample_id": f"gsm8k::{index}", "dataset": "gsm8k", "pole": "R",
                     "question": record["question"], "options": options, "answer_index": correct, "source_split": "test"})
    return rows


def symbolic_rows(seed: int) -> tuple[list[dict[str, Any]], set[int]]:
    by_original: dict[int, list[dict[str, Any]]] = {}
    with (DATA / "gsm-symbolic_data" / "GSM_symbolic.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_original.setdefault(int(row["original_id"]), []).append(row)
    rows, originals = [], set()
    for original_id, variants in sorted(by_original.items()):
        variant = sorted(variants, key=lambda row: stable_key(seed, f"symbolic::{original_id}::{row['instance']}"))[0]
        answer = final_number(variant["answer"])
        options, correct = numeric_options(answer, f"symbolic::{original_id}::{variant['instance']}")
        rows.append({"sample_id": f"gsm_symbolic::{original_id}::{variant['instance']}", "dataset": "gsm_symbolic",
                     "pole": "R", "question": variant["question"], "options": options,
                     "answer_index": correct, "source_split": "symbolic", "original_gsm8k_index": original_id})
        originals.add(original_id)
    return rows, originals


def mgsm_rows() -> list[dict[str, Any]]:
    rows = []
    path = DATA / "mgsm" / "mgsm_en.tsv"
    with path.open(encoding="utf-8") as handle:
        for index, values in enumerate(csv.reader(handle, delimiter="\t")):
            if len(values) != 2:
                raise RuntimeError(f"Bad MGSM row {index}")
            answer = final_number(values[1])
            options, correct = numeric_options(answer, f"mgsm_en::{index}")
            rows.append({"sample_id": f"mgsm_en::{index}", "dataset": "mgsm_en", "pole": "R",
                         "question": values[0], "options": options, "answer_index": correct, "source_split": "test"})
    return rows


def strict_candidates(root: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(root / model / "tables" / "causal_candidate_results.csv")
    return frame[frame["functional_homologue_pass"].astype(str).str.lower() == "true"].copy()


def component_means(root: Path, model: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in ("frozen_candidates.csv", "frozen_controls.csv"):
        frame = pd.read_csv(root / model / "manifests" / name)
        for row in frame.to_dict("records"):
            if row["component_type"] == "neuron" and pd.notna(row["pooled_activation_mean"]):
                values[str(row["component_id"])] = float(row["pooled_activation_mean"])
    return values


def model_manifest(model: str) -> dict[str, Any]:
    if model == "Meta-Llama-3-8B":
        base = read_json(HERE / "candidate_manifest_v1_frozen.json")
        r_pass = set(strict_candidates(R_ROOT, model)["component_id"])
        m_pass = set(strict_candidates(META_M_ROOT, model)["component_id"])
        candidates = []
        for row in base["candidates"]:
            cid = row["component_id"]
            directions = (["R"] if cid in r_pass else []) + (["M"] if cid in m_pass else [])
            if directions:
                candidates.append({**row, "target_poles": directions})
        return {"status": "FROZEN_BEFORE_V2_RESULTS", "model": model, "candidates": candidates}
    roots = [("R", R_ROOT), ("M", M_ROOT)]
    accumulated: dict[str, dict[str, Any]] = {}
    for pole, root in roots:
        causal = strict_candidates(root, model)
        controls = pd.read_csv(root / model / "manifests" / "frozen_controls.csv")
        means = component_means(root, model)
        for result in causal.to_dict("records"):
            cid = str(result["component_id"])
            owned = controls[controls["owner_candidate_id"] == cid]
            matched = str(owned[owned["control_kind"] == "matched"].iloc[0]["component_id"])
            random = str(owned[owned["control_kind"] == "random"].iloc[0]["component_id"])
            if cid not in accumulated:
                needed = {key: means[key] for key in (cid, matched, random) if key in means}
                accumulated[cid] = {"component_id": cid, "source": f"{pole}_directed",
                                    "matched_control": matched, "random_control": random,
                                    "neuron_means": needed, "target_poles": [pole]}
            elif pole not in accumulated[cid]["target_poles"]:
                accumulated[cid]["target_poles"].append(pole)
    return {"status": "FROZEN_BEFORE_V2_RESULTS", "model": model,
            "candidates": [accumulated[key] for key in sorted(accumulated)]}


def main() -> None:
    design = read_json(DESIGN)
    seed, n, primary = int(design["dataset_seed"]), int(design["dataset_samples_each"]), int(design["primary_samples_each"])
    symbolic, excluded = symbolic_rows(seed)
    sources = {"ceval_h": ceval_rows(), "gsm8k": gsm8k_rows(excluded),
               "gsm_symbolic": symbolic, "mgsm_en": mgsm_rows()}
    records = []
    for dataset, source in sources.items():
        selected = choose(source, n, seed, dataset)
        for index, row in enumerate(selected):
            records.append({**row, "evaluation_split": "primary" if index < primary else "confirmation"})
    if len({row["sample_id"] for row in records}) != len(records):
        raise RuntimeError("Duplicate sample ids")
    write_json(ASSETS / "evaluation_records_frozen.json", records)
    for model in [row["name"] for row in design["models"]]:
        write_json(ASSETS / "candidate_manifests" / f"{model}.json", model_manifest(model))
    provenance = {"design_sha256": sha(DESIGN), "records_sha256": sha(ASSETS / "evaluation_records_frozen.json"),
                  "counts": {key: len(value) for key, value in sources.items()},
                  "selected": {dataset: {split: sum(r["dataset"] == dataset and r["evaluation_split"] == split for r in records)
                                           for split in ("primary", "confirmation")} for dataset in sources},
                  "gsm8k_symbolic_overlap_removed": len(excluded),
                  "candidate_counts": {model["name"]: len(read_json(ASSETS / "candidate_manifests" / f"{model['name']}.json")["candidates"])
                                       for model in design["models"]}}
    write_json(ASSETS / "provenance.json", provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
