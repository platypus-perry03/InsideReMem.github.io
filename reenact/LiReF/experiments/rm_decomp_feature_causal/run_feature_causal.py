#!/usr/bin/env python3
"""Pre-confirmatory pipeline for LiReF Stage D causal feature identification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_A_DIR = SCRIPT_DIR.parent / "rm_decomp"
STAGE_B_DIR = SCRIPT_DIR.parent / "rm_decomp_b"
for source in (STAGE_A_DIR, STAGE_B_DIR):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from core import load_model_and_tokenizer, release_model, sha256_file, sha256_text  # noqa: E402
from stage_b_core import load_directions, model_parameter_checksum  # noqa: E402


LABELS = ("A", "B", "C", "D")
CONDITIONS = ("A", "B", "C", "D")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temporary, path)


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(path)
    if args.gpu_id is not None:
        config["gpu_id"] = args.gpu_id
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    config["config_path"] = str(path.resolve())
    config["config_hash"] = canonical_hash({k: v for k, v in config.items() if k not in {"config_path", "config_hash"}})
    return config


def paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["output_root"])
    return {
        "root": root,
        "manifests": root / "manifests",
        "tables": root / "tables",
        "status": root / "status",
        "logs": root / "logs",
        "audit": root / "human_audit",
        "pool": root / "manifests" / "calibration_pool.jsonl",
        "calibration_items": root / "manifests" / "calibration_items.csv.gz",
        "calibration_results": root / "tables" / "calibration_results.csv.gz",
        "frozen_chains": root / "manifests" / "frozen_chain_split.json",
        "pilot_items": root / "manifests" / "pilot_items.csv",
        "confirmatory_items": root / "manifests" / "confirmatory_items.csv",
    }


def ensure_dirs(p: dict[str, Path]) -> None:
    for key in ("root", "manifests", "tables", "status", "logs", "audit"):
        p[key].mkdir(parents=True, exist_ok=True)


def status(p: dict[str, Path], phase: str, state: str = "PASS", **values: Any) -> None:
    write_json(p["status"] / f"{phase}.json", {"phase": phase, "status": state, "timestamp": utc_now(), **values})


def require(p: dict[str, Path], phase: str) -> None:
    target = p["status"] / f"{phase}.json"
    if not target.exists() or read_json(target).get("status") != "PASS":
        raise RuntimeError(f"Required phase is not PASS: {phase}")


def option_block(options: list[str]) -> str:
    return "\n".join(f"{label}. {option}" for label, option in zip(LABELS, options))


def question_with_options(stem: str, options: list[str]) -> str:
    return f"{stem}\n\nOptions:\n{option_block(options)}\nAnswer with only the option letter."


def slot(value: str) -> str:
    """Remove terminal sentence punctuation before inserting a value into prose."""
    return str(value).rstrip(" .?!")


def yz_fact(country: str, capital: str, family: int) -> str:
    country, capital = slot(country), slot(capital)
    return [
        f"The capital of {country} is {capital}.", f"{capital} is the capital city of {country}.",
        f"{country} has {capital} as its capital.", f"For {country}, the capital is {capital}.",
        f"The national capital associated with {country} is {capital}.", f"{capital} serves as {country}'s capital.",
    ][family]


def xy_fact(entity: str, country: str, family: int) -> str:
    entity, country = slot(entity), slot(country)
    return [
        f"{entity} is in {country}.", f"{entity} is associated with the country {country}.",
        f"The country connected to {entity} is {country}.", f"The country associated with {entity} is {country}.",
        f"{entity} belongs to or originates in {country}.", f"The relevant country for {entity} is {country}.",
    ][family]


def direct_stems(country: str) -> list[str]:
    country = slot(country)
    return [
        f"What is the capital of {country}?",
        f"Which city is the capital of {country}?",
        f"Name {country}'s capital city.",
    ]


def country_stems(entity: str) -> list[str]:
    entity = slot(entity)
    return [
        f"In what country is {entity}?",
        f"Which country is {entity} associated with?",
        f"What country is commonly associated with {entity}?",
    ]


def family_text(chain: dict[str, Any], condition: str, family: int) -> str:
    x, y, z = slot(chain["entity"]), slot(chain["country"]), slot(chain["capital"])
    dx, dy, dz = slot(chain["distractor_entity"]), slot(chain["distractor_country"]), slot(chain["distractor_capital"])
    fact_yz = yz_fact(y, z, family)
    fact_xy = xy_fact(x, y, family)
    distractor_yz = yz_fact(dy, dz, family)
    distractor_xy = xy_fact(dx, dy, family)
    direct = [
        f"What is the capital of {y}?", f"Which city is {y}'s capital?",
        f"Name the capital city of {y}.", f"For {y}, which city is the capital?",
        f"Which national capital is associated with {y}?", f"What city serves as {y}'s capital?",
    ][family]
    composition = [
        f"What is the capital of the country where {x} is located?",
        f"Which capital city belongs to the country associated with {x}?",
        f"Name the capital of the country connected to {x}.",
        f"For the country associated with {x}, which city is its capital?",
        f"Which national capital corresponds to {x}'s country?",
        f"What city serves as the capital of the country relevant to {x}?",
    ][family]
    if condition == "A":
        stem = f"Use only facts relevant to the question.\nFact 1: {distractor_xy}\nFact 2: {fact_yz}\nQuestion: {direct}"
    elif condition == "B":
        stem = f"Use only facts relevant to the question.\nFact 1: {fact_xy}\nFact 2: {fact_yz}\nQuestion: {composition}"
    elif condition == "C":
        stem = f"Use only facts relevant to the question.\nFact 1: {distractor_xy}\nFact 2: {distractor_yz}\nQuestion: {direct}"
    elif condition == "D":
        stem = f"Use only facts relevant to the question.\nFact 1: {distractor_xy}\nFact 2: {distractor_yz}\nQuestion: {composition}"
    else:
        raise ValueError(condition)
    return question_with_options(stem, chain["options"])


def pilot_text(chain: dict[str, Any], condition: str) -> str:
    x, y, z = slot(chain["entity"]), slot(chain["country"]), slot(chain["capital"])
    dx, dy, dz = slot(chain["distractor_entity"]), slot(chain["distractor_country"]), slot(chain["distractor_capital"])
    direct = f"Identify the capital city of {y}."
    composition = f"Identify the capital city of the country linked to {x}."
    if condition == "A":
        stem = f"Given: {dx} is linked to {dy}; {z} is {y}'s capital.\n{direct}"
    elif condition == "B":
        stem = f"Given: {x} is linked to {y}; {z} is {y}'s capital.\n{composition}"
    elif condition == "C":
        stem = f"Given: {dx} is linked to {dy}; {dz} is {dy}'s capital.\n{direct}"
    else:
        stem = f"Given: {dx} is linked to {dy}; {dz} is {dy}'s capital.\n{composition}"
    return question_with_options(stem, chain["options"])


def load_popqa_chains(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_json(Path(config["popqa_path"]))
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_subject_property: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subj_id"])].append(row)
        by_subject_property[(str(row["subj_id"]), str(row["prop"]))].append(row)
    raw: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for first in rows:
        if first["prop"] != "country":
            continue
        if len({str(v["obj_id"]) for v in by_subject_property[(str(first["subj_id"]), "country")]}) != 1:
            continue
        seconds = [v for v in by_subject.get(str(first["obj_id"]), []) if v["prop"] == "capital"]
        if len({str(v["obj_id"]) for v in seconds}) != 1:
            continue
        second = seconds[0]
        if len({str(first["subj_id"]), str(first["obj_id"]), str(second["obj_id"])}) != 3:
            continue
        raw.append((first, second))
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in raw:
        grouped[str(pair[0]["obj_id"])].append(pair)
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for country_id in sorted(grouped):
        selected.extend(sorted(grouped[country_id], key=lambda pair: (-int(pair[0]["s_pop"]), str(pair[0]["subj_id"])))[: int(config["pool_country_cap"])])
    capitals = sorted({str(second["obj"]) for _, second in selected})
    countries = sorted({str(first["obj"]) for first, _ in selected})
    surface_countries: dict[str, set[str]] = defaultdict(set)
    for first, _ in selected:
        entity_title = str(first.get("s_wiki_title") or first["subj"])
        surface_countries[entity_title.casefold()].add(str(first["obj"]))
    output: list[dict[str, Any]] = []
    for index, (first, second) in enumerate(sorted(selected, key=lambda pair: (-int(pair[0]["s_pop"]), str(pair[0]["subj_id"])))):
        chain_id = f"FC{index:04d}"
        seed = int(hashlib.sha256(f"{config['seed']}::{chain_id}".encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        foils = [v for v in capitals if v != str(second["obj"])]
        rng.shuffle(foils)
        options = [str(second["obj"]), *foils[:3]]
        rng.shuffle(options)
        country_foils = [v for v in countries if v != str(first["obj"])]
        rng.shuffle(country_foils)
        country_options = [str(first["obj"]), *country_foils[:3]]
        rng.shuffle(country_options)
        entity_title = str(first.get("s_wiki_title") or first["subj"])
        output.append({
            "chain_id": chain_id, "entity": entity_title, "source_entity_surface": str(first["subj"]), "country": str(first["obj"]),
            "capital": str(second["obj"]), "entity_id": str(first["subj_id"]),
            "country_id": str(first["obj_id"]), "capital_id": str(second["obj_id"]),
            "entity_popularity": int(first["s_pop"]), "country_popularity": int(first["o_pop"]),
            "capital_popularity": int(second["o_pop"]), "options": options,
            "correct_label": LABELS[options.index(str(second["obj"]))],
            "country_options": country_options,
            "country_correct_label": LABELS[country_options.index(str(first["obj"]))],
            "entity_surface_country_count": len(surface_countries[entity_title.casefold()]),
            "entity_surface_country_unambiguous": len(surface_countries[entity_title.casefold()]) == 1,
            "source_row_ids": [int(first["id"]), int(second["id"])],
        })
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    by_family_shape: dict[int, dict[tuple[int, int], list[dict[str, Any]]]] = {family: defaultdict(list) for family in range(6)}
    for chain in output:
        chain["entity_token_length"] = len(tokenizer.encode(chain["entity"], add_special_tokens=False))
        chain["country_token_length"] = len(tokenizer.encode(chain["country"], add_special_tokens=False))
        chain["capital_token_length"] = len(tokenizer.encode(chain["capital"], add_special_tokens=False))
        for family in range(6):
            shape = (len(tokenizer.encode(xy_fact(chain["entity"], chain["country"], family), add_special_tokens=False)),
                     len(tokenizer.encode(yz_fact(chain["country"], chain["capital"], family), add_special_tokens=False)))
            by_family_shape[family][shape].append(chain)
    matched: list[dict[str, Any]] = []
    for chain in output:
        chain = dict(chain)
        chain["distractors_by_family"] = {}
        for family in range(6):
            shape = (len(tokenizer.encode(xy_fact(chain["entity"], chain["country"], family), add_special_tokens=False)),
                     len(tokenizer.encode(yz_fact(chain["country"], chain["capital"], family), add_special_tokens=False)))
            alternatives = [row for row in by_family_shape[family][shape]
                            if row["country_id"] != chain["country_id"]
                            and row["capital"] not in chain["options"]
                            and row["entity_surface_country_unambiguous"]]
            exact_fact_length = bool(alternatives)
            if not alternatives:
                alternatives = [row for row in output
                                if row["country_id"] != chain["country_id"]
                                and row["capital"] not in chain["options"]
                                and row["entity_surface_country_unambiguous"]]
            alternatives.sort(key=lambda row: canonical_hash([config["seed"], chain["chain_id"], family, row["chain_id"]]))
            if alternatives:
                distractor = alternatives[0]
                chain["distractors_by_family"][str(family)] = {"chain_id": distractor["chain_id"], "entity": distractor["entity"],
                                                               "country": distractor["country"], "capital": distractor["capital"],
                                                               "fact_length_shape_exact": exact_fact_length}
        distractor = chain["distractors_by_family"]["0"]
        chain.update({"distractor_chain_id": distractor["chain_id"], "distractor_entity": distractor["entity"],
                      "distractor_country": distractor["country"], "distractor_capital": distractor["capital"]})
        matched.append(chain)
    return matched


def calibration_rows(chain: dict[str, Any], prompt_template: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edge, stems in (("r1", country_stems(chain["entity"])), ("r2", direct_stems(chain["country"]))):
        answer = chain["country"] if edge == "r1" else chain["capital"]
        options = list(chain["country_options"] if edge == "r1" else chain["options"])
        label = LABELS[options.index(answer)]
        for template_index, stem in enumerate(stems):
            question = question_with_options(stem, options)
            rows.append({"chain_id": chain["chain_id"], "item_type": edge, "template_index": template_index,
                         "prompt": prompt_template.format(question=question), "correct_label": label})
    for condition in ("A", "B"):
        question = family_text(chain, condition, 0)
        rows.append({"chain_id": chain["chain_id"], "item_type": f"supplied_{condition}", "template_index": 0,
                     "prompt": prompt_template.format(question=question), "correct_label": chain["correct_label"]})
    return rows


def prepare(config: dict[str, Any], p: dict[str, Path]) -> None:
    ensure_dirs(p)
    chains = load_popqa_chains(config)
    if len(chains) < 200:
        raise RuntimeError(f"Initial calibration pool must contain at least 200 chains, got {len(chains)}")
    with p["pool"].open("w", encoding="utf-8") as handle:
        for chain in chains:
            handle.write(json.dumps(chain, ensure_ascii=False) + "\n")
    items = pd.DataFrame(row for chain in chains for row in calibration_rows(chain, config["prompt_template"]))
    write_csv(p["calibration_items"], items)
    parameter_hash, parameter_files = model_parameter_checksum(Path(config["model_path"]))
    causal_conditions = read_json(Path(config["causal_root"]) / "manifests" / "intervention_conditions.json")["conditions"]
    frozen_components = []
    for candidate in config["primary_candidates"]:
        hits = [row for row in causal_conditions if row["owner_candidate_id"] == candidate and row["component_role"] == "candidate"]
        if not hits:
            raise RuntimeError(f"Candidate missing from frozen causal manifest: {candidate}")
        frozen_components.append(hits[0]["components"][0])
    design = {
        "run_id": config["run_id"], "factorial_conditions": {
            "A": "supplied+direct", "B": "supplied+composition", "C": "parametric+direct", "D": "parametric+composition"
        },
        "chain_definition": "entity --country--> country --capital--> capital; direct uses country→capital, composition uses entity→country→capital",
        "length_control": "All four conditions contain two structurally parallel facts. Parametric conditions contain an irrelevant distractor chain with exactly matched entity/country/capital token counts. A/C and B/D are exactly token-length matched; reasoning exact-match eligibility additionally requires entity and country names to have equal token counts.",
        "same_answer_and_options_within_chain": True,
        "calibration_pool_amendment": "v2_d01 stopped before Pilot because 149/528 chains passed versus 184 required. v2_d02 expanded the pre-feature pool. v2_d02 fixed double punctuation in v2_d03. v2_d04 removed multi-country surface conflicts. Full AI audit of v2_d04 found 74 chains whose short subject surface differed from the source Wikipedia title. v2_d05 used source s_wiki_title. Full AI linguistic audit of v2_d05 found an ill-formed family-04 relation phrase and five distractor capitals overlapping answer options. v2_d06 corrects the phrase and excludes such distractors before any confirmatory inference; only byte-identical calibration prompts are reused.",
        "entity_surface_rule": "Use frozen PopQA s_wiki_title when non-empty, otherwise subj. This disambiguates entities such as Mary, Saône-et-Loire and Paste (magazine).",
        "linguistic_quality_filter": "A target or distractor entity surface is eligible only when all PopQA country rows with that case-folded surface agree on one country. Short names alone are not automatically rejected and remain subject to blind human review.",
        "effects": {"E_R": "0.5*((S_B-S_A)+(S_D-S_C))", "E_M": "0.5*((S_A-S_C)+(S_B-S_D))"},
        "frozen_components": frozen_components, "calibration_thresholds": config["calibration"],
        "statistical_thresholds": config["statistics"], "scope_gate": config["scope_gate"],
        "hashes": {"config": config["config_hash"], "prompt": sha256_text(config["prompt_template"]),
                   "popqa": sha256_file(Path(config["popqa_path"])), "model_parameters": parameter_hash,
                   "model_parameter_files": parameter_files,
                   "code": {path.name: sha256_file(path) for path in sorted(SCRIPT_DIR.iterdir()) if path.is_file()},
                   "stage_a_direction": sha256_file(Path(config["stage_a_root"]) / "checkpoints" / "discovery_liref_directions.pt"),
                   "causal_conditions": sha256_file(Path(config["causal_root"]) / "manifests" / "intervention_conditions.json")},
        "interpretation_boundary": "The factorial manipulation targets composition demand and parametric factual-memory dependence; it does not establish reasoning or memorization in general.",
    }
    write_json(p["manifests"] / "frozen_design.json", design)
    write_json(p["manifests"] / "frozen_config.json", config)
    status(p, "prepare", chain_count=len(chains), calibration_item_count=len(items), design_hash=sha256_file(p["manifests"] / "frozen_design.json"))
    print(f"Prepared {len(chains)} chains and {len(items)} calibration items", flush=True)


def load_chain_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.inference_mode()
def score_answer_labels(model: Any, tokenizer: Any, device: torch.device, prompts: list[str], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    label_ids = []
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Answer label is not one token: {label} -> {ids}")
        label_ids.append(ids[0])
    probabilities, predictions = [], []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(prompts[start:start + batch_size], return_tensors="pt", padding="longest", return_token_type_ids=False)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, label_ids]
        probs = torch.softmax(logits.float(), dim=-1)
        probabilities.append(probs.cpu().numpy())
        predictions.append(probs.argmax(dim=-1).cpu().numpy())
        print(f"calibration {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return np.concatenate(probabilities), np.concatenate(predictions)


def calibrate(config: dict[str, Any], p: dict[str, Path]) -> None:
    require(p, "prepare")
    items = pd.read_csv(p["calibration_items"])
    reuse_path = Path(config.get("calibration_reuse_root", "")) / "tables" / "calibration_results.csv.gz"
    reused: dict[tuple[str, str], dict[str, Any]] = {}
    if reuse_path.exists():
        prior = pd.read_csv(reuse_path)
        reused = {(str(row["prompt"]), str(row["correct_label"])): row.to_dict() for _, row in prior.iterrows()}
    missing_indices = [i for i, row in items.iterrows() if (str(row["prompt"]), str(row["correct_label"])) not in reused]
    probs = np.zeros((len(items), len(LABELS)), dtype=np.float32)
    predictions = np.zeros(len(items), dtype=np.int64)
    for i, row in items.iterrows():
        prior = reused.get((str(row["prompt"]), str(row["correct_label"])))
        if prior is not None:
            probs[i] = [float(prior[f"p_{label}"]) for label in LABELS]
            predictions[i] = LABELS.index(str(prior["predicted_label"]))
    model = None
    if missing_indices:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when calibration contains fresh prompts")
        device = torch.device(f"cuda:{config['gpu_id']}")
        try:
            model, tokenizer = load_model_and_tokenizer(config, device)
            fresh_probs, fresh_predictions = score_answer_labels(model, tokenizer, device, items.iloc[missing_indices]["prompt"].tolist(), int(config["batch_size"]))
            probs[missing_indices] = fresh_probs
            predictions[missing_indices] = fresh_predictions
        finally:
            release_model(model)
    for index, label in enumerate(LABELS):
        items[f"p_{label}"] = probs[:, index]
    items["predicted_label"] = [LABELS[int(v)] for v in predictions]
    items["correct_probability"] = [probs[i, LABELS.index(label)] for i, label in enumerate(items["correct_label"])]
    items["strongest_foil_probability"] = [max(np.delete(probs[i], LABELS.index(label))) for i, label in enumerate(items["correct_label"])]
    items["correct"] = items["predicted_label"] == items["correct_label"]
    write_csv(p["calibration_results"], items)
    status(p, "calibrate", rows=len(items), reused_rows=len(items) - len(missing_indices), fresh_rows=len(missing_indices),
           accuracy=float(items["correct"].mean()), result_hash=sha256_file(p["calibration_results"]))


def edge_pass(frame: pd.DataFrame, threshold: dict[str, Any]) -> bool:
    return bool(
        int(frame["correct"].sum()) == int(threshold["paraphrase_correct_required"])
        and float(frame["correct_probability"].mean()) >= float(threshold["mean_correct_probability_min"])
        and float(frame["correct_probability"].min()) >= float(threshold["individual_correct_probability_min"])
        and int((frame["correct_probability"] > frame["strongest_foil_probability"]).sum()) == int(threshold["foil_dominance_required"])
        and int((frame["predicted_label"] == frame["correct_label"]).sum()) == int(threshold["output_consistency_required"])
    )


def round_robin_chains(chains: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chain in chains:
        groups[chain["country_id"]].append(chain)
    for values in groups.values():
        values.sort(key=lambda row: (-int(row["entity_popularity"]), row["chain_id"]))
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    depth = 0
    while len(selected) < count:
        changed = False
        for key in keys:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                changed = True
                if len(selected) == count:
                    break
        if not changed:
            break
        depth += 1
    return selected


def build_factorial_items(chains: list[dict[str, Any]], split: str, prompt_template: str, model_path: str | None = None) -> pd.DataFrame:
    output = []
    tokenizer = AutoTokenizer.from_pretrained(model_path or read_json(SCRIPT_DIR / "config.json")["model_path"], trust_remote_code=True)
    for index, chain in enumerate(chains):
        if split == "pilot":
            family = -1
        else:
            exact_families = []
            available_families = sorted(int(value) for value in chain["distractors_by_family"])
            for candidate_family in available_families:
                candidate = dict(chain)
                distractor = chain["distractors_by_family"][str(candidate_family)]
                candidate.update({"distractor_chain_id": distractor["chain_id"], "distractor_entity": distractor["entity"],
                                  "distractor_country": distractor["country"], "distractor_capital": distractor["capital"]})
                lengths = [len(tokenizer.encode(prompt_template.format(question=family_text(candidate, condition, candidate_family)))) for condition in CONDITIONS]
                if lengths[0] == lengths[1] and lengths[2] == lengths[3]:
                    exact_families.append(candidate_family)
            preferred = index if index < 6 and str(index) in chain["distractors_by_family"] else available_families[index % len(available_families)]
            family = exact_families[index % len(exact_families)] if index >= 6 and exact_families else preferred
        active_chain = dict(chain)
        active_distractor = chain["distractors_by_family"]["0" if split == "pilot" else str(family)]
        active_chain.update({"distractor_chain_id": active_distractor["chain_id"], "distractor_entity": active_distractor["entity"],
                             "distractor_country": active_distractor["country"], "distractor_capital": active_distractor["capital"]})
        chain_rows = []
        for condition in CONDITIONS:
            question = pilot_text(active_chain, condition) if split == "pilot" else family_text(active_chain, condition, family)
            prompt = prompt_template.format(question=question)
            chain_rows.append({
                "chain_id": chain["chain_id"], "split": split, "condition": condition,
                "memory_factor": "supplied" if condition in {"A", "B"} else "parametric",
                "reasoning_factor": "direct" if condition in {"A", "C"} else "composition",
                "template_family": "pilot_only" if split == "pilot" else f"confirmatory_{family + 1:02d}",
                "entity": chain["entity"], "country": chain["country"], "capital": chain["capital"],
                "distractor_chain_id": active_chain["distractor_chain_id"], "distractor_entity": active_chain["distractor_entity"],
                "distractor_country": active_chain["distractor_country"], "distractor_capital": active_chain["distractor_capital"],
                "options_json": json.dumps(chain["options"], ensure_ascii=False), "correct_label": chain["correct_label"],
                "question": question, "prompt": prompt, "token_length": len(tokenizer.encode(prompt)),
            })
        reasoning_exact = chain_rows[0]["token_length"] == chain_rows[1]["token_length"] and chain_rows[2]["token_length"] == chain_rows[3]["token_length"]
        memory_exact = chain_rows[0]["token_length"] == chain_rows[2]["token_length"] and chain_rows[1]["token_length"] == chain_rows[3]["token_length"]
        for row in chain_rows:
            row["reasoning_exact_length_matched"] = reasoning_exact
            row["memory_exact_length_matched"] = memory_exact
        output.extend(chain_rows)
    return pd.DataFrame(output)


def freeze(config: dict[str, Any], p: dict[str, Path]) -> None:
    require(p, "calibrate")
    chains = {row["chain_id"]: row for row in load_chain_jsonl(p["pool"])}
    results = pd.read_csv(p["calibration_results"])
    passing = []
    summary_rows = []
    for chain_id, frame in results.groupby("chain_id", sort=True):
        r1 = frame[frame["item_type"] == "r1"]
        r2 = frame[frame["item_type"] == "r2"]
        supplied = frame[frame["item_type"].str.startswith("supplied_")]
        row = {"chain_id": chain_id, "r1_pass": edge_pass(r1, config["calibration"]),
               "r2_pass": edge_pass(r2, config["calibration"]),
               "supplied_accuracy": float(supplied["correct"].mean())}
        row["pass"] = row["r1_pass"] and row["r2_pass"] and row["supplied_accuracy"] >= float(config["calibration"]["supplied_accuracy_required"])
        summary_rows.append(row)
        row["quality_eligible"] = bool(chains[chain_id]["entity_surface_country_unambiguous"])
        if row["pass"] and row["quality_eligible"]:
            passing.append(chains[chain_id])
    required_count = int(config["pilot_chain_count"]) + int(config["confirmatory_chain_count"])
    selected = round_robin_chains(passing, required_count)
    write_csv(p["tables"] / "calibration_chain_summary.csv", pd.DataFrame(summary_rows))
    if len(selected) < required_count:
        status(p, "freeze", "FAIL", passing_chain_count=len(passing), required_chain_count=required_count)
        raise RuntimeError(f"Only {len(passing)} calibrated chains pass; {required_count} required")
    pilot = selected[: int(config["pilot_chain_count"])]
    confirmatory = selected[int(config["pilot_chain_count"]):]
    payload = {"pilot": pilot, "confirmatory": confirmatory, "pilot_reused_in_confirmatory": False,
               "selection_rule": "country round-robin then frozen popularity/chain-id order", "frozen_at": utc_now()}
    write_json(p["frozen_chains"], payload)
    write_csv(p["pilot_items"], build_factorial_items(pilot, "pilot", config["prompt_template"], config["model_path"]))
    write_csv(p["confirmatory_items"], build_factorial_items(confirmatory, "confirmatory", config["prompt_template"], config["model_path"]))
    status(p, "freeze", passing_chain_count=len(passing), pilot_chain_count=len(pilot), confirmatory_chain_count=len(confirmatory),
           represented_country_count=len({row["country_id"] for row in confirmatory}), frozen_hash=sha256_file(p["frozen_chains"]))


class PilotCapture:
    def __init__(self, model: Any, components: list[dict[str, Any]]) -> None:
        self.model = model
        self.components = components
        self.final: torch.Tensor | None = None
        self.values: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def install(self) -> None:
        def final_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.final = tensor.detach()[:, -1, :].clone()
        self.handles.append(self.model.model.layers[31].register_forward_hook(final_hook))
        for component in self.components:
            layer = self.model.model.layers[int(component["module_index"])]
            module = layer.self_attn.o_proj if component["component_type"] == "head" else layer.mlp.down_proj
            self.handles.append(module.register_forward_pre_hook(self._component_hook(component)))

    def _component_hook(self, component: dict[str, Any]):
        component_id = component["component_id"]
        index = int(component["component_index"])
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            values = args[0].detach()[:, -1, :]
            if component["component_type"] == "head":
                head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
                value = values[:, index * head_dim:(index + 1) * head_dim].float().norm(dim=-1)
            else:
                value = values[:, index].float()
            self.values[component_id] = value.cpu()
        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def pilot(config: dict[str, Any], p: dict[str, Path]) -> None:
    require(p, "freeze")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pilot")
    device = torch.device(f"cuda:{config['gpu_id']}")
    items = pd.read_csv(p["pilot_items"])
    design = read_json(p["manifests"] / "frozen_design.json")
    components = design["frozen_components"]
    directions = load_directions(Path(config["stage_a_root"]) / "checkpoints" / "discovery_liref_directions.pt")
    direction = torch.as_tensor(directions[31], device=device, dtype=torch.float32)
    rows = []
    model = None
    try:
        model, tokenizer = load_model_and_tokenizer(config, device)
        capture = PilotCapture(model, components)
        capture.install()
        for start in range(0, len(items), int(config["batch_size"])):
            batch = items.iloc[start:start + int(config["batch_size"])]
            encoded = tokenizer(batch["prompt"].tolist(), return_tensors="pt", padding="longest", return_token_type_ids=False)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            model(**encoded, use_cache=False, return_dict=True)
            if capture.final is None:
                raise RuntimeError("Final state was not captured")
            scores = torch.mv(capture.final.float(), direction).cpu().numpy()
            for offset, (_, source) in enumerate(batch.iterrows()):
                row = source.to_dict()
                row["score"] = float(scores[offset])
                for component in components:
                    row[f"state::{component['component_id']}"] = float(capture.values[component["component_id"]][offset])
                rows.append(row)
            print(f"pilot {min(start + int(config['batch_size']), len(items))}/{len(items)}", flush=True)
        capture.remove()
    finally:
        release_model(model)
    response = pd.DataFrame(rows)
    write_csv(p["tables"] / "pilot_responses.csv.gz", response)
    wide = response.pivot(index="chain_id", columns="condition", values="score")
    wide["E_R"] = 0.5 * ((wide["B"] - wide["A"]) + (wide["D"] - wide["C"]))
    wide["E_M"] = 0.5 * ((wide["A"] - wide["C"]) + (wide["B"] - wide["D"]))
    estimates = []
    for effect in ("E_R", "E_M"):
        values = wide[effect].to_numpy(float)
        estimates.append({"effect": effect, "n": len(values), "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                          "paired_dz": float(values.mean() / values.std(ddof=1)) if values.std(ddof=1) else None})
    write_csv(p["tables"] / "pilot_effect_estimates.csv", pd.DataFrame(estimates))
    status(p, "pilot", pilot_is_nonconfirmatory=True, response_hash=sha256_file(p["tables"] / "pilot_responses.csv.gz"),
           thresholds_changed_from_pilot=False, candidates_changed_from_pilot=False)


def audit_package(config: dict[str, Any], p: dict[str, Path]) -> None:
    require(p, "pilot")
    items = pd.read_csv(p["confirmatory_items"])
    if items["chain_id"].nunique() < 160 or items["template_family"].nunique() < 6:
        raise RuntimeError("Confirmatory minimum chains/template families not met")
    checks = []
    for chain_id, frame in items.groupby("chain_id", sort=True):
        checks.append({
            "chain_id": chain_id, "has_four_conditions": set(frame["condition"]) == set(CONDITIONS),
            "same_answer": frame["capital"].nunique() == 1, "same_correct_label": frame["correct_label"].nunique() == 1,
            "same_options": frame["options_json"].nunique() == 1, "no_empty_text": bool(frame["question"].str.len().gt(0).all()),
            "no_double_terminal_punctuation": not bool(frame["question"].str.contains(r"[.!?]{2,}", regex=True).any()),
        })
    checks_frame = pd.DataFrame(checks)
    checks_frame["automated_pass"] = checks_frame.drop(columns="chain_id").all(axis=1)
    write_csv(p["audit"] / "automated_pre_audit.csv", checks_frame)
    if not bool(checks_frame["automated_pass"].all()):
        raise RuntimeError("Automated pre-audit failed")
    wide = items.pivot(index=["chain_id", "entity", "country", "capital", "correct_label", "options_json", "template_family"], columns="condition", values="question").reset_index()
    for column in ("fact_accuracy", "grammar_ok", "single_feature_validity", "same_answer_valid", "distractors_valid", "approve", "reviewer_comment"):
        wide[column] = ""
    write_csv(p["audit"] / "reviewer_1_blind.csv", wide)
    write_csv(p["audit"] / "reviewer_2_blind.csv", wide)
    instructions = """# Stage D blind linguistic audit\n\n모델 출력과 Pilot 결과를 보지 말고 두 검수자가 독립적으로 작성합니다. 각 chain의 A/B/C/D에 대해 사실 정확성, 문법, direct-vs-composition 조작 타당성, 동일 정답, distractor 무관성을 확인하고 각 항목에 1 또는 0을 입력합니다. `approve`는 모든 항목이 1일 때만 1입니다. 검수자는 서로의 파일을 보지 않습니다. 두 파일 작성 후 별도 adjudication으로 불일치를 해결하고 `audit_final_status.json`을 생성해야 합니다.\n\n현재 파일 생성은 요청 승인이 아니라 실제 문항 품질 검수를 위한 것입니다.\n"""
    (p["audit"] / "AUDIT_INSTRUCTIONS_KO.md").write_text(instructions, encoding="utf-8")
    write_json(p["audit"] / "audit_final_status.json", {"status": "PENDING_TWO_BLIND_HUMAN_REVIEWS", "reviewer_1": "PENDING", "reviewer_2": "PENDING", "approved_chain_count": 0})
    write_json(p["manifests"] / "confirmatory_go.json", {"approved": False, "reason": "Two-reviewer blind linguistic audit has not passed"})
    status(p, "audit_package", "STOP_FOR_HUMAN_AUDIT", confirmatory_chain_count=int(items["chain_id"].nunique()),
           automated_pre_audit="PASS", human_audit="PENDING", confirmatory_executed=False)
    print("STOP_FOR_HUMAN_AUDIT: confirmatory inference is locked", flush=True)


def enforce_confirmatory_gate(p: dict[str, Path]) -> None:
    audit = read_json(p["audit"] / "audit_final_status.json") if (p["audit"] / "audit_final_status.json").exists() else {}
    go = read_json(p["manifests"] / "confirmatory_go.json") if (p["manifests"] / "confirmatory_go.json").exists() else {}
    if audit.get("status") != "PASS" or go.get("approved") is not True:
        raise RuntimeError("Confirmatory is locked until two blind human reviews PASS and confirmatory_go.json approved=true")
    raise RuntimeError("Gate passed, but confirmatory intervention implementation belongs to the post-audit continuation and was not executed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["prepare", "calibrate", "freeze", "pilot", "audit-package", "confirmatory"])
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)
    p = paths(config)
    ensure_dirs(p)
    functions = {"prepare": prepare, "calibrate": calibrate, "freeze": freeze, "pilot": pilot, "audit-package": audit_package}
    if args.phase == "confirmatory":
        enforce_confirmatory_gate(p)
    else:
        functions[args.phase](config, p)


if __name__ == "__main__":
    main()
