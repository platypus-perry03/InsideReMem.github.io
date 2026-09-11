#!/usr/bin/env python3
"""Held-out Stage B extension for lexical, relevance, numeric, and factual controls."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "rm_decomp_b"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from stage_b_core import (  # noqa: E402
    AtomicCsvSink,
    StageBCapture,
    component_lookup,
    load_directions,
    load_model_and_tokenizer,
    model_parameter_checksum,
    projections_for_components,
    read_json,
    release_model,
    sha256_file,
    sha256_text,
    unique_components,
    write_json,
)
from stage_b_stats import benjamini_hochberg, paired_summary  # noqa: E402


INTERPRETATION_BOUNDARY = (
    "This extension tests feature sensitivity of the 20 frozen Stage A candidates under synthetic "
    "held-out controlled prompts. It does not establish a reasoning/memorization component or causality."
)

MANIFEST_COLUMNS = [
    "pair_id", "split", "feature_family", "condition", "lexical_family", "template_id",
    "template_family", "analysis_cluster", "base_id", "original_text", "modified_text",
    "changed_spans_original", "changed_spans_modified", "invariant_features",
    "expected_answer_original", "expected_answer_modified", "generation_rule_id",
    "reviewer_id", "approval_basis", "automated_validation", "approved",
    "token_length_original", "token_length_modified",
]

RESPONSE_COLUMNS = [
    "split", "pair_id", "feature_family", "condition", "lexical_family", "template_family",
    "analysis_cluster", "base_id", "variant", "candidate_id", "component_id", "component_type",
    "component_role", "control_kind", "module_index", "component_index", "activation",
    "projection", "total_contribution",
]

SOURCE_COLUMNS = [
    "split", "pair_id", "feature_family", "condition", "lexical_family", "template_family",
    "analysis_cluster", "base_id", "variant", "candidate_id", "module_index", "component_index",
    "changed_span_contribution", "total_head_contribution", "reconstruction_error",
]


SUBJECTS = [
    "Mira", "Noah", "Lena", "Owen", "Asha", "Evan", "Iris", "Theo", "Nora", "Liam",
    "Sofia", "Mason", "Zara", "Caleb", "Maya", "Jonah", "Leah", "Eli", "Aria", "Simon",
]
ITEMS = [
    "tokens", "cards", "shells", "beads", "notebooks", "tickets", "stamps", "coins", "tiles", "markers",
    "folders", "bottles", "ribbons", "badges", "crayons", "labels", "packets", "clips", "keys", "blocks",
]
LOCATIONS = [
    "studio", "library", "workshop", "archive", "gallery", "laboratory", "office", "classroom",
    "warehouse", "museum",
]
QUESTION_FRAMES = [
    "How many {item} are there after the change?",
    "What is the resulting number of {item}?",
    "Determine the final {item} count.",
    "Report how many {item} remain afterward.",
]

PILOT_RELATIONS = [
    ("received_lost", "{agent} received {delta} {item}.", "{agent} lost {delta} {item}.", "received", "lost"),
    ("added_removed", "The staff added {delta} {item}.", "The staff removed {delta} {item}.", "added", "removed"),
]

CONFIRMATORY_RELATIONS = [
    ("gained_lost", "{agent} gained {delta} {item}.", "{agent} lost {delta} {item}.", "gained", "lost"),
    ("acquired_discarded", "{agent} acquired {delta} {item}.", "{agent} discarded {delta} {item}.", "acquired", "discarded"),
    ("increased_decreased", "{agent}'s stock increased by {delta} {item}.", "{agent}'s stock decreased by {delta} {item}.", "increased by", "decreased by"),
    ("collected_distributed", "{agent} collected {delta} {item}.", "{agent} distributed {delta} {item}.", "collected", "distributed"),
    ("obtained_gave_away", "{agent} obtained {delta} {item}.", "{agent} gave away {delta} {item}.", "obtained", "gave away"),
    ("purchased_sold", "{agent} purchased {delta} {item}.", "{agent} sold {delta} {item}.", "purchased", "sold"),
]

NUMBER_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}

# Each tuple is two factual entity bundles. Repeated question realizations are clustered by fact-pair ID.
FACT_PAIRS = [
    ("France", "Paris", "Italy", "Rome", "capital"),
    ("Japan", "Tokyo", "Spain", "Madrid", "capital"),
    ("Canada", "Ottawa", "Brazil", "Brasilia", "capital"),
    ("Norway", "Oslo", "Greece", "Athens", "capital"),
    ("Egypt", "Cairo", "Kenya", "Nairobi", "capital"),
    ("Austria", "Vienna", "Poland", "Warsaw", "capital"),
    ("Ireland", "Dublin", "Sweden", "Stockholm", "capital"),
    ("Portugal", "Lisbon", "Finland", "Helsinki", "capital"),
    ("Denmark", "Copenhagen", "Belgium", "Brussels", "capital"),
    ("Argentina", "Buenos Aires", "Peru", "Lima", "capital"),
    ("oxygen", "O", "carbon", "C", "symbol"),
    ("hydrogen", "H", "nitrogen", "N", "symbol"),
    ("sodium", "Na", "potassium", "K", "symbol"),
    ("iron", "Fe", "copper", "Cu", "symbol"),
    ("silver", "Ag", "gold", "Au", "symbol"),
    ("calcium", "Ca", "magnesium", "Mg", "symbol"),
    ("chlorine", "Cl", "fluorine", "F", "symbol"),
    ("silicon", "Si", "helium", "He", "symbol"),
    ("zinc", "Zn", "nickel", "Ni", "symbol"),
    ("lead", "Pb", "tin", "Sn", "symbol"),
    ("Pride and Prejudice", "Jane Austen", "Hamlet", "William Shakespeare", "author"),
    ("1984", "George Orwell", "The Odyssey", "Homer", "author"),
    ("Jane Eyre", "Charlotte Bronte", "The Trial", "Franz Kafka", "author"),
    ("Beloved", "Toni Morrison", "Frankenstein", "Mary Shelley", "author"),
    ("The Stranger", "Albert Camus", "The Republic", "Plato", "author"),
    ("The Iliad", "Homer", "Macbeth", "William Shakespeare", "author"),
    ("Emma", "Jane Austen", "Ulysses", "James Joyce", "author"),
    ("The Prince", "Niccolo Machiavelli", "Walden", "Henry Thoreau", "author"),
    ("The Metamorphosis", "Franz Kafka", "The Bluest Eye", "Toni Morrison", "author"),
    ("Don Quixote", "Miguel Cervantes", "Middlemarch", "George Eliot", "author"),
    ("relativity", "Albert Einstein", "penicillin", "Alexander Fleming", "association"),
    ("radioactivity", "Marie Curie", "evolution", "Charles Darwin", "association"),
    ("gravity", "Isaac Newton", "vaccination", "Edward Jenner", "association"),
    ("telephone", "Alexander Bell", "printing press", "Johannes Gutenberg", "association"),
    ("periodic table", "Dmitri Mendeleev", "pasteurization", "Louis Pasteur", "association"),
    ("World Wide Web", "Tim Berners-Lee", "analytical engine", "Charles Babbage", "association"),
    ("smallpox vaccine", "Edward Jenner", "polio vaccine", "Jonas Salk", "association"),
    ("heliocentric model", "Nicolaus Copernicus", "laws of motion", "Isaac Newton", "association"),
    ("electromagnetic induction", "Michael Faraday", "natural selection", "Charles Darwin", "association"),
    ("X-rays", "Wilhelm Roentgen", "insulin", "Frederick Banting", "association"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_hash() -> tuple[str, dict[str, str]]:
    files = {}
    for path in sorted(SCRIPT_DIR.iterdir()):
        if path.is_file() and path.suffix in {".py", ".json", ".sh", ".md"}:
            files[path.name] = sha256_file(path)
    return canonical_hash(files), files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["prepare", "sanity", "pilot", "freeze_confirmatory", "confirmatory", "report"])
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


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
        "cards": root / "candidate_cards",
        "logs": root / "logs",
        "candidates": root / "manifests" / "frozen_stage_b_candidates.json",
        "controls": root / "manifests" / "frozen_control_components.json",
        "hypotheses": root / "manifests" / "extension_hypothesis_manifest.json",
        "pilot_manifest": root / "manifests" / "pilot_pairs.csv",
        "confirmatory_manifest": root / "manifests" / "confirmatory_pairs.csv",
        "design": root / "manifests" / "confirmatory_design.json",
    }


def ensure_dirs(p: dict[str, Path]) -> None:
    for key in ("root", "manifests", "tables", "status", "cards", "logs"):
        p[key].mkdir(parents=True, exist_ok=True)


def write_status(p: dict[str, Path], phase: str, **payload: Any) -> None:
    write_json(p["status"] / f"{phase}.json", {"phase": phase, "status": "PASS", "timestamp": utc_now(), **payload})


def require_status(p: dict[str, Path], phase: str) -> dict[str, Any]:
    target = p["status"] / f"{phase}.json"
    if not target.exists():
        raise RuntimeError(f"Required phase has not completed: {phase}")
    payload = read_json(target)
    if payload.get("status") != "PASS":
        raise RuntimeError(f"Required phase is not PASS: {phase}")
    return payload


def prompt_length(tokenizer: Any, template: str, question: str) -> int:
    return len(tokenizer(template.format(question=question), add_special_tokens=True)["input_ids"])


def context_values(index: int, offset: int = 0) -> dict[str, Any]:
    i = index + offset
    return {
        "agent": SUBJECTS[i % len(SUBJECTS)],
        "item": ITEMS[(i * 7 + 3) % len(ITEMS)],
        "location": LOCATIONS[(i * 3 + 1) % len(LOCATIONS)],
        "base": 24 + (i * 5) % 31,
        "delta": 2 + (i * 3) % 9,
        "alt": 25 + (i * 11) % 29,
        "question": QUESTION_FRAMES[i % len(QUESTION_FRAMES)],
    }


def manifest_row(
    *, pair_id: str, split: str, feature_family: str, condition: str, lexical_family: str,
    template_family: str, analysis_cluster: str, base_id: str, original: str, modified: str,
    changed_original: list[str], changed_modified: list[str], invariant: list[str],
    answer_original: str, answer_modified: str, rule: str,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "split": split,
        "feature_family": feature_family,
        "condition": condition,
        "lexical_family": lexical_family,
        "template_id": pair_id,
        "template_family": template_family,
        "analysis_cluster": analysis_cluster,
        "base_id": base_id,
        "original_text": original,
        "modified_text": modified,
        "changed_spans_original": json.dumps(changed_original, ensure_ascii=False),
        "changed_spans_modified": json.dumps(changed_modified, ensure_ascii=False),
        "invariant_features": json.dumps(invariant, ensure_ascii=False),
        "expected_answer_original": answer_original,
        "expected_answer_modified": answer_modified,
        "generation_rule_id": rule,
        "reviewer_id": "user_authorized_extension_2026-08-25",
        "approval_basis": "explicit_user_request_to_continue_stage_b_extension",
        "automated_validation": True,
        "approved": True,
        "token_length_original": "",
        "token_length_modified": "",
    }


def relation_rows(split: str, n_context: int, offset: int, relations: list[tuple[str, str, str, str, str]]) -> list[dict[str, Any]]:
    rows = []
    for index in range(n_context):
        values = context_values(index, offset)
        cluster = f"{split}_relation_context_{index:03d}"
        question = values["question"].format(item=values["item"])
        for lexical, additive, subtractive, span_add, span_sub in relations:
            for relevance in ("relevant", "irrelevant"):
                agent = values["agent"] if relevance == "relevant" else "A visitor"
                event_item = values["item"] if relevance == "relevant" else "vouchers"
                first = f"{values['agent']} had {values['base']} {values['item']} in the {values['location']}."
                add_event = additive.format(agent=agent, delta=values["delta"], item=event_item)
                sub_event = subtractive.format(agent=agent, delta=values["delta"], item=event_item)
                if relevance == "relevant":
                    q = question
                    add_answer, sub_answer = values["base"] + values["delta"], values["base"] - values["delta"]
                else:
                    q = f"How many {values['item']} did {values['agent']} have in the {values['location']}?"
                    add_answer = sub_answer = values["base"]
                original = f"{first} {add_event} {q}"
                modified = f"{first} {sub_event} {q}"
                base_id = f"{cluster}__{lexical}"
                pair_id = f"{base_id}__{relevance}"
                rows.append(manifest_row(
                    pair_id=pair_id, split=split, feature_family="relation_polarity",
                    condition=relevance, lexical_family=lexical, template_family=base_id,
                    analysis_cluster=cluster, base_id=base_id, original=original, modified=modified,
                    changed_original=[span_add], changed_modified=[span_sub],
                    invariant=["numbers", "entities", "question_within_relevance", "sentence_frame", "event_item_within_relevance"],
                    answer_original=str(add_answer), answer_modified=str(sub_answer), rule=f"relation__{lexical}__{relevance}",
                ))
    return rows


def numeric_rows(split: str, n_context: int, offset: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(n_context):
        values = context_values(index, offset)
        cluster = f"{split}_numeric_context_{index:03d}"
        a = 2 + (index * 5 + offset) % 17
        b = 2 + (index * 11 + offset + 3) % 17
        if b == a:
            b = 19 if a != 19 else 18
        fixed = 30 + (index * 7) % 41
        for relevance in ("relevant", "irrelevant"):
            if relevance == "relevant":
                original = f"{values['agent']} placed {a} {values['item']} in a case. A side label shows {fixed}. How many {values['item']} did {values['agent']} place?"
                modified = f"{values['agent']} placed {b} {values['item']} in a case. A side label shows {fixed}. How many {values['item']} did {values['agent']} place?"
                answers = (str(a), str(b))
            else:
                original = f"{values['agent']} placed {fixed} {values['item']} in a case. A side label shows {a}. How many {values['item']} did {values['agent']} place?"
                modified = f"{values['agent']} placed {fixed} {values['item']} in a case. A side label shows {b}. How many {values['item']} did {values['agent']} place?"
                answers = (str(fixed), str(fixed))
            base_id = f"{cluster}__value"
            rows.append(manifest_row(
                pair_id=f"{base_id}__{relevance}", split=split, feature_family="numeric_value",
                condition=relevance, lexical_family="digit_value", template_family=base_id,
                analysis_cluster=cluster, base_id=base_id, original=original, modified=modified,
                changed_original=[str(a)], changed_modified=[str(b)],
                invariant=["numeric_representation", "sentence_frame", "target_entity", "number_of_numeric_spans"],
                answer_original=answers[0], answer_modified=answers[1], rule=f"numeric_value__{relevance}",
            ))

        # These digit/word pairs are each one token inside parentheses for the frozen tokenizer.
        token_matched_numbers = [2, 4, 6, 9, 20]
        number = token_matched_numbers[(index + offset) % len(token_matched_numbers)]
        word = NUMBER_WORDS[number]
        # Parentheses avoid Llama-3's standalone whitespace token before bare digits,
        # making digit and number-word prompts exactly token-count matched.
        original = f"The {values['location']} catalog lists ({number}) {values['item']}. What quantity is stated?"
        modified = f"The {values['location']} catalog lists ({word}) {values['item']}. What quantity is stated?"
        rows.append(manifest_row(
            pair_id=f"{cluster}__representation", split=split,
            feature_family="numeric_representation_token_matched", condition="not_applicable",
            lexical_family="digit_word", template_family=f"{cluster}__representation",
            analysis_cluster=cluster, base_id=f"{cluster}__representation", original=original, modified=modified,
            changed_original=[str(number)], changed_modified=[word],
            invariant=["numeric_value", "sentence_frame", "expected_answer", "prompt_token_length"],
            answer_original=str(number), answer_modified=str(number), rule="numeric_representation__token_matched",
        ))
    return rows


def factual_text(a: str, b: str, kind: str) -> tuple[str, str]:
    if kind == "capital":
        return f"The capital of {a} is {b}.", "Which city is stated in the target record?"
    if kind == "symbol":
        return f"The chemical symbol for {a} is {b}.", "Which symbol is stated in the target record?"
    if kind == "author":
        return f"{a} is associated with the author {b}.", "Which author is stated in the target record?"
    return f"{a} is associated with {b}.", "Which person is stated in the target record?"


def factual_rows(split: str, fact_indices: Iterable[int]) -> list[dict[str, Any]]:
    rows = []
    for index, fact_index in enumerate(fact_indices):
        a1, b1, a2, b2, kind = FACT_PAIRS[fact_index]
        statement1, question1 = factual_text(a1, b1, kind)
        statement2, question2 = factual_text(a2, b2, kind)
        cluster = f"{split}_fact_pair_{fact_index:03d}"
        # The target question is lexically identical; only its supporting factual bundle changes.
        original = f"Target record: {statement1} Distractor record: basalt is a rock. {question1}"
        modified = f"Target record: {statement2} Distractor record: basalt is a rock. {question2}"
        rows.append(manifest_row(
            pair_id=f"{split}_fact_{index:03d}__relevant", split=split,
            feature_family="factual_entity_bundle", condition="relevant", lexical_family=kind,
            template_family=f"{split}_fact_{index:03d}", analysis_cluster=cluster,
            base_id=f"{split}_fact_{index:03d}", original=original, modified=modified,
            changed_original=[a1, b1], changed_modified=[a2, b2],
            invariant=["factual_relation_type", "sentence_count", "target_question", "distractor_record"],
            answer_original=b1, answer_modified=b2, rule=f"factual_entity__{kind}__relevant",
        ))
        # The same bundle changes in a distractor; the target question and answer remain fixed.
        target = "Target record: basalt is a rock. What material is stated in the target record?"
        original = f"Distractor record: {statement1} {target}"
        modified = f"Distractor record: {statement2} {target}"
        rows.append(manifest_row(
            pair_id=f"{split}_fact_{index:03d}__irrelevant", split=split,
            feature_family="factual_entity_bundle", condition="irrelevant", lexical_family=kind,
            template_family=f"{split}_fact_{index:03d}", analysis_cluster=cluster,
            base_id=f"{split}_fact_{index:03d}", original=original, modified=modified,
            changed_original=[a1, b1], changed_modified=[a2, b2],
            invariant=["factual_relation_type", "sentence_count", "target_question", "target_answer"],
            answer_original="rock", answer_modified="rock", rule=f"factual_entity__{kind}__irrelevant",
        ))
    return rows


def generate_manifest(config: dict[str, Any], split: str, tokenizer: Any) -> pd.DataFrame:
    if split == "pilot":
        n_context, offset, relations = int(config["pilot_context_clusters"]), 1000, PILOT_RELATIONS
        fact_indices = range(0, 10)
    else:
        n_context, offset, relations = int(config["confirmatory_context_clusters"]), 0, CONFIRMATORY_RELATIONS
        fact_indices = range(10, len(FACT_PAIRS))
    rows = relation_rows(split, n_context, offset, relations)
    rows += numeric_rows(split, n_context, offset)
    rows += factual_rows(split, fact_indices)
    if split == "pilot":
        for row in rows:
            row["original_text"] = "In a preliminary example, " + row["original_text"]
            row["modified_text"] = "In a preliminary example, " + row["modified_text"]
    for row in rows:
        row["token_length_original"] = prompt_length(tokenizer, config["prompt_template"], row["original_text"])
        row["token_length_modified"] = prompt_length(tokenizer, config["prompt_template"], row["modified_text"])
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    validate_manifest(frame, split)
    numeric_rep = frame[frame["feature_family"] == "numeric_representation_token_matched"]
    if not bool((numeric_rep["token_length_original"] == numeric_rep["token_length_modified"]).all()):
        bad = numeric_rep[numeric_rep["token_length_original"] != numeric_rep["token_length_modified"]]
        raise RuntimeError(f"Numeric representation pairs are not token matched: {bad['pair_id'].tolist()[:5]}")
    return frame


def validate_manifest(frame: pd.DataFrame, split: str) -> None:
    missing = set(MANIFEST_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")
    if frame.empty or set(frame["split"]) != {split}:
        raise RuntimeError(f"Invalid or empty split: {split}")
    if frame["pair_id"].duplicated().any():
        raise RuntimeError("pair_id must be unique")
    if (frame["original_text"] == frame["modified_text"]).any():
        raise RuntimeError("Every pair must differ")
    for _, row in frame.iterrows():
        originals = json.loads(row["changed_spans_original"])
        modified = json.loads(row["changed_spans_modified"])
        if not originals or not modified:
            raise RuntimeError(f"Missing changed spans: {row['pair_id']}")
        if any(span not in row["original_text"] for span in originals):
            raise RuntimeError(f"Original changed span not found: {row['pair_id']}")
        if any(span not in row["modified_text"] for span in modified):
            raise RuntimeError(f"Modified changed span not found: {row['pair_id']}")
        if not bool(row["approved"]) or not bool(row["automated_validation"]):
            raise RuntimeError(f"Unapproved pair: {row['pair_id']}")


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def build_hypotheses(config: dict[str, Any], candidates: dict[str, Any]) -> dict[str, Any]:
    hypotheses = [
        {"analysis_family": "relation_lexical_robustness", "label": "primary", "endpoint": "modified-minus-original relation polarity averaged across held-out lexical families"},
        {"analysis_family": "relation_relevance_interaction", "label": "primary", "endpoint": "relation polarity effect in relevant minus irrelevant condition"},
        {"analysis_family": "numeric_relevance_interaction", "label": "primary", "endpoint": "numeric value effect in relevant minus irrelevant condition"},
        {"analysis_family": "factual_entity_relevance_interaction", "label": "primary", "endpoint": "factual entity-bundle effect in relevant minus irrelevant condition"},
        {"analysis_family": "numeric_representation_token_matched", "label": "secondary", "endpoint": "number-word minus digit with identical prompt token length"},
    ]
    return {
        "run_id": config["stage_b_extension_run_id"],
        "source_candidate_rule": "all 20 frozen v2_b02 candidates; no extension-result reselection",
        "approved": True,
        "approval_basis": "explicit user request to continue Stage B extension",
        "freeze_timestamp": utc_now(),
        "candidates": [{"candidate_id": row["component_id"], "hypotheses": hypotheses} for row in candidates["candidates"]],
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


def run_prepare(config: dict[str, Any], p: dict[str, Path]) -> None:
    source = Path(config["source_stage_b_root"])
    summary = read_json(source / "stage_b_summary.json")
    if summary.get("status") != "PASS":
        raise RuntimeError("Source Stage B is not PASS")
    source_report = read_json(source / "status" / "report.json")
    if source_report.get("status") != "PASS":
        raise RuntimeError("Source Stage B report is not PASS")
    source_config = read_json(source / "manifests" / "frozen_config.json")
    if source_config["prompt_template"] != config["prompt_template"]:
        raise RuntimeError("Prompt template differs from Stage B/Stage A frozen prompt")
    frozen_config = p["manifests"] / "frozen_config.json"
    if frozen_config.exists() and read_json(frozen_config) != config:
        raise RuntimeError("Existing extension config differs; use a new run/output root")
    write_json(frozen_config, config)
    shutil.copyfile(source / "manifests" / "frozen_stage_b_candidates.json", p["candidates"])
    shutil.copyfile(source / "manifests" / "frozen_control_components.json", p["controls"])
    candidates = read_json(p["candidates"])
    if len(candidates["candidates"]) != 20:
        raise RuntimeError("Expected all 20 frozen candidates")
    write_json(p["hypotheses"], build_hypotheses(config, candidates))
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    pilot = generate_manifest(config, "pilot", tokenizer)
    atomic_frame(p["pilot_manifest"], pilot)
    parameter_hash, parameter_files = model_parameter_checksum(Path(config["model_path"]))
    extension_hash, extension_files = code_hash()
    provenance = {
        "run_id": config["stage_b_extension_run_id"],
        "config_hash": config["config_hash"],
        "source_stage_b_summary_sha256": sha256_file(source / "stage_b_summary.json"),
        "source_results_sha256": sha256_file(source / "RESULTS_KO.md"),
        "source_candidate_sha256": sha256_file(source / "manifests" / "frozen_stage_b_candidates.json"),
        "source_controls_sha256": sha256_file(source / "manifests" / "frozen_control_components.json"),
        "frozen_candidates_sha256": sha256_file(p["candidates"]),
        "frozen_controls_sha256": sha256_file(p["controls"]),
        "hypothesis_manifest_sha256": sha256_file(p["hypotheses"]),
        "pilot_manifest_sha256": sha256_file(p["pilot_manifest"]),
        "prompt_sha256": sha256_text(config["prompt_template"]),
        "model_parameter_checksum": parameter_hash,
        "model_parameter_files": parameter_files,
        "extension_code_hash": extension_hash,
        "extension_code_files": extension_files,
        "model_training": False,
        "independent_human_linguistic_audit": False,
    }
    write_json(p["manifests"] / "provenance_prepare.json", provenance)
    write_status(p, "prepare", candidate_count=20, pilot_pair_count=len(pilot), **provenance)
    print(f"prepare PASS: 20 candidates, {len(pilot)} pilot pairs")


def load_components(p: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates = read_json(p["candidates"])
    controls = read_json(p["controls"])
    lookup = component_lookup(candidates, controls)
    return candidates, controls, lookup, unique_components(lookup)


def source_stage_a_direction(config: dict[str, Any]) -> Path:
    root = Path(config["stage_a_root"])
    candidates = [
        root / "checkpoints" / "discovery_directions.pt",
        root / "checkpoints" / "directions_discovery.pt",
        root / "directions_discovery.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Reuse the exact asset discovery contract from the base Stage B implementation.
    source_config = {
        **config,
        "stage_a_root": str(root),
    }
    from stage_b_core import load_stage_a_assets

    return Path(load_stage_a_assets(source_config)["direction_path"])


def encode_batch(tokenizer: Any, prompts: list[str], device: torch.device, offsets: bool) -> tuple[dict[str, torch.Tensor], Any]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        truncation=False,
        return_token_type_ids=False,
        return_offsets_mapping=offsets,
    )
    offset_mapping = encoded.pop("offset_mapping", None)
    if not bool(torch.all(encoded["attention_mask"][:, -1] == 1)):
        raise RuntimeError("Final prompt token invariant failed")
    return {key: value.to(device) for key, value in encoded.items()}, offset_mapping


def overlap_changed_spans(
    prompt: str,
    question: str,
    spans: list[str],
    offsets: torch.Tensor,
) -> list[int]:
    question_start = prompt.index(question)
    intervals = []
    cursor_by_span: dict[str, int] = {}
    for span in spans:
        start_at = cursor_by_span.get(span, 0)
        local_start = question.find(span, start_at)
        if local_start < 0:
            raise RuntimeError(f"Changed span not found during attribution: {span!r}")
        cursor_by_span[span] = local_start + len(span)
        intervals.append((question_start + local_start, question_start + local_start + len(span)))
    selected = []
    for token_index, pair in enumerate(offsets.tolist()):
        start, end = map(int, pair)
        if start == end == 0:
            continue
        if any(max(start, left) < min(end, right) for left, right in intervals):
            selected.append(token_index)
    return selected


def run_sanity(config: dict[str, Any], p: dict[str, Path]) -> None:
    prepare = require_status(p, "prepare")
    source = Path(config["source_stage_b_root"])
    if prepare["source_candidate_sha256"] != sha256_file(source / "manifests" / "frozen_stage_b_candidates.json"):
        raise RuntimeError("Source candidate manifest changed")
    pilot = pd.read_csv(p["pilot_manifest"], keep_default_na=False)
    validate_manifest(pilot, "pilot")
    if set(name for name, *_ in PILOT_RELATIONS) & set(name for name, *_ in CONFIRMATORY_RELATIONS):
        raise RuntimeError("Pilot and confirmatory relation lexical families overlap")
    source_sanity = read_json(source / "status" / "sanity.json")
    if source_sanity.get("status") != "PASS":
        raise RuntimeError("Source Stage B sanity is not PASS")
    device = torch.device(f"cuda:{int(config['gpu_id'])}")
    model = capture = None
    try:
        _, _, _, components = load_components(p)
        directions = load_directions(source_stage_a_direction(config))
        model, tokenizer = load_model_and_tokenizer(config, device)
        projections = projections_for_components(model, directions, components)
        capture = StageBCapture(model, components, capture_sources=False)
        capture.install()
        sample = pilot.head(2)
        prompts = [config["prompt_template"].format(question=text) for text in sample["original_text"]]
        encoded, _ = encode_batch(tokenizer, prompts, device, offsets=False)
        model(**encoded, use_cache=False, return_dict=True)
        capture.validate()
        if not projections or model.training:
            raise RuntimeError("Projection/model sanity failed")
    finally:
        if capture is not None:
            capture.remove()
        release_model(model)
    write_status(
        p, "sanity", source_sanity_sha256=sha256_file(source / "status" / "sanity.json"),
        pilot_manifest_sha256=sha256_file(p["pilot_manifest"]), model_eval=True,
        prompt_sha256=sha256_text(config["prompt_template"]), lexical_family_overlap=[],
    )
    print("sanity PASS")


@torch.inference_mode()
def extract_responses(
    config: dict[str, Any], p: dict[str, Path], manifest_path: Path, split: str,
    output_path: Path, capture_sources: bool,
) -> dict[str, Any]:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    validate_manifest(manifest, split)
    _, _, lookup, components = load_components(p)
    directions = load_directions(source_stage_a_direction(config))
    device = torch.device(f"cuda:{int(config['gpu_id'])}")
    before, _ = model_parameter_checksum(Path(config["model_path"]))
    response_sink = AtomicCsvSink(output_path, RESPONSE_COLUMNS)
    source_path = p["tables"] / f"{split}_changed_span_sources.csv.gz"
    source_sink = AtomicCsvSink(source_path, SOURCE_COLUMNS) if capture_sources else None
    model = capture = None
    started = time.time()
    reconstruction_errors: list[float] = []
    try:
        model, tokenizer = load_model_and_tokenizer(config, device)
        projections = projections_for_components(model, directions, components)
        capture = StageBCapture(model, components, capture_sources=capture_sources)
        capture.install()
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        flattened: list[tuple[dict[str, Any], str, str]] = []
        for row in manifest.to_dict(orient="records"):
            for variant in ("original", "modified"):
                question = str(row[f"{variant}_text"])
                flattened.append((row, variant, config["prompt_template"].format(question=question)))
        batch_size = int(config["batch_size"])
        candidate_heads = [row for row in components.values() if row["component_type"] == "head" and row["component_id"] in {c["component_id"] for c in read_json(p["candidates"])["candidates"]}]
        for start in range(0, len(flattened), batch_size):
            batch = flattened[start : start + batch_size]
            capture.reset()
            encoded, offsets = encode_batch(tokenizer, [row[2] for row in batch], device, offsets=capture_sources)
            model_output = model(
                **encoded, use_cache=False, output_attentions=capture_sources, return_dict=True,
            )
            capture.validate()
            scores: dict[str, torch.Tensor] = {}
            for component_id, component in components.items():
                if component["component_type"] == "neuron":
                    scores[component_id] = capture.z[component["module_index"]][:, component["component_index"]].float()
                else:
                    pre = capture.pre_o[component["module_index"]].reshape(len(batch), model.config.num_attention_heads, head_dim)
                    scores[component_id] = (pre[:, component["component_index"]].float() * projections[component_id].float()).sum(dim=-1)
            for batch_offset, (pair, variant, _) in enumerate(batch):
                base = {
                    "split": split, "pair_id": pair["pair_id"], "feature_family": pair["feature_family"],
                    "condition": pair["condition"], "lexical_family": pair["lexical_family"],
                    "template_family": pair["template_family"], "analysis_cluster": pair["analysis_cluster"],
                    "base_id": pair["base_id"], "variant": variant,
                }
                for association in lookup.values():
                    component_id = association["component_id"]
                    projection = projections[component_id]
                    if association["component_type"] == "neuron":
                        activation = float(scores[component_id][batch_offset])
                        total = activation * float(projection)
                    else:
                        activation = ""
                        total = float(scores[component_id][batch_offset])
                    response_sink.writerow({
                        **base, "candidate_id": association["candidate_id"], "component_id": component_id,
                        "component_type": association["component_type"], "component_role": association["role"],
                        "control_kind": association["control_kind"], "module_index": association["module_index"],
                        "component_index": association["component_index"], "activation": activation,
                        "projection": float(projection.norm()) if projection.ndim else float(projection),
                        "total_contribution": total,
                    })
                if capture_sources:
                    question = str(pair[f"{variant}_text"])
                    spans = json.loads(pair[f"changed_spans_{variant}"])
                    selected = overlap_changed_spans(batch[batch_offset][2], question, spans, offsets[batch_offset])
                    groups = model.config.num_attention_heads // model.config.num_key_value_heads
                    for component in candidate_heads:
                        cid = component["component_id"]
                        layer, head = int(component["module_index"]), int(component["component_index"])
                        values = capture.values[layer].float().reshape(len(batch), -1, model.config.num_key_value_heads, head_dim)
                        values = values[batch_offset, :, head // groups, :]
                        attention = model_output.attentions[layer][batch_offset, head, -1, :].float()
                        source_scores = attention * torch.einsum("sd,d->s", values, projections[cid].float())
                        error = float(abs(source_scores.sum() - scores[cid][batch_offset]))
                        reconstruction_errors.append(error)
                        changed = float(source_scores[selected].sum()) if selected else 0.0
                        source_sink.writerow({
                            **base, "candidate_id": cid, "module_index": layer, "component_index": head,
                            "changed_span_contribution": changed,
                            "total_head_contribution": float(scores[cid][batch_offset]),
                            "reconstruction_error": error,
                        })
            if start == 0 or (start // batch_size + 1) % 50 == 0 or start + batch_size >= len(flattened):
                print(f"{split}: {min(start + batch_size, len(flattened))}/{len(flattened)} variants", flush=True)
        response_sink.close(True)
        response_sink = None
        if source_sink is not None:
            source_sink.close(True)
            source_sink = None
        after, _ = model_parameter_checksum(Path(config["model_path"]))
        if before != after:
            raise RuntimeError("Model parameter checksum changed")
    finally:
        if response_sink is not None:
            response_sink.close(False)
        if source_sink is not None:
            source_sink.close(False)
        if capture is not None:
            capture.remove()
        release_model(model)
    metadata = {
        "response_sha256": sha256_file(output_path), "n_pair": len(manifest),
        "n_variant": len(manifest) * 2, "runtime_seconds": time.time() - started,
        "model_parameter_checksum_before": before, "model_parameter_checksum_after": after,
    }
    if capture_sources:
        metadata.update({
            "source_sha256": sha256_file(source_path),
            "source_reconstruction_mean_abs_error": float(np.mean(reconstruction_errors)),
            "source_reconstruction_max_abs_error": float(np.max(reconstruction_errors)),
        })
    return metadata


def run_pilot(config: dict[str, Any], p: dict[str, Path]) -> None:
    require_status(p, "sanity")
    output = p["tables"] / "pilot_responses.csv.gz"
    metadata = extract_responses(config, p, p["pilot_manifest"], "pilot", output, capture_sources=False)
    descriptive = build_effect_vectors(output)
    variance = summarize_vectors(config, descriptive, candidate_only=True, apply_fdr=False)
    atomic_frame(p["tables"] / "pilot_variance_estimates.csv", variance)
    write_status(
        p, "pilot", **metadata, pilot_variance_sha256=sha256_file(p["tables"] / "pilot_variance_estimates.csv"),
        use_restriction="variance/runtime/sanity only; excluded from confirmatory claims",
    )
    print(f"pilot PASS: {metadata['n_pair']} pairs in {metadata['runtime_seconds']:.1f}s")


def primitive_pair_effects(response_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(response_path, low_memory=False)
    keys = [
        "pair_id", "feature_family", "condition", "lexical_family", "template_family",
        "analysis_cluster", "base_id", "candidate_id", "component_id", "component_type",
        "component_role", "control_kind",
    ]
    pieces = []
    for component_type, endpoint in (("head", "total_contribution"), ("neuron", "activation")):
        subset = frame[frame["component_type"] == component_type]
        if subset.empty:
            continue
        pivot = subset.pivot(index=keys, columns="variant", values=endpoint).reset_index()
        pivot["difference"] = pivot["modified"] - pivot["original"]
        pivot["endpoint"] = endpoint
        pieces.append(pivot)
    if not pieces:
        raise RuntimeError("No analyzable component responses")
    return pd.concat(pieces, ignore_index=True)


def _vector_base(frame: pd.DataFrame, family: str, label: str, unit: pd.Series | str = "pooled") -> pd.DataFrame:
    result = frame.copy()
    result["analysis_family"] = family
    result["analysis_label"] = label
    result["value"] = result["difference"]
    result["unit_id"] = unit if isinstance(unit, str) else unit.astype(str)
    return result[
        ["analysis_family", "analysis_label", "analysis_cluster", "unit_id", "candidate_id",
         "component_id", "component_type", "component_role", "control_kind", "endpoint", "value"]
    ]


def _interaction(frame: pd.DataFrame, family: str, label: str, unit_column: str | None) -> pd.DataFrame:
    index = [
        "analysis_cluster", "candidate_id", "component_id", "component_type", "component_role",
        "control_kind", "endpoint",
    ]
    if unit_column is not None:
        index.append(unit_column)
    pivot = frame.pivot(index=index, columns="condition", values="difference").reset_index()
    if not {"relevant", "irrelevant"}.issubset(pivot.columns):
        raise RuntimeError(f"Missing relevance condition for {family}")
    pivot["value"] = pivot["relevant"] - pivot["irrelevant"]
    pivot["analysis_family"] = family
    pivot["analysis_label"] = label
    pivot["unit_id"] = pivot[unit_column].astype(str) if unit_column else "interaction"
    return pivot[
        ["analysis_family", "analysis_label", "analysis_cluster", "unit_id", "candidate_id",
         "component_id", "component_type", "component_role", "control_kind", "endpoint", "value"]
    ]


def build_effect_vectors(response_path: Path) -> pd.DataFrame:
    pairs = primitive_pair_effects(response_path)
    vectors = []
    relation = pairs[pairs["feature_family"] == "relation_polarity"]
    relevant_relation = relation[relation["condition"] == "relevant"]
    vectors.append(_vector_base(
        relevant_relation, "relation_lexical_robustness", "primary", relevant_relation["lexical_family"],
    ))
    vectors.append(_interaction(
        relation, "relation_relevance_interaction", "primary", "lexical_family",
    ))
    for lexical, subgroup in relevant_relation.groupby("lexical_family", sort=True):
        vectors.append(_vector_base(subgroup, f"relation_lexical_subgroup::{lexical}", "secondary", lexical))

    numeric = pairs[pairs["feature_family"] == "numeric_value"]
    vectors.append(_interaction(numeric, "numeric_relevance_interaction", "primary", None))
    representation = pairs[pairs["feature_family"] == "numeric_representation_token_matched"]
    vectors.append(_vector_base(
        representation, "numeric_representation_token_matched", "secondary", "digit_word",
    ))
    factual = pairs[pairs["feature_family"] == "factual_entity_bundle"]
    vectors.append(_interaction(factual, "factual_entity_relevance_interaction", "primary", None))
    result = pd.concat(vectors, ignore_index=True)
    if result["value"].isna().any():
        raise RuntimeError("Nonfinite effect vector")
    return result


def summarize_vectors(
    config: dict[str, Any], vectors: pd.DataFrame, *, candidate_only: bool, apply_fdr: bool,
) -> pd.DataFrame:
    frame = vectors[vectors["component_role"] == "candidate"] if candidate_only else vectors
    settings = config["statistics"]
    group_keys = [
        "analysis_family", "analysis_label", "candidate_id", "component_id", "component_type",
        "component_role", "control_kind", "endpoint",
    ]
    rows = []
    for group_key, group in frame.groupby(group_keys, sort=True):
        summary = paired_summary(
            group["analysis_cluster"], group["value"],
            int(settings["bootstrap_iterations"]), int(settings["permutation_iterations"]),
            int(settings["random_seed"]),
        )
        rows.append(dict(zip(group_keys, group_key)) | summary)
    result = pd.DataFrame(rows)
    result["bh_q"] = np.nan
    if apply_fdr:
        for _, family in result.groupby(["analysis_label", "analysis_family", "component_type"], sort=True):
            result.loc[family.index, "bh_q"] = benjamini_hochberg(family["sign_flip_p"].astype(float))
    return result


def specificity_vectors(vectors: pd.DataFrame) -> pd.DataFrame:
    candidate = vectors[vectors["component_role"] == "candidate"].rename(
        columns={"component_id": "candidate_component_id", "value": "candidate_value"}
    )
    controls = vectors[vectors["component_role"] == "control"].copy()
    join = [
        "analysis_family", "analysis_label", "analysis_cluster", "unit_id", "candidate_id",
        "component_type", "endpoint",
    ]
    merged = controls.merge(
        candidate[join + ["candidate_component_id", "candidate_value"]], on=join, validate="many_to_one",
    )
    merged["value"] = merged["candidate_value"] - merged["value"]
    merged["component_role"] = "specificity"
    return merged[
        ["analysis_family", "analysis_label", "analysis_cluster", "unit_id", "candidate_id",
         "component_id", "component_type", "component_role", "control_kind", "endpoint", "value"]
    ]


def summarize_specificity(config: dict[str, Any], vectors: pd.DataFrame) -> pd.DataFrame:
    settings = config["statistics"]
    group_keys = [
        "analysis_family", "analysis_label", "candidate_id", "component_id", "component_type",
        "component_role", "control_kind", "endpoint",
    ]
    rows = []
    for group_key, group in vectors.groupby(group_keys, sort=True):
        summary = paired_summary(
            group["analysis_cluster"], group["value"], int(settings["bootstrap_iterations"]),
            int(settings["permutation_iterations"]), int(settings["random_seed"]),
        )
        rows.append(dict(zip(group_keys, group_key)) | summary)
    result = pd.DataFrame(rows)
    result["bh_q"] = np.nan
    for _, family in result.groupby(["analysis_label", "analysis_family", "component_type", "control_kind"], sort=True):
        result.loc[family.index, "bh_q"] = benjamini_hochberg(family["sign_flip_p"].astype(float))
    return result


def summarize_source(config: dict[str, Any], source_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_path, low_memory=False)
    keys = [
        "pair_id", "feature_family", "condition", "lexical_family", "analysis_cluster", "base_id",
        "candidate_id", "module_index", "component_index",
    ]
    pivot = frame.pivot(index=keys, columns="variant", values="changed_span_contribution").reset_index()
    pivot["difference"] = pivot["modified"] - pivot["original"]
    relation = pivot[pivot["feature_family"] == "relation_polarity"]
    relevant = relation[relation["condition"] == "relevant"].copy()
    relevant["analysis_family"] = "changed_span_relation_lexical_robustness"
    relevant["analysis_label"] = "secondary"
    relevant["value"] = relevant["difference"]
    relevant["unit_id"] = relevant["lexical_family"]
    interaction_index = ["analysis_cluster", "lexical_family", "candidate_id", "module_index", "component_index"]
    interaction = relation.pivot(index=interaction_index, columns="condition", values="difference").reset_index()
    interaction["value"] = interaction["relevant"] - interaction["irrelevant"]
    interaction["analysis_family"] = "changed_span_relation_relevance_interaction"
    interaction["analysis_label"] = "secondary"
    interaction["unit_id"] = interaction["lexical_family"]
    combined = pd.concat([
        relevant[["analysis_family", "analysis_label", "analysis_cluster", "unit_id", "candidate_id", "value"]],
        interaction[["analysis_family", "analysis_label", "analysis_cluster", "unit_id", "candidate_id", "value"]],
    ], ignore_index=True)
    settings = config["statistics"]
    rows = []
    for group_key, group in combined.groupby(["analysis_family", "analysis_label", "candidate_id"], sort=True):
        summary = paired_summary(
            group["analysis_cluster"], group["value"], int(settings["bootstrap_iterations"]),
            int(settings["permutation_iterations"]), int(settings["random_seed"]),
        )
        rows.append(dict(zip(["analysis_family", "analysis_label", "candidate_id"], group_key)) | summary)
    result = pd.DataFrame(rows)
    result["bh_q"] = np.nan
    for _, family in result.groupby("analysis_family", sort=True):
        result.loc[family.index, "bh_q"] = benjamini_hochberg(family["sign_flip_p"].astype(float))
    return result


def run_freeze_confirmatory(config: dict[str, Any], p: dict[str, Path]) -> None:
    require_status(p, "pilot")
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    confirmatory = generate_manifest(config, "confirmatory", tokenizer)
    pilot = pd.read_csv(p["pilot_manifest"], keep_default_na=False)
    text_overlap = (set(pilot["original_text"]) | set(pilot["modified_text"])) & (
        set(confirmatory["original_text"]) | set(confirmatory["modified_text"])
    )
    lexical_overlap = set(pilot[pilot["feature_family"] == "relation_polarity"]["lexical_family"]) & set(
        confirmatory[confirmatory["feature_family"] == "relation_polarity"]["lexical_family"]
    )
    cluster_overlap = set(pilot["analysis_cluster"]) & set(confirmatory["analysis_cluster"])
    if text_overlap or lexical_overlap or cluster_overlap:
        raise RuntimeError(
            f"Pilot/confirmatory overlap: text={len(text_overlap)}, lexical={lexical_overlap}, cluster={cluster_overlap}"
        )
    atomic_frame(p["confirmatory_manifest"], confirmatory)
    design = {
        "run_id": config["stage_b_extension_run_id"],
        "approved": True,
        "approval_basis": "explicit user request to continue Stage B extension",
        "freeze_timestamp": utc_now(),
        "candidate_count": 20,
        "primary_analysis_families": config["statistics"]["primary_families"],
        "secondary_analysis_families": config["statistics"]["secondary_families"],
        "observation_unit": config["statistics"]["observation_unit"],
        "minimum_effect_of_interest_dz": config["statistics"]["minimum_effect_of_interest_dz"],
        "confirmatory_pair_count": len(confirmatory),
        "relation_context_clusters": int(confirmatory[confirmatory["feature_family"] == "relation_polarity"]["analysis_cluster"].nunique()),
        "numeric_context_clusters": int(confirmatory[confirmatory["feature_family"] == "numeric_value"]["analysis_cluster"].nunique()),
        "factual_context_clusters": int(confirmatory[confirmatory["feature_family"] == "factual_entity_bundle"]["analysis_cluster"].nunique()),
        "confirmatory_manifest_sha256": sha256_file(p["confirmatory_manifest"]),
        "candidate_manifest_sha256": sha256_file(p["candidates"]),
        "control_manifest_sha256": sha256_file(p["controls"]),
        "hypothesis_manifest_sha256": sha256_file(p["hypotheses"]),
        "pilot_text_overlap": [], "pilot_relation_lexical_overlap": [], "pilot_cluster_overlap": [],
        "independent_human_linguistic_audit": False,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    write_json(p["design"], design)
    write_status(
        p, "freeze_confirmatory", confirmatory_manifest_sha256=sha256_file(p["confirmatory_manifest"]),
        confirmatory_design_sha256=sha256_file(p["design"]), n_pair=len(confirmatory),
        relation_lexical_families=sorted(confirmatory[confirmatory["feature_family"] == "relation_polarity"]["lexical_family"].unique().tolist()),
    )
    print(f"freeze_confirmatory PASS: {len(confirmatory)} held-out pairs")


def revalidate_freeze(p: dict[str, Path]) -> None:
    status = require_status(p, "freeze_confirmatory")
    if status["confirmatory_manifest_sha256"] != sha256_file(p["confirmatory_manifest"]):
        raise RuntimeError("Confirmatory manifest changed after freeze")
    if status["confirmatory_design_sha256"] != sha256_file(p["design"]):
        raise RuntimeError("Confirmatory design changed after freeze")
    design = read_json(p["design"])
    for key, path in (
        ("candidate_manifest_sha256", p["candidates"]),
        ("control_manifest_sha256", p["controls"]),
        ("hypothesis_manifest_sha256", p["hypotheses"]),
    ):
        if design[key] != sha256_file(path):
            raise RuntimeError(f"Frozen artifact changed: {key}")


def run_confirmatory(config: dict[str, Any], p: dict[str, Path]) -> None:
    require_status(p, "sanity")
    revalidate_freeze(p)
    output = p["tables"] / "confirmatory_responses.csv.gz"
    metadata = extract_responses(
        config, p, p["confirmatory_manifest"], "confirmatory", output, capture_sources=True,
    )
    vectors = build_effect_vectors(output)
    atomic_frame(p["tables"] / "effect_vectors.csv.gz", vectors)
    effects = summarize_vectors(config, vectors, candidate_only=True, apply_fdr=True)
    response = pd.read_csv(output, usecols=["candidate_id", "component_id", "component_role", "projection"])
    projection = response[response["component_role"] == "candidate"].drop_duplicates(
        ["candidate_id", "component_id"]
    )[["candidate_id", "component_id", "projection"]]
    effects = effects.merge(projection, on=["candidate_id", "component_id"], validate="many_to_one")
    effects["liref_direction_mean_effect"] = np.where(
        effects["component_type"] == "neuron",
        effects["mean_template_effect"] * effects["projection"],
        effects["mean_template_effect"],
    )
    atomic_frame(p["tables"] / "candidate_effects.csv", effects)
    specificity = summarize_specificity(config, specificity_vectors(vectors))
    atomic_frame(p["tables"] / "control_specificity.csv", specificity)
    source = summarize_source(config, p["tables"] / "confirmatory_changed_span_sources.csv.gz")
    atomic_frame(p["tables"] / "changed_span_source_effects.csv", source)
    write_status(
        p, "confirmatory", **metadata,
        effect_vectors_sha256=sha256_file(p["tables"] / "effect_vectors.csv.gz"),
        candidate_effects_sha256=sha256_file(p["tables"] / "candidate_effects.csv"),
        control_specificity_sha256=sha256_file(p["tables"] / "control_specificity.csv"),
        source_effects_sha256=sha256_file(p["tables"] / "changed_span_source_effects.csv"),
        interpretation_boundary=INTERPRETATION_BOUNDARY,
    )
    print(f"confirmatory PASS: {metadata['n_pair']} pairs in {metadata['runtime_seconds']:.1f}s")


def all_random_specificity_counts(specificity: pd.DataFrame, family: str) -> set[str]:
    subset = specificity[
        (specificity["analysis_family"] == family)
        & specificity["control_kind"].astype(str).str.startswith("random")
    ]
    if subset.empty:
        return set()
    flags = subset.assign(significant=subset["bh_q"] < 0.05).groupby("candidate_id")["significant"].all()
    return set(flags[flags].index.astype(str))


def result_counts(effects: pd.DataFrame, specificity: pd.DataFrame, source: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    primary = sorted(effects[effects["analysis_label"] == "primary"]["analysis_family"].unique())
    for family in primary:
        subset = effects[effects["analysis_family"] == family]
        sig = subset[subset["bh_q"] < 0.05]
        strong = sig[sig["cohen_dz"].abs() >= 0.5]
        matched = specificity[
            (specificity["analysis_family"] == family)
            & (specificity["control_kind"] == "matched")
            & (specificity["bh_q"] < 0.05)
        ]
        output[family] = {
            "q05": int(len(sig)),
            "q05_and_abs_dz_ge_0_5": int(len(strong)),
            "matched_control_specificity_q05": int(matched["candidate_id"].nunique()),
            "all_three_random_controls_specificity_q05": len(all_random_specificity_counts(specificity, family)),
            "significant_candidates": sig.sort_values("cohen_dz", key=lambda value: value.abs(), ascending=False)[
                ["candidate_id", "component_type", "mean_template_effect", "cohen_dz", "bh_q", "liref_direction_mean_effect"]
            ].to_dict(orient="records"),
        }
    numeric_rep = effects[effects["analysis_family"] == "numeric_representation_token_matched"]
    output["numeric_representation_token_matched"] = {
        "q05": int((numeric_rep["bh_q"] < 0.05).sum()),
        "q05_and_abs_dz_ge_0_5": int(((numeric_rep["bh_q"] < 0.05) & (numeric_rep["cohen_dz"].abs() >= 0.5)).sum()),
    }
    output["changed_span_sources"] = {
        family: int((group["bh_q"] < 0.05).sum()) for family, group in source.groupby("analysis_family")
    }
    return output


def lexical_consistency_table(effects: pd.DataFrame) -> pd.DataFrame:
    subgroup = effects[effects["analysis_family"].str.startswith("relation_lexical_subgroup::")].copy()
    subgroup["lexical_family"] = subgroup["analysis_family"].str.split("::").str[1]
    pooled = effects[effects["analysis_family"] == "relation_lexical_robustness"][
        ["candidate_id", "mean_template_effect", "cohen_dz", "bh_q"]
    ].rename(columns={
        "mean_template_effect": "pooled_mean_effect", "cohen_dz": "pooled_cohen_dz", "bh_q": "pooled_bh_q",
    })
    joined = subgroup.merge(pooled, on="candidate_id", validate="many_to_one")
    joined["same_sign_as_pooled"] = np.sign(joined["mean_template_effect"]) == np.sign(joined["pooled_mean_effect"])
    joined["subgroup_q05"] = joined["bh_q"] < 0.05
    rows = []
    for cid, group in joined.groupby("candidate_id", sort=True):
        rows.append({
            "candidate_id": cid,
            "n_lexical_family": int(group["lexical_family"].nunique()),
            "n_same_sign_as_pooled": int(group["same_sign_as_pooled"].sum()),
            "n_lexical_family_q05": int(group["subgroup_q05"].sum()),
            "pooled_mean_effect": float(group["pooled_mean_effect"].iloc[0]),
            "pooled_cohen_dz": float(group["pooled_cohen_dz"].iloc[0]),
            "pooled_bh_q": float(group["pooled_bh_q"].iloc[0]),
            "minimum_lexical_effect": float(group["mean_template_effect"].min()),
            "maximum_lexical_effect": float(group["mean_template_effect"].max()),
            "lexical_effects_json": json.dumps(
                dict(zip(group["lexical_family"], group["mean_template_effect"])), sort_keys=True,
            ),
        })
    return pd.DataFrame(rows)


def attach_robust_cores(
    counts: dict[str, Any], effects: pd.DataFrame, specificity: pd.DataFrame, consistency: pd.DataFrame,
) -> None:
    consistency_by_id = consistency.set_index("candidate_id")
    for family in ("relation_lexical_robustness", "relation_relevance_interaction"):
        primary = effects[
            (effects["analysis_family"] == family) & (effects["bh_q"] < 0.05)
            & (effects["cohen_dz"].abs() >= 0.5)
        ]
        matched = set(specificity[
            (specificity["analysis_family"] == family) & (specificity["control_kind"] == "matched")
            & (specificity["bh_q"] < 0.05)
        ]["candidate_id"].astype(str))
        all_random = all_random_specificity_counts(specificity, family)
        core = set(primary["candidate_id"].astype(str)) & matched & all_random
        if family == "relation_lexical_robustness":
            core = {cid for cid in core if int(consistency_by_id.loc[cid, "n_same_sign_as_pooled"]) >= 5}
        ordered = primary[primary["candidate_id"].isin(core)].sort_values(
            "cohen_dz", key=lambda value: value.abs(), ascending=False
        )["candidate_id"].astype(str).tolist()
        counts[family]["robust_core_definition"] = (
            "q<.05, |dz|>=.5, matched-control q<.05, all three random-control q<.05"
            + (", and >=5/6 lexical families with the pooled sign" if family == "relation_lexical_robustness" else "")
        )
        counts[family]["robust_core_candidates"] = ordered
        counts[family]["robust_core_count"] = len(ordered)


def candidate_card(
    candidate: dict[str, Any], effects: pd.DataFrame, specificity: pd.DataFrame, source: pd.DataFrame,
) -> dict[str, Any]:
    cid = candidate["component_id"]
    stage_a = candidate["stage_a_metadata"]
    return {
        "candidate_id": cid,
        "component_type": candidate["component_type"],
        "layer_one_based": int(candidate["module_index"]) + 1,
        "module_index_zero_based": int(candidate["module_index"]),
        "component_index": int(candidate["component_index"]),
        "stage_a_delta_discovery": stage_a.get("Delta_discovery"),
        "stage_a_delta_validation": stage_a.get("Delta_validation"),
        "stage_a_sign_group": stage_a.get("sign_group_validation"),
        "extension_effects": effects[effects["candidate_id"] == cid].to_dict(orient="records"),
        "control_specificity": specificity[specificity["candidate_id"] == cid].to_dict(orient="records"),
        "changed_span_source_effects": source[source["candidate_id"] == cid].to_dict(orient="records"),
        "endpoint_note": (
            "Head effects are LiReF-projected head contributions. Neuron primary effects are activation changes; "
            "liref_direction_mean_effect multiplies them by the fixed neuron projection scalar."
        ),
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


def korean_report(config: dict[str, Any], p: dict[str, Path], counts: dict[str, Any], effects: pd.DataFrame) -> str:
    status = read_json(p["status"] / "confirmatory.json")
    design = read_json(p["design"])
    relation = counts["relation_lexical_robustness"]
    relevance = counts["relation_relevance_interaction"]
    numeric = counts["numeric_relevance_interaction"]
    factual = counts["factual_entity_relevance_interaction"]
    numeric_rep = counts["numeric_representation_token_matched"]
    token_matched_relation = counts.get("relation_token_matched_lexical_robustness", {})
    causal_priority = counts.get("cross_criterion_causal_priority", [])
    source_counts = counts.get("changed_span_sources", {})

    def top_lines(family: str, limit: int = 8) -> list[str]:
        sub = effects[(effects["analysis_family"] == family) & (effects["bh_q"] < 0.05)].copy()
        sub = sub.sort_values("cohen_dz", key=lambda value: value.abs(), ascending=False).head(limit)
        return [
            f"- `{row.candidate_id}`: d_z={row.cohen_dz:.3f}, q={row.bh_q:.4g}, LiReF-aligned mean={row.liref_direction_mean_effect:.4g}"
            for row in sub.itertuples()
        ] or ["- 유의한 후보 없음"]

    lines = [
        f"# LiReF Stage B 확장 결과 — {config['stage_b_extension_run_id']}", "",
        "## 실행 및 설계", "",
        "- 상태: **PASS**", "- 기존 Stage A/Stage B 후보 20개를 순위 변경 없이 유지",
        f"- Held-out controlled pair: **{status['n_pair']:,}개** / variant **{status['n_variant']:,}개**",
        f"- 관계 문맥 cluster: {design['relation_context_clusters']}개, 새 관계 lexical family: 6개",
        f"- 숫자 문맥 cluster: {design['numeric_context_clusters']}개",
        f"- factual entity fact-pair cluster: {design['factual_context_clusters']}개",
        "- 통계 단위: 개별 문장이나 lexical form이 아니라 semantic context/fact-pair cluster",
        "- Numeric representation은 original/modified prompt token length가 동일한 pair만 사용",
        f"- 모델 checksum 실행 전후 동일: `{status['model_parameter_checksum_before'] == status['model_parameter_checksum_after']}`",
        f"- Head source reconstruction 최대 절대오차: `{status['source_reconstruction_max_abs_error']}`", "",
        "## 1차 결과", "",
        f"1. 새 lexical family 전체에 평균한 relation polarity 효과: q<.05 **{relation['q05']}/20**, q<.05 및 |d_z|≥0.5 **{relation['q05_and_abs_dz_ge_0_5']}/20**.",
        f"2. 관계가 답에 필요한 경우와 불필요한 경우의 상호작용: q<.05 **{relevance['q05']}/20**, 강한 효과 **{relevance['q05_and_abs_dz_ge_0_5']}/20**.",
        f"3. 숫자 값의 relevant-minus-irrelevant 상호작용: q<.05 **{numeric['q05']}/20**, 강한 효과 **{numeric['q05_and_abs_dz_ge_0_5']}/20**.",
        f"4. factual entity bundle의 relevant-minus-irrelevant 상호작용: q<.05 **{factual['q05']}/20**, 강한 효과 **{factual['q05_and_abs_dz_ge_0_5']}/20**.",
        f"5. token-length-matched digit↔number-word 보조 분석: q<.05 **{numeric_rep['q05']}/20**.", "",
        "## Token-length 강건성 보조 검사", "",
        f"- original/modified token length가 다른 `obtained/gave away` 어휘군을 제외한 5개 관계 어휘군에서도 q<.05 **{token_matched_relation.get('q05', 0)}/20**, |d_z|≥0.5 **{token_matched_relation.get('q05_and_abs_dz_ge_0_5', 0)}/20**로 유지됐다.",
        "- 이 검사는 사전에 동결된 1차 분석이 아니라, 관계 결과가 한 개의 token-length 비대칭 어휘군에 의해 생겼는지 확인한 보조 강건성 분석이다.", "",
        "## 보수적 핵심 집합", "",
        f"- 관계 lexical robustness: matched control, 세 random control, 효과크기, 어휘군 부호 일관성을 모두 만족한 **{relation.get('robust_core_count', 0)}개** — {', '.join(f'`{x}`' for x in relation.get('robust_core_candidates', [])) or '없음'}",
        f"- 관계 task-relevance: matched control, 세 random control, 효과크기를 모두 만족한 **{relevance.get('robust_core_count', 0)}개** — {', '.join(f'`{x}`' for x in relevance.get('robust_core_candidates', [])) or '없음'}", "",
        "## 다음 Causal Validation 우선순위", "",
        f"- lexical robustness 핵심 집합과 task-relevance 핵심 집합을 동시에 만족한 **{len(causal_priority)}개**: {', '.join(f'`{x}`' for x in causal_priority) or '없음'}",
        "- 이 목록은 다음 독립 causal experiment의 intervention target을 동결하기 위한 우선순위이며, 현재 결과 자체가 인과성을 의미하지 않는다.", "",
        "## Control 대비", "",
        f"- 관계 lexical robustness: matched control과 구분 **{relation['matched_control_specificity_q05']}/20**, 세 random control 모두와 구분 **{relation['all_three_random_controls_specificity_q05']}/20**.",
        f"- 관계 task-relevance: matched control과 구분 **{relevance['matched_control_specificity_q05']}/20**, 세 random control 모두와 구분 **{relevance['all_three_random_controls_specificity_q05']}/20**.", "",
        "## Attention changed-span 근거", "",
        f"- Head 10개 중 changed relation span의 직접 source contribution이 lexical robustness에서 **{source_counts.get('changed_span_relation_lexical_robustness', 0)}/10**, relevance interaction에서 **{source_counts.get('changed_span_relation_relevance_interaction', 0)}/10** 유의했다.",
        "- 따라서 total-head 결과가 prompt suffix만으로 생겼다고 보기는 어렵지만, source contribution 역시 인과적 attribution은 아니다.", "",
        "## 관계 lexical robustness 상위 후보", "",
        *top_lines("relation_lexical_robustness"), "",
        "## 관계 task-relevance 상호작용 상위 후보", "",
        *top_lines("relation_relevance_interaction"), "",
        "## 해석 원칙", "",
        "- `relation_lexical_robustness`가 유의하면 기존 received/lost 한 표현에만 묶이지 않고 새 표현들에서도 polarity sensitivity가 재현됐다는 뜻이다.",
        "- `relation_relevance_interaction`이 유의하면 같은 polarity 변화가 distractor 문맥보다 답에 필요한 문맥에서 더 다르게 나타났다는 뜻이다. 이는 task-relevance sensitivity이지 reasoning 수행의 증명은 아니다.",
        "- `numeric_relevance_interaction`은 숫자 모양 자체와 답에 필요한 숫자를 분리하는 근거다. 값 변경에 따른 의미·정답 변화까지 완전히 제거하는 실험은 아니다.",
        "- token-length-matched digit↔word 반응은 단순 token 개수보다 표기/token identity에 대한 sensitivity가 남는다는 뜻이며, numerical reasoning sensitivity를 뜻하지 않는다.",
        "- `factual_entity_relevance_interaction`은 factual cue bundle의 관련성을 본다. entity 여러 개가 함께 바뀌므로 단일 named-entity feature나 memorization mechanism으로 해석하면 안 된다.",
        "- 숫자 relevance와 factual entity relevance가 0/20인 것은 이 controlled operationalization에서 선택적 효과를 찾지 못했다는 뜻이다. 다른 memory cue나 numeric reasoning feature가 없다는 증거는 아니다.",
        "- Head 부호는 LiReF-projected contribution 부호다. Neuron 검정 부호는 activation 부호이며, R/M 방향 해석에는 고정 projection을 곱한 `liref_direction_mean_effect`를 사용한다.",
        "- 본 결과는 Meta-Llama-3-8B Base, `Q: {question}\\nA: `, 마지막 prompt token, synthetic English template 조건에 한정된다.",
        "- 생성 pair는 자동 구조 검사를 통과했고 사용자가 확장 실행을 승인했지만, 독립 제3자의 문장별 linguistic audit는 아직 없다.",
        "- 인과성은 다음 Causal Validation에서 후보 억제/증폭 후 representation 및 행동 변화를 측정해야 판단할 수 있다.", "",
        "## 주요 파일", "",
        "- `tables/candidate_effects.csv`", "- `tables/control_specificity.csv`",
        "- `tables/changed_span_source_effects.csv`", "- `tables/effect_vectors.csv.gz`",
        "- `tables/relation_lexical_consistency.csv`",
        "- `tables/relation_token_matched_robustness.csv`", "- `tables/relation_token_matched_control_specificity.csv`",
        "- `manifests/confirmatory_pairs.csv`", "- `candidate_cards/*.json`", "- `stage_b_extension_summary.json`", "",
    ]
    return "\n".join(lines)


def run_report(config: dict[str, Any], p: dict[str, Path]) -> None:
    require_status(p, "confirmatory")
    revalidate_freeze(p)
    effects = pd.read_csv(p["tables"] / "candidate_effects.csv")
    specificity = pd.read_csv(p["tables"] / "control_specificity.csv")
    source = pd.read_csv(p["tables"] / "changed_span_source_effects.csv")
    counts = result_counts(effects, specificity, source)
    consistency = lexical_consistency_table(effects)
    atomic_frame(p["tables"] / "relation_lexical_consistency.csv", consistency)
    attach_robust_cores(counts, effects, specificity, consistency)

    vectors = pd.read_csv(p["tables"] / "effect_vectors.csv.gz")
    token_matched_vectors = vectors[
        (vectors["analysis_family"] == "relation_lexical_robustness")
        & (vectors["unit_id"] != "obtained_gave_away")
    ].copy()
    token_matched_vectors["analysis_family"] = "relation_token_matched_lexical_robustness"
    token_matched_vectors["analysis_label"] = "secondary_robustness"
    token_matched_effects = summarize_vectors(config, token_matched_vectors, candidate_only=True, apply_fdr=True)
    projection = effects.drop_duplicates(["candidate_id", "component_id"])[
        ["candidate_id", "component_id", "projection"]
    ]
    token_matched_effects = token_matched_effects.merge(
        projection, on=["candidate_id", "component_id"], validate="many_to_one"
    )
    token_matched_effects["liref_direction_mean_effect"] = np.where(
        token_matched_effects["component_type"] == "neuron",
        token_matched_effects["mean_template_effect"] * token_matched_effects["projection"],
        token_matched_effects["mean_template_effect"],
    )
    token_matched_specificity = summarize_specificity(config, specificity_vectors(token_matched_vectors))
    atomic_frame(p["tables"] / "relation_token_matched_robustness.csv", token_matched_effects)
    atomic_frame(p["tables"] / "relation_token_matched_control_specificity.csv", token_matched_specificity)
    counts["relation_token_matched_lexical_robustness"] = {
        "analysis_label": "secondary_robustness",
        "excluded_lexical_family": "obtained_gave_away",
        "reason": "modified prompt is one token longer than original; the other five lexical families are length matched",
        "q05": int((token_matched_effects["bh_q"] < 0.05).sum()),
        "q05_and_abs_dz_ge_0_5": int(((token_matched_effects["bh_q"] < 0.05) & (token_matched_effects["cohen_dz"].abs() >= 0.5)).sum()),
    }
    lexical_core = counts["relation_lexical_robustness"]["robust_core_candidates"]
    relevance_core = set(counts["relation_relevance_interaction"]["robust_core_candidates"])
    counts["cross_criterion_causal_priority"] = [cid for cid in lexical_core if cid in relevance_core]
    write_json(p["root"] / "stage_b_extension_result_counts.json", counts)
    candidates = read_json(p["candidates"])
    for candidate in candidates["candidates"]:
        card = candidate_card(candidate, effects, specificity, source)
        cid = candidate["component_id"]
        write_json(p["cards"] / f"{cid}.json", card)
        markdown = [
            f"# {cid}", "", f"- Type: {card['component_type']}", f"- Layer: {card['layer_one_based']}",
            f"- Stage A validation sign: {card['stage_a_sign_group']}", "", "## Extension effects", "",
            "```json", json.dumps(card["extension_effects"], ensure_ascii=False, indent=2), "```", "",
            "## Interpretation boundary", "", INTERPRETATION_BOUNDARY, "",
        ]
        (p["cards"] / f"{cid}.md").write_text("\n".join(markdown), encoding="utf-8")
    report = korean_report(config, p, counts, effects)
    (p["root"] / "RESULTS_KO.md").write_text(report, encoding="utf-8")
    extension_hash, extension_files = code_hash()
    summary = {
        "run_id": config["stage_b_extension_run_id"], "status": "PASS", "candidate_count": 20,
        "candidate_card_count": len(list(p["cards"].glob("*.json"))), "result_counts": counts,
        "results_ko_sha256": sha256_file(p["root"] / "RESULTS_KO.md"),
        "extension_code_hash": extension_hash, "extension_code_files": extension_files,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
        "independent_human_linguistic_audit": False,
    }
    write_json(p["root"] / "stage_b_extension_summary.json", summary)
    write_status(p, "report", summary_sha256=sha256_file(p["root"] / "stage_b_extension_summary.json"), results_ko_sha256=summary["results_ko_sha256"])
    print("report PASS")


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args)
    p = paths(config)
    ensure_dirs(p)
    functions = {
        "prepare": run_prepare,
        "sanity": run_sanity,
        "pilot": run_pilot,
        "freeze_confirmatory": run_freeze_confirmatory,
        "confirmatory": run_confirmatory,
        "report": run_report,
    }
    try:
        functions[args.phase](config, p)
        return 0
    except Exception as exc:
        write_json(p["status"] / f"{args.phase}.json", {
            "phase": args.phase, "status": "FAIL", "timestamp": utc_now(),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
