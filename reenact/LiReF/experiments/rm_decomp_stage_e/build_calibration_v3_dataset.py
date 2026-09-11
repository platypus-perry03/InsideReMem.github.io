#!/usr/bin/env python3
"""Build and automatically audit the frozen Stage E Calibration v3 dataset.

This is a dataset-only utility. It may load the hash-locked local tokenizer,
but it must never load model weights, run a forward pass, use a GPU, inspect
LiReF/candidate states, install hooks, or perform an intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
STAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "AI" / "reenact" / "models" / "Meta-Llama-3-8B"
DESIGN_PATH = STAGE_DIR / "calibration_v3_design_frozen.json"
DEFAULT_OUTPUT_DIR = STAGE_DIR / "calibration_v3_assets"

EXPECTED_DESIGN_SHA256 = "c60a579729376d391582dbc03af9cfd3ba0a1e1743a9e9a884967aacc177adfc"
EXPECTED_TOKENIZER_HASHES = {
    "tokenizer.json": "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
    "tokenizer_config.json": "690727b4fed286383df1c7ca5e805124cb70c6eb4529f807c7b2e60ff741da7e",
    "special_tokens_map.json": "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
}
PRIOR_DATASETS = {
    "v2": (
        STAGE_DIR / "calibration_v2_assets" / "calibration_v2_dataset_draft.json",
        "c58390cdcb0f7282e36c918b193db69a0733851cfd07c291ab59a6fe12df1c87",
    ),
    "v2_1": (
        STAGE_DIR / "calibration_v2_1_assets" / "calibration_v2_1_dataset_draft.json",
        "cfc5a628d75e05c17d4c4ff3907f477d6f6179892453f1e2c3927c8aeed10640",
    ),
    "v2_1_1": (
        STAGE_DIR / "calibration_v2_1_1_assets" / "calibration_v2_1_1_dataset_draft.json",
        "322d80f3c2c0723f6a4e8b0a968b30baac23e28cf68b6c60c5a7a95a5bca7420",
    ),
}

OUTPUT_FILENAMES = (
    "calibration_v3_static_schema_check.json",
    "calibration_v3_dataset_draft.json",
    "calibration_v3_automatic_audit.json",
    "calibration_v3_dataset_manifest.json",
)

PRIOR_NAMES = {
    "Ava", "Liam", "Nora", "Owen", "Maya", "Noah", "Lena", "Evan",
    "Iris", "Milo", "Sara", "Theo", "Emma", "Lucas", "Zoe", "Henry",
    "Mina", "Felix", "Aria", "Caleb", "Daria", "Jonah", "Kiara", "Mason",
    "Priya", "Rowan", "Tessa", "Victor", "Willa", "Xavier", "Yara", "Bruno",
    "Chloe", "Dante", "Freya", "Gavin", "Hazel", "Isaac", "Julia", "Karim",
    "Layla", "Quinn",
}

NAMES = [
    "Amara", "Bastian", "Celia", "Devon", "Elara", "Finn", "Gia", "Hugo",
    "Ines", "Jules", "Kaia", "Leon", "Maren", "Niko", "Orla", "Pavel",
    "Rhea", "Soren", "Uma", "Veda", "Wynn", "Xena", "Yuri", "Zane",
]

LABEL_PAIRS = [
    ("amber", "azure"),
    ("coral", "ivory"),
    ("north", "south"),
    ("upper", "lower"),
    ("inner", "outer"),
    ("alpha", "omega"),
    ("cedar", "maple"),
    ("pearl", "flint"),
]


def spec(
    template_id: str,
    mapping: str,
    arithmetic_case: str,
    selector_case: str,
    ledger: str,
    question: str,
    add_word: str = "ADD",
    sub_word: str = "SUB",
) -> dict[str, str]:
    return {
        "id": template_id,
        "mapping": mapping,
        "arithmetic_case": arithmetic_case,
        "selector_case": selector_case,
        "ledger": ledger,
        "question": question,
        "add_word": add_word,
        "sub_word": sub_word,
    }


TEMPLATES: dict[str, list[dict[str, str]]] = {
    "object_count": [
        spec(
            "v3_obj_lantern_dossier",
            "To track {name}'s {label} lanterns, consult case {case_key}.",
            "Case {case_key} calls {value_key_1} the starting-count key and {value_key_2} the change-count key; its instruction is {operation}.",
            "Case {case_key} bears selector {active_tag}; entry {tag_1} names key {value_key_1}, whereas entry {tag_2} names key {value_key_2}.",
            "The lantern ledger assigns {key_1}={value_1}, {key_2}={value_2}, {key_3}={value_3}, and {key_4}={value_4}.",
            "How many {target_label} lanterns does {name} have now?",
        ),
        spec(
            "v3_obj_scarf_registry",
            "The registry sends {name}'s {label} scarf total to case {case_key}.",
            "Within {case_key}, {value_key_1} identifies the original total and {value_key_2} identifies the adjustment; follow command {operation}.",
            "Within {case_key}, selector {active_tag} chooses between {tag_1}:{value_key_1} and {tag_2}:{value_key_2}.",
            "The scarf value table reads {key_1} {value_1}; {key_2} {value_2}; {key_3} {value_3}; {key_4} {value_4}.",
            "What is the current number of {target_label} scarves for {name}?",
        ),
        spec(
            "v3_obj_magnet_portfolio",
            "For {name}, portfolio case {case_key} governs the {label} magnets.",
            "Portfolio case {case_key} designates {value_key_1} for the baseline and {value_key_2} for the change, with operation {operation}.",
            "Portfolio case {case_key} shows seal {active_tag}; seal row {tag_1} refers to {value_key_1}, and seal row {tag_2} refers to {value_key_2}.",
            "Its magnet lookup table contains {key_1}->{value_1}, {key_2}->{value_2}, {key_3}->{value_3}, {key_4}->{value_4}.",
            "How many magnets are in {name}'s {target_label} group now?",
        ),
        spec(
            "v3_obj_postcard_journal",
            "Journal case {case_key} is the reference for {name}'s {label} postcards.",
            "Case {case_key} gives start pointer {value_key_1}, change pointer {value_key_2}, and directive {operation}.",
            "Case {case_key} displays marker {active_tag}; marker {tag_1} selects pointer {value_key_1}, while marker {tag_2} selects pointer {value_key_2}.",
            "The postcard journal decodes the pointers as {key_1}:{value_1}, {key_2}:{value_2}, {key_3}:{value_3}, {key_4}:{value_4}.",
            "What is {name}'s present {target_label} postcard count?",
        ),
        spec(
            "v3_obj_button_compendium",
            "In the compendium, {name}'s {label} buttons are handled by case {case_key}.",
            "Case {case_key} labels {value_key_1} as START and {value_key_2} as CHANGE and specifies {operation}.",
            "Case {case_key} carries tag {active_tag}; its {tag_1} line leads to {value_key_1}, and its {tag_2} line leads to {value_key_2}.",
            "The button keybook gives {key_1} value {value_1}, {key_2} value {value_2}, {key_3} value {value_3}, and {key_4} value {value_4}.",
            "What number of {target_label} buttons currently belongs to {name}?",
        ),
        spec(
            "v3_obj_gear_inventory",
            "{name}'s {label} gear inventory is controlled through case {case_key}.",
            "For computation, {case_key} points first to {value_key_1} and then to {value_key_2}; execute {operation} on those values.",
            "For selection, {case_key} announces code {active_tag}; code {tag_1} points to {value_key_1} and code {tag_2} points to {value_key_2}.",
            "The gear ledger maps {key_1} to {value_1}, {key_2} to {value_2}, {key_3} to {value_3}, and {key_4} to {value_4}.",
            "How many {target_label} gears are currently in {name}'s inventory?",
        ),
        spec(
            "v3_obj_mug_directory",
            "Directory case {case_key} covers the {label} mugs assigned to {name}.",
            "Case {case_key} details identify initial key {value_key_1}, update key {value_key_2}, and rule {operation}.",
            "Case {case_key} details identify active badge {active_tag}; badge {tag_1} links with {value_key_1}, and badge {tag_2} links with {value_key_2}.",
            "The mug directory resolves {key_1} as {value_1}, {key_2} as {value_2}, {key_3} as {value_3}, and {key_4} as {value_4}.",
            "What is the current {target_label} mug total assigned to {name}?",
        ),
        spec(
            "v3_obj_flag_workbook",
            "The workbook associates {name}'s {label} flags with case {case_key}.",
            "Case {case_key}'s page lists source key {value_key_1}, shift key {value_key_2}, and action {operation}.",
            "Case {case_key}'s page lists chosen symbol {active_tag}; symbol {tag_1} addresses {value_key_1}, while symbol {tag_2} addresses {value_key_2}.",
            "A separate flag table states {key_1}={value_1} | {key_2}={value_2} | {key_3}={value_3} | {key_4}={value_4}.",
            "How many flags does {name}'s {target_label} set contain now?",
        ),
    ],
    "points_balance": [
        spec(
            "v3_pts_quiz_scorecard",
            "{name}'s {label} quiz score is routed through case {case_key}.",
            "Score case {case_key} uses {value_key_1} for the opening score and {value_key_2} for its change; apply {operation}.",
            "Score case {case_key} has selector {active_tag}; option {tag_1} uses {value_key_1}, and option {tag_2} uses {value_key_2}.",
            "The quiz scorecard records {key_1}:{value_1}, {key_2}:{value_2}, {key_3}:{value_3}, and {key_4}:{value_4}.",
            "What is {name}'s current {target_label} quiz score?",
        ),
        spec(
            "v3_pts_arcade_tally",
            "Arcade tally case {case_key} tracks {name}'s {label} points.",
            "Tally case {case_key} pairs base locator {value_key_1} with adjustment locator {value_key_2} under instruction {operation}.",
            "Tally case {case_key} presents token {active_tag}; token {tag_1} selects {value_key_1}, whereas token {tag_2} selects {value_key_2}.",
            "The arcade table translates {key_1} into {value_1}, {key_2} into {value_2}, {key_3} into {value_3}, and {key_4} into {value_4}.",
            "How many {target_label} arcade points does {name} have now?",
        ),
        spec(
            "v3_pts_reward_sheet",
            "On {name}'s reward sheet, the {label} balance belongs to case {case_key}.",
            "Case {case_key} identifies beginning-value key {value_key_1} and adjustment key {value_key_2}, followed by {operation}.",
            "Case {case_key} displays selector stamp {active_tag}; stamp {tag_1} refers to {value_key_1}, and stamp {tag_2} refers to {value_key_2}.",
            "The reward ledger says {key_1} has {value_1}, {key_2} has {value_2}, {key_3} has {value_3}, and {key_4} has {value_4}.",
            "What is the current {target_label} reward balance for {name}?",
        ),
        spec(
            "v3_pts_league_record",
            "League record case {case_key} stores {name}'s {label} point account.",
            "Record case {case_key} marks {value_key_1} as the prior-point key and {value_key_2} as the point-change key and orders {operation}.",
            "Record case {case_key} marks selector {active_tag}; branch {tag_1} leads to {value_key_1}, and branch {tag_2} leads to {value_key_2}.",
            "The league values are {key_1}->{value_1}; {key_2}->{value_2}; {key_3}->{value_3}; {key_4}->{value_4}.",
            "What is {name}'s {target_label} league-point total now?",
        ),
        spec(
            "v3_pts_loyalty_folio",
            "Folio case {case_key} is assigned to {name}'s {label} loyalty score.",
            "For arithmetic, folio case {case_key} provides origin key {value_key_1}, movement key {value_key_2}, and command {operation}.",
            "For lookup, folio case {case_key} provides seal {active_tag}; seal {tag_1} chooses {value_key_1}, while seal {tag_2} chooses {value_key_2}.",
            "The loyalty key table lists {key_1} {value_1}, {key_2} {value_2}, {key_3} {value_3}, and {key_4} {value_4}.",
            "How many {target_label} loyalty points are credited to {name} now?",
        ),
        spec(
            "v3_pts_contest_docket",
            "The contest docket links {name}'s {label} score to case {case_key}.",
            "Docket case {case_key} supplies starting key {value_key_1}, change key {value_key_2}, and operator {operation}.",
            "Docket case {case_key} supplies active sign {active_tag}; sign {tag_1} names {value_key_1}, and sign {tag_2} names {value_key_2}.",
            "The contest key ledger contains {key_1}={value_1}; {key_2}={value_2}; {key_3}={value_3}; {key_4}={value_4}.",
            "What is the present {target_label} contest score for {name}?",
        ),
        spec(
            "v3_pts_merit_index",
            "Merit index case {case_key} governs the {label} credits held by {name}.",
            "Index case {case_key} declares baseline reference {value_key_1}, revision reference {value_key_2}, and instruction {operation}.",
            "Index case {case_key} declares chosen emblem {active_tag}; emblem {tag_1} resolves to {value_key_1}, and emblem {tag_2} resolves to {value_key_2}.",
            "The merit reference table gives {key_1}:{value_1} / {key_2}:{value_2} / {key_3}:{value_3} / {key_4}:{value_4}.",
            "What is {name}'s current {target_label} merit-credit balance?",
        ),
        spec(
            "v3_pts_challenge_book",
            "Challenge book case {case_key} covers {name}'s {label} point line.",
            "Case {case_key} sets {value_key_1} as INPUT, {value_key_2} as CHANGE, and {operation} as the operation.",
            "Case {case_key} sets {active_tag} as the selector; {tag_1} routes to {value_key_1}, while {tag_2} routes to {value_key_2}.",
            "The challenge lookup reads {key_1}->{value_1}, {key_2}->{value_2}, {key_3}->{value_3}, {key_4}->{value_4}.",
            "How many points are on {name}'s {target_label} challenge line now?",
        ),
    ],
    "temperature": [
        spec(
            "v3_tmp_terrarium_monitor",
            "At {name}'s lab, the {label} terrarium uses case {case_key}.",
            "Temperature case {case_key} assigns {value_key_1} to the initial reading and {value_key_2} to the degree change; perform {operation}.",
            "Temperature case {case_key} carries selector {active_tag}; entry {tag_1} calls for {value_key_1}, and entry {tag_2} calls for {value_key_2}.",
            "The terrarium scale table states {key_1}:{value_1}, {key_2}:{value_2}, {key_3}:{value_3}, and {key_4}:{value_4} degrees.",
            "What is the {target_label} terrarium's current temperature in {name}'s lab?",
            "WARM",
            "COOL",
        ),
        spec(
            "v3_tmp_kiln_tracker",
            "{name}'s {label} kiln is represented by tracker case {case_key}.",
            "Tracker case {case_key} points to starting-temperature key {value_key_1} and shift key {value_key_2}, with directive {operation}.",
            "Tracker case {case_key} shows active mark {active_tag}; mark {tag_1} points to {value_key_1}, whereas mark {tag_2} points to {value_key_2}.",
            "The kiln tracker resolves {key_1} as {value_1}, {key_2} as {value_2}, {key_3} as {value_3}, and {key_4} as {value_4} degrees.",
            "What temperature does {name}'s {target_label} kiln have now?",
            "WARM",
            "COOL",
        ),
        spec(
            "v3_tmp_cooler_station",
            "Station case {case_key} controls the {label} cooler in {name}'s facility.",
            "Station case {case_key} defines original-reading key {value_key_1}, variation key {value_key_2}, and action {operation}.",
            "Station case {case_key} displays signal {active_tag}; signal {tag_1} selects {value_key_1}, and signal {tag_2} selects {value_key_2}.",
            "A cooler conversion chart gives {key_1}={value_1}, {key_2}={value_2}, {key_3}={value_3}, and {key_4}={value_4} degrees.",
            "What is the current temperature of the {target_label} cooler in {name}'s facility?",
            "WARM",
            "COOL",
        ),
        spec(
            "v3_tmp_reactor_panel",
            "Panel case {case_key} corresponds to {name}'s {label} reactor.",
            "For its reading, case {case_key} names source key {value_key_1}, change key {value_key_2}, and rule {operation}.",
            "For its reading, case {case_key} names selector rune {active_tag}; rune {tag_1} uses {value_key_1}, while rune {tag_2} uses {value_key_2}.",
            "The reactor table associates {key_1} with {value_1}, {key_2} with {value_2}, {key_3} with {value_3}, and {key_4} with {value_4} degrees.",
            "What temperature is {name}'s {target_label} reactor at now?",
            "WARM",
            "COOL",
        ),
        spec(
            "v3_tmp_greenhouse_sensor",
            "{name}'s {label} greenhouse reading belongs to sensor case {case_key}.",
            "Sensor case {case_key} provides baseline key {value_key_1}, adjustment key {value_key_2}, and command {operation}.",
            "Sensor case {case_key} provides selected glyph {active_tag}; glyph {tag_1} addresses {value_key_1}, and glyph {tag_2} addresses {value_key_2}.",
            "The greenhouse ledger records {key_1}:{value_1}; {key_2}:{value_2}; {key_3}:{value_3}; {key_4}:{value_4} degrees.",
            "What is the {target_label} greenhouse temperature for {name} now?",
            "WARM",
            "COOL",
        ),
        spec(
            "v3_tmp_vault_gauge",
            "Gauge case {case_key} is assigned to the {label} vault at {name}'s site.",
            "Gauge case {case_key} lists initial key {value_key_1}, change key {value_key_2}, and instruction {operation}.",
            "Gauge case {case_key} lists active crest {active_tag}; crest {tag_1} links to {value_key_1}, and crest {tag_2} links to {value_key_2}.",
            "The vault gauge table lists {key_1} {value_1}, {key_2} {value_2}, {key_3} {value_3}, and {key_4} {value_4} degrees.",
            "What temperature does the {target_label} vault at {name}'s site have now?",
            "WARM",
            "COOL",
        ),
        spec(
            "v3_tmp_basin_console",
            "Console case {case_key} monitors {name}'s {label} basin.",
            "Console case {case_key} uses {value_key_1} for the earlier temperature, {value_key_2} for the change, and {operation} for the update.",
            "Console case {case_key} uses flag {active_tag} for selection; flag {tag_1} refers to {value_key_1}, whereas flag {tag_2} refers to {value_key_2}.",
            "The basin console decodes {key_1}->{value_1}, {key_2}->{value_2}, {key_3}->{value_3}, and {key_4}->{value_4} degrees.",
            "What is {name}'s {target_label} basin temperature now?",
            "WARM",
            "COOL",
        ),
        spec(
            "v3_tmp_module_thermometer",
            "Thermometer case {case_key} covers the {label} module managed by {name}.",
            "Thermometer case {case_key} identifies reference key {value_key_1}, degree-shift key {value_key_2}, and operation {operation}.",
            "Thermometer case {case_key} identifies selector badge {active_tag}; badge {tag_1} resolves through {value_key_1}, and badge {tag_2} resolves through {value_key_2}.",
            "The module thermometer key gives {key_1}={value_1} | {key_2}={value_2} | {key_3}={value_3} | {key_4}={value_4} degrees.",
            "What is the current temperature of the {target_label} module managed by {name}?",
            "WARM",
            "COOL",
        ),
    ],
}


KEY_ONSETS = [
    "Ba", "Be", "Bi", "Bo", "Ca", "Ce", "Ci", "Co", "Da", "De", "Di", "Do",
    "Fa", "Fe", "Fi", "Fo", "Ga", "Ge", "Gi", "Go", "Ha", "He", "Hi", "Ho",
    "Ja", "Je", "Ji", "Jo", "La", "Le", "Li", "Lo", "Ma", "Me", "Mi", "Mo",
    "Na", "Ne", "Ni", "No", "Pa", "Pe", "Pi", "Po", "Ra", "Re", "Ri", "Ro",
    "Sa", "Se", "Si", "So", "Ta", "Te", "Ti", "To", "Va", "Ve", "Vi", "Vo",
]
KEY_ENDINGS = ["dor", "fen", "gis", "hal", "jor", "kel", "lin", "mor", "nus", "pel", "rin", "sov"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_keys(mapping: dict[str, Any], keys: list[str], section: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise RuntimeError(f"Frozen manifest section {section} missing keys: {missing}")


def require_section(design: dict[str, Any], name: str, keys: list[str]) -> dict[str, Any]:
    value = design.get(name)
    if not isinstance(value, dict):
        raise RuntimeError(f"Frozen manifest section {name} must be an object")
    require_keys(value, keys, name)
    return value


def validate_oa_matrix(matrix: list[list[int]]) -> dict[str, Any]:
    if len(matrix) != 8 or any(len(row) != 5 for row in matrix):
        raise RuntimeError("Frozen OA matrix must be exactly 8x5")
    if any(value not in (0, 1) for row in matrix for value in row):
        raise RuntimeError("Frozen OA matrix must be binary")
    column_counts: dict[str, dict[str, int]] = {}
    for index, name in enumerate("ABCDE"):
        counts = {str(level): sum(row[index] == level for row in matrix) for level in (0, 1)}
        if counts != {"0": 4, "1": 4}:
            raise RuntimeError(f"OA factor {name} is not 4:4: {counts}")
        column_counts[name] = counts
    pair_counts: dict[str, dict[str, int]] = {}
    for left, right in combinations(range(5), 2):
        counts = {
            f"{a}{b}": sum(row[left] == a and row[right] == b for row in matrix)
            for a in (0, 1) for b in (0, 1)
        }
        if set(counts.values()) != {2}:
            raise RuntimeError(f"OA factor pair {left},{right} is not strength-2: {counts}")
        pair_counts[f"{'ABCDE'[left]}{'ABCDE'[right]}"] = counts
    if not all(row[3] == (row[0] ^ row[1]) and row[4] == (row[0] ^ row[2]) for row in matrix):
        raise RuntimeError("Frozen OA derived columns do not match D=A xor B and E=A xor C")
    return {"column_counts": column_counts, "pair_counts": pair_counts, "pass": True}


def validate_schema_compatibility(design: dict[str, Any]) -> dict[str, Any]:
    required_top = [
        "schema_version", "design_id", "status", "source_design", "feature",
        "prompt_output_contract", "pair_invariants", "multi_hop_matching",
        "within_condition_exposure_matched_scoring", "numeric_design",
        "arithmetic_primary_foil", "teacher_forced_scoring", "counterbalance",
        "dataset_design", "acceptance_criteria", "cluster_statistics",
        "automatic_audit", "near_duplicate_nonreuse", "ai_audit", "human_audit",
        "liref_dataset_role", "safety", "next_allowed_work",
    ]
    require_keys(design, required_top, "root")
    feature = require_section(
        design, "feature",
        ["operationalized_name", "perfect_exposure_invariance_claim_allowed"],
    )
    prompt = require_section(
        design, "prompt_output_contract",
        ["template", "do_sample", "num_beams", "max_new_tokens", "normalization_order"],
    )
    exposure = require_section(
        design, "within_condition_exposure_matched_scoring",
        ["absolute_exposure_regime_equal_across_conditions", "cross_condition_canonical_answer_as_primary_alternative_allowed"],
    )
    numeric = require_section(
        design, "numeric_design",
        ["candidate_value_min", "candidate_value_max", "delta_min", "delta_max",
         "candidate_pair_distance_formula", "candidate_pair_values_same_parity",
         "answer_digit_length", "answer_continuation_token_count", "numeric_seed"],
    )
    foil = require_section(
        design, "arithmetic_primary_foil",
        ["rule", "add_or_warming_correct", "add_or_warming_alternative",
         "subtract_or_cooling_correct", "subtract_or_cooling_alternative"],
    )
    scoring = require_section(
        design, "teacher_forced_scoring",
        ["primary_metric", "arithmetic_margin_formula", "selector_margin_formula",
         "template_contrast_formula", "secondary_metrics"],
    )
    counterbalance = require_section(
        design, "counterbalance",
        ["design_type", "factor_count", "frames_per_template", "factors",
         "matrix_columns", "matrix", "matrix_substitution_allowed"],
    )
    dataset = require_section(
        design, "dataset_design",
        ["lexical_families", "lexical_family_count", "template_families_per_lexical_family",
         "independent_template_family_count", "frames_per_template_family",
         "paired_item_count", "prompt_count", "prompts_per_condition_per_lexical_family",
         "dataset_seed", "reuse_in_pilot", "reuse_in_confirmatory"],
    )
    acceptance = require_section(
        design, "acceptance_criteria",
        ["all_hard_criteria_required", "result_dependent_template_or_family_exclusion_allowed",
         "candidate_forced_choice", "one_token_generation",
         "maximum_absolute_mean_template_contrast_nats", "maximum_absolute_cluster_dz",
         "hard_same_sign_template_count_gate_enabled"],
    )
    human = require_section(
        design, "human_audit",
        ["required", "waiver_allowed", "primary_reviewer_count",
         "pairs_per_primary_reviewer", "blind_to"],
    )
    safety = require_section(
        design, "safety",
        ["execution_allowed", "dataset_generation_allowed_after_freeze",
         "model_loading_allowed", "baseline_calibration_execution_allowed",
         "load_liref_direction", "capture_hidden_states", "forward_hooks",
         "activation_or_weight_intervention", "stage_e_pilot_allowed"],
    )
    oa_report = validate_oa_matrix(counterbalance["matrix"])
    fc = acceptance["candidate_forced_choice"]
    generation = acceptance["one_token_generation"]
    checks = {
        "design_identity": design["schema_version"] == "3.0" and design["design_id"] == "stage_e_baseline_calibration_v3",
        "frozen_status": design["status"] == "design_frozen_execution_not_approved",
        "feature_name": feature["operationalized_name"] == "arithmetic composition vs selector-guided value retrieval after matched multi-hop binding",
        "exposure_claim_limited": feature["perfect_exposure_invariance_claim_allowed"] is False,
        "within_condition_scoring": exposure["absolute_exposure_regime_equal_across_conditions"] is False and exposure["cross_condition_canonical_answer_as_primary_alternative_allowed"] is False,
        "prompt_contract": prompt["template"] == "Q: {question}\nAnswer with one Arabic numeral only.\nA: " and prompt["do_sample"] is False and prompt["num_beams"] == 1 and prompt["max_new_tokens"] == 1,
        "numeric_contract": numeric["candidate_value_min"] == 30 and numeric["candidate_value_max"] == 69 and numeric["delta_min"] == 2 and numeric["delta_max"] == 9 and numeric["candidate_pair_distance_formula"] == "2_times_delta" and numeric["candidate_pair_values_same_parity"] is True and numeric["answer_digit_length"] == 2 and numeric["answer_continuation_token_count"] == 1 and numeric["numeric_seed"] == 20260831,
        "foil_contract": foil["rule"] == "wrong_operation_result" and foil["add_or_warming_correct"] == "start_plus_delta" and foil["add_or_warming_alternative"] == "start_minus_delta" and foil["subtract_or_cooling_correct"] == "start_minus_delta" and foil["subtract_or_cooling_alternative"] == "start_plus_delta",
        "primary_metric": scoring["primary_metric"] == "within_condition_exposure_matched_template_level_candidate_log_odds_contrast",
        "oa_contract": counterbalance["design_type"] == "OA(8,5,2,2)_strength_2_orthogonal_counterbalance" and counterbalance["factor_count"] == 5 and counterbalance["frames_per_template"] == 8 and counterbalance["matrix_substitution_allowed"] is False,
        "dataset_counts": dataset["lexical_families"] == ["object_count", "points_balance", "temperature"] and dataset["lexical_family_count"] == 3 and dataset["template_families_per_lexical_family"] == 8 and dataset["independent_template_family_count"] == 24 and dataset["frames_per_template_family"] == 8 and dataset["paired_item_count"] == 192 and dataset["prompt_count"] == 384 and dataset["prompts_per_condition_per_lexical_family"] == 64 and dataset["dataset_seed"] == 20260831,
        "template_coverage": set(dataset["lexical_families"]) == set(TEMPLATES) and all(len(TEMPLATES[family]) == 8 for family in TEMPLATES),
        "forced_choice_thresholds": fc["minimum_correct_count"] == 40 and fc["maximum_correct_count"] == 56 and fc["denominator"] == 64 and fc["maximum_absolute_condition_count_gap"] == 6,
        "generation_thresholds": generation["minimum_correct_count"] == 16 and generation["maximum_correct_count"] == 56 and generation["denominator"] == 64 and generation["maximum_absolute_condition_count_gap"] == 6,
        "contrast_thresholds": acceptance["maximum_absolute_mean_template_contrast_nats"] == 0.40 and acceptance["maximum_absolute_cluster_dz"] == 0.30 and acceptance["hard_same_sign_template_count_gate_enabled"] is False,
        "no_post_result_selection": acceptance["result_dependent_template_or_family_exclusion_allowed"] is False,
        "human_gate": human["required"] is True and human["waiver_allowed"] is False and human["primary_reviewer_count"] == 2 and human["pairs_per_primary_reviewer"] == 192,
        "dataset_build_allowed": safety["dataset_generation_allowed_after_freeze"] is True,
        "execution_closed": safety["execution_allowed"] is False and safety["model_loading_allowed"] is False and safety["baseline_calibration_execution_allowed"] is False,
        "forbidden_analysis_closed": safety["load_liref_direction"] is False and safety["capture_hidden_states"] is False and safety["forward_hooks"] is False and safety["activation_or_weight_intervention"] is False and safety["stage_e_pilot_allowed"] is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Frozen v3 manifest compatibility failed: {failed}")
    expected_pairs = dataset["lexical_family_count"] * dataset["template_families_per_lexical_family"] * dataset["frames_per_template_family"]
    if expected_pairs != dataset["paired_item_count"] or expected_pairs * 2 != dataset["prompt_count"]:
        raise RuntimeError("Frozen v3 dataset counts are internally inconsistent")
    if len(NAMES) != 24 or len(set(NAMES)) != 24 or set(NAMES) & PRIOR_NAMES:
        raise RuntimeError("v3 entity names must be 24 unique names not used in prior datasets")
    template_ids = [item["id"] for family in TEMPLATES.values() for item in family]
    if len(template_ids) != 24 or len(set(template_ids)) != 24:
        raise RuntimeError("v3 template IDs must be 24 unique values")
    return {
        "schema_version": "1.0",
        "status": "v3_static_schema_compatibility_pass",
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "builder_sha256": sha256_file(Path(__file__)),
        "semantic_checks": checks,
        "oa_validation": oa_report,
        "all_checks_pass": True,
        "tokenizer_trust_remote_code": False,
        "tokenizer_local_files_only": True,
        "model_weights_loaded": False,
        "model_forward_performed": False,
        "gpu_used": False,
        "baseline_calibration_execution_allowed": False,
        "stage_e_pilot_allowed": False,
    }


def validate_design(design: dict[str, Any]) -> dict[str, Any]:
    actual_hash = sha256_file(DESIGN_PATH)
    if actual_hash != EXPECTED_DESIGN_SHA256:
        raise RuntimeError(f"Frozen v3 design hash mismatch: {actual_hash}")
    for version, (path, expected_hash) in PRIOR_DATASETS.items():
        actual = sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(f"Locked prior dataset hash mismatch for {version}: {actual}")
    return validate_schema_compatibility(design)


def load_tokenizer() -> Any:
    actual_hashes = {name: sha256_file(MODEL_DIR / name) for name in EXPECTED_TOKENIZER_HASHES}
    if actual_hashes != EXPECTED_TOKENIZER_HASHES:
        raise RuntimeError(f"Locked tokenizer hash mismatch: {actual_hashes}")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required only for tokenizer-only dataset generation") from error
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=False,
        local_files_only=True,
    )
    return tokenizer


def numeric_mentions(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"(?<![A-Za-z0-9])-?\d+(?![A-Za-z0-9])", text)]


def token_indices_for_span(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [index for index, (left, right) in enumerate(offsets) if right > start and left < end]


def answer_suffix_token_indices(tokenizer: Any, prompt: str, answer: int) -> tuple[list[int], list[int]]:
    joint = prompt + str(answer)
    encoded = tokenizer(joint, add_special_tokens=True, return_offsets_mapping=True)
    indices = token_indices_for_span(
        [tuple(value) for value in encoded["offset_mapping"]],
        len(prompt),
        len(joint),
    )
    return indices, [int(encoded["input_ids"][index]) for index in indices]


def question_skeleton(question: str, target: str) -> str:
    if question.count(target) != 1:
        return "INVALID_TARGET_COUNT"
    return question.replace(target, "<TARGET>")


def choose_numeric_blocks(design: dict[str, Any]) -> dict[tuple[str, int], dict[str, dict[str, int]]]:
    numeric = design["numeric_design"]
    seed = numeric["numeric_seed"]
    candidates: list[dict[str, int]] = []
    for start in range(numeric["candidate_value_min"], numeric["candidate_value_max"] + 1):
        for delta in range(numeric["delta_min"], numeric["delta_max"] + 1):
            low, high = start - delta, start + delta
            if low < numeric["candidate_value_min"] or high > numeric["candidate_value_max"]:
                continue
            if len({low, high, start, delta}) != 4:
                continue
            candidates.append({"low": low, "high": high, "start": start, "delta": delta})
    candidates.sort(key=lambda item: sha256_text(f"{seed}|numeric-pair|{item['low']}|{item['high']}|{item['start']}|{item['delta']}"))
    blocks: dict[tuple[str, int], dict[str, dict[str, int]]] = {}
    used_pair_combinations: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    cursor = 0
    for family in design["dataset_design"]["lexical_families"]:
        for template_index in range(design["dataset_design"]["template_families_per_lexical_family"]):
            chosen: tuple[dict[str, int], dict[str, int]] | None = None
            for left_index in range(cursor, len(candidates)):
                left = candidates[left_index]
                left_values = set(left.values())
                for right in candidates:
                    if left is right or left_values & set(right.values()):
                        continue
                    signature = (
                        tuple(left[key] for key in ("low", "high", "start", "delta")),
                        tuple(right[key] for key in ("low", "high", "start", "delta")),
                    )
                    if signature in used_pair_combinations:
                        continue
                    chosen = (left, right)
                    used_pair_combinations.add(signature)
                    cursor = (left_index + 1) % len(candidates)
                    break
                if chosen is not None:
                    break
            if chosen is None:
                raise RuntimeError("Unable to choose deterministic collision-free numeric pair block")
            blocks[(family, template_index)] = {"P": dict(chosen[0]), "Q": dict(chosen[1])}
    return blocks


def generated_key_candidates() -> list[str]:
    values = [f"{left}{right}" for left in KEY_ONSETS for right in KEY_ENDINGS]
    if len(values) != len(set(values)) or not all(re.fullmatch(r"[A-Za-z]+", value) for value in values):
        raise RuntimeError("Generated key candidate pool is invalid")
    return values


def choose_key_sets(tokenizer: Any, count: int, seed: int) -> list[tuple[str, ...]]:
    prior_key_words = {
        "Kappa", "Tango", "Cedar", "Maple", "Amber", "Falcon", "Heron", "Lotus",
        "Quartz", "Raven", "Sable", "Umber", "Wren", "Zenith", "Delta", "Echo",
        "Indigo", "Jasper", "Onyx", "Pearl", "Silver", "Topaz", "Birch", "Coral",
        "Dahlia", "Elmwood", "Fable", "Garnet", "Harbor", "Ivory", "Juniper", "Linden",
    }
    candidates = [value for value in generated_key_candidates() if value not in prior_key_words]
    candidates.sort(key=lambda value: sha256_text(f"{seed}|key|{value}"))
    by_token_count: dict[int, list[str]] = defaultdict(list)
    for value in candidates:
        token_count = len(tokenizer(value, add_special_tokens=False)["input_ids"])
        by_token_count[token_count].append(value)
    sets: list[tuple[str, ...]] = []
    for token_count in sorted(by_token_count, key=lambda key: (-len(by_token_count[key]), key)):
        values = by_token_count[token_count]
        for start in range(0, len(values) - 7, 8):
            sets.append(tuple(values[start:start + 8]))
            if len(sets) == count:
                return sets
    raise RuntimeError("Insufficient globally unique eight-key sets with equal tokenizer length")


def choose_label_pairs(
    tokenizer: Any,
    templates: list[dict[str, str]],
    names: list[str],
    seed: int,
) -> list[tuple[str, str]]:
    """Choose labels whose complete question encodings are exactly matched.

    Equal standalone token counts are insufficient because a tokenizer can merge
    a label differently at a possessive or punctuation boundary. Selection is
    therefore deterministic and template-specific, using the exact frozen
    question form and offsets around the changed target span.
    """
    if len(templates) != len(names):
        raise RuntimeError("Template/name count mismatch during label selection")
    selected: list[tuple[str, str]] = []
    for index, (template, name) in enumerate(zip(templates, names)):
        candidates = sorted(
            LABEL_PAIRS,
            key=lambda pair: sha256_text(
                f"{seed}|label|{index}|{template['id']}|{pair[0]}|{pair[1]}"
            ),
        )
        valid: list[tuple[str, str]] = []
        for pair in candidates:
            encoded_questions = []
            span_counts = []
            for label in pair:
                question = template["question"].format(name=name, target_label=label)
                probe = f"Context. {question}"
                encoded = tokenizer(probe, add_special_tokens=True, return_offsets_mapping=True)
                label_start = probe.index(label, len("Context. "))
                label_end = label_start + len(label)
                offsets = [tuple(value) for value in encoded["offset_mapping"]]
                encoded_questions.append(len(encoded["input_ids"]))
                span_counts.append(len(token_indices_for_span(offsets, label_start, label_end)))
            if encoded_questions[0] == encoded_questions[1] and span_counts[0] == span_counts[1]:
                valid.append(pair)
        if not valid:
            raise RuntimeError(f"No context-matched label pair for template {template['id']}")
        selected.append(valid[0])
    return selected


def replace_known_tokens(text: str, values: set[str], placeholder: str) -> str:
    for value in sorted(values, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z]){re.escape(value)}(?![A-Za-z])", placeholder, text, flags=re.I)
    return text


def normalized_template_tokens(
    text: str,
    names: set[str],
    labels: set[str],
    keys: set[str],
) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = replace_known_tokens(normalized, {value.lower() for value in names}, " NAME ")
    normalized = replace_known_tokens(normalized, {value.lower() for value in labels}, " LABEL ")
    normalized = replace_known_tokens(normalized, {value.lower() for value in keys}, " KEY ")
    normalized = re.sub(r"(?<![A-Za-z0-9])-?\d+(?![A-Za-z0-9])", " NUM ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return normalized.split()


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)}


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def pair_reference_text(pair: dict[str, Any]) -> str:
    conditions = pair.get("conditions", {})
    questions = []
    for name in ("relevant", "irrelevant", "arithmetic", "selector"):
        if name in conditions:
            questions.append(conditions[name]["question"])
    return " ".join([pair["context"], *questions])


def collect_pair_placeholders(pair: dict[str, Any]) -> tuple[set[str], set[str]]:
    labels: set[str] = set()
    keys: set[str] = set()
    for field in (
        "label_pair", "direct_attribute", "transformed_attribute", "arithmetic_attribute",
        "selector_attribute",
    ):
        value = pair.get(field)
        if isinstance(value, list):
            labels.update(str(item) for item in value)
        elif isinstance(value, str):
            labels.add(value)
    for field in ("record_key_pair", "case_key_pair", "all_case_and_value_keys", "tag_pair"):
        value = pair.get(field)
        if isinstance(value, list):
            keys.update(str(item) for item in value)
    return labels, keys


def compute_near_duplicate_audit(
    design: dict[str, Any],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    rule = design["near_duplicate_nonreuse"]
    n = int(rule["normalized_word_ngram_n"])
    threshold = float(rule["jaccard_fail_threshold_inclusive"])
    all_names = set(NAMES) | PRIOR_NAMES
    all_labels = {label for pair in LABEL_PAIRS for label in pair} | {"red", "blue", "green", "black"}
    all_keys: set[str] = set()
    prior_templates: dict[str, dict[str, dict[str, Any]]] = {}
    for version, (path, _) in PRIOR_DATASETS.items():
        dataset = json.loads(path.read_text(encoding="utf-8"))
        seen: set[str] = set()
        prior_templates[version] = {}
        for pair in dataset["pairs"]:
            template_id = pair["template_family_id"]
            labels, keys = collect_pair_placeholders(pair)
            all_labels.update(labels)
            all_keys.update(keys)
            if template_id in seen:
                continue
            seen.add(template_id)
            text = pair_reference_text(pair)
            tokens = normalized_template_tokens(text, all_names, all_labels, all_keys | keys)
            prior_templates[version][template_id] = {
                "tokens": tokens,
                "hash": sha256_text(" ".join(tokens)),
                "ngrams": ngrams(tokens, n),
            }

    for pair in pairs:
        _, keys = collect_pair_placeholders(pair)
        all_keys.update(keys)

    first_by_template: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        first_by_template.setdefault(pair["template_family_id"], pair)
    reports: dict[str, Any] = {}
    v3_hashes: dict[str, str] = {}
    for template_id, pair in sorted(first_by_template.items()):
        labels, keys = collect_pair_placeholders(pair)
        tokens = normalized_template_tokens(
            pair_reference_text(pair), all_names, all_labels | labels, all_keys | keys
        )
        skeleton_hash = sha256_text(" ".join(tokens))
        grams = ngrams(tokens, n)
        comparisons = []
        for version, templates in prior_templates.items():
            for prior_id, info in templates.items():
                similarity = jaccard(grams, info["ngrams"])
                comparisons.append({
                    "version": version,
                    "template_family_id": prior_id,
                    "jaccard": similarity,
                    "exact_hash": skeleton_hash == info["hash"],
                })
        strongest = max(comparisons, key=lambda item: item["jaccard"])
        passes = not any(item["exact_hash"] or item["jaccard"] >= threshold for item in comparisons)
        reports[template_id] = {
            "normalized_skeleton_sha256": skeleton_hash,
            "normalized_token_count": len(tokens),
            "strongest_prior_match": strongest,
            "threshold_inclusive": threshold,
            "pass": passes,
        }
        v3_hashes[template_id] = skeleton_hash
    if len(set(v3_hashes.values())) != len(v3_hashes):
        raise RuntimeError("Duplicate normalized skeleton across distinct v3 template families")
    failed = [template_id for template_id, report in reports.items() if not report["pass"]]
    if failed:
        raise RuntimeError(f"Prior-to-v3 near-duplicate audit failed: {failed}")
    return {
        "normalization": rule["normalization"],
        "word_ngram_n": n,
        "jaccard_fail_threshold_inclusive": threshold,
        "comparison_scope": rule["comparison_scope"],
        "template_reports": reports,
        "all_cross_version_checks_pass": True,
        "all_v3_template_skeletons_unique": True,
    }


def ensure_output_targets(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise RuntimeError(f"Refusing to overwrite existing outputs: {[str(path) for path in existing]}")


def format_mapping(template: dict[str, str], name: str, label: str, case_key: str) -> str:
    return template["mapping"].format(name=name, label=label, case_key=case_key)


def build(output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    schema_report = validate_design(design)
    ensure_output_targets(output_dir, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / OUTPUT_FILENAMES[0]
    atomic_json(schema_path, schema_report)

    tokenizer = load_tokenizer()
    tokenizer_hashes = {name: sha256_file(MODEL_DIR / name) for name in EXPECTED_TOKENIZER_HASHES}
    blocks = choose_numeric_blocks(design)
    key_sets = choose_key_sets(
        tokenizer,
        design["dataset_design"]["independent_template_family_count"],
        design["dataset_design"]["dataset_seed"],
    )
    ordered_templates = [
        template
        for family in design["dataset_design"]["lexical_families"]
        for template in TEMPLATES[family]
    ]
    label_pairs = choose_label_pairs(
        tokenizer,
        ordered_templates,
        NAMES[: len(ordered_templates)],
        design["dataset_design"]["dataset_seed"],
    )
    matrix = design["counterbalance"]["matrix"]
    prompt_template = design["prompt_output_contract"]["template"]
    pairs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    template_cursor = 0

    for family_index, family in enumerate(design["dataset_design"]["lexical_families"]):
        for template_index, template in enumerate(TEMPLATES[family]):
            name = NAMES[family_index * 8 + template_index]
            label_pair = label_pairs[template_cursor]
            keys = key_sets[template_cursor]
            case_pair = keys[0:2]
            value_keys_p = keys[2:4]
            value_keys_q = keys[4:6]
            tag_pair = keys[6:8]
            numeric_block = blocks[(family, template_index)]
            template_cursor += 1

            for frame_index, row in enumerate(matrix, start=1):
                factor_a, factor_b, factor_c, factor_d, factor_e = row
                arithmetic_pair_name, selector_pair_name = (
                    ("P", "Q") if factor_a == 0 else ("Q", "P")
                )
                arithmetic_pair = numeric_block[arithmetic_pair_name]
                selector_pair = numeric_block[selector_pair_name]
                arithmetic_value_keys = value_keys_p if arithmetic_pair_name == "P" else value_keys_q
                selector_value_keys = value_keys_q if selector_pair_name == "Q" else value_keys_p
                operation = "add" if factor_b == 0 else "subtract"
                operation_word = template["add_word"] if factor_b == 0 else template["sub_word"]
                arithmetic_correct = arithmetic_pair["high"] if factor_b == 0 else arithmetic_pair["low"]
                arithmetic_alternative = arithmetic_pair["low"] if factor_b == 0 else arithmetic_pair["high"]
                selector_correct = selector_pair["low"] if factor_c == 0 else selector_pair["high"]
                selector_alternative = selector_pair["high"] if factor_c == 0 else selector_pair["low"]
                active_tag = tag_pair[factor_c]
                arithmetic_label, selector_label = (
                    label_pair if factor_d == 0 else tuple(reversed(label_pair))
                )
                arithmetic_case_key, selector_case_key = (
                    case_pair if factor_d == 0 else tuple(reversed(case_pair))
                )

                arithmetic_mapping = format_mapping(template, name, arithmetic_label, arithmetic_case_key)
                selector_mapping = format_mapping(template, name, selector_label, selector_case_key)
                arithmetic_case = template["arithmetic_case"].format(
                    case_key=arithmetic_case_key,
                    value_key_1=arithmetic_value_keys[0],
                    value_key_2=arithmetic_value_keys[1],
                    operation=operation_word,
                )
                selector_case = template["selector_case"].format(
                    case_key=selector_case_key,
                    active_tag=active_tag,
                    tag_1=tag_pair[0],
                    tag_2=tag_pair[1],
                    value_key_1=selector_value_keys[0],
                    value_key_2=selector_value_keys[1],
                )
                arithmetic_block = f"{arithmetic_mapping} {arithmetic_case}"
                selector_block = f"{selector_mapping} {selector_case}"
                if factor_e == 0:
                    channel_blocks = [arithmetic_block, selector_block]
                    ledger_entries = [
                        (arithmetic_value_keys[0], arithmetic_pair["start"]),
                        (arithmetic_value_keys[1], arithmetic_pair["delta"]),
                        (selector_value_keys[0], selector_pair["low"]),
                        (selector_value_keys[1], selector_pair["high"]),
                    ]
                else:
                    channel_blocks = [selector_block, arithmetic_block]
                    ledger_entries = [
                        (selector_value_keys[0], selector_pair["low"]),
                        (selector_value_keys[1], selector_pair["high"]),
                        (arithmetic_value_keys[0], arithmetic_pair["start"]),
                        (arithmetic_value_keys[1], arithmetic_pair["delta"]),
                    ]
                ledger = template["ledger"].format(
                    key_1=ledger_entries[0][0], value_1=ledger_entries[0][1],
                    key_2=ledger_entries[1][0], value_2=ledger_entries[1][1],
                    key_3=ledger_entries[2][0], value_3=ledger_entries[2][1],
                    key_4=ledger_entries[3][0], value_4=ledger_entries[3][1],
                )
                context = " ".join([*channel_blocks, ledger])
                arithmetic_question = template["question"].format(
                    name=name, target_label=arithmetic_label
                )
                selector_question = template["question"].format(
                    name=name, target_label=selector_label
                )
                pair_id = f"calv3_{family}_{template_index:02d}_f{frame_index:02d}"
                condition_payload: dict[str, Any] = {}
                prompt_lengths: dict[str, int] = {}
                target_token_counts: dict[str, int] = {}

                for condition, question, correct, alternative, target_label, source_block, role in (
                    ("arithmetic", arithmetic_question, arithmetic_correct, arithmetic_alternative,
                     arithmetic_label, arithmetic_block, "relevant"),
                    ("selector", selector_question, selector_correct, selector_alternative,
                     selector_label, selector_block, "irrelevant"),
                ):
                    full_question_text = f"{context} {question}"
                    full_prompt = prompt_template.format(question=full_question_text)
                    encoded = tokenizer(full_prompt, add_special_tokens=True, return_offsets_mapping=True)
                    offsets = [tuple(value) for value in encoded["offset_mapping"]]
                    question_start = full_prompt.index(question)
                    target_start = question_start + question.index(target_label)
                    target_end = target_start + len(target_label)
                    source_start = full_prompt.index(source_block)
                    source_end = source_start + len(source_block)
                    target_indices = token_indices_for_span(offsets, target_start, target_end)
                    source_indices = token_indices_for_span(offsets, source_start, source_end)
                    answer_indices, answer_ids = answer_suffix_token_indices(tokenizer, full_prompt, correct)
                    alternative_indices, alternative_ids = answer_suffix_token_indices(tokenizer, full_prompt, alternative)
                    prompt_lengths[condition] = len(encoded["input_ids"])
                    target_token_counts[condition] = len(target_indices)
                    condition_payload[condition] = {
                        "condition": condition,
                        "legacy_role": role,
                        "question": question,
                        "full_question_text": full_question_text,
                        "full_prompt": full_prompt,
                        "target_attribute": target_label,
                        "canonical_answer": str(correct),
                        "primary_alternative_answer": str(alternative),
                        "accepted_answers": [str(correct)],
                        "canonical_answer_continuation_token_count": len(answer_indices),
                        "canonical_answer_continuation_token_ids": answer_ids,
                        "alternative_answer_continuation_token_count": len(alternative_indices),
                        "alternative_answer_continuation_token_ids": alternative_ids,
                        "changed_question_span": [
                            question.index(target_label),
                            question.index(target_label) + len(target_label),
                        ],
                        "changed_question_prompt_token_indices": target_indices,
                        "changed_question_token_ids": [
                            int(encoded["input_ids"][index]) for index in target_indices
                        ],
                        "source_block_prompt_token_indices": source_indices,
                        "source_block_token_ids": [
                            int(encoded["input_ids"][index]) for index in source_indices
                        ],
                        "prompt_token_count": len(encoded["input_ids"]),
                    }

                context_numbers = numeric_mentions(context)
                arithmetic_numbers = numeric_mentions(arithmetic_block)
                selector_numbers = numeric_mentions(ledger)
                all_key_token_counts = [
                    len(tokenizer(key, add_special_tokens=False)["input_ids"]) for key in keys
                ]
                checks = {
                    "arithmetic_correct_absent": context_numbers.count(arithmetic_correct) == 0,
                    "arithmetic_alternative_absent": context_numbers.count(arithmetic_alternative) == 0,
                    "selector_correct_once_in_ledger": selector_numbers.count(selector_correct) == 1,
                    "selector_alternative_once_in_ledger": selector_numbers.count(selector_alternative) == 1,
                    "arithmetic_operands_each_once_in_ledger": context_numbers.count(arithmetic_pair["start"]) == 1 and context_numbers.count(arithmetic_pair["delta"]) == 1,
                    "wrong_operation_primary_foil": arithmetic_alternative == (arithmetic_pair["start"] - arithmetic_pair["delta"] if factor_b == 0 else arithmetic_pair["start"] + arithmetic_pair["delta"]),
                    "numeric_collision_absence": len({arithmetic_correct, arithmetic_alternative, arithmetic_pair["start"], arithmetic_pair["delta"], selector_correct, selector_alternative}) == 6,
                    "arithmetic_case_has_two_value_keys": all(arithmetic_case.count(key) == 1 for key in arithmetic_value_keys),
                    "selector_case_has_two_entries": all(selector_case.count(key) == 1 for key in selector_value_keys) and all(selector_case.count(tag) >= 1 for tag in tag_pair),
                    "selector_active_tag_explicit": selector_case.count(active_tag) >= 2,
                    "case_keys_each_appear_twice": context.count(arithmetic_case_key) == 2 and context.count(selector_case_key) == 2,
                    "value_keys_each_appear_twice": all(context.count(key) == 2 for key in (*value_keys_p, *value_keys_q)),
                    "all_keys_alphabetic": all(re.fullmatch(r"[A-Za-z]+", key) for key in keys),
                    "all_keys_equal_token_count": len(set(all_key_token_counts)) == 1,
                    "exact_prompt_token_count_match": prompt_lengths["arithmetic"] == prompt_lengths["selector"],
                    "question_differs_only_at_target_label": question_skeleton(arithmetic_question, arithmetic_label) == question_skeleton(selector_question, selector_label),
                    "target_label_token_count_match": target_token_counts["arithmetic"] == target_token_counts["selector"],
                    "canonical_answers_one_token": all(condition_payload[name_]["canonical_answer_continuation_token_count"] == 1 for name_ in ("arithmetic", "selector")),
                    "primary_alternatives_one_token": all(condition_payload[name_]["alternative_answer_continuation_token_count"] == 1 for name_ in ("arithmetic", "selector")),
                    "source_block_tokens_nonempty": all(condition_payload[name_]["source_block_token_ids"] for name_ in ("arithmetic", "selector")),
                    "answer_regex_contract": all(re.fullmatch(r"[0-9]+", condition_payload[name_]["canonical_answer"]) for name_ in ("arithmetic", "selector")),
                    "arithmetic_block_has_no_numeric_literal": len(arithmetic_numbers) == 0,
                }
                failed_checks = [name_ for name_, passed in checks.items() if not passed]
                if failed_checks:
                    failures.append({"pair_id": pair_id, "failed_checks": failed_checks})

                pairs.append({
                    "pair_id": pair_id,
                    "dataset_version": "3.0",
                    "lexical_family": family,
                    "template_family_id": template["id"],
                    "frame_index": frame_index,
                    "oa_row": list(row),
                    "factors": {
                        "A_candidate_pair_assignment": factor_a,
                        "B_arithmetic_operation": factor_b,
                        "C_selector_active_entry": factor_c,
                        "D_label_case_role": factor_d,
                        "E_channel_block_order": factor_e,
                    },
                    "numeric_block_id": f"v3_block_{family}_{template_index:02d}",
                    "numeric_block": numeric_block,
                    "arithmetic_candidate_pair": arithmetic_pair_name,
                    "selector_candidate_pair": selector_pair_name,
                    "operation": operation,
                    "operation_word": operation_word,
                    "selector_active_entry": factor_c + 1,
                    "name": name,
                    "label_pair": list(label_pair),
                    "arithmetic_attribute": arithmetic_label,
                    "selector_attribute": selector_label,
                    "case_key_pair": list(case_pair),
                    "arithmetic_case_key": arithmetic_case_key,
                    "selector_case_key": selector_case_key,
                    "value_key_pair_P": list(value_keys_p),
                    "value_key_pair_Q": list(value_keys_q),
                    "tag_pair": list(tag_pair),
                    "all_case_and_value_keys": list(keys),
                    "channel_block_order": "arithmetic_first" if factor_e == 0 else "selector_first",
                    "context": context,
                    "arithmetic_mapping_sentence": arithmetic_mapping,
                    "arithmetic_case_sentence": arithmetic_case,
                    "arithmetic_block": arithmetic_block,
                    "selector_mapping_sentence": selector_mapping,
                    "selector_case_sentence": selector_case,
                    "selector_block": selector_block,
                    "value_ledger_sentence": ledger,
                    "start": arithmetic_pair["start"],
                    "delta": arithmetic_pair["delta"],
                    "arithmetic_answer": str(arithmetic_correct),
                    "arithmetic_primary_foil": str(arithmetic_alternative),
                    "selector_answer": str(selector_correct),
                    "selector_primary_foil": str(selector_alternative),
                    "conditions": condition_payload,
                    "automatic_audit": checks,
                    "ai_audit_status": "pending",
                    "human_audit_required": True,
                    "human_reviewer_1_status": "pending",
                    "human_reviewer_2_status": "pending",
                    "final_human_review_status": "pending",
                    "pilot_reuse_allowed": False,
                    "confirmatory_reuse_allowed": False,
                })

    near_duplicate_audit = compute_near_duplicate_audit(design, pairs)
    template_audits: dict[str, dict[str, Any]] = {}
    for template_id in sorted({pair["template_family_id"] for pair in pairs}):
        selected = [pair for pair in pairs if pair["template_family_id"] == template_id]
        rows = [tuple(pair["oa_row"]) for pair in selected]
        factor_level_checks = {
            f"factor_{name}_4_to_4": [row[index] for row in rows].count(0) == 4 and [row[index] for row in rows].count(1) == 4
            for index, name in enumerate("ABCDE")
        }
        pairwise_checks: dict[str, bool] = {}
        for left, right in combinations(range(5), 2):
            pairwise_checks[f"pair_{'ABCDE'[left]}{'ABCDE'[right]}_strength_2"] = all(
                sum(row[left] == a and row[right] == b for row in rows) == 2
                for a in (0, 1) for b in (0, 1)
            )
        role_counts: dict[str, dict[str, int]] = {}
        for pair_name in ("P", "Q"):
            role_counts[pair_name] = {
                "arithmetic_correct_low": sum(pair["arithmetic_candidate_pair"] == pair_name and pair["operation"] == "subtract" for pair in selected),
                "arithmetic_correct_high": sum(pair["arithmetic_candidate_pair"] == pair_name and pair["operation"] == "add" for pair in selected),
                "selector_correct_low": sum(pair["selector_candidate_pair"] == pair_name and pair["selector_active_entry"] == 1 for pair in selected),
                "selector_correct_high": sum(pair["selector_candidate_pair"] == pair_name and pair["selector_active_entry"] == 2 for pair in selected),
            }
        checks = {
            "pair_count_eight": len(selected) == 8,
            "exact_frozen_oa_rows": rows == [tuple(row) for row in matrix],
            "unique_oa_rows": len(set(rows)) == 8,
            **factor_level_checks,
            **pairwise_checks,
            "candidate_pair_P_roles_balanced": set(role_counts["P"].values()) == {2},
            "candidate_pair_Q_roles_balanced": set(role_counts["Q"].values()) == {2},
            "cross_version_near_duplicate_pass": near_duplicate_audit["template_reports"][template_id]["pass"],
        }
        template_audits[template_id] = {"checks": checks, "numeric_role_counts": role_counts}
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            failures.append({"template_family_id": template_id, "failed_checks": failed_checks})

    expected_pairs = design["dataset_design"]["paired_item_count"]
    family_counts = {family: sum(pair["lexical_family"] == family for pair in pairs) for family in design["dataset_design"]["lexical_families"]}
    operation_counts = {
        family: {
            operation: sum(pair["lexical_family"] == family and pair["operation"] == operation for pair in pairs)
            for operation in ("add", "subtract")
        }
        for family in design["dataset_design"]["lexical_families"]
    }
    global_checks = {
        "pair_count_192": len(pairs) == expected_pairs,
        "prompt_count_384": len(pairs) * 2 == design["dataset_design"]["prompt_count"],
        "template_family_count_24": len(template_audits) == 24,
        "unique_pair_ids": len({pair["pair_id"] for pair in pairs}) == len(pairs),
        "each_family_has_64_pairs": all(count == 64 for count in family_counts.values()),
        "each_family_add_sub_32_to_32": all(counts == {"add": 32, "subtract": 32} for counts in operation_counts.values()),
        "all_pair_checks_pass": all(all(pair["automatic_audit"].values()) for pair in pairs),
        "all_template_checks_pass": all(all(report["checks"].values()) for report in template_audits.values()),
        "all_names_new_and_unique": len(set(NAMES)) == 24 and not bool(set(NAMES) & PRIOR_NAMES),
        "all_key_strings_globally_unique": len({key for key_set in key_sets for key in key_set}) == 192,
        "all_cross_version_near_duplicate_checks_pass": near_duplicate_audit["all_cross_version_checks_pass"],
        "all_v3_template_skeletons_unique": near_duplicate_audit["all_v3_template_skeletons_unique"],
    }
    if not all(global_checks.values()):
        failures.append({"global_failed_checks": [name for name, passed in global_checks.items() if not passed]})
    if failures:
        raise RuntimeError(f"Automatic v3 audit failed: {failures[:12]}")

    dataset_path = output_dir / OUTPUT_FILENAMES[1]
    audit_path = output_dir / OUTPUT_FILENAMES[2]
    manifest_path = output_dir / OUTPUT_FILENAMES[3]
    dataset = {
        "schema_version": "3.0",
        "status": "dataset_created_automatic_audit_pass_ai_and_human_audits_pending",
        "design_id": design["design_id"],
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "operationalized_feature": design["feature"]["operationalized_name"],
        "pair_count": len(pairs),
        "prompt_count": len(pairs) * 2,
        "independent_template_family_count": len(template_audits),
        "random_seed": design["dataset_design"]["dataset_seed"],
        "model_results_unavailable_at_generation": True,
        "human_audit_required": True,
        "human_audit_waiver_allowed": False,
        "pilot_reuse_allowed": False,
        "confirmatory_reuse_allowed": False,
        "pairs": pairs,
    }
    atomic_json(dataset_path, dataset)
    pair_check_totals = {
        check: sum(int(pair["automatic_audit"][check]) for pair in pairs)
        for check in sorted(pairs[0]["automatic_audit"])
    }
    audit = {
        "schema_version": "3.0",
        "status": "automatic_shortcut_nonreuse_OA_counterbalance_structural_tokenizer_audit_pass",
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "builder_sha256": sha256_file(Path(__file__)),
        "prior_dataset_hashes": {version: expected for version, (_, expected) in PRIOR_DATASETS.items()},
        "tokenizer_hashes": tokenizer_hashes,
        "pair_count": len(pairs),
        "prompt_count": len(pairs) * 2,
        "template_family_count": len(template_audits),
        "family_counts": family_counts,
        "operation_counts": operation_counts,
        "pair_check_pass_counts": pair_check_totals,
        "template_audits": template_audits,
        "near_duplicate_audit": near_duplicate_audit,
        "global_checks": global_checks,
        "all_automatic_checks_pass": True,
        "ai_audit_complete": False,
        "independent_human_audit_complete": False,
        "model_weights_loaded": False,
        "model_forward_performed": False,
        "gpu_used": False,
        "baseline_calibration_execution_allowed": False,
        "stage_e_pilot_allowed": False,
    }
    atomic_json(audit_path, audit)
    dataset_manifest = {
        "schema_version": "3.0",
        "status": "dataset_created_automatic_audit_pass_ai_and_human_audits_pending",
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "builder_sha256": sha256_file(Path(__file__)),
        "static_schema_check_sha256": sha256_file(schema_path),
        "dataset_sha256": sha256_file(dataset_path),
        "automatic_audit_sha256": sha256_file(audit_path),
        "prior_dataset_hashes": {version: expected for version, (_, expected) in PRIOR_DATASETS.items()},
        "tokenizer_hashes": tokenizer_hashes,
        "pair_count": len(pairs),
        "prompt_count": len(pairs) * 2,
        "template_family_count": len(template_audits),
        "frames_per_template_family": len(matrix),
        "random_seed": design["dataset_design"]["dataset_seed"],
        "automatic_audit_pass": True,
        "ai_audit_complete": False,
        "independent_human_audit_complete": False,
        "human_reviewer_1_id": None,
        "human_reviewer_2_id": None,
        "pilot_reuse_allowed": False,
        "confirmatory_reuse_allowed": False,
        "baseline_calibration_execution_allowed": False,
        "stage_e_pilot_allowed": False,
        "tokenizer_trust_remote_code": False,
        "tokenizer_local_files_only": True,
        "model_weights_loaded": False,
        "model_forward_performed": False,
        "gpu_used": False,
    }
    atomic_json(manifest_path, dataset_manifest)
    return dataset_manifest


def main() -> None:
    args = parse_args()
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if args.validate_only:
        report = validate_design(design)
        output_dir = args.output_dir.resolve()
        ensure_output_targets(output_dir, args.overwrite)
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / OUTPUT_FILENAMES[0], report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return
    result = build(args.output_dir.resolve(), overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
