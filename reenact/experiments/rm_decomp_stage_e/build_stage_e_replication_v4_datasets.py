#!/usr/bin/env python3
"""Build two disjoint, model-free Stage E replication v4 datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
STAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "AI" / "reenact" / "models" / "Meta-Llama-3-8B"
DESIGN_PATH = STAGE_DIR / "stage_e_independent_replication_v4_design_frozen.json"
CATALOG_PATH = STAGE_DIR / "stage_e_replication_v4_template_catalog_frozen.json"
CATALOG_AMENDMENT_PATH = STAGE_DIR / "stage_e_replication_v4_0_1_catalog_amendment_frozen.json"
PRIOR_V3_PATH = STAGE_DIR / "calibration_v3_assets" / "calibration_v3_dataset_draft.json"
DEFAULT_OUTPUT_DIR = STAGE_DIR / "stage_e_replication_v4_assets"

EXPECTED_HASHES = {
    "design": "0382a059f2ac3578446e772939a10dc6911d11b7a90bb4cb0f7bd78ed5ebe106",
    "catalog": "d49fab4ceb9be75ec6d9ec2549b433abcbb108fb57862d12659232a3b7fe186b",
    "catalog_amendment": "0cd26c72db2c3510e4013c4427e25ab3622796b3f6e9f202100c5e7e4e5b68cb",
    "prior_v3": "d2187c0623ba9752776cf0251dee3dabf9d80ac04e339cf3eb4bd1d1b42761a1",
    "tokenizer.json": "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
    "tokenizer_config.json": "690727b4fed286383df1c7ca5e805124cb70c6eb4529f807c7b2e60ff741da7e",
    "special_tokens_map.json": "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
}
OA_MATRIX = (
    (0, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 1, 1),
    (0, 1, 0, 1, 0, 1),
    (0, 1, 1, 1, 1, 0),
    (1, 0, 0, 1, 1, 0),
    (1, 0, 1, 1, 0, 1),
    (1, 1, 0, 0, 1, 1),
    (1, 1, 1, 0, 0, 0),
)
FAMILIES = ("points_balance", "temperature")
POOLS = ("calibration", "replication")
CONDITIONS = ("arithmetic", "selector")
SEED = 20260901

KEY_ONSETS = (
    "Ba", "Be", "Bi", "Bo", "Ca", "Ce", "Ci", "Co", "Da", "De", "Di", "Do",
    "Fa", "Fe", "Fi", "Fo", "Ga", "Ge", "Gi", "Go", "Ha", "He", "Hi", "Ho",
    "Ja", "Je", "Ji", "Jo", "Ka", "Ke", "Ki", "Ko", "La", "Le", "Li", "Lo",
    "Ma", "Me", "Mi", "Mo", "Na", "Ne", "Ni", "No", "Pa", "Pe", "Pi", "Po",
    "Ra", "Re", "Ri", "Ro", "Sa", "Se", "Si", "So", "Ta", "Te", "Ti", "To",
    "Va", "Ve", "Vi", "Vo", "Za", "Ze", "Zi", "Zo",
)
KEY_ENDINGS = ("bex", "dor", "fen", "gis", "hal", "jor", "kel", "lin", "mor", "nus", "pel", "rin", "sov", "tal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_locked_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    actual = {
        "design": sha256_file(DESIGN_PATH),
        "catalog": sha256_file(CATALOG_PATH),
        "catalog_amendment": sha256_file(CATALOG_AMENDMENT_PATH),
        "prior_v3": sha256_file(PRIOR_V3_PATH),
        **{
            name: sha256_file(MODEL_DIR / name)
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        },
    }
    if actual != EXPECTED_HASHES:
        raise RuntimeError(f"Locked input hash mismatch: {actual}")
    design = load_json(DESIGN_PATH)
    catalog = load_json(CATALOG_PATH)
    catalog_amendment = load_json(CATALOG_AMENDMENT_PATH)
    if catalog_amendment["base_catalog_sha256"] != EXPECTED_HASHES["catalog"]:
        raise RuntimeError("Catalog amendment does not lock the expected base catalog")
    for replacement in catalog_amendment["replacements"]:
        pool = replacement["pool"]
        index = replacement["pair_index_zero_based"]
        if catalog["pools"][pool]["label_pairs"][index] != replacement["old"]:
            raise RuntimeError("Catalog amendment old label pair does not match base catalog")
        catalog["pools"][pool]["label_pairs"][index] = replacement["new"]
    prior = load_json(PRIOR_V3_PATH)
    checks = {
        "design_frozen": design["status"] == "design_frozen_dataset_build_allowed_model_execution_not_authorized",
        "catalog_frozen": catalog["status"] == "template_catalog_frozen_model_execution_not_authorized",
        "seed": design["numeric_design"]["seed"] == SEED == catalog["random_seed"],
        "pool_counts": all(design["pools"][pool]["pair_count"] == 128 for pool in POOLS),
        "model_execution_closed": design["permissions"]["model_loading_allowed"] is False
        and design["permissions"]["gpu_forward_allowed"] is False,
        "intervention_closed": design["permissions"]["intervention_allowed"] is False,
        "catalog_sizes": len(catalog["calibration_styles"]) == 8
        and len(catalog["replication_styles"]) == 8,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen design semantic check failed: {checks}")
    return design, catalog, prior


def validate_oa() -> dict[str, Any]:
    if len(OA_MATRIX) != 8 or any(len(row) != 6 for row in OA_MATRIX):
        raise RuntimeError("OA matrix must be 8x6")
    factor_counts = []
    for column in range(6):
        counts = Counter(row[column] for row in OA_MATRIX)
        if counts != Counter({0: 4, 1: 4}):
            raise RuntimeError(f"OA column {column} imbalance: {counts}")
        factor_counts.append(dict(counts))
    pair_counts = {}
    for left, right in combinations(range(6), 2):
        counts = Counter((row[left], row[right]) for row in OA_MATRIX)
        if set(counts.values()) != {2} or len(counts) != 4:
            raise RuntimeError(f"OA pair {left},{right} imbalance: {counts}")
        pair_counts[f"{left}_{right}"] = {f"{a}{b}": counts[(a, b)] for a in (0, 1) for b in (0, 1)}
    return {"pass": True, "factor_counts": factor_counts, "pair_counts": pair_counts}


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False, use_fast=True
    )


def suffix_token_ids(tokenizer: Any, prompt: str, continuation: str) -> list[int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    joint_ids = tokenizer(prompt + continuation, add_special_tokens=True)["input_ids"]
    if joint_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Prompt token prefix changed when answer continuation was appended")
    return [int(value) for value in joint_ids[len(prompt_ids) :]]


def make_global_key_assignments(tokenizer: Any, template_keys: list[str]) -> dict[str, list[str]]:
    candidates = [onset + ending for onset in KEY_ONSETS for ending in KEY_ENDINGS]
    candidates.sort(key=lambda value: hashlib.sha256(f"{SEED}|key|{value}".encode()).hexdigest())
    by_length: dict[int, list[str]] = defaultdict(list)
    for value in candidates:
        by_length[len(tokenizer.encode(" " + value, add_special_tokens=False))].append(value)
    required = len(template_keys) * 8
    eligible = sorted(
        ((length, values) for length, values in by_length.items() if len(values) >= required),
        key=lambda item: (item[0], -len(item[1])),
    )
    if not eligible:
        raise RuntimeError(f"No uniform tokenizer-length key pool has {required} values")
    token_length, values = eligible[0]
    output = {}
    for index, template_key in enumerate(template_keys):
        output[template_key] = values[index * 8 : (index + 1) * 8]
    if len({value for group in output.values() for value in group}) != required:
        raise RuntimeError("Generated keys are not globally unique")
    if token_length < 1:
        raise RuntimeError("Invalid key token length")
    return output


def numeric_block(template_key: str, used: set[tuple[int, ...]]) -> dict[str, int]:
    rng = random.Random(stable_int(f"{SEED}|numeric|{template_key}"))
    for _ in range(100000):
        start = rng.randint(40, 69)
        delta = rng.randint(2, 9)
        low, high = start - delta, start + delta
        selector_1 = rng.randint(30, 79)
        selector_2 = rng.randint(30, 79)
        candidate_values = {low, high, selector_1, selector_2}
        if len(candidate_values) != 4:
            continue
        if selector_1 in {start, delta} or selector_2 in {start, delta}:
            continue
        if any(value < 10 or value > 99 for value in candidate_values):
            continue
        signature = (start, delta, low, high, selector_1, selector_2)
        if signature in used:
            continue
        used.add(signature)
        return {
            "start": start,
            "delta": delta,
            "low": low,
            "high": high,
            "selector_1": selector_1,
            "selector_2": selector_2,
        }
    raise RuntimeError(f"Could not create numeric block for {template_key}")


def numeric_mentions(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"(?<![A-Za-z0-9])-?\d+(?![A-Za-z0-9])", text)]


def render_pair(
    *, pool: str, family: str, style: dict[str, str], style_index: int,
    frame_index: int, oa_row: tuple[int, ...], name: str,
    labels: tuple[str, str], item: str, keys: list[str], block: dict[str, int],
    tokenizer: Any,
) -> dict[str, Any]:
    operation_bit, active_bit, arithmetic_choice_bit, selector_choice_bit, label_bit, order_bit = oa_row
    operation = "ADD" if operation_bit == 0 else "SUBTRACT"
    arithmetic_answer = block["high"] if operation_bit == 0 else block["low"]
    arithmetic_foil = block["low"] if operation_bit == 0 else block["high"]
    selector_answer = block["selector_1"] if active_bit == 0 else block["selector_2"]
    selector_foil = block["selector_2"] if active_bit == 0 else block["selector_1"]
    arithmetic_correct_choice = "A" if arithmetic_choice_bit == 0 else "B"
    selector_correct_choice = "A" if selector_choice_bit == 0 else "B"
    arithmetic_label, selector_label = labels if label_bit == 0 else (labels[1], labels[0])
    arithmetic_case, selector_case, tag_1, tag_2, start_key, change_key, selector_key_1, selector_key_2 = keys
    active_tag = tag_1 if active_bit == 0 else tag_2

    arithmetic_choices = {
        arithmetic_correct_choice: arithmetic_answer,
        "B" if arithmetic_correct_choice == "A" else "A": arithmetic_foil,
    }
    selector_key_for_choice = {
        selector_correct_choice: selector_key_1 if active_bit == 0 else selector_key_2,
        "B" if selector_correct_choice == "A" else "A": selector_key_2 if active_bit == 0 else selector_key_1,
    }
    common = {"name": name, "item": item}
    arithmetic_mapping = style["mapping"].format(
        **common, label=arithmetic_label, case_key=arithmetic_case
    )
    arithmetic_case_text = style["arithmetic"].format(
        case_key=arithmetic_case,
        key_1=start_key,
        key_2=change_key,
        operation=operation,
        choice_a=arithmetic_choices["A"],
        choice_b=arithmetic_choices["B"],
    )
    selector_mapping = style["mapping"].format(
        **common, label=selector_label, case_key=selector_case
    )
    selector_case_text = style["selector"].format(
        case_key=selector_case,
        active_tag=active_tag,
        tag_1=tag_1,
        tag_2=tag_2,
        key_1=selector_key_1,
        key_2=selector_key_2,
        choice_a=selector_key_for_choice["A"],
        choice_b=selector_key_for_choice["B"],
    )
    ledger = style["ledger"].format(
        key_1=start_key,
        value_1=block["start"],
        key_2=change_key,
        value_2=block["delta"],
        key_3=selector_key_1,
        value_3=block["selector_1"],
        key_4=selector_key_2,
        value_4=block["selector_2"],
    )
    arithmetic_block = f"{arithmetic_mapping} {arithmetic_case_text}"
    selector_block = f"{selector_mapping} {selector_case_text}"
    ordered_blocks = (
        (arithmetic_block, selector_block) if order_bit == 0 else (selector_block, arithmetic_block)
    )
    context = f"{ordered_blocks[0]} {ordered_blocks[1]} {ledger}"
    questions = {
        "arithmetic": style["question"].format(
            **common, target_label=arithmetic_label
        ),
        "selector": style["question"].format(
            **common, target_label=selector_label
        ),
    }
    conditions = {}
    for condition, correct_choice, answer, foil in (
        ("arithmetic", arithmetic_correct_choice, arithmetic_answer, arithmetic_foil),
        ("selector", selector_correct_choice, selector_answer, selector_foil),
    ):
        prompt = (
            f"{context}\nQuestion: {questions[condition]}\n"
            "Answer with A or B only.\nAnswer:"
        )
        alternative_choice = "B" if correct_choice == "A" else "A"
        correct_continuation = " " + correct_choice
        alternative_continuation = " " + alternative_choice
        conditions[condition] = {
            "question": questions[condition],
            "full_prompt": prompt,
            "correct_choice": correct_choice,
            "alternative_choice": alternative_choice,
            "correct_continuation": correct_continuation,
            "alternative_continuation": alternative_continuation,
            "correct_choice_token_ids": suffix_token_ids(tokenizer, prompt, correct_continuation),
            "alternative_choice_token_ids": suffix_token_ids(tokenizer, prompt, alternative_continuation),
            "canonical_numeric_result": answer,
            "foil_numeric_result": foil,
        }
    prompt_lengths = {
        condition: len(tokenizer(conditions[condition]["full_prompt"], add_special_tokens=True)["input_ids"])
        for condition in CONDITIONS
    }
    pair_id = f"v4_{pool}_{family}_{style_index:02d}_f{frame_index:02d}"
    template_id = f"v4_{pool}_{family}_{style['style_id']}"
    return {
        "pair_id": pair_id,
        "pool": pool,
        "lexical_family": family,
        "matched_style_index": style_index,
        "template_family_id": template_id,
        "frame_index": frame_index,
        "oa_row": list(oa_row),
        "factors": {
            "arithmetic_operation": operation,
            "selector_active_entry": active_bit + 1,
            "arithmetic_correct_choice": arithmetic_correct_choice,
            "selector_correct_choice": selector_correct_choice,
            "label_role_orientation": label_bit,
            "block_order": "arithmetic_first" if order_bit == 0 else "selector_first",
        },
        "name": name,
        "labels": list(labels),
        "arithmetic_label": arithmetic_label,
        "selector_label": selector_label,
        "item": item,
        "keys": keys,
        "numeric_block": block,
        "context": context,
        "arithmetic_block": arithmetic_block,
        "selector_block": selector_block,
        "ledger": ledger,
        "conditions": conditions,
        "prompt_token_lengths": prompt_lengths,
        "automatic_audit": {},
        "ai_audit_status": "pending",
        "human_audit": "not_performed",
    }


def normalized_skeleton(text: str, dynamic_values: list[str]) -> str:
    normalized = text.lower()
    for value in sorted(set(dynamic_values), key=len, reverse=True):
        normalized = re.sub(rf"\b{re.escape(value.lower())}\b", "<x>", normalized)
    normalized = re.sub(r"-?\d+", "<n>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z<>]+", text.lower())
    return {tuple(words[index : index + n]) for index in range(max(0, len(words) - n + 1))}


def jaccard(left: set[Any], right: set[Any]) -> float:
    return 0.0 if not left and not right else len(left & right) / len(left | right)


def audit_pair(pair: dict[str, Any], tokenizer: Any) -> dict[str, bool]:
    arithmetic = pair["conditions"]["arithmetic"]
    selector = pair["conditions"]["selector"]
    block = pair["numeric_block"]
    operation = pair["factors"]["arithmetic_operation"]
    arithmetic_expected = block["start"] + block["delta"] if operation == "ADD" else block["start"] - block["delta"]
    selector_expected = block["selector_1"] if pair["factors"]["selector_active_entry"] == 1 else block["selector_2"]
    mentions = numeric_mentions(pair["context"])
    question_a = arithmetic["question"].replace(pair["arithmetic_label"], "<TARGET>")
    question_s = selector["question"].replace(pair["selector_label"], "<TARGET>")
    checks = {
        "arithmetic_answer_correct": arithmetic["canonical_numeric_result"] == arithmetic_expected,
        "selector_answer_correct": selector["canonical_numeric_result"] == selector_expected,
        "wrong_operation_foil_correct": arithmetic["foil_numeric_result"]
        == (block["start"] - block["delta"] if operation == "ADD" else block["start"] + block["delta"]),
        "numeric_candidates_pairwise_distinct": len(
            {
                arithmetic["canonical_numeric_result"], arithmetic["foil_numeric_result"],
                selector["canonical_numeric_result"], selector["foil_numeric_result"],
            }
        ) == 4,
        "candidate_numeric_exposure_once_each": all(
            mentions.count(value) == 1
            for value in (
                arithmetic["canonical_numeric_result"], arithmetic["foil_numeric_result"],
                selector["canonical_numeric_result"], selector["foil_numeric_result"],
            )
        ),
        "operands_exposed_once": mentions.count(block["start"]) == 1
        and mentions.count(block["delta"]) == 1,
        "choice_tokens_one_token": all(
            len(condition[field]) == 1
            for condition in (arithmetic, selector)
            for field in ("correct_choice_token_ids", "alternative_choice_token_ids")
        ),
        "choice_token_ids_exact": {
            arithmetic["correct_choice_token_ids"][0], arithmetic["alternative_choice_token_ids"][0],
            selector["correct_choice_token_ids"][0], selector["alternative_choice_token_ids"][0],
        } == {362, 426},
        "prompt_token_length_exact_match": pair["prompt_token_lengths"]["arithmetic"]
        == pair["prompt_token_lengths"]["selector"],
        "question_differs_only_target": question_a == question_s,
        "shared_context": arithmetic["full_prompt"].startswith(pair["context"])
        and selector["full_prompt"].startswith(pair["context"]),
        "answer_instruction_exact": all(
            condition["full_prompt"].endswith("Answer with A or B only.\nAnswer:")
            for condition in (arithmetic, selector)
        ),
        "no_prior_result_fields": pair["ai_audit_status"] == "pending",
        "human_audit_disclosed": pair["human_audit"] == "not_performed",
    }
    return checks


def audit_dataset(dataset: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    pairs = dataset["pairs"]
    per_pair = {}
    for pair in pairs:
        checks = audit_pair(pair, tokenizer)
        pair["automatic_audit"] = checks
        per_pair[pair["pair_id"]] = checks
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["template_family_id"]].append(pair)
    template_checks = {}
    for template_id, rows in grouped.items():
        factor_values = {
            "operation": Counter(row["factors"]["arithmetic_operation"] for row in rows),
            "active": Counter(row["factors"]["selector_active_entry"] for row in rows),
            "arithmetic_choice": Counter(row["factors"]["arithmetic_correct_choice"] for row in rows),
            "selector_choice": Counter(row["factors"]["selector_correct_choice"] for row in rows),
            "label_role": Counter(row["factors"]["label_role_orientation"] for row in rows),
            "block_order": Counter(row["factors"]["block_order"] for row in rows),
        }
        template_checks[template_id] = {
            "eight_frames": len(rows) == 8,
            "oa_rows_exact": sorted(tuple(row["oa_row"]) for row in rows) == sorted(OA_MATRIX),
            "each_factor_4_4": all(sorted(counter.values()) == [4, 4] for counter in factor_values.values()),
            "unique_pair_ids": len({row["pair_id"] for row in rows}) == 8,
        }
    all_pair_checks = all(all(checks.values()) for checks in per_pair.values())
    all_template_checks = all(all(checks.values()) for checks in template_checks.values())
    family_counts = Counter(pair["lexical_family"] for pair in pairs)
    result = {
        "schema_version": "4.0",
        "pool": dataset["pool"],
        "pair_count": len(pairs),
        "prompt_count": 2 * len(pairs),
        "template_count": len(grouped),
        "family_pair_counts": dict(family_counts),
        "per_pair_checks": per_pair,
        "template_checks": template_checks,
        "global_checks": {
            "pair_count_128": len(pairs) == 128,
            "prompt_count_256": 2 * len(pairs) == 256,
            "template_count_16": len(grouped) == 16,
            "family_counts_64_each": family_counts == Counter({"points_balance": 64, "temperature": 64}),
            "all_pair_checks_pass": all_pair_checks,
            "all_template_checks_pass": all_template_checks,
            "full_prompts_unique": len(
                {
                    condition["full_prompt"]
                    for pair in pairs
                    for condition in pair["conditions"].values()
                }
            ) == 256,
        },
    }
    result["all_checks_pass"] = all(result["global_checks"].values())
    return result


def build_datasets(design: dict[str, Any], catalog: dict[str, Any], tokenizer: Any) -> dict[str, dict[str, Any]]:
    template_keys = [f"{pool}|{family}|{index}" for pool in POOLS for family in FAMILIES for index in range(8)]
    key_assignments = make_global_key_assignments(tokenizer, template_keys)
    numeric_used: set[tuple[int, ...]] = set()
    datasets = {}
    for pool in POOLS:
        styles = catalog[f"{pool}_styles"]
        pairs = []
        for family in FAMILIES:
            for index, style in enumerate(styles):
                template_key = f"{pool}|{family}|{index}"
                block = numeric_block(template_key, numeric_used)
                name = catalog["pools"][pool]["names"][index]
                labels = tuple(catalog["pools"][pool]["label_pairs"][index])
                item = catalog["family_items"][family][index]
                for frame_index, oa_row in enumerate(OA_MATRIX, start=1):
                    pairs.append(
                        render_pair(
                            pool=pool,
                            family=family,
                            style=style,
                            style_index=index + 1,
                            frame_index=frame_index,
                            oa_row=oa_row,
                            name=name,
                            labels=labels,
                            item=item,
                            keys=key_assignments[template_key],
                            block=block,
                            tokenizer=tokenizer,
                        )
                    )
        datasets[pool] = {
            "schema_version": "4.0",
            "dataset_id": f"stage_e_v4_{pool}_pool_20260830",
            "status": "dataset_created_automatic_audit_pending_ai_audits",
            "pool": pool,
            "design_sha256": EXPECTED_HASHES["design"],
            "catalog_sha256": EXPECTED_HASHES["catalog"],
            "catalog_amendment_sha256": EXPECTED_HASHES["catalog_amendment"],
            "random_seed": SEED,
            "pair_count": len(pairs),
            "prompt_count": 2 * len(pairs),
            "template_count": 16,
            "human_audit": "not_performed",
            "human_audited_evidence": False,
            "pairs": pairs,
        }
    return datasets


def cross_dataset_audit(
    datasets: dict[str, dict[str, Any]], prior: dict[str, Any]
) -> dict[str, Any]:
    calibration = datasets["calibration"]["pairs"]
    replication = datasets["replication"]["pairs"]
    prompts = {
        pool: {
            condition["full_prompt"]
            for pair in datasets[pool]["pairs"]
            for condition in pair["conditions"].values()
        }
        for pool in POOLS
    }
    prior_prompts = {
        condition["full_prompt"]
        for pair in prior["pairs"]
        for condition in pair["conditions"].values()
    }
    prior_skeletons = []
    for pair in prior["pairs"][::8]:
        dynamic = [
            pair.get("name", ""), pair.get("arithmetic_attribute", ""),
            pair.get("selector_attribute", ""), *pair.get("all_case_and_value_keys", []),
        ]
        prior_skeletons.append(
            ngrams(normalized_skeleton(pair["context"] + " " + pair["conditions"]["arithmetic"]["question"], dynamic))
        )
    max_prior_jaccard = 0.0
    for pool in POOLS:
        for pair in datasets[pool]["pairs"][::8]:
            dynamic = [pair["name"], *pair["labels"], *pair["keys"]]
            skeleton = ngrams(
                normalized_skeleton(pair["context"] + " " + pair["conditions"]["arithmetic"]["question"], dynamic)
            )
            for prior_skeleton in prior_skeletons:
                max_prior_jaccard = max(max_prior_jaccard, jaccard(skeleton, prior_skeleton))
    calibration_numeric = {
        tuple(pair["numeric_block"].values()) for pair in calibration[::8]
    }
    replication_numeric = {
        tuple(pair["numeric_block"].values()) for pair in replication[::8]
    }
    checks = {
        "pair_ids_disjoint": not ({pair["pair_id"] for pair in calibration} & {pair["pair_id"] for pair in replication}),
        "template_ids_disjoint": not ({pair["template_family_id"] for pair in calibration} & {pair["template_family_id"] for pair in replication}),
        "full_prompts_disjoint": not (prompts["calibration"] & prompts["replication"]),
        "numeric_blocks_disjoint": not (calibration_numeric & replication_numeric),
        "names_disjoint": not ({pair["name"] for pair in calibration} & {pair["name"] for pair in replication}),
        "labels_disjoint": not ({label for pair in calibration for label in pair["labels"]} & {label for pair in replication for label in pair["labels"]}),
        "no_exact_prior_v3_prompt_reuse": not ((prompts["calibration"] | prompts["replication"]) & prior_prompts),
        "prior_v3_max_normalized_5gram_jaccard_below_0_80": max_prior_jaccard < 0.80,
    }
    return {
        "checks": checks,
        "max_prior_v3_normalized_5gram_jaccard": max_prior_jaccard,
        "all_checks_pass": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    design, catalog, prior = validate_locked_inputs()
    oa_report = validate_oa()
    if args.validate_only:
        print(json.dumps({"status": "static_validation_pass", "oa": oa_report, "model_loaded": False, "gpu_used": False}, indent=2))
        return
    output_dir = args.output_dir.resolve()
    output_paths = {
        "calibration_dataset": output_dir / "calibration_pool_dataset.json",
        "replication_dataset": output_dir / "replication_pool_dataset.json",
        "calibration_audit": output_dir / "calibration_pool_automatic_audit.json",
        "replication_audit": output_dir / "replication_pool_automatic_audit.json",
        "cross_pool_audit": output_dir / "cross_pool_nonreuse_audit.json",
        "manifest": output_dir / "dataset_manifest.json",
    }
    if not args.overwrite and any(path.exists() for path in output_paths.values()):
        raise RuntimeError("Refusing to overwrite an existing v4 dataset artifact")
    tokenizer = load_tokenizer()
    datasets = build_datasets(design, catalog, tokenizer)
    audits = {
        pool: audit_dataset(datasets[pool], tokenizer) for pool in POOLS
    }
    cross = cross_dataset_audit(datasets, prior)
    if not all(audit["all_checks_pass"] for audit in audits.values()) or not cross["all_checks_pass"]:
        raise RuntimeError("Dataset automatic audit failed")
    for pool in POOLS:
        datasets[pool]["status"] = "dataset_created_automatic_audit_pass_ai_audits_pending"
    atomic_json(output_paths["calibration_dataset"], datasets["calibration"])
    atomic_json(output_paths["replication_dataset"], datasets["replication"])
    atomic_json(output_paths["calibration_audit"], audits["calibration"])
    atomic_json(output_paths["replication_audit"], audits["replication"])
    atomic_json(output_paths["cross_pool_audit"], cross)
    manifest = {
        "schema_version": "4.0",
        "status": "datasets_created_automatic_audits_pass_ai_audits_pending",
        "design_sha256": EXPECTED_HASHES["design"],
        "catalog_sha256": EXPECTED_HASHES["catalog"],
        "catalog_amendment_sha256": EXPECTED_HASHES["catalog_amendment"],
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "random_seed": SEED,
        "oa_validation": oa_report,
        "output_sha256": {
            name: sha256_file(path) for name, path in output_paths.items() if name != "manifest"
        },
        "model_loaded": False,
        "model_forward_performed": False,
        "gpu_used": False,
        "liref_loaded": False,
        "baseline_calibration_execution_allowed": False,
        "replication_execution_allowed": False,
        "intervention_allowed": False,
        "human_audit": "not_performed",
    }
    atomic_json(output_paths["manifest"], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
