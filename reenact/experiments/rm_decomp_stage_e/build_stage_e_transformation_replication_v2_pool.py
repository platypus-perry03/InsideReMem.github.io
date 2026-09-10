#!/usr/bin/env python3
"""Build the item-independent MMLU-Pro candidate pool without model outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = STAGE_DIR / "stage_e_transformation_replication_v2_assets"
FULL_DATASET = ROOT / "liref" / "dataset" / "mmlu-pro"
USED_3000 = ROOT / "liref" / "dataset" / "mmlu-pro-3000samples.json"
USED_600 = ROOT / "liref" / "dataset" / "mmlu-pro-600samples.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> None:
    from datasets import load_from_disk

    full = load_from_disk(str(FULL_DATASET))["test"]
    used_3000 = load_json(USED_3000)
    used_600 = load_json(USED_600)
    used_ids = {str(row["question_id"]) for row in used_3000 + used_600}
    used_text = {normalize_question(str(row["question"])) for row in used_3000 + used_600}

    rows: list[dict[str, Any]] = []
    excluded_id = 0
    excluded_text = 0
    excluded_pool_duplicate_text = 0
    seen_candidate_text: set[str] = set()
    for row in full:
        question_id = str(row["question_id"])
        normalized = normalize_question(str(row["question"]))
        by_id = question_id in used_ids
        by_text = normalized in used_text
        if by_id or by_text:
            excluded_id += int(by_id)
            excluded_text += int(by_text)
            continue
        if normalized in seen_candidate_text:
            excluded_pool_duplicate_text += 1
            continue
        seen_candidate_text.add(normalized)
        rows.append({
            "candidate_id": f"TRV2-{len(rows):05d}",
            "question_id": question_id,
            "question": str(row["question"]),
            "options": [str(value) for value in row["options"]],
            "answer": str(row["answer"]),
            "answer_index": int(row["answer_index"]),
            "category": str(row["category"]),
            "src": str(row["src"]),
        })

    # MMLU-Pro contains repeated normalized question texts under distinct IDs.
    # Content non-reuse is stricter than ID-only exclusion and is intentional.
    if len(rows) != 8656:
        raise RuntimeError(f"Expected 8,656 unique content-nonoverlapping test items, found {len(rows)}")
    ids = [row["question_id"] for row in rows]
    texts = [normalize_question(row["question"]) for row in rows]
    if len(ids) != len(set(ids)) or len(texts) != len(set(texts)):
        raise RuntimeError("Candidate pool contains duplicate IDs or normalized questions")

    full_path = ASSET_DIR / "candidate_pool_private.jsonl"
    atomic_jsonl(full_path, rows)
    blind_rows = [{"candidate_id": row["candidate_id"], "question": row["question"], "options": row["options"]} for row in rows]
    blind_path = ASSET_DIR / "candidate_pool_blind.jsonl"
    atomic_jsonl(blind_path, blind_rows)

    audit = {
        "schema_id": "stage_e_transformation_replication_v2_pool_audit",
        "status": "PASS",
        "full_mmlu_test_rows": len(full),
        "used_3000_rows": len(used_3000),
        "used_600_rows": len(used_600),
        "excluded_by_id_count": excluded_id,
        "excluded_by_normalized_text_count": excluded_text,
        "excluded_candidate_pool_duplicate_text_count": excluded_pool_duplicate_text,
        "candidate_rows": len(rows),
        "unique_question_ids": len(set(ids)),
        "unique_normalized_questions": len(set(texts)),
        "blind_fields": ["candidate_id", "question", "options"],
        "blind_forbidden_fields_absent": True,
        "full_dataset_arrow_sha256": sha256_file(FULL_DATASET / "test" / "data-00000-of-00001.arrow"),
        "used_3000_sha256": sha256_file(USED_3000),
        "used_600_sha256": sha256_file(USED_600),
        "private_pool_sha256": sha256_file(full_path),
        "blind_pool_sha256": sha256_file(blind_path),
    }
    atomic_json(ASSET_DIR / "candidate_pool_audit.json", audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="build and audit the candidate pool")
    args = parser.parse_args()
    if not args.build:
        parser.error("--build is required")
    build()


if __name__ == "__main__":
    main()
