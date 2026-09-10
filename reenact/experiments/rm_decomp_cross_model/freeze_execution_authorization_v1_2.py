#!/usr/bin/env python3
"""Create the one-time hash-locked execution authorization for M-selective v1.2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
DESIGN_PATH = HERE / "design_v1_2_m_selective_frozen.json"
IMPLEMENTATION_PATH = HERE / "run_cross_model_m_selective_v1_2.py"
STATIC_REVIEW_PATH = HERE / "STATIC_REVIEW_V1_2_M_SELECTIVE.md"
OUTPUT_PATH = HERE / "execution_authorization_v1_2_m_selective_frozen.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"Authorization already exists; refusing overwrite: {OUTPUT_PATH}")
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    model_locks = {}
    for entry in design["models"]:
        model_path = WORKSPACE / entry["model_path"]
        shards = sorted(model_path.glob("*.safetensors"))
        if not shards:
            raise RuntimeError(f"No model shards: {model_path}")
        model_locks[entry["name"]] = {
            "config_sha256": sha256_file(model_path / "config.json"),
            "index_sha256": sha256_file(model_path / "model.safetensors.index.json"),
            "shards": {path.name: sha256_file(path) for path in shards},
        }
    inputs = {
        "dataset_sha256": sha256_file(WORKSPACE / design["dataset_path"]),
        "split_sha256": sha256_file(WORKSPACE / design["split_path"]),
        "layerwise_liref_sha256": {
            entry["name"]: sha256_file(WORKSPACE / design["layerwise_root"] / entry["name"] / "liref_vectors_heldout.pt")
            for entry in design["models"]
        },
    }
    payload = {
        "authorization_id": "cross_model_m_selective_homologue_v1_2_20260831_01",
        "status": "FROZEN_EXECUTION_AUTHORIZED",
        "authorized_at": "2026-08-31",
        "authorized_models": [entry["name"] for entry in design["models"]],
        "gpu_map": {entry["name"]: entry["gpu"] for entry in design["models"]},
        "batch_size": {entry["name"]: entry["batch_size"] for entry in design["models"]},
        "dtype": design["dtype"],
        "design_sha256": sha256_file(DESIGN_PATH),
        "implementation_sha256": sha256_file(IMPLEMENTATION_PATH),
        "static_review_sha256": sha256_file(STATIC_REVIEW_PATH),
        "inputs": inputs,
        "model_locks": model_locks,
        "allowed": [
            "Discovery-only negative-tail M-selective component screening",
            "held-out validation of frozen candidates",
            "last-token candidate/control suppression",
            "scalar/statistical output",
        ],
        "forbidden": [
            "candidate replacement after held-out inspection",
            "positive-tail R-selective candidate search",
            "new input-feature search",
            "full hidden-state persistence",
            "weight updates",
            "automatic result.pdf modification",
        ],
        "result_pdf_update_allowed": False,
    }
    write_json(OUTPUT_PATH, payload)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
