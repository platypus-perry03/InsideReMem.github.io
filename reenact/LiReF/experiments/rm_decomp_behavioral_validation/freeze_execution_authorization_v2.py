#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DESIGN = HERE / "design_v2_frozen.json"
RUNNER = HERE / "run_behavioral_validation_v2.py"
REVIEW = HERE / "STATIC_REVIEW_V2.md"
OUTPUT = HERE / "execution_authorization_v2_frozen.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    design = json.loads(DESIGN.read_text())
    manifests = resolve(design["manifest_root"])
    model_locks = {}
    for entry in design["models"]:
        model_path = resolve(entry["model_path"])
        model_locks[entry["name"]] = {
            "config_sha256": sha(model_path / "config.json"),
            "index_sha256": sha(model_path / "model.safetensors.index.json"),
            "shards": {path.name: sha(path) for path in sorted(model_path.glob("*.safetensors"))},
        }
    payload = {
        "status": "FROZEN_EXECUTION_AUTHORIZED",
        "study_id": design["study_id"],
        "design_sha256": sha(DESIGN),
        "implementation_sha256": sha(RUNNER),
        "static_review_sha256": sha(REVIEW),
        "records_sha256": sha(resolve(design["dataset_asset"])),
        "manifest_sha256": {entry["name"]: sha(manifests / f"{entry['name']}.json") for entry in design["models"]},
        "model_locks": model_locks,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"authorization": str(OUTPUT), "models": list(model_locks)}, indent=2))


if __name__ == "__main__":
    main()
