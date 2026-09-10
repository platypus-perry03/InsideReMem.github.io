from __future__ import annotations

import importlib.util
import json
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = STAGE_DIR / "run_stage_e_transformation_replication_v2_annotations.py"
POOL_DIR = STAGE_DIR / "stage_e_transformation_replication_v2_assets"


def load_runner():
    spec = importlib.util.spec_from_file_location("transformation_replication_v2", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_score_parser_uses_final_score():
    module = load_runner()
    assert module.parse_score("Analysis sentence.\nSCORE=0.75") == 0.75
    assert module.parse_score("SCORE=0.20\ncorrection SCORE=0.80") == 0.80
    assert module.parse_score("score unavailable") is None


def test_transformation_parser_is_strict():
    module = load_runner()
    assert module.parse_transformation("T=Y") == "Y"
    assert module.parse_transformation(" T=N ") == "N"
    assert module.parse_transformation("T=Y because arithmetic") is None


def test_score_bins_are_frozen():
    module = load_runner()
    assert module.score_bin(0.0) == "zero"
    assert module.score_bin(0.4) == "low_nonzero"
    assert module.score_bin(0.8) == "high_subpointnine"
    assert module.score_bin(0.9) == "pointnine_or_one"


def test_candidate_pool_blind_schema():
    path = POOL_DIR / "candidate_pool_blind.jsonl"
    with path.open(encoding="utf-8") as handle:
        first = json.loads(next(handle))
    assert set(first) == {"candidate_id", "question", "options"}
    forbidden = {"answer", "answer_index", "category", "src", "memory_reason_score", "liref", "component"}
    assert not forbidden.intersection(first)


def test_pool_audit_passes_and_is_unique():
    audit = json.loads((POOL_DIR / "candidate_pool_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["candidate_rows"] == 8656
    assert audit["unique_question_ids"] == 8656
    assert audit["unique_normalized_questions"] == 8656


def test_annotation_runner_has_no_study_hooks_or_intervention():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    forbidden = ["register_forward_hook", "discovery_liref_directions", "L31N13336", "L29H00030", "patching", "suppression"]
    for token in forbidden:
        assert token not in source


def test_kappa_and_balanced_accuracy_helpers():
    module = load_runner()
    assert module.cohen_kappa([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert module.balanced_accuracy([0, 0, 1, 1], [0, 1, 1, 1]) == 0.75
