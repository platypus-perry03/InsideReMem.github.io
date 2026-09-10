from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_stage_e_natural_feature_discovery_v1.py"
SPEC = importlib.util.spec_from_file_location("natural_feature_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NaturalFeatureDiscoveryTests(unittest.TestCase):
    def test_parse_exact_contract(self) -> None:
        result = MODULE.parse_annotation("MODE=DER;T=Y;C=N;S=N;K=Y;A=IND")
        self.assertEqual(result["answer_mode"], "DER")
        self.assertEqual(result["answer_directness"], "IND")

    def test_invalid_annotation_rejected(self) -> None:
        self.assertIsNone(MODULE.parse_annotation("Reasoning problem"))
        self.assertIsNone(MODULE.parse_annotation("MODE=DER;T=MAYBE;C=N;S=N;K=Y;A=IND"))

    def test_blind_item_has_no_outcome_leakage(self) -> None:
        record = {
            "question": "What follows?", "options": ["A", "B"], "answer": "A",
            "answer_index": 0, "memory_reason_score": 1.0, "category": "math", "src": "secret",
        }
        item = MODULE.blind_item(record, "NF0000")
        self.assertEqual(set(item), {"annotation_id", "question", "options"})
        self.assertFalse(MODULE.FORBIDDEN_ANNOTATION_FIELDS.intersection(item))

    def test_feature_mapping(self) -> None:
        row = {
            "answer_mode": "RET", "transformation_required": "N",
            "composition_required": "N", "multi_step_required": "N",
            "external_knowledge_required": "Y", "answer_directness": "DIR",
        }
        result = MODULE.annotation_to_features(row)
        self.assertEqual(result["mode_derivation_vs_retrieval"], 0.0)
        self.assertEqual(result["external_knowledge_required"], 1.0)
        self.assertEqual(result["answer_indirect"], 0.0)

    def test_uncertain_mapping_is_missing(self) -> None:
        row = {
            "answer_mode": "MIX", "transformation_required": "UNC",
            "composition_required": "UNC", "multi_step_required": "UNC",
            "external_knowledge_required": "UNC", "answer_directness": "UNC",
        }
        self.assertTrue(all(math.isnan(value) for value in MODULE.annotation_to_features(row).values()))

    def test_bh_adjust(self) -> None:
        adjusted = MODULE.bh_adjust([0.01, 0.04, 0.03, 0.20])
        self.assertAlmostEqual(adjusted[0], 0.04)
        self.assertAlmostEqual(adjusted[1], 0.05333333333333334)
        self.assertAlmostEqual(adjusted[2], 0.05333333333333334)
        self.assertAlmostEqual(adjusted[3], 0.20)

    def test_kappa_identity(self) -> None:
        self.assertEqual(MODULE.cohen_kappa(["Y", "N", "Y"], ["Y", "N", "Y"]), 1.0)

    def test_hook_source_is_read_only(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("return None", source)
        self.assertNotIn("register_forward_pre_hook", source)
        self.assertNotIn("copy_(", source)
        self.assertNotIn("add_(", source)


if __name__ == "__main__":
    unittest.main()
