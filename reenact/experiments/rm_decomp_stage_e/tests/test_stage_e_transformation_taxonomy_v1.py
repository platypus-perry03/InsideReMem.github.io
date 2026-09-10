import importlib.util
import math
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run_stage_e_transformation_taxonomy_v1.py"
SPEC = importlib.util.spec_from_file_location("taxonomy_v1", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TransformationTaxonomyV1Tests(unittest.TestCase):
    def test_parse_valid_arithmetic(self):
        self.assertEqual(
            MODULE.parse_annotation("T=Y;TYPE=ARITH;STEPS=2"),
            {"transformation": "Y", "subtype": "ARITH", "steps": "2"},
        )

    def test_parse_valid_none(self):
        self.assertEqual(
            MODULE.parse_annotation("T=N;TYPE=NONE;STEPS=0"),
            {"transformation": "N", "subtype": "NONE", "steps": "0"},
        )

    def test_parse_rejects_inconsistent_fields(self):
        self.assertIsNone(MODULE.parse_annotation("T=N;TYPE=LOGIC;STEPS=0"))
        self.assertIsNone(MODULE.parse_annotation("T=Y;TYPE=ARITH;STEPS=0"))
        self.assertIsNone(MODULE.parse_annotation("T=N;TYPE=NONE;STEPS=1"))

    def test_kappa_identity(self):
        self.assertAlmostEqual(MODULE.cohen_kappa(["Y", "N", "Y"], ["Y", "N", "Y"]), 1.0)
        self.assertAlmostEqual(MODULE.weighted_kappa(["0", "1", "2", "3P"], ["0", "1", "2", "3P"]), 1.0)

    def test_balanced_accuracy(self):
        value = MODULE.balanced_accuracy(["Y", "Y", "N", "N"], ["Y", "N", "N", "N"])
        self.assertAlmostEqual(value, 0.75)

    def test_bh_adjust(self):
        adjusted = MODULE.bh_adjust([0.01, 0.04, 0.03])
        self.assertTrue(all(math.isfinite(value) for value in adjusted))
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.04)

    def test_blind_render_excludes_outcomes(self):
        text = MODULE.render_item({"annotation_id": "X", "question": "What follows?", "options": ["A", "B"]})
        for forbidden in ("memory_reason_score", "LiReF", "component_L29H00030", "correct answer"):
            self.assertNotIn(forbidden, text)

    def test_runner_has_no_external_api_client(self):
        source = RUNNER.read_text(encoding="utf-8")
        for forbidden in ("api.openai.com", "OpenAI(", "requests.post", "urllib.request", "httpx"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
