import importlib.util
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run_stage_e_transformation_taxonomy_v1_1.py"
SPEC = importlib.util.spec_from_file_location("taxonomy_v1_1", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TransformationTaxonomyV11Tests(unittest.TestCase):
    def test_valid_parse(self):
        self.assertEqual(
            MODULE.parse_annotation("TYPE=ARITH;STEPS=1"),
            {"subtype": "ARITH", "steps": "1"},
        )

    def test_invalid_unc_mismatch(self):
        self.assertIsNone(MODULE.parse_annotation("TYPE=UNC;STEPS=1"))
        self.assertIsNone(MODULE.parse_annotation("TYPE=LOGIC;STEPS=UNC"))

    def test_valid_unc(self):
        self.assertEqual(
            MODULE.parse_annotation("TYPE=UNC;STEPS=UNC"),
            {"subtype": "UNC", "steps": "UNC"},
        )

    def test_cohen_kappa_identity(self):
        values = ["ARITH", "LOGIC", "FORMAL", "ARITH"]
        self.assertAlmostEqual(MODULE.cohen_kappa(values, values), 1.0)

    def test_weighted_kappa_identity(self):
        values = ["1", "2", "3P", "1"]
        self.assertAlmostEqual(MODULE.weighted_kappa(values, values), 1.0)

    def test_concentration(self):
        rows = {
            "a": {"subtype": "ARITH", "steps": "1"},
            "b": {"subtype": "LOGIC", "steps": "2"},
        }
        result = MODULE.concentration(["a", "b"], rows)
        self.assertEqual(result["subtype_max_fraction"], 0.5)
        self.assertEqual(result["step_max_fraction"], 0.5)

    def test_blind_render_has_no_outcomes(self):
        text = MODULE.render_item({"question": "Q?", "options": ["x", "y"]})
        self.assertIn("Question", text)
        self.assertNotIn("memory_reason_score", text)
        self.assertNotIn("LiReF", text)

    def test_forbidden_capabilities_absent(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("api.openai.com", source)
        self.assertNotIn("OPENAI_API_KEY", source)
        self.assertNotIn("register_forward_hook", source)
        self.assertNotIn("patching", source.lower())


if __name__ == "__main__":
    unittest.main()
