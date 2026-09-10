import importlib.util
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run_stage_e_transformation_primitives_v1_2.py"
SPEC = importlib.util.spec_from_file_location("transformation_primitives_v1_2", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TransformationPrimitivesV12Tests(unittest.TestCase):
    def test_valid_multi_label_parse(self):
        value = MODULE.parse_annotation("NUM=Y;RULE=Y;REL=N;COND=N;CAUS=N;INTER=Y")
        self.assertEqual(value, {"NUM": "Y", "RULE": "Y", "REL": "N", "COND": "N", "CAUS": "N", "INTER": "Y"})

    def test_invalid_parse(self):
        self.assertIsNone(MODULE.parse_annotation("NUM=MAYBE;RULE=N"))

    def test_kappa_identity(self):
        self.assertAlmostEqual(MODULE.cohen_kappa(["Y", "N", "Y"], ["Y", "N", "Y"]), 1.0)

    def test_primitive_metrics(self):
        ids = ["a", "b", "c", "d"]
        a = {key: {"NUM": value} for key, value in zip(ids, ["Y", "Y", "N", "N"])}
        b = {key: {"NUM": value} for key, value in zip(ids, ["Y", "Y", "N", "N"])}
        result = MODULE.primitive_metrics(ids, a, b, "NUM")
        self.assertEqual(result["raw_agreement"], 1.0)
        self.assertEqual(result["pooled_positive_prevalence"], 0.5)

    def test_preflight_usable(self):
        metrics = {"pooled_positive_prevalence": 0.5, "raw_agreement": 0.9, "cohen_kappa": 0.7}
        self.assertEqual(MODULE.reliable_preflight_status(metrics), "USABLE")

    def test_preflight_low_prevalence(self):
        metrics = {"pooled_positive_prevalence": 0.01, "raw_agreement": 0.99, "cohen_kappa": 0.8}
        self.assertEqual(MODULE.reliable_preflight_status(metrics), "INSUFFICIENT_PREVALENCE")

    def test_blind_render(self):
        text = MODULE.render_item({"question": "Q?", "options": ["x", "y"]})
        self.assertNotIn("memory_reason_score", text)
        self.assertNotIn("LiReF", text)

    def test_forbidden_execution_capabilities_absent(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("api.openai.com", source)
        self.assertNotIn("OPENAI_API_KEY", source)
        self.assertNotIn("register_forward_hook", source)


if __name__ == "__main__":
    unittest.main()
