import importlib.util
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run_stage_e_transformation_primitives_v1_3.py"
SPEC = importlib.util.spec_from_file_location("transformation_primitives_v1_3", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TransformationPrimitivesV13Tests(unittest.TestCase):
    def test_all_six_primitives_have_questions(self):
        self.assertEqual(set(MODULE.PRIMITIVE_QUESTIONS), set(MODULE.BASE.PRIMITIVES))

    def test_questions_are_binary(self):
        self.assertTrue(all(text.endswith("?") for text in MODULE.PRIMITIVE_QUESTIONS.values()))

    def test_parent_label_is_not_a_scored_primitive(self):
        self.assertNotIn("transformation_required", MODULE.PRIMITIVE_QUESTIONS)

    def test_base_reliability_threshold_still_applies(self):
        metrics = {"pooled_positive_prevalence": 0.5, "raw_agreement": 0.81, "cohen_kappa": 0.51}
        self.assertEqual(MODULE.BASE.reliable_preflight_status(metrics), "USABLE")

    def test_low_agreement_is_unreliable(self):
        metrics = {"pooled_positive_prevalence": 0.5, "raw_agreement": 0.79, "cohen_kappa": 0.7}
        self.assertEqual(MODULE.BASE.reliable_preflight_status(metrics), "UNRELIABLE")

    def test_blind_prompt_has_no_internal_outcomes(self):
        source = MODULE.SYSTEM + " ".join(MODULE.PRIMITIVE_QUESTIONS.values())
        self.assertNotIn("memory_reason_score", source)
        self.assertNotIn("LiReF", source)

    def test_no_free_generation_call(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("model.generate", source)

    def test_forbidden_external_capabilities_absent(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("api.openai.com", source)
        self.assertNotIn("OPENAI_API_KEY", source)
        self.assertNotIn("register_forward_hook", source)


if __name__ == "__main__":
    unittest.main()
