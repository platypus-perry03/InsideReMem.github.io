#!/usr/bin/env python3
"""Model-free tests for the frozen Baseline Calibration v3.1 evaluator."""

from __future__ import annotations

import importlib.util
import json
import math
import unittest
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = STAGE_DIR / "run_calibration_v3_1_baseline.py"
DESIGN_PATH = STAGE_DIR / "calibration_v3_design_frozen.json"


def load_runner():
    specification = importlib.util.spec_from_file_location("calibration_v3_1_runner", RUNNER_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the v3.1 baseline runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


RUNNER = load_runner()
DESIGN = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))


def make_condition(margin: float, forced_choice: bool, generation: bool) -> dict:
    return {
        "margin_nats": margin,
        "forced_choice_correct": forced_choice,
        "generation_correct": generation,
        "generation_valid_format": True,
        "correct_log_probability": -1.0,
        "correct_geometric_probability": math.exp(-1.0),
    }


def synthetic_results(*, selector_ceiling: bool = False) -> list[dict]:
    results = []
    for family in RUNNER.EXPECTED_FAMILIES:
        for template_index in range(8):
            for frame_index in range(8):
                # 48/64 forced-choice and 32/64 generation in the balanced case.
                arithmetic_fc = frame_index < 6
                selector_fc = True if selector_ceiling else frame_index < 6
                generation = frame_index < 4
                # Identical condition margins produce D_k=0 for all clusters.
                arithmetic_margin = 1.0 if arithmetic_fc else -1.0
                selector_margin = 1.0 if selector_fc else -1.0
                results.append(
                    {
                        "pair_id": f"{family}_{template_index}_{frame_index}",
                        "lexical_family": family,
                        "template_family_id": f"{family}_template_{template_index}",
                        "conditions": {
                            "arithmetic": make_condition(
                                arithmetic_margin, arithmetic_fc, generation
                            ),
                            "selector": make_condition(
                                selector_margin, selector_fc, generation
                            ),
                        },
                    }
                )
    return results


class BaselineV31ModelFreeTests(unittest.TestCase):
    def test_locked_inputs_preflight_without_model_runtime(self) -> None:
        preflight = RUNNER.validate_locked_inputs()
        self.assertEqual(preflight["pair_count"], 192)
        self.assertFalse(preflight["execution_allowed"])
        self.assertFalse(preflight["model_runtime_imported"])
        self.assertFalse(preflight["gpu_used"])

    def test_balanced_synthetic_results_pass_all_families(self) -> None:
        summary = RUNNER.summarize_results(synthetic_results(), DESIGN)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["passed_families"], list(RUNNER.EXPECTED_FAMILIES))
        self.assertEqual(summary["failed_families"], [])
        for family in RUNNER.EXPECTED_FAMILIES:
            result = summary["family_summaries"][family]
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["mean_template_contrast_nats"], 0.0)
            self.assertEqual(result["cluster_dz"], 0.0)
            self.assertTrue(all(result["checks"].values()))

    def test_selector_ceiling_fails_all_families(self) -> None:
        summary = RUNNER.summarize_results(
            synthetic_results(selector_ceiling=True), DESIGN
        )
        self.assertEqual(summary["status"], "FAIL")
        self.assertEqual(summary["passed_families"], [])
        self.assertEqual(summary["failed_families"], list(RUNNER.EXPECTED_FAMILIES))
        for family in RUNNER.EXPECTED_FAMILIES:
            checks = summary["family_summaries"][family]["checks"]
            self.assertFalse(checks["forced_choice_selector_count_range"])
            self.assertFalse(checks["forced_choice_condition_count_gap"])

    def test_normalization_accepts_only_one_arabic_numeral_token_text(self) -> None:
        self.assertEqual(RUNNER.normalize_generated_token(" ４２ "), "42")
        self.assertEqual(RUNNER.normalize_generated_token("42"), "42")
        self.assertEqual(RUNNER.normalize_generated_token("42 lanterns"), "")
        self.assertEqual(RUNNER.normalize_generated_token("42\n"), "42")

    def test_dz_edge_cases_follow_frozen_definition(self) -> None:
        self.assertEqual(RUNNER.compute_cluster_dz([0.0] * 8), (0.0, True, "all_exactly_zero"))
        dz, valid, reason = RUNNER.compute_cluster_dz([1.0] * 8)
        self.assertIsNone(dz)
        self.assertFalse(valid)
        self.assertEqual(reason, "near_zero_sd_nonzero_mean")


if __name__ == "__main__":
    unittest.main()
