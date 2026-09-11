#!/usr/bin/env python3
"""Model-free tests for the frozen Stage E v4 behavioral Calibration runner."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = STAGE_DIR / "run_stage_e_replication_v4_calibration.py"
SPEC = importlib.util.spec_from_file_location("stage_e_v4_calibration", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def synthetic_results(*, fail_points_generation: bool = False):
    rows = []
    for family in RUNNER.EXPECTED_FAMILIES:
        for template_index in range(8):
            for frame_index in range(8):
                index = template_index * 8 + frame_index
                arithmetic_fc = index < 50
                selector_fc = index < 50
                arithmetic_gen = index < 40
                selector_gen = index < 40
                if fail_points_generation and family == "points_balance":
                    arithmetic_gen = False
                # Alternating, exactly balanced cluster contrasts give mean=dz=0.
                arithmetic_margin = 0.1 if template_index % 2 == 0 else -0.1
                selector_margin = arithmetic_margin
                def condition(margin, fc, gen):
                    return {
                        "margin_nats": margin,
                        "forced_choice_correct": fc,
                        "generation_correct": gen,
                        "generation_valid_format": True,
                        "correct_probability": 0.6,
                    }
                rows.append({
                    "pair_id": f"{family}_{template_index}_{frame_index}",
                    "lexical_family": family,
                    "template_family_id": f"{family}_template_{template_index}",
                    "frame_index": frame_index + 1,
                    "conditions": {
                        "arithmetic": condition(arithmetic_margin, arithmetic_fc, arithmetic_gen),
                        "selector": condition(selector_margin, selector_fc, selector_gen),
                    },
                })
    return rows


class StageEV4CalibrationTests(unittest.TestCase):
    def test_locked_preflight_is_model_free_and_passes(self):
        result = RUNNER.validate_locked_inputs()
        self.assertEqual(result["status"], "preflight_pass_execution_requires_separate_authorization")
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["gpu_used"])
        self.assertFalse(result["replication_pool_accessed"])
        self.assertTrue(all(result["checks"].values()))

    def test_generation_normalization_accepts_only_single_choice(self):
        self.assertEqual(RUNNER.normalize_generation(" A"), "A")
        self.assertEqual(RUNNER.normalize_generation("b"), "B")
        self.assertEqual(RUNNER.normalize_generation("A."), "")
        self.assertEqual(RUNNER.normalize_generation("answer A"), "")

    def test_all_frozen_gates_can_pass(self):
        design = json.loads(RUNNER.DESIGN_PATH.read_text())
        summary = RUNNER.summarize(synthetic_results(), design)
        self.assertEqual(summary["passed_families"], ["points_balance", "temperature"])
        self.assertTrue(summary["primary_replication_gate_open"])
        self.assertTrue(summary["interaction_replication_gate_open"])

    def test_points_failure_closes_primary_and_interaction_gates(self):
        design = json.loads(RUNNER.DESIGN_PATH.read_text())
        summary = RUNNER.summarize(synthetic_results(fail_points_generation=True), design)
        self.assertIn("points_balance", summary["failed_families"])
        self.assertFalse(summary["primary_replication_gate_open"])
        self.assertFalse(summary["interaction_replication_gate_open"])

    def test_cluster_dz_contract(self):
        value, valid, reason = RUNNER.cluster_dz([0.0] * 8)
        self.assertEqual(value, 0.0)
        self.assertTrue(valid)
        self.assertEqual(reason, "all_exactly_zero")
        value, valid, _ = RUNNER.cluster_dz([1.0] * 8)
        self.assertIsNone(value)
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
