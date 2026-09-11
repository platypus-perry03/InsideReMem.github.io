#!/usr/bin/env python3
"""Model-free tests for the limited same-sample Pilot v3.2 runner."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = STAGE_DIR / "run_stage_e_limited_pilot_v3_2.py"
SPEC = importlib.util.spec_from_file_location("pilot_v3_2_runner", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load Pilot v3.2 runner")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


ENDPOINT_MULTIPLIERS = {
    "layer31_liref_projection": 1.0,
    "L31N13336_contribution": 2.0,
    "L29H00030_contribution": -1.0,
    "L30H00006_contribution": 0.5,
    "L29H00031_contribution": -0.25,
    "L31N13336_signed_z": 3.0,
    "L31N13336_absolute_z": 4.0,
    "L29H00030_pre_o_l2_norm": 5.0,
    "L30H00006_pre_o_l2_norm": 6.0,
    "L29H00031_pre_o_l2_norm": 7.0,
}
FAMILY_EFFECTS = {
    "object_count": 1.0,
    "points_balance": 2.0,
    "temperature": -1.0,
}


def synthetic_prompt_rows(dataset: dict) -> list[dict]:
    rows = []
    for pair in dataset["pairs"]:
        base_effect = FAMILY_EFFECTS[pair["lexical_family"]]
        # Centered frame perturbation proves that aggregation is frame -> template.
        frame_perturbation = (pair["frame_index"] - 4.5) / 100.0
        for condition in runner.CONDITIONS:
            sign = 1.0 if condition == "arithmetic" else 0.0
            row = {"pair_id": pair["pair_id"], "condition": condition}
            for endpoint, multiplier in ENDPOINT_MULTIPLIERS.items():
                row[endpoint] = sign * multiplier * (base_effect + frame_perturbation)
            rows.append(row)
    return rows


class PilotV32PureFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = runner.load_json(runner.DATASET_PATH)
        cls.baseline = runner.load_json(runner.BASELINE_RESULTS_PATH)
        cls.prompt_rows = synthetic_prompt_rows(cls.dataset)
        cls.pair_rows = runner.pair_prompt_rows(
            cls.prompt_rows, cls.dataset, cls.baseline
        )
        cls.template_rows = runner.template_cluster_rows(cls.pair_rows)

    def test_locked_preflight_remains_model_free_and_execution_closed(self) -> None:
        result = runner.validate_locked_inputs()
        self.assertEqual(result["status"], "preflight_pass_execution_still_requires_separate_authorization")
        self.assertFalse(result["model_runtime_imported"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["model_forward_performed"])
        self.assertFalse(result["gpu_used"])
        self.assertFalse(result["pilot_execution_allowed"])

    def test_all_pairs_and_template_clusters_are_preserved(self) -> None:
        self.assertEqual(len(self.pair_rows), 192)
        self.assertEqual(len({row["pair_id"] for row in self.pair_rows}), 192)
        self.assertEqual(len(self.template_rows), 24)
        self.assertTrue(all(row["pair_count"] == 8 for row in self.template_rows))
        family_counts = {
            family: sum(row["lexical_family"] == family for row in self.template_rows)
            for family in runner.EXPECTED_FAMILIES
        }
        self.assertEqual(family_counts, {family: 8 for family in runner.EXPECTED_FAMILIES})

    def test_difference_orientation_is_arithmetic_minus_selector(self) -> None:
        first_pair = self.dataset["pairs"][0]
        row = next(row for row in self.pair_rows if row["pair_id"] == first_pair["pair_id"])
        expected = FAMILY_EFFECTS[first_pair["lexical_family"]] + (
            first_pair["frame_index"] - 4.5
        ) / 100.0
        self.assertAlmostEqual(row["layer31_liref_projection"], expected)
        self.assertAlmostEqual(row["L31N13336_contribution"], 2.0 * expected)
        self.assertAlmostEqual(row["L29H00030_contribution"], -expected)

    def test_family_overall_and_interaction_aggregation(self) -> None:
        result = runner.aggregate_primary(self.template_rows)
        endpoint = "layer31_liref_projection"
        for family, expected in FAMILY_EFFECTS.items():
            observed = result["family_effects"][family][endpoint]
            self.assertAlmostEqual(observed["mean_effect"], expected)
            self.assertEqual(observed["cluster_count"], 8)
            self.assertFalse(observed["confirmatory_significance_claim_allowed"])
        overall = result["overall_effects"][endpoint]
        self.assertAlmostEqual(overall["mean_effect"], 2.0 / 3.0)
        self.assertAlmostEqual(overall["equal_weight_family_mean"], 2.0 / 3.0)
        self.assertTrue(overall["aggregation_equivalence_check"])
        interactions = result["interaction_effects"][endpoint]
        self.assertAlmostEqual(
            interactions["object_count_minus_points_balance"]["effect_difference"],
            -1.0,
        )
        self.assertAlmostEqual(
            interactions["points_balance_minus_temperature"]["effect_difference"],
            3.0,
        )
        self.assertFalse(result["hard_pass_fail_applied"])

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        first = runner.aggregate_primary(self.template_rows)
        second = runner.aggregate_primary(self.template_rows)
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_repetitions"], 10000)
        self.assertEqual(first["bootstrap_seed"], 20260831)

    def test_secondary_splits_cannot_replace_primary(self) -> None:
        result = runner.secondary_diagnostics(self.pair_rows, self.template_rows)
        self.assertFalse(result["secondary_results_may_replace_primary"])
        self.assertFalse(result["secondary_p_values_computed"])
        self.assertFalse(result["candidate_selection_from_secondary_allowed"])
        self.assertEqual(set(result["operation"]["overall"]), {"add", "subtract"})
        self.assertEqual(
            sum(level["pair_count"] for level in result["operation"]["overall"].values()),
            192,
        )

    def test_duplicate_or_missing_prompt_scalar_rows_fail_closed(self) -> None:
        malformed = self.prompt_rows[:-1] + [dict(self.prompt_rows[0])]
        with self.assertRaisesRegex(RuntimeError, "missing, duplicated, or unexpected"):
            runner.pair_prompt_rows(malformed, self.dataset, self.baseline)

    def test_statistical_edge_cases(self) -> None:
        self.assertEqual(runner.sample_dz([0.0] * 8), (0.0, "all_exactly_zero"))
        value, reason = runner.sample_dz([1.0] * 8)
        self.assertIsNone(value)
        self.assertEqual(reason, "near_zero_sd_nonzero_mean")
        self.assertAlmostEqual(runner.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(runner.spearman([1, 1, 1], [1, 2, 3]))
        self.assertTrue(math.isfinite(runner.percentile([0.0, 1.0], 0.5)))


if __name__ == "__main__":
    unittest.main()
