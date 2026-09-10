#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load_module("bv2_builder", HERE / "build_behavioral_validation_v2_assets.py")
runner = load_module("bv2_runner", HERE / "run_behavioral_validation_v2.py")


class BehavioralValidationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = json.loads((HERE / "design_v2_frozen.json").read_text())
        cls.records = json.loads((HERE / "v2_assets/evaluation_records_frozen.json").read_text())

    def test_dataset_balance_and_splits(self):
        for dataset in self.design["task_poles"]:
            rows = [row for row in self.records if row["dataset"] == dataset]
            self.assertEqual(len(rows), 100)
            self.assertEqual(sum(row["evaluation_split"] == "primary" for row in rows), 70)
            self.assertEqual(sum(row["evaluation_split"] == "confirmation" for row in rows), 30)

    def test_task_poles_are_frozen_family_labels(self):
        for row in self.records:
            self.assertEqual(row["pole"], self.design["task_poles"][row["dataset"]])

    def test_answers_and_ids(self):
        self.assertEqual(len({row["sample_id"] for row in self.records}), 400)
        for row in self.records:
            self.assertEqual(len(row["options"]), 4)
            self.assertIn(row["answer_index"], range(4))
            self.assertEqual(len(set(row["options"])), 4)

    def test_gsm8k_symbolic_no_source_overlap(self):
        gsm_ids = {int(row["sample_id"].split("::")[1]) for row in self.records if row["dataset"] == "gsm8k"}
        symbolic_ids = {int(row["original_gsm8k_index"]) for row in self.records if row["dataset"] == "gsm_symbolic"}
        self.assertTrue(gsm_ids.isdisjoint(symbolic_ids))

    def test_numeric_option_builder_is_deterministic(self):
        self.assertEqual(builder.numeric_options("18", "x"), builder.numeric_options("18", "x"))
        options, index = builder.numeric_options("18", "x")
        self.assertEqual(options[index], "18")

    def test_model_manifests(self):
        expected = {"Meta-Llama-3-8B": 13, "Mistral-7B-v0.3": 15, "OLMo-2-1124-7B": 17, "gemma-2-9b": 2}
        for model, count in expected.items():
            manifest = runner.validate_manifest(self.design, model)
            self.assertEqual(len(manifest["candidates"]), count)

    def test_intervention_contract(self):
        self.assertEqual(self.design["intervention_position"], "last_prompt_token_only")
        self.assertEqual(self.design["candidate_alphas"], [0.5, 1.0])

    def test_conditional_gate(self):
        gate = self.design["conditional_gate"]
        self.assertIn("launch_other_three_models", gate["meta_strict_primary"])
        self.assertEqual(gate["meta_no_signal_or_baseline_fail"], "stop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
