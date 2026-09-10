#!/usr/bin/env python3
"""Model-free contract tests for Meta-Llama behavioral validation v1."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / "run_meta_llama_behavioral_validation_v1.py"
SPEC = importlib.util.spec_from_file_location("behavioral_v1", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(6, 6, bias=False)
        with torch.no_grad():
            self.self_attn.o_proj.weight.copy_(torch.eye(8))
            self.mlp.down_proj.weight.copy_(torch.eye(6))


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer()])
        self.config = SimpleNamespace(head_dim=2, hidden_size=8, num_attention_heads=4)


class BehavioralValidationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = runner.load_design()
        cls.manifest = runner.validate_candidate_manifest(cls.design)

    def test_frozen_manifest_has_13_unique_candidates(self) -> None:
        rows = self.manifest["candidates"]
        ids = [row["component_id"] for row in rows]
        self.assertEqual(len(ids), 13)
        self.assertEqual(len(set(ids)), 13)
        self.assertEqual(sum("H" in value for value in ids), 5)
        self.assertEqual(sum("N" in value for value in ids), 8)

    def test_component_parser(self) -> None:
        self.assertEqual(runner.parse_component("L29H00030"), ("head", 29, 30))
        self.assertEqual(runner.parse_component("L31N13336"), ("neuron", 31, 13336))
        with self.assertRaises(ValueError):
            runner.parse_component("L29H30")

    def test_prompt_contains_all_choices_and_terminal_answer(self) -> None:
        prompt = runner.build_prompt(
            {"question": "Q?", "options": ["first", "second", "third"]}, self.design
        )
        self.assertIn("A. first", prompt)
        self.assertIn("B. second", prompt)
        self.assertIn("C. third", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_head_suppression_changes_only_selected_last_token_block(self) -> None:
        model = FakeModel()
        values = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)
        expected = values.clone()
        expected[:, -1, 2:4] *= 0.5
        intervention = runner.Intervention(model, "L00H00001", 0.5, None)
        intervention.install()
        output = model.model.layers[0].self_attn.o_proj(values)
        intervention.remove()
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(runner.registered_hook_count(model), 0)

    def test_neuron_mean_ablation_changes_only_selected_last_token(self) -> None:
        model = FakeModel()
        values = torch.arange(36, dtype=torch.float32).reshape(2, 3, 6)
        expected = values.clone()
        expected[:, -1, 3] = 10.0
        intervention = runner.Intervention(model, "L00N00003", 1.0, 10.0)
        intervention.install()
        output = model.model.layers[0].mlp.down_proj(values)
        intervention.remove()
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(runner.registered_hook_count(model), 0)

    def test_condition_plan_is_deduplicated_and_complete(self) -> None:
        plan = runner.condition_plan(self.manifest, self.design)
        physical = [(row["component_id"], row["alpha"]) for row in plan]
        self.assertEqual(len(physical), len(set(physical)))
        for candidate in self.manifest["candidates"]:
            self.assertIn((candidate["component_id"], 0.5), physical)
            self.assertIn((candidate["component_id"], 1.0), physical)
            self.assertIn((candidate["matched_control"], 1.0), physical)
            self.assertIn((candidate["random_control"], 1.0), physical)

    def test_baseline_gate_passes_solvable_groups(self) -> None:
        rows = []
        for group in (0, 1):
            for index in range(20):
                rows.append({
                    "group": group,
                    "forced_choice_correct": int(index < 10),
                    "option_count": 4,
                    "correct_probability": 0.4,
                    "correct_log_probability": -1.0,
                    "correct_vs_best_foil_margin": 0.2,
                    "vocab_top1_is_valid_label": 1,
                    "vocab_top1_is_correct_label": int(index < 10),
                })
        self.assertEqual(runner.baseline_summary(pd.DataFrame(rows), self.design)["status"], "PASS")

    def test_baseline_gate_fails_chance_level_groups(self) -> None:
        rows = []
        for group in (0, 1):
            for index in range(20):
                rows.append({
                    "group": group,
                    "forced_choice_correct": int(index < 5),
                    "option_count": 4,
                    "correct_probability": 0.25,
                    "correct_log_probability": -1.4,
                    "correct_vs_best_foil_margin": 0.0,
                    "vocab_top1_is_valid_label": 1,
                    "vocab_top1_is_correct_label": int(index < 5),
                })
        self.assertEqual(runner.baseline_summary(pd.DataFrame(rows), self.design)["status"], "FAIL")

    def test_directional_p_values_detect_consistent_damage(self) -> None:
        base = np.ones(12, dtype=int)
        changed = np.zeros(12, dtype=int)
        self.assertLess(runner.mcnemar_damage_p(base, changed), 0.01)
        self.assertLess(runner.signflip_p(np.ones(20), 5000, 20260901), 0.01)

    def test_pdf_auto_update_is_forbidden(self) -> None:
        self.assertIs(self.design["automatic_pdf_update_allowed"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
