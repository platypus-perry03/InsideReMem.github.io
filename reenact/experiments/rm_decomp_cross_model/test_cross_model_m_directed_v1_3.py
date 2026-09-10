"""Model-free tests for the frozen M-directed cross-model runner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_cross_model_m_directed_v1_3 as runner  # noqa: E402


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = SimpleNamespace(o_proj=nn.Linear(8, 8, bias=False))
        self.mlp = SimpleNamespace(down_proj=nn.Linear(6, 8, bias=False))


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2)
        self.model = SimpleNamespace(layers=[_Layer()])


class MDirectedCrossModelRunnerTests(unittest.TestCase):
    def test_whole_depth_includes_every_layer(self) -> None:
        self.assertEqual(runner.eligible_layers(32, "all_transformer_blocks"), list(range(32)))
        self.assertEqual(runner.eligible_layers(42, "all_transformer_blocks"), list(range(42)))

    def test_depth_bands_do_not_allocate_candidates(self) -> None:
        design = {"depth_bins": {"early_max": 1 / 3, "middle_max": 2 / 3}}
        self.assertEqual(runner.depth_band(0.2, design), "early")
        self.assertEqual(runner.depth_band(0.5, design), "middle")
        self.assertEqual(runner.depth_band(0.9, design), "late")

    def test_component_id_roundtrip(self) -> None:
        self.assertEqual(runner.parse_component("L31N13336"), ("neuron", 31, 13336))
        self.assertEqual(runner.parse_component("L29H00030"), ("head", 29, 30))
        with self.assertRaises(RuntimeError):
            runner.parse_component("L29H30")

    def test_bh_is_bounded_and_order_correct(self) -> None:
        q = runner.bh(np.asarray([0.04, 0.001, 0.02]))
        np.testing.assert_allclose(q, [0.04, 0.003, 0.03])

    def test_gap_reduction_sign(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        base = np.asarray([0.0, 0.0, 2.0, 2.0])
        changed = np.asarray([0.0, 0.0, 1.0, 1.0])
        result = runner.gap(base, changed, labels)
        self.assertEqual(result["G_base"], 2.0)
        self.assertEqual(result["G_intervention"], 1.0)
        self.assertEqual(result["gap_reduction"], 1.0)

    def test_head_intervention_changes_only_last_token_block(self) -> None:
        model = _Model()
        intervention = runner.Intervention(model, "L00H00001", 0.5, None)
        intervention.install()
        values = torch.ones(2, 3, 8)
        output = model.model.layers[0].self_attn.o_proj(values)
        intervention.remove()
        direct = model.model.layers[0].self_attn.o_proj
        expected_input = values.clone()
        expected_input[:, -1, 4:8] *= 0.5
        torch.testing.assert_close(output, direct(expected_input))
        self.assertEqual(len(direct._forward_pre_hooks), 0)

    def test_neuron_intervention_mean_ablation_last_token_only(self) -> None:
        model = _Model()
        intervention = runner.Intervention(model, "L00N00002", 1.0, 3.0)
        intervention.install()
        values = torch.ones(2, 3, 6)
        output = model.model.layers[0].mlp.down_proj(values)
        intervention.remove()
        expected_input = values.clone()
        expected_input[:, -1, 2] = 3.0
        torch.testing.assert_close(output, model.model.layers[0].mlp.down_proj(expected_input))

    def test_discovery_reset_preserves_hook_target_dictionaries(self) -> None:
        model = _Model()
        capture = runner.DiscoveryCapture(model, [0])
        pre_o_target = capture.pre_o
        z_target = capture.z
        capture.pre_o[0] = torch.ones(1)
        capture.z[0] = torch.ones(1)
        capture.reset()
        self.assertIs(capture.pre_o, pre_o_target)
        self.assertIs(capture.z, z_target)
        self.assertEqual(capture.pre_o, {})
        self.assertEqual(capture.z, {})

    def test_candidate_selection_is_discovery_only_and_deterministic(self) -> None:
        rows = []
        for component_type, letter, width in (("head", "H", 20), ("neuron", "N", 30)):
            for index in range(width):
                rows.append({
                    "component_id": f"L31{letter}{index:05d}", "component_type": component_type,
                    "module_index": 31, "component_index": index, "relative_layer_depth": 1.0,
                    "Delta_discovery": float(index + 1), "abs_Delta_discovery": float(index + 1),
                    "memory_mean_discovery": -float(index + 2),
                    "reasoning_mean_discovery": -1.0,
                    "writer_scale_proxy": float(index + 2), "pooled_activation_mean": 0.0,
                })
        frame = pd.DataFrame(rows)
        design = {"discovery_max_candidates_per_component_type": 5, "seed": 20260831, "random_controls_per_candidate": 3}
        candidates_a, controls_a = runner.select_candidates_and_controls([frame], design)
        candidates_b, controls_b = runner.select_candidates_and_controls([frame], design)
        self.assertEqual(len(candidates_a), 10)
        self.assertTrue((candidates_a["Delta_discovery"] > 0).all())
        self.assertTrue((candidates_a["memory_mean_discovery"] < 0).all())
        self.assertEqual(candidates_a[candidates_a["component_type"] == "head"].iloc[0]["component_id"], "L31H00019")
        self.assertEqual(candidates_a[candidates_a["component_type"] == "neuron"].iloc[0]["component_id"], "L31N00029")
        pd.testing.assert_frame_equal(candidates_a, candidates_b)
        pd.testing.assert_frame_equal(controls_a, controls_b)
        self.assertEqual(set(controls_a["control_kind"]), {"matched", "random"})

    def test_candidate_selection_can_return_zero(self) -> None:
        frame = pd.DataFrame([{
            "component_id": "L00H00000", "component_type": "head", "module_index": 0,
            "component_index": 0, "relative_layer_depth": 0.1, "depth_band": "early",
            "Delta_discovery": 1.0, "abs_Delta_discovery": 1.0,
            "memory_mean_discovery": 1.0, "reasoning_mean_discovery": 2.0,
            "writer_scale_proxy": 1.0, "pooled_activation_mean": 0.0,
        }, {
            "component_id": "L00N00000", "component_type": "neuron", "module_index": 0,
            "component_index": 0, "relative_layer_depth": 0.1, "depth_band": "early",
            "Delta_discovery": 1.0, "abs_Delta_discovery": 1.0,
            "memory_mean_discovery": 1.0, "reasoning_mean_discovery": 2.0,
            "writer_scale_proxy": 1.0, "pooled_activation_mean": 0.0,
        }])
        design = {"discovery_max_candidates_per_component_type": 5, "seed": 20260831, "random_controls_per_candidate": 3}
        candidates, controls = runner.select_candidates_and_controls([frame], design)
        self.assertTrue(candidates.empty)
        self.assertTrue(controls.empty)

    def test_frozen_design_prohibits_pdf_update(self) -> None:
        design = runner.load_design(HERE / "design_v1_3_m_directed_frozen.json")
        self.assertIs(design["result_pdf_update_allowed"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
