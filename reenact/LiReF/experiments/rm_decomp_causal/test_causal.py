import unittest

import numpy as np

from run_causal import apply_component_edit, build_conditions, component_from_id, gap_metrics, make_mediation_pairs


class CausalTests(unittest.TestCase):
    def test_component_id(self):
        self.assertEqual(component_from_id("L25H00021"), {"component_id": "L25H00021", "component_type": "head", "module_index": 25, "component_index": 21})
        self.assertEqual(component_from_id("L31N13336")["component_index"], 13336)

    def test_edit_math(self):
        x = np.arange(12, dtype=np.float32).reshape(1, 1, 12)
        y = apply_component_edit(x, "head", 1, 0.5, 3, 0.0)
        np.testing.assert_allclose(y[..., 3:6], x[..., 3:6] * 0.5)
        z = apply_component_edit(x, "neuron", 4, 1.0, 3, -2.0)
        self.assertEqual(float(z[0, -1, 4]), -2.0)

    def test_gap_sign_and_abs(self):
        labels = np.array([0, 0, 1, 1])
        base = np.array([0.0, 0.0, 2.0, 2.0])
        intervention = np.array([0.0, 0.0, 1.0, 1.0])
        result = gap_metrics(base, intervention, labels)
        self.assertEqual(result["G_base"], 2.0)
        self.assertEqual(result["delta_abs_G"], -1.0)

    def test_mediation_pairs_are_balanced(self):
        rows = make_mediation_pairs(8)
        self.assertEqual(len(rows), 16)
        self.assertEqual({row["relevance"] for row in rows}, {"relevant", "irrelevant"})
        self.assertTrue(all(row["original_text"] != row["modified_text"] for row in rows))

    def test_full20_conditions_have_no_joint_and_cover_every_candidate(self):
        candidates = [f"L31N{i:05d}" for i in range(20)]
        controls = []
        for index, candidate in enumerate(candidates):
            controls.append({"candidate_id": candidate, "control_id": f"L30N{index:05d}", "control_kind": "matched"})
        config = {
            "causal_candidates": candidates,
            "candidate_alphas": [0.0, 0.5, 1.0],
            "control_alpha": 1.0,
            "include_joint_intervention": False,
        }
        conditions = build_conditions(config, {"controls": controls})
        self.assertEqual(sum(row.component_role == "candidate" for row in conditions), 60)
        self.assertEqual(sum(row.component_role == "control" for row in conditions), 20)
        self.assertFalse(any(row.component_role == "joint" for row in conditions))


if __name__ == "__main__":
    unittest.main()
