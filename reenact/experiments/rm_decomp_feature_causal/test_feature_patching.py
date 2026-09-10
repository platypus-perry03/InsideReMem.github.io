from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("run_feature_patching.py")
SPEC = importlib.util.spec_from_file_location("run_feature_patching", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FeaturePatchingTests(unittest.TestCase):
    def test_component_parser(self) -> None:
        self.assertEqual(MODULE.component_from_id("L29H00030")["component_index"], 30)
        self.assertEqual(MODULE.component_from_id("L31N13336")["component_type"], "neuron")

    def test_patch_operation_set_is_bidirectional(self) -> None:
        self.assertEqual(len(MODULE.PATCH_OPERATIONS), 8)
        self.assertIn("A_to_B", MODULE.PATCH_OPERATIONS)
        self.assertIn("B_to_A", MODULE.PATCH_OPERATIONS)
        self.assertIn("A_to_C", MODULE.PATCH_OPERATIONS)
        self.assertIn("C_to_A", MODULE.PATCH_OPERATIONS)

    def test_feature_vectors(self) -> None:
        frame = pd.DataFrame(
            {
                "chain_id": ["x"] * 4,
                "condition": ["A", "B", "C", "D"],
                "score": [1.0, 3.0, 0.0, 2.0],
            }
        )
        _, vectors = MODULE.feature_vectors(frame)
        self.assertTrue(np.allclose(vectors["E_R"], [2.0]))
        self.assertTrue(np.allclose(vectors["E_M"], [1.0]))

    def test_patch_vectors_zero_when_patch_has_no_effect(self) -> None:
        baseline = pd.DataFrame(
            {"A": [1.0], "B": [3.0], "C": [0.0], "D": [2.0]}, index=["x"]
        )
        feature = {"E_R": np.array([2.0]), "E_M": np.array([1.0])}
        rows = []
        targets = {"A_to_B": 3.0, "C_to_D": 2.0, "B_to_A": 1.0, "D_to_C": 0.0,
                   "A_to_C": 0.0, "B_to_D": 2.0, "C_to_A": 1.0, "D_to_B": 3.0}
        for operation, score in targets.items():
            rows.append({"chain_id": "x", "operation": operation, "score": score})
        vectors = MODULE.patch_vectors(baseline, pd.DataFrame(rows), feature)
        for values in vectors.values():
            self.assertTrue(np.allclose(values, [0.0]))


if __name__ == "__main__":
    unittest.main()
