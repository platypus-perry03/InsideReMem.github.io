from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from stage_b_core import (  # noqa: E402
    CONTROLLED_COLUMNS,
    FeatureExtractor,
    validate_backend_tolerances,
    validate_controlled_manifest,
    validate_feature_schema,
)
from stage_b_stats import benjamini_hochberg, paired_summary, template_effects  # noqa: E402
if str(PACKAGE_DIR) in sys.path:
    sys.path.remove(str(PACKAGE_DIR))
sys.path.insert(0, str(PACKAGE_DIR))
from run import summarize_controlled  # noqa: E402


class FeatureSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads((PACKAGE_DIR / "feature_schema.json").read_text(encoding="utf-8"))

    def test_schema_is_fully_materialized(self) -> None:
        validate_feature_schema(self.schema)

    def test_feature_extraction(self) -> None:
        extractor = FeatureExtractor(self.schema)
        result = extractor.summarize("John has 3 apples and receives five more; 3 + 5 = 8.")
        self.assertGreaterEqual(result["numeric_span_count"], 5)
        self.assertGreaterEqual(result["relation_span_count"], 2)
        self.assertGreaterEqual(result["operator_span_count"], 2)

    def test_placeholder_is_rejected(self) -> None:
        invalid = json.loads(json.dumps(self.schema))
        invalid["textual_operator_lexicon"] = ["..."]
        with self.assertRaises(RuntimeError):
            validate_feature_schema(invalid)


class GateTests(unittest.TestCase):
    def test_backend_tolerance_null_is_rejected(self) -> None:
        values = {
            "logit_max_abs_tolerance": 0.1,
            "hidden_state_max_abs_tolerance": 0.1,
            "hidden_state_cosine_tolerance": 0.99,
            "head_reconstruction_mean_tolerance": 0.1,
            "head_reconstruction_max_tolerance": 0.1,
            "source_reconstruction_mean_tolerance": 0.1,
            "source_reconstruction_max_tolerance": None,
        }
        with self.assertRaises(RuntimeError):
            validate_backend_tolerances({"backend_tolerances": values})

    def test_unapproved_controlled_manifest_is_blocked(self) -> None:
        row = {column: "value" for column in CONTROLLED_COLUMNS}
        row.update({"pair_id": "p1", "split": "confirmatory", "approved": False, "human_validated": False, "reviewer_id": ""})
        frame = pd.DataFrame([row])
        with self.assertRaises(RuntimeError):
            validate_controlled_manifest(frame, "confirmatory", require_approved=True)

    def test_template_family_is_required(self) -> None:
        row = {column: "value" for column in CONTROLLED_COLUMNS}
        row.update({"pair_id": "p1", "split": "pilot", "template_family": ""})
        frame = pd.DataFrame([row])
        with self.assertRaises(RuntimeError):
            validate_controlled_manifest(frame, "pilot", require_approved=False)


class StatisticsTests(unittest.TestCase):
    def test_template_is_observation_unit(self) -> None:
        ids, effects = template_effects(["a", "a", "b", "b"], [1.0, 3.0, -1.0, 1.0])
        self.assertEqual(ids.tolist(), ["a", "b"])
        np.testing.assert_allclose(effects, [2.0, 0.0])

    def test_paired_summary_is_deterministic(self) -> None:
        kwargs = dict(
            template_ids=["a", "b", "c", "d"],
            differences=[1.0, 2.0, 1.5, 2.5],
            bootstrap_iterations=1000,
            permutation_iterations=1000,
            seed=7,
        )
        self.assertEqual(paired_summary(**kwargs), paired_summary(**kwargs))

    def test_bh_preserves_nan(self) -> None:
        result = benjamini_hochberg([0.01, np.nan, 0.04])
        self.assertTrue(np.isnan(result[1]))
        np.testing.assert_allclose(result[[0, 2]], [0.02, 0.04])

    def test_neuron_inference_uses_activation_not_deterministic_contribution(self) -> None:
        rows = []
        for template, original, modified in (("t1", 1.0, 2.0), ("t2", 2.0, 4.0)):
            for role, component, control_kind, scale in (
                ("candidate", "L00N00001", "candidate", 1.0),
                ("control", "L00N00002", "matched", 0.5),
            ):
                for variant, value in (("original", original), ("modified", modified)):
                    rows.append({
                        "pair_id": f"{template}-p", "hypothesis_id": "h1", "feature_family": "numeric_value",
                        "template_family": template, "candidate_id": "L00N00001", "component_id": component,
                        "component_type": "neuron", "component_role": role, "control_kind": control_kind,
                        "variant": variant, "activation": value * scale, "total_contribution": value * scale * 3.0,
                    })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "responses.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            config = {"statistics": {"bootstrap_iterations": 100, "permutation_iterations": 100, "random_seed": 1}}
            result = summarize_controlled(config, source, "confirmatory", root / "effects.csv", root / "specificity.csv")
            self.assertEqual(set(result["endpoint"]), {"activation"})
            self.assertTrue((root / "specificity.csv").exists())


if __name__ == "__main__":
    unittest.main()
