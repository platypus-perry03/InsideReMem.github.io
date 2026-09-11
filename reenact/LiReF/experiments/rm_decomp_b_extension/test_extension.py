from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_extension as extension


class Args:
    gpu_id = None
    batch_size = None


class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = extension.load_config(ROOT / "config.json", Args())
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.config["model_path"], trust_remote_code=True)
        cls.pilot = extension.generate_manifest(cls.config, "pilot", cls.tokenizer)
        cls.confirmatory = extension.generate_manifest(cls.config, "confirmatory", cls.tokenizer)

    def test_frozen_candidate_scale_design(self) -> None:
        self.assertEqual(len(self.pilot), 90)
        self.assertEqual(len(self.confirmatory), 1260)
        relation = self.confirmatory[self.confirmatory["feature_family"] == "relation_polarity"]
        self.assertEqual(relation["analysis_cluster"].nunique(), 80)
        self.assertEqual(relation["lexical_family"].nunique(), 6)

    def test_pilot_confirmatory_are_disjoint(self) -> None:
        pilot_text = set(self.pilot["original_text"]) | set(self.pilot["modified_text"])
        confirm_text = set(self.confirmatory["original_text"]) | set(self.confirmatory["modified_text"])
        self.assertFalse(pilot_text & confirm_text)
        self.assertFalse(set(self.pilot["analysis_cluster"]) & set(self.confirmatory["analysis_cluster"]))
        pilot_lex = set(self.pilot.query("feature_family == 'relation_polarity'")["lexical_family"])
        confirm_lex = set(self.confirmatory.query("feature_family == 'relation_polarity'")["lexical_family"])
        self.assertFalse(pilot_lex & confirm_lex)

    def test_numeric_representation_is_prompt_token_matched(self) -> None:
        frame = self.confirmatory[
            self.confirmatory["feature_family"] == "numeric_representation_token_matched"
        ]
        self.assertTrue((frame["token_length_original"] == frame["token_length_modified"]).all())


class ContrastTests(unittest.TestCase):
    def test_relevance_interaction_is_relevant_minus_irrelevant(self) -> None:
        rows = []
        for condition, original, modified in (
            ("relevant", 1.0, 4.0),
            ("irrelevant", 2.0, 3.0),
        ):
            for variant, value in (("original", original), ("modified", modified)):
                rows.append({
                    "split": "confirmatory", "pair_id": f"p-{condition}",
                    "feature_family": "relation_polarity", "condition": condition,
                    "lexical_family": "lex", "template_family": "t", "analysis_cluster": "c",
                    "base_id": "b", "variant": variant, "candidate_id": "L00H00000",
                    "component_id": "L00H00000", "component_type": "head",
                    "component_role": "candidate", "control_kind": "candidate", "module_index": 0,
                    "component_index": 0, "activation": "", "projection": 1.0,
                    "total_contribution": value,
                })
        # Add the other required families so build_effect_vectors has complete inputs.
        for family, conditions in (
            ("numeric_value", ("relevant", "irrelevant")),
            ("factual_entity_bundle", ("relevant", "irrelevant")),
        ):
            for condition in conditions:
                for variant, value in (("original", 0.0), ("modified", 0.0)):
                    row = dict(rows[0])
                    row.update({"pair_id": f"{family}-{condition}", "feature_family": family,
                                "condition": condition, "variant": variant, "total_contribution": value})
                    rows.append(row)
        for variant, value in (("original", 0.0), ("modified", 0.0)):
            row = dict(rows[0])
            row.update({"pair_id": "rep", "feature_family": "numeric_representation_token_matched",
                        "condition": "not_applicable", "variant": variant, "total_contribution": value})
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            vectors = extension.build_effect_vectors(path)
        interaction = vectors[vectors["analysis_family"] == "relation_relevance_interaction"]
        self.assertEqual(interaction.iloc[0]["value"], 2.0)


if __name__ == "__main__":
    unittest.main()
