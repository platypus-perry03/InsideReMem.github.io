import unittest

import numpy as np
import pandas as pd

from run_pre_stage_c import bh_adjust, extract_memory_features, ols_hc3
from score_linguistic_audit import cohen_kappa, parse_bool


SCHEMA = {
    "capitalized_exclusions": ["Who", "What", "The"],
    "date_words": ["january", "month"],
    "person_cues": ["who", "author", "born"],
    "location_cues": ["where", "capital"],
    "temporal_cues": ["when", "year", "born"],
    "factual_association_phrases": ["author of", "born in"],
    "attribute_cues": ["who", "author", "capital", "born"],
}


class PreStageCTests(unittest.TestCase):
    def test_feature_extraction(self):
        result = extract_memory_features("Who was born in Paris in 1984?", SCHEMA)
        self.assertEqual(result["has_capitalized_span"], 1)
        self.assertEqual(result["has_year"], 1)
        self.assertEqual(result["has_person_cue"], 1)
        self.assertEqual(result["has_entity_attribute_proxy"], 1)

    def test_bh(self):
        adjusted = bh_adjust([0.01, 0.02, 0.5])
        self.assertTrue(np.allclose(adjusted, [0.03, 0.03, 0.5]))

    def test_ols_recovers_signal(self):
        rng = np.random.default_rng(1)
        feature = np.tile([0.0, 1.0], 100)
        x = np.column_stack([np.ones(200), feature, rng.normal(size=200)])
        y = 0.8 * feature + rng.normal(scale=0.2, size=200)
        result = ols_hc3(y, x, 1)
        self.assertGreater(result["beta"], 0.7)
        self.assertLess(result["p_value"], 0.001)

    def test_audit_parsing_and_kappa(self):
        self.assertTrue(parse_bool("true", "field", "A1"))
        self.assertFalse(parse_bool("FALSE", "field", "A1"))
        left = pd.Series([True, True, False, False])
        self.assertEqual(cohen_kappa(left, left), 1.0)


if __name__ == "__main__":
    unittest.main()
