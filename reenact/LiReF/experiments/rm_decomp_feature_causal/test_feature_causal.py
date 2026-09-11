import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_feature_causal.py")
SPEC = importlib.util.spec_from_file_location("run_feature_causal", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StageDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace(gpu_id=None, batch_size=None)
        cls.config = MODULE.load_config(Path(__file__).with_name("config.json"), args)
        cls.chains = MODULE.load_popqa_chains(cls.config)

    def test_chain_pool_and_unique_answer(self):
        self.assertGreaterEqual(len(self.chains), 200)
        for chain in self.chains:
            self.assertEqual(len(chain["options"]), 4)
            self.assertEqual(len(set(chain["options"])), 4)
            self.assertEqual(chain["options"][MODULE.LABELS.index(chain["correct_label"])], chain["capital"])

    def test_calibration_has_eight_items(self):
        rows = MODULE.calibration_rows(self.chains[0], self.config["prompt_template"])
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row["item_type"] == "r1" for row in rows), 3)
        self.assertEqual(sum(row["item_type"] == "r2" for row in rows), 3)

    def test_factorial_conditions_share_answer_and_options(self):
        frame = MODULE.build_factorial_items(self.chains[:6], "confirmatory", self.config["prompt_template"])
        self.assertEqual(set(frame["condition"]), set(MODULE.CONDITIONS))
        self.assertEqual(frame["template_family"].nunique(), 6)
        for _, group in frame.groupby("chain_id"):
            self.assertEqual(group["capital"].nunique(), 1)
            self.assertEqual(group["correct_label"].nunique(), 1)
            self.assertEqual(group["options_json"].nunique(), 1)

    def test_confirmatory_gate_rejects_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = {"audit": root / "audit", "manifests": root / "manifests"}
            p["audit"].mkdir()
            p["manifests"].mkdir()
            (p["audit"] / "audit_final_status.json").write_text(json.dumps({"status": "PENDING"}))
            (p["manifests"] / "confirmatory_go.json").write_text(json.dumps({"approved": False}))
            with self.assertRaises(RuntimeError):
                MODULE.enforce_confirmatory_gate(p)


if __name__ == "__main__":
    unittest.main()
