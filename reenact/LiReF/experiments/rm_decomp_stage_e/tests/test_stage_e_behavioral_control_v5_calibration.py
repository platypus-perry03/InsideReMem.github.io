from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


RUNNER_PATH=Path(__file__).resolve().parents[1]/"run_stage_e_behavioral_control_v5_calibration.py"
SPEC=importlib.util.spec_from_file_location("v5_calibration",RUNNER_PATH)
RUNNER=importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(RUNNER)


def rows(fail_points=False):
    output=[]
    for family in RUNNER.FAMILIES:
        for template in range(4):
            for frame in range(8):
                index=template*8+frame
                fc=index<24; generation=index<24
                if fail_points and family=="points_balance":fc=index<16; generation=index<16
                def c():return {"margin_nats":.1,"forced_choice_correct":fc,"generation_correct":generation,
                                "generation_valid_format":True,"correct_probability":.6}
                output.append({"pair_id":f"{family}_{template}_{frame}","lexical_family":family,
                               "template_family_id":f"{family}_{template}","frame_index":frame+1,
                               "conditions":{"arithmetic":c(),"selector":c()}})
    return output


class Tests(unittest.TestCase):
    def test_preflight(self):
        result=RUNNER.preflight(); self.assertTrue(all(result["checks"].values())); self.assertFalse(result["model_loaded"])

    def test_frozen_hard_gates_pass(self):
        summary=RUNNER.summarize(rows(),json.loads(RUNNER.DESIGN_PATH.read_text()))
        self.assertEqual(summary["passed_families"],["points_balance","temperature"])
        self.assertTrue(summary["primary_replication_gate_open"]); self.assertTrue(summary["interaction_replication_gate_open"])

    def test_floor_closes_replication(self):
        summary=RUNNER.summarize(rows(True),json.loads(RUNNER.DESIGN_PATH.read_text()))
        self.assertIn("points_balance",summary["failed_families"]); self.assertFalse(summary["primary_replication_gate_open"])

    def test_dz_is_descriptive_only(self):
        summary=RUNNER.summarize(rows(),json.loads(RUNNER.DESIGN_PATH.read_text()))
        self.assertFalse(summary["cluster_dz_used_for_pass_fail"])
        self.assertNotIn("cluster_dz",summary["family_summaries"]["points_balance"]["checks"])

    def test_bootstrap_contract(self):
        self.assertEqual(RUNNER.BOOTSTRAP_REPETITIONS,10000); self.assertEqual(RUNNER.BOOTSTRAP_SEED,20260902)


if __name__=="__main__":unittest.main()
