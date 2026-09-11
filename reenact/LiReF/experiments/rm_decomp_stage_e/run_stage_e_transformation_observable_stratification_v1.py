#!/usr/bin/env python3
"""Model-free stratification of the frozen Transformation feature by numeric presence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
PARENT_RUNNER = STAGE_DIR / "run_stage_e_natural_feature_discovery_v1.py"
OUTPUT_DIR = ROOT / "liref_outputs" / "rm_decomp" / "v2" / "e_transformation_observable_stratification_v1"
PRIMARY_ENDPOINTS = ("layer31_liref", "component_L29H00030", "component_L30H00006")
SECONDARY_ENDPOINTS = ("component_L31N13336", "component_L29H00031")


def load_parent() -> tuple[Any, Any, Any]:
    spec = importlib.util.spec_from_file_location("natural_feature_for_numeric_stratification", PARENT_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    manifest = module.load_and_verify_manifest()
    return module, manifest, module.load_analysis_frame(manifest)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def association_rows(module: Any, frame: Any, stratum: str) -> Any:
    import pandas as pd
    if stratum == "numeric":
        work = frame[frame["has_numeric"] == 1].copy()
    elif stratum == "nonnumeric":
        work = frame[frame["has_numeric"] == 0].copy()
    else:
        raise ValueError(stratum)
    rows = []
    for endpoint in (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS):
        result = module.ols_hc3(work, "transformation_required", endpoint, adjust_for_label=True)
        rows.append({"stratum": stratum, "endpoint": endpoint, **result})
    return pd.DataFrame(rows)


def phase_discovery() -> None:
    import numpy as np
    module, _manifest, frame = load_parent()
    discovery = frame[frame["analysis_split"] == "discovery"].copy()
    numeric = association_rows(module, discovery, "numeric")
    nonnumeric = association_rows(module, discovery, "nonnumeric")
    numeric["count_gate"] = (numeric["n_absent"] >= 100) & (numeric["n_present"] >= 100)
    numeric["primary_test"] = numeric["endpoint"].isin(PRIMARY_ENDPOINTS) & numeric["count_gate"]
    numeric["q_discovery_primary_bh"] = np.nan
    mask = numeric["primary_test"]
    numeric.loc[mask, "q_discovery_primary_bh"] = module.bh_adjust(numeric.loc[mask, "p"].tolist())
    numeric["discovery_supported"] = numeric["primary_test"] & (numeric["q_discovery_primary_bh"] < 0.05) & ((numeric["ci_low"] > 0) | (numeric["ci_high"] < 0))
    numeric_path = OUTPUT_DIR / "tables" / "discovery_numeric_only_associations.csv"
    nonnumeric_path = OUTPUT_DIR / "tables" / "discovery_nonnumeric_descriptive.csv"
    atomic_csv(numeric_path, numeric); atomic_csv(nonnumeric_path, nonnumeric)

    counts = frame.groupby(["analysis_split", "has_numeric", "transformation_required"], dropna=False).size().reset_index(name="n")
    categories = frame.groupby(["analysis_split", "category", "transformation_required"], dropna=False).size().reset_index(name="n")
    counts_path = OUTPUT_DIR / "tables" / "numeric_transformation_counts.csv"
    categories_path = OUTPUT_DIR / "tables" / "category_transformation_counts.csv"
    atomic_csv(counts_path, counts); atomic_csv(categories_path, categories)
    selected = [{"endpoint": row.endpoint, "discovery_beta": float(row.beta)} for row in numeric[numeric["discovery_supported"]].itertuples(index=False)]
    atomic_json(OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json", {
        "status": "DISCOVERY_COMPLETE_HELDOUT_NOT_ANALYZED", "selected_endpoints": selected,
        "numeric_table_sha256": module.sha256_file(numeric_path),
        "nonnumeric_table_sha256": module.sha256_file(nonnumeric_path),
        "counts_sha256": module.sha256_file(counts_path), "categories_sha256": module.sha256_file(categories_path),
        "heldout_used_for_selection": False, "external_api_used": False,
    })


def phase_heldout() -> None:
    import pandas as pd
    module, _manifest, frame = load_parent()
    selection_path = OUTPUT_DIR / "manifests" / "discovery_selection_frozen.json"; selection = json.loads(selection_path.read_text())
    numeric_path = OUTPUT_DIR / "tables" / "discovery_numeric_only_associations.csv"
    if selection.get("status") != "DISCOVERY_COMPLETE_HELDOUT_NOT_ANALYZED" or module.sha256_file(numeric_path) != selection["numeric_table_sha256"]:
        raise RuntimeError("Discovery selection is not frozen")
    validation = frame[frame["analysis_split"] == "validation"].copy()
    all_rows = association_rows(module, validation, "numeric"); selected_rows = []
    for spec in selection["selected_endpoints"]:
        row = all_rows[all_rows["endpoint"] == spec["endpoint"]].iloc[0].to_dict()
        row["discovery_beta"] = spec["discovery_beta"]; selected_rows.append(row)
    heldout = pd.DataFrame(selected_rows)
    if len(heldout):
        heldout["count_gate"] = (heldout["n_absent"] >= 50) & (heldout["n_present"] >= 50)
        heldout["q_heldout_selected_bh"] = module.bh_adjust(heldout["p"].tolist())
        heldout["same_sign"] = heldout["beta"] * heldout["discovery_beta"] > 0
        heldout["heldout_supported"] = heldout["count_gate"] & heldout["same_sign"] & (heldout["q_heldout_selected_bh"] < 0.05) & ((heldout["ci_low"] > 0) | (heldout["ci_high"] < 0))
    heldout_path = OUTPUT_DIR / "tables" / "heldout_numeric_only_checks.csv"; atomic_csv(heldout_path, heldout)
    atomic_json(OUTPUT_DIR / "transformation_observable_stratification_summary.json", {
        "status": "COMPLETE_POST_DISCOVERY_OBSERVATIONAL_STRATIFICATION",
        "selected_endpoint_count": len(selection["selected_endpoints"]),
        "heldout_supported_endpoints": [] if not len(heldout) else [{"endpoint": row.endpoint, "beta": float(row.beta)} for row in heldout[heldout["heldout_supported"]].itertuples(index=False)],
        "nonnumeric_transformation_total": 26,
        "interpretation": "Tests whether the association persists among numeric-present items; does not identify a causal primitive.",
        "external_api_used": False, "new_model_forward": False, "heldout_is_independent_confirmatory": False,
        "causal_claim_allowed": False, "heldout_sha256": module.sha256_file(heldout_path),
    })


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("discovery", "heldout")); args = parser.parse_args()
    if args.phase == "discovery": phase_discovery()
    else: phase_heldout()


if __name__ == "__main__":
    main()
