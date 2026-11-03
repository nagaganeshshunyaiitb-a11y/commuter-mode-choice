"""Test 12: the whole pipeline, workbook to optimised policy, in a temp folder."""
import json

import pytest


@pytest.mark.slow
def test_pipeline_runs_end_to_end(tmp_path):
    import run_pipeline
    summary = run_pipeline.run(root=tmp_path, fast=True, n_synth=600)
    res = tmp_path / "reports" / "results"
    for f in ["audit_summary.json", "cv_summary.csv", "test_metrics.csv", "model_selection.json",
              "synthetic_method_comparison.csv", "baseline_shares_synthetic.csv", "scenario_combined.csv",
              "optimisation_result.json", "pipeline_summary.json"]:
        assert (res / f).exists(), f
    assert (tmp_path / "models" / "best_model" / "model.joblib").exists()
    assert (tmp_path / "data" / "synthetic" / "synthetic_population.csv").exists()
    opt = json.loads((res / "optimisation_result.json").read_text())
    assert opt["verification"]["matches"]
    assert opt["objective_sustainable_share_pct"] >= opt["baseline_sustainable_pct"] - 1e-9
    assert summary["no_change_max_abs_diff"] == 0.0
