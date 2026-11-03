"""Test 10 (constraints) and Test 11 (independent verification)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import policy_optimisation as PO
from src.prediction import load_frozen


def test_optimiser_respects_bounds_and_budget(frozen, population):
    model, _ = frozen
    obj = PO.Objective(model, population.sample(800, random_state=0).reset_index(drop=True))
    best, table, _ = PO.grid_search(obj, step=10.0, budget=1.5)
    levers = {k: float(best[k]) for k in PO.LEVERS}
    assert PO.check_constraints(levers, budget=1.5) == []
    assert (table["effort"] <= 1.5 + 1e-9).all()
    x, val, _, _ = PO.de_search(obj, budget=1.5, maxiter=20)
    assert PO.check_constraints(dict(zip(PO.LEVERS, x)), budget=1.5, tol=1e-4) == []
    assert val >= float(best["objective"]) - 1e-6      # DE should not do worse than the grid


def test_fast_evaluator_equals_the_general_path(frozen, population):
    model, _ = frozen
    obj = PO.Objective(model, population)
    if not obj.fast:
        pytest.skip("frozen model has no shortcut evaluator")
    generic = PO.GenericEvaluator(model, population)
    rng = np.random.default_rng(1)
    for _ in range(5):
        lev = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in C.POLICY_BOUNDS.items()}
        assert np.allclose(obj.evaluator.probs(lev), generic.probs(lev), atol=1e-12)


def test_reported_optimum_is_verified_independently():
    p = C.Paths().results / "optimisation_result.json"
    if not p.exists():
        pytest.skip("run the pipeline first")
    res = json.loads(p.read_text())
    model, _ = load_frozen()
    pop = pd.read_csv(C.Paths().synthetic / "synthetic_population.csv")
    v = PO.verify_solution(res["optimal_levers"], res["objective_sustainable_share_pct"] / 100, model, pop)
    assert v["matches"] and v["constraint_problems"] == []
    assert v["non_negative_times"] and v["non_negative_costs"] and v["availability_unchanged"]


def test_optimum_is_not_beaten_by_any_benchmark():
    p = C.Paths().results / "optimisation_benchmarks.csv"
    if not p.exists():
        pytest.skip("run the pipeline first")
    b = pd.read_csv(p)
    opt = b.loc[b["policy"] == "Optimised policy", "sustainable_share_pct"].iloc[0]
    assert (b["sustainable_share_pct"] <= opt + 1e-9).all()
