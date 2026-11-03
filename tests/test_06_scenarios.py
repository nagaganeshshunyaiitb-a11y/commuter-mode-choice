"""Test 8 (scenario simulation) and Test 9 (no-change scenario)."""
import numpy as np

from src import config as C
from src import scenario_simulation as SS
from src.prediction import predict_probabilities


def test_bus_time_and_fare_change_the_probabilities(frozen, population):
    model, _ = frozen
    base = predict_probabilities(model, population)
    new = predict_probabilities(model, SS.apply_policy(population, time_pct={"bus": -20}, cost_pct={"bus": -20}))
    has_bus = population["Bus_AV"] == 1
    assert (new.loc[has_bus, "P_bus"] > base.loc[has_bus, "P_bus"]).all()
    assert (new.loc[~has_bus, "P_bus"] == 0).all()
    assert np.allclose(new.sum(axis=1), 1)


def test_no_change_scenario_equals_baseline(frozen, population):
    model, _ = frozen
    base = predict_probabilities(model, population).to_numpy()
    same, _, _ = SS.simulate(model, population)
    zero_levers = SS.levers_to_policy({k: 0.0 for k in C.POLICY_BOUNDS})
    zero, _, _ = SS.simulate(model, population, **zero_levers)
    assert np.array_equal(same.to_numpy(), base)
    assert np.array_equal(zero.to_numpy(), base)


def test_apply_policy_does_not_modify_its_input(population):
    before = population.copy()
    out = SS.apply_policy(population, time_pct={"bus": -30}, cost_pct={"car": 50}, bus_av_extension=0.5)
    assert population.equals(before)
    assert (out[list(C.TIME_COLS.values())] >= 0).all().all()
    assert np.allclose(out["TC_Car"], population["TC_Car"] * 1.5)


def test_bus_extension_is_nested(population):
    small = SS.apply_policy(population, bus_av_extension=0.25)["Bus_AV"]
    large = SS.apply_policy(population, bus_av_extension=0.50)["Bus_AV"]
    assert (large >= small).all()
    gained = (small == 1) & (population["Bus_AV"] == 0)
    assert abs(gained.sum() - round(0.25 * (population["Bus_AV"] == 0).sum())) <= 1
