"""Policy scenarios: change travel attributes, re-predict with the frozen model.

Every scenario is a model-based counterfactual. It tells us what the fitted
model implies if an attribute changes while everything else about the person
stays the same; it is not evidence that the real intervention would produce
the same behaviour change.
"""
import numpy as np
import pandas as pd

from . import config as C
from .prediction import predict_probabilities, summarise


def apply_policy(df, time_pct=None, cost_pct=None, bus_av_extension=0.0, walk_av_km=None, seed=C.SEED):
    """Return a modified copy of df.

    time_pct / cost_pct : {mode: signed percentage change}, e.g. {"bus": -10}
    bus_av_extension    : share (0-1) of people without a bus option who get one
    walk_av_km          : walking becomes available for trips up to this distance
    """
    out = df.copy()
    for m, pct in (time_pct or {}).items():
        col = C.TIME_COLS[m]
        out[col] = np.maximum(out[col] * (1 + pct / 100.0), 0.0)
    for m, pct in (cost_pct or {}).items():
        col = C.COST_COLS[m]
        out[col] = np.maximum(out[col] * (1 + pct / 100.0), 0.0)
    if bus_av_extension > 0:
        # fixed random order, so a larger extension always contains a smaller one
        order = np.random.default_rng(seed).random(len(out))
        no_bus = (out["Bus_AV"] == 0).to_numpy()
        ranks = pd.Series(np.where(no_bus, order, np.inf)).rank(method="first").to_numpy()
        n_new = int(round(bus_av_extension * no_bus.sum()))
        out.loc[no_bus & (ranks <= n_new), "Bus_AV"] = 1
    if walk_av_km is not None:
        out.loc[out["Distance_km"] <= walk_av_km, "Walk_AV"] = 1
    return out


def levers_to_policy(levers):
    """Map optimisation levers (positive numbers, in percent) to signed changes."""
    time_pct, cost_pct = {}, {}
    if levers.get("bus_time_red"):
        time_pct["bus"] = -levers["bus_time_red"]
    if levers.get("train_time_red"):
        time_pct["train"] = -levers["train_time_red"]
    if levers.get("bus_fare_red"):
        cost_pct["bus"] = -levers["bus_fare_red"]
    if levers.get("train_fare_red"):
        cost_pct["train"] = -levers["train_fare_red"]
    if levers.get("car_cost_inc"):
        cost_pct["car"] = levers["car_cost_inc"]
    return {"time_pct": time_pct, "cost_pct": cost_pct}


def simulate(model, pop, **policy):
    P = predict_probabilities(model, apply_policy(pop, **policy))
    table, sus = summarise(P)
    return P, table, sus


def run_scenarios(model, pop, scenarios):
    """scenarios: list of (name, policy-kwargs). Returns one row per scenario."""
    rows = []
    for name, policy in scenarios:
        _, table, sus = simulate(model, pop, **policy)
        row = {"scenario": name}
        row.update({f"{m}_pct": v for m, v in zip(C.MODES, table["expected_share_pct"])})
        row.update(sus)
        rows.append(row)
    out = pd.DataFrame(rows)
    base = out.iloc[0]
    out["sustainable_change_pp"] = out["expected_sustainable_share_pct"] - base["expected_sustainable_share_pct"]
    return out


# ---------------------------------------------------------------------------
# The four experiments of the brief
# ---------------------------------------------------------------------------

def experiment_cost(model, pop):
    S = [("Baseline", {})]
    for p in (5, 10, 15, 20, 30, 50):
        S.append((f"Bus fare -{p}%", {"cost_pct": {"bus": -p}}))
    for p in (5, 10, 15, 20, 30, 50):
        S.append((f"Rail fare -{p}%", {"cost_pct": {"train": -p}}))
    for p in (5, 10, 15, 20, 30, 50):
        S.append((f"Car cost +{p}%", {"cost_pct": {"car": p}}))
    for p in (10, 20, 30):
        S.append((f"Two-wheeler cost +{p}%", {"cost_pct": {"twow": p}}))
    for p in (10, 20, 30):
        S.append((f"Auto fare +{p}%", {"cost_pct": {"auto": p}}))
    S.append(("Bus fare -10% and car cost +10%", {"cost_pct": {"bus": -10, "car": 10}}))
    return run_scenarios(model, pop, S)


def experiment_time(model, pop):
    S = [("Baseline", {})]
    for p in (5, 10, 15, 20, 30):
        S.append((f"Bus time -{p}%", {"time_pct": {"bus": -p}}))
    for p in (5, 10, 15, 20):
        S.append((f"Rail time -{p}%", {"time_pct": {"train": -p}}))
    for p in (10, 20, 30):
        S.append((f"Car time +{p}%", {"time_pct": {"car": p}}))
    for p in (10, 20, 30):
        S.append((f"Two-wheeler time +{p}%", {"time_pct": {"twow": p}}))
    return run_scenarios(model, pop, S)


def experiment_availability(model, pop):
    """Extrapolation scenarios: the availability changes put people into choice
    sets the survey does not contain for them. Suburban rail is already
    available to everyone, so nothing can be learned about rail availability."""
    S = [("Baseline", {})]
    for p in (0.10, 0.25, 0.50, 1.00):
        S.append((f"Bus offered to {int(p * 100)}% of those without it", {"bus_av_extension": p}))
    for km in (2.0, 3.0):
        S.append((f"Walking available for all trips up to {km:g} km", {"walk_av_km": km}))
    return run_scenarios(model, pop, S)


def experiment_combined(model, pop):
    S = [
        ("Baseline", {}),
        ("Bus package: time -15%, fare -20%", {"time_pct": {"bus": -15}, "cost_pct": {"bus": -20}}),
        ("Rail package: time -10%, fare -20%", {"time_pct": {"train": -10}, "cost_pct": {"train": -20}}),
        ("PT package: bus and rail packages together",
         {"time_pct": {"bus": -15, "train": -10}, "cost_pct": {"bus": -20, "train": -20}}),
        ("Push and pull: PT package + car cost +25%",
         {"time_pct": {"bus": -15, "train": -10}, "cost_pct": {"bus": -20, "train": -20, "car": 25}}),
        ("All levers at their upper bounds",
         levers_to_policy({k: hi for k, (lo, hi) in C.POLICY_BOUNDS.items()})),
    ]
    return run_scenarios(model, pop, S)
