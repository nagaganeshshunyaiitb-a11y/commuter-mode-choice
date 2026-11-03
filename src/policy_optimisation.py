"""Bounded policy optimisation.

    maximise   S(x) = (1/N) sum_n [P_n(walk) + P_n(bus) + P_n(rail)]
    subject to lo_k <= x_k <= hi_k                       (policy bounds)
               sum_k x_k / hi_k <= B                     (optional effort budget)

x holds the five levers in config.POLICY_BOUNDS. There is no cost data in the
project, so B is a dimensionless "share of the maximum intervention" budget,
not money. It is only used as a sensitivity analysis; the main result is the
purely bounded problem.

Grid search is the baseline. Differential evolution is run as a second,
independent search, and the reported optimum is recomputed from the files on
disk by verify_solution().
"""
import itertools
import time

import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, differential_evolution

from . import config as C
from .choice_models import MultinomialLogit, NestedLogit, build_design
from .prediction import predict_probabilities, summarise
from .scenario_simulation import apply_policy, levers_to_policy

LEVERS = list(C.POLICY_BOUNDS)
SUS_IDX = [C.MODES.index(m) for m in C.SUSTAINABLE]
# which data column each lever scales, and in which direction
LEVER_COLUMN = {
    "bus_time_red": (C.TIME_COLS["bus"], -1), "bus_fare_red": (C.COST_COLS["bus"], -1),
    "train_time_red": (C.TIME_COLS["train"], -1), "train_fare_red": (C.COST_COLS["train"], -1),
    "car_cost_inc": (C.COST_COLS["car"], +1),
}


def effort(x, bounds=C.POLICY_BOUNDS):
    return float(sum(v / bounds[k][1] for k, v in zip(LEVERS, x) if bounds[k][1] > 0))


class GenericEvaluator:
    """Works for any fitted model: modify the data, call predict_proba."""

    def __init__(self, model, pop):
        self.model, self.pop = model, pop

    def probs(self, levers):
        return predict_probabilities(self.model, apply_policy(self.pop, **levers_to_policy(levers))).to_numpy()


class FastChoiceEvaluator:
    """Exact shortcut for the MNL / NL: a lever multiplies one data column, and
    that column enters the design array linearly, so only those cells change."""

    def __init__(self, model, pop):
        self.model = model
        self.X0, self.A = build_design(pop, model.terms_)
        self.cells = {}
        for lever, (col, _) in LEVER_COLUMN.items():
            self.cells[lever] = [(C.MODES.index(m), k) for k, (_, alts) in enumerate(model.terms_)
                                 for m, c in alts.items() if c == col]

    def probs(self, levers):
        X = self.X0.copy()
        for lever, val in levers.items():
            if not val:
                continue
            sign = LEVER_COLUMN[lever][1]
            for j, k in self.cells[lever]:
                X[:, j, k] = self.X0[:, j, k] * (1 + sign * val / 100.0)
        if isinstance(self.model, NestedLogit):
            return np.where(self.A > 0, np.exp(self.model._log_probs(self.model.coef_, X, self.A)), 0.0)
        return MultinomialLogit._probs(self.model.coef_, X, self.A)


class Objective:
    def __init__(self, model, pop):
        fast = isinstance(model, (MultinomialLogit, NestedLogit))
        self.evaluator = FastChoiceEvaluator(model, pop) if fast else GenericEvaluator(model, pop)
        self.fast = fast
        self.n_evals = 0

    def levers(self, x):
        return dict(zip(LEVERS, map(float, x)))

    def __call__(self, x):
        self.n_evals += 1
        P = self.evaluator.probs(self.levers(x))
        return float(P[:, SUS_IDX].sum(axis=1).mean())


def grid_search(obj, bounds=C.POLICY_BOUNDS, step=C.GRID_STEP, budget=None):
    axes = [np.arange(lo, hi + 1e-9, step) for lo, hi in (bounds[k] for k in LEVERS)]
    rows = []
    t0 = time.time()
    for combo in itertools.product(*axes):
        e = effort(combo, bounds)
        if budget is not None and e > budget + 1e-9:
            continue
        rows.append((*combo, e, obj(combo)))
    table = pd.DataFrame(rows, columns=LEVERS + ["effort", "objective"])
    best = table.loc[table["objective"].idxmax()]
    return best, table, time.time() - t0


def de_search(obj, bounds=C.POLICY_BOUNDS, budget=None, seed=C.SEED, maxiter=200):
    b = [bounds[k] for k in LEVERS]
    kw = {}
    if budget is not None:
        kw["constraints"] = NonlinearConstraint(lambda x: effort(x, bounds), -np.inf, budget)
    t0 = time.time()
    res = differential_evolution(lambda x: -obj(x), b, seed=seed, maxiter=maxiter, tol=1e-10, popsize=20,
                                 polish=obj.fast, init="sobol", **kw)
    x = np.clip(res.x, [lo for lo, _ in b], [hi for _, hi in b])
    return x, -res.fun, res, time.time() - t0


def check_constraints(levers, bounds=C.POLICY_BOUNDS, budget=None, tol=1e-6):
    problems = []
    for k, v in levers.items():
        lo, hi = bounds[k]
        if v < lo - tol or v > hi + tol:
            problems.append(f"{k}={v:.4f} outside [{lo}, {hi}]")
    if budget is not None and effort([levers[k] for k in LEVERS], bounds) > budget + tol:
        problems.append("effort budget exceeded")
    return problems


def verify_solution(levers, reported_objective, model, pop, bounds=C.POLICY_BOUNDS, budget=None):
    """Recompute the objective the slow, general way (modify the data frame,
    call the frozen model) and check every constraint."""
    policy = levers_to_policy(levers)
    new = apply_policy(pop, **policy)
    P = predict_probabilities(model, new)         # raises if availability or sum-to-one fails
    S = float(P[[f"P_{m}" for m in C.SUSTAINABLE]].sum(axis=1).mean())
    times = new[list(C.TIME_COLS.values())].to_numpy()
    costs = new[list(C.COST_COLS.values())].to_numpy()
    av_unchanged = (new[[C.AV_COLS[m] for m in C.MODES]].to_numpy() == pop[[C.AV_COLS[m] for m in C.MODES]].to_numpy()).all()
    return {
        "recomputed_objective": S,
        "reported_objective": reported_objective,
        "abs_difference": abs(S - reported_objective),
        "matches": abs(S - reported_objective) < 1e-8,
        "constraint_problems": check_constraints(levers, bounds, budget),
        "non_negative_times": bool((times >= 0).all()),
        "non_negative_costs": bool((costs >= 0).all()),
        "availability_unchanged": bool(av_unchanged),
        "probabilities_valid": True,
    }


def compare_with_benchmarks(obj, best_x, bounds=C.POLICY_BOUNDS):
    """Optimum against the baseline, each lever alone at its bound and a few
    hand-made packages. The optimum should never be beaten."""
    zero = np.zeros(len(LEVERS))
    rows = [("Baseline (no intervention)", zero)]
    for i, k in enumerate(LEVERS):
        x = zero.copy(); x[i] = bounds[k][1]
        rows.append((f"Only {C.POLICY_LABELS[k].split(' (')[0].lower()} at bound", x))
    half = np.array([bounds[k][1] / 2 for k in LEVERS])
    rows.append(("All levers at half their bound", half))
    rows.append(("Public transport levers at bound, no car pricing",
                 np.array([bounds[k][1] if k != "car_cost_inc" else 0 for k in LEVERS])))
    rows.append(("Optimised policy", np.asarray(best_x, float)))
    return pd.DataFrame([{"policy": n, **dict(zip(LEVERS, x)), "effort": effort(x, bounds),
                          "sustainable_share_pct": 100 * obj(x)} for n, x in rows])


def budget_frontier(grid_table, budgets):
    """Best grid point under each effort budget (the objective does not depend
    on the budget, so one pass over the grid is enough)."""
    rows = []
    for B in budgets:
        ok = grid_table[grid_table["effort"] <= B + 1e-9]
        b = ok.loc[ok["objective"].idxmax()]
        rows.append({"effort_budget": B, **{k: float(b[k]) for k in LEVERS}, "effort_used": float(b["effort"]),
                     "sustainable_share_pct": 100 * float(b["objective"])})
    return pd.DataFrame(rows)


def walk_weight_sensitivity(frozen, train, y, pop, weights=(1.5, 2.0, 2.5, 3.0), bounds=C.POLICY_BOUNDS):
    """How much do the optimisation results depend on the walk-time weight
    assumed when estimating the choice model? Refits the frozen model's class
    and specification for each weight; the frozen model itself is untouched."""
    from sklearn.base import clone
    rows = []
    zero = np.zeros(len(LEVERS))
    top = np.array([bounds[k][1] for k in LEVERS])
    for w in weights:
        scale = w / C.WALK_TIME_WEIGHT          # same as estimating with weight w
        tr, pp = train.copy(), pop.copy()
        tr[C.TIME_COLS["walk"]] *= scale
        pp[C.TIME_COLS["walk"]] *= scale
        m = clone(frozen).fit(tr, y)
        obj = Objective(m, pp)
        base = obj(zero)
        s = m.summary_["estimate"]
        row = {"walk_weight": w, "log_likelihood": round(m.loglik_, 3), "B_time": s["B_time"], "B_cost": s["B_cost"],
               "VOT_INR_per_hour": 60 * s["B_time"] / s["B_cost"], "baseline_pct": 100 * base,
               "all_levers_at_bound_pct": 100 * obj(top)}
        for i, k in enumerate(LEVERS):
            x = zero.copy(); x[i] = bounds[k][1]
            row[f"gain_{k}_pp"] = 100 * (obj(x) - base)
        rows.append(row)
    return pd.DataFrame(rows)


def shares_for(obj, x):
    P = obj.evaluator.probs(obj.levers(x))
    table, sus = summarise(pd.DataFrame(P, columns=[f"P_{m}" for m in C.MODES]))
    return table, sus
