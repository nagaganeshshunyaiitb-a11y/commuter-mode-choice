"""Metrics, the cross-validated comparison, test-set evaluation, calibration,
the simulation-readiness check and the pre-declared selection rule."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support)
from sklearn.model_selection import RepeatedStratifiedKFold

from . import config as C
from .scenario_simulation import apply_policy

LABELS = list(range(len(C.MODES)))


def expected_calibration_error(y, P, bins=10):
    """Top-label ECE: |accuracy - confidence| averaged over confidence bins."""
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def metrics(y, P):
    y = np.asarray(y)
    pred = P.argmax(axis=1)
    onehot = np.eye(len(LABELS))[y]
    observed_share = onehot.mean(axis=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_precision": precision_recall_fscore_support(y, pred, labels=LABELS, average="macro",
                                                           zero_division=0)[0],
        "macro_recall": precision_recall_fscore_support(y, pred, labels=LABELS, average="macro",
                                                        zero_division=0)[1],
        "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, labels=LABELS, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "log_loss": log_loss(y, np.clip(P, 1e-15, 1), labels=LABELS),
        "brier": float(((P - onehot) ** 2).sum(axis=1).mean()),
        "ece": expected_calibration_error(y, P),
        "share_mae_pp": float(100 * np.abs(P.mean(axis=0) - observed_share).mean()),
    }


def _fit_fold(name, est, X, y, tr, va, rep, fold):
    m = clone(est).fit(X.iloc[tr], y[tr])
    P = m.predict_proba(X.iloc[va])
    res = {"model": name, "repeat": rep, "fold": fold, **metrics(y[va], P)}
    return res, (name, rep, va, P)


def cross_validate_models(models, X, y, seed=C.CV_SEED, splits=C.CV_SPLITS, repeats=C.CV_REPEATS, n_jobs=-1):
    """Fit every (tuned) model on the same repeated stratified folds.

    Returns per-fold metrics and out-of-fold probabilities (one array per
    repeat and model, used for the calibration plots)."""
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)
    tasks = []
    for k, (tr, va) in enumerate(cv.split(X, y)):
        rep, fold = divmod(k, splits)
        for name, est in models.items():
            tasks.append(delayed(_fit_fold)(name, est, X, y, tr, va, rep, fold))
    out = Parallel(n_jobs=n_jobs)(tasks)
    folds = pd.DataFrame([o[0] for o in out])
    oof = {}
    for name, rep, va, P in (o[1] for o in out):
        arr = oof.setdefault(name, np.zeros((repeats, len(y), len(LABELS))))
        arr[rep, va] = P
    return folds, oof


def summarise_cv(folds):
    cols = [c for c in folds.columns if c not in ("model", "repeat", "fold")]
    g = folds.groupby("model")[cols]
    mean, sd, n = g.mean(), g.std(ddof=1), g.size()
    se = sd.div(np.sqrt(n), axis=0)
    out = pd.concat({"mean": mean, "sd": sd, "se": se}, axis=1)
    out.columns = [f"{m}_{s}" for s, m in out.columns]
    return out


def evaluate_on_test(model, X_test, y_test, n_boot=1000, seed=C.SEED):
    P = model.predict_proba(X_test)
    res = metrics(y_test, P)
    pred = P.argmax(axis=1)
    rng = np.random.default_rng(seed)
    boot = {"accuracy": [], "macro_f1": [], "balanced_accuracy": [], "log_loss": []}
    n = len(y_test)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y_test[i])) < len(LABELS):
            continue
        b = metrics(y_test[i], P[i])
        for k in boot:
            boot[k].append(b[k])
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot.items()}
    p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=LABELS, zero_division=0)
    per_class = pd.DataFrame({"mode": [C.MODE_LABELS[m] for m in C.MODES], "precision": p, "recall": r,
                              "f1": f, "support": s})
    cm = confusion_matrix(y_test, pred, labels=LABELS)
    return res, ci, per_class, cm, P


def reliability_table(y, P, bins=10):
    """One-vs-rest reliability data for every mode (pooled predictions)."""
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for j, m in enumerate(C.MODES):
        pj = P[:, j]
        hit = (y == j).astype(float)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (pj > lo) & (pj <= hi) if lo > 0 else (pj >= lo) & (pj <= hi)
            if sel.sum() >= 5:
                rows.append({"mode": m, "bin_mid": (lo + hi) / 2, "mean_predicted": pj[sel].mean(),
                             "observed_frequency": hit[sel].mean(), "n": int(sel.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulation readiness
# ---------------------------------------------------------------------------
READINESS_PROBES = {
    # lever: (policy kwargs, affected mode, expected direction of its probability)
    "bus time": ({"time_pct": {"bus": -C.READINESS_SHOCK_PCT}}, "bus", +1),
    "bus fare": ({"cost_pct": {"bus": -C.READINESS_SHOCK_PCT}}, "bus", +1),
    "rail time": ({"time_pct": {"train": -C.READINESS_SHOCK_PCT}}, "train", +1),
    "rail fare": ({"cost_pct": {"train": -C.READINESS_SHOCK_PCT}}, "train", +1),
    "car cost": ({"cost_pct": {"car": C.READINESS_SHOCK_PCT}}, "car", -1),
}


def simulation_readiness(model, X):
    """Does the model respond to each policy lever, and in the direction that
    consumer theory requires? (Cheaper or faster bus must not make the bus
    less likely, a dearer car must not make the car more likely.)

    The check uses the training rows only and is part of the pre-declared
    selection rule because the chosen model's whole purpose is simulation."""
    base = model.predict_proba(X)
    rows = []
    for lever, (policy, mode, sign) in READINESS_PROBES.items():
        j = C.MODES.index(mode)
        avail = X[C.AV_COLS[mode]].to_numpy() == 1
        d = (model.predict_proba(apply_policy(X, **policy))[:, j] - base[:, j])[avail]
        wrong = np.mean(sign * d < -1e-9)
        moved = np.mean(np.abs(d) > 1e-6)
        mean_change = float(d.mean())
        ok = (sign * mean_change > 1e-4) and (wrong <= C.READINESS_MAX_WRONG_SHARE)
        rows.append({"lever": lever, "mode": mode, "mean_change_pp": 100 * mean_change,
                     "share_responding": moved, "share_wrong_direction": wrong, "passes": ok})
    table = pd.DataFrame(rows)
    return bool(table["passes"].all()), table


# ---------------------------------------------------------------------------
# Selection rule (see config for the written-down version)
# ---------------------------------------------------------------------------

def rank_models(cv_summary):
    s = cv_summary.copy()
    best = s["macro_f1_mean"].idxmax()
    short = s[s["macro_f1_mean"] >= s.loc[best, "macro_f1_mean"] - s.loc[best, "macro_f1_se"]]
    best_ba = short["balanced_accuracy_mean"].idxmax()
    short = short[short["balanced_accuracy_mean"] >=
                  short.loc[best_ba, "balanced_accuracy_mean"] - short.loc[best_ba, "balanced_accuracy_se"]]
    # inside the shortlist: lowest log loss, with near-ties (within 1 SE) broken by simplicity
    order = list(short.sort_values("log_loss_mean").index)
    lead = order[0]
    tied = [m for m in order if s.loc[m, "log_loss_mean"] <= s.loc[lead, "log_loss_mean"] + s.loc[lead, "log_loss_se"]]
    tied.sort(key=C.COMPLEXITY_ORDER.index)
    head = tied + [m for m in order if m not in tied]
    rest = [m for m in s.sort_values("macro_f1_mean", ascending=False).index if m not in head]
    return head + rest, list(short.index)


def select_model(cv_summary, readiness):
    """readiness: {model name: passes (bool)}. Returns (winner, log)."""
    ranking, shortlist = rank_models(cv_summary)
    log = {"ranking": ranking, "shortlist_after_f1_and_ba": shortlist, "skipped": []}
    for name in ranking:
        if readiness.get(name, False):
            log["winner"] = name
            return name, log
        log["skipped"].append(name)
    raise RuntimeError("no candidate passed the simulation-readiness check")
