"""Prediction with the frozen model: individual probabilities and population shares."""
import json

import joblib
import numpy as np
import pandas as pd

from . import config as C


def load_frozen(paths=None):
    paths = paths or C.Paths()
    model = joblib.load(paths.best_model / "model.joblib")
    card = json.loads((paths.best_model / "model_card.json").read_text())
    return model, card


def predict_probabilities(model, df):
    """Six probabilities per person. Unavailable modes are exactly zero and
    each row sums to one over the available modes."""
    P = np.asarray(model.predict_proba(df), float)
    A = df[[C.AV_COLS[m] for m in C.MODES]].to_numpy(float)
    if np.any(P[A == 0] > 1e-12):
        raise ValueError("model gave positive probability to an unavailable mode")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("probabilities do not sum to one")
    return pd.DataFrame(P, columns=[f"P_{m}" for m in C.MODES], index=df.index)


def summarise(P):
    """Population-level summary of a probability table.

    expected share  = mean probability (the quantity optimised)
    hard share      = share of people whose most likely mode it is
    """
    probs = P.to_numpy()
    n = len(probs)
    hard = np.bincount(probs.argmax(axis=1), minlength=len(C.MODES)) / n
    expected = probs.mean(axis=0)
    sus = [C.MODES.index(m) for m in C.SUSTAINABLE]
    out = pd.DataFrame({
        "mode": [C.MODE_LABELS[m] for m in C.MODES],
        "expected_share_pct": 100 * expected,
        "expected_count": expected * n,
        "hard_share_pct": 100 * hard,
        "hard_count": hard * n,
    })
    sustainable = {
        "expected_sustainable_share_pct": float(100 * probs[:, sus].sum(axis=1).mean()),
        "hard_sustainable_share_pct": float(100 * np.isin(probs.argmax(axis=1), sus).mean()),
    }
    return out, sustainable


def sustainable_probability(P):
    return P[[f"P_{m}" for m in C.SUSTAINABLE]].sum(axis=1)
