"""Test 3: metrics are computed on the right partitions, with no leakage."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import model_evaluation as ME
from src import preprocessing as PP


def test_train_and_test_are_disjoint_and_stratified(cleaned, split):
    clean = cleaned[0]
    train, test = split
    assert set(train["ID"]).isdisjoint(test["ID"])
    assert len(train) + len(test) == len(clean)
    full = np.bincount(PP.target(clean)) / len(clean)
    te = np.bincount(PP.target(test), minlength=6) / len(test)
    assert np.max(np.abs(full - te)) < 0.02


def test_saved_partitions_match_the_split(split):
    p = C.Paths().processed
    if not (p / "train.csv").exists():
        pytest.skip("run the pipeline first")
    train, test = split
    assert set(pd.read_csv(p / "train.csv")["ID"]) == set(train["ID"])
    assert set(pd.read_csv(p / "test.csv")["ID"]) == set(test["ID"])


def test_frozen_model_was_fitted_on_training_rows_only(frozen, split):
    """Refitting the frozen model's configuration on the training rows must give
    back exactly the frozen model; a model that had seen test rows would not."""
    from sklearn.base import clone
    model, card = frozen
    train, test = split
    assert card["n_train"] == len(train)
    refit = clone(model).fit(train, PP.target(train))
    assert np.allclose(refit.predict_proba(test), model.predict_proba(test), atol=1e-6)


def test_cv_folds_never_touch_test_rows():
    p = C.Paths().results / "cv_fold_metrics.csv"
    if not p.exists():
        pytest.skip("run the pipeline first")
    folds = pd.read_csv(p)
    assert folds.groupby("model").size().nunique() == 1          # same folds for every model
    assert folds["repeat"].max() + 1 == C.CV_REPEATS


def test_metric_function_on_a_toy_case():
    y = np.array([0, 1, 2, 3, 4, 5])
    P = np.eye(6) * 0.9 + 0.02
    P = P / P.sum(axis=1, keepdims=True)
    m = ME.metrics(y, P)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["balanced_accuracy"] == 1
    assert m["log_loss"] == pytest.approx(-np.log(P[0, 0]))


def test_selection_rule_skips_models_that_fail_readiness():
    s = pd.DataFrame({
        "macro_f1_mean": [0.60, 0.50], "macro_f1_se": [0.01, 0.01],
        "balanced_accuracy_mean": [0.6, 0.5], "balanced_accuracy_se": [0.01, 0.01],
        "log_loss_mean": [0.8, 0.9], "log_loss_se": [0.01, 0.01]}, index=["XGBoost", "MNL"])
    winner, log = ME.select_model(s, {"XGBoost": False, "MNL": True})
    assert winner == "MNL" and log["skipped"] == ["XGBoost"]
