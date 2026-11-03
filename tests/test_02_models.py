"""Test 2 (all candidates train) and checks of the choice-model estimators."""
import numpy as np
import pytest
from scipy import optimize
from sklearn.model_selection import StratifiedKFold

from src import choice_models as CM
from src import model_training as MT
from src import preprocessing as PP


def test_every_candidate_trains_and_predicts(split):
    train, test = split
    y = PP.target(train)
    for name, (est, space, kind, _) in MT.candidates().items():
        # first value of every hyperparameter is enough to show the model trains
        params = {k: v[0] for k, v in space.items()}
        m = est.set_params(**params).fit(train, y)
        P = m.predict_proba(test)
        assert P.shape == (len(test), 6), name
        assert np.allclose(P.sum(axis=1), 1), name
    nl = CM.NestedLogit("GC+CORE", "SHARED{auto,bus,train}").fit(train, y)
    assert nl.converged_


def test_mnl_matches_statsmodels_conditional_logit(split):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    train, _ = split
    y = PP.target(train)
    m = CM.MultinomialLogit("GC+CORE").fit(train, y)
    X, A = CM.build_design(train, m.terms_)
    rows = np.argwhere(A > 0)
    sm = ConditionalLogit((rows[:, 1] == y[rows[:, 0]]).astype(float), X[rows[:, 0], rows[:, 1], :],
                          groups=rows[:, 0]).fit(method="bfgs", maxiter=5000, gtol=1e-8, disp=0)
    assert abs(sm.llf - m.loglik_) < 1e-3
    assert np.max(np.abs(sm.params - m.coef_)) < 0.02


def test_mnl_reproduces_sample_shares(split):
    """With a full set of constants the MNL predicts the training shares exactly."""
    train, _ = split
    y = PP.target(train)
    m = CM.MultinomialLogit("GC+CORE").fit(train, y)
    assert np.allclose(m.predict_proba(train).mean(axis=0), np.bincount(y) / len(y), atol=1e-6)


def test_nested_logit_gradient_is_correct(split):
    train, _ = split
    y = PP.target(train)
    nl = CM.NestedLogit("GC+CORE", "SHARED{auto,bus,train} + PRIVATE{car,twow}")
    X, A = nl._prepare(train)
    nl.nest_matrix_, nl.groups_, nl.free_ = nl._nest_matrix()
    theta = np.r_[CM.MultinomialLogit("GC+CORE").fit(train, y).coef_, 0.6, 0.8]
    _, g = nl._negll_grad(theta, X, A, y)
    num = optimize.approx_fprime(theta, lambda t: nl._negll_grad(t, X, A, y)[0], 1e-6)
    assert np.max(np.abs(g - num) / np.maximum(1, np.abs(num))) < 1e-3


def test_nested_logit_with_lambda_one_is_the_mnl(split):
    train, _ = split
    y = PP.target(train)
    mnl = CM.MultinomialLogit("GC+CORE").fit(train, y)
    nl = CM.NestedLogit("GC+CORE", "SHARED{auto,bus,train}", lambda_bounds=(1.0, 1.0)).fit(train, y)
    assert abs(nl.loglik_ - mnl.loglik_) < 1e-4


def test_behavioural_specification_has_negative_time_and_cost(split):
    train, _ = split
    m = CM.MultinomialLogit("GC+CORE").fit(train, PP.target(train))
    assert CM.signs_ok(m)
