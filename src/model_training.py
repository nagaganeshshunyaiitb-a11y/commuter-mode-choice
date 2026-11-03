"""Candidate models, their search spaces and hyperparameter tuning.

Tuning only ever sees the training partition: every search below uses
repeated stratified K-fold on the training rows and macro-F1 as the score.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config as C
from .choice_models import MultinomialLogit, NestedLogit, SPEC_NAMES
from .feature_engineering import CATEGORICAL, NUMERIC, FeatureBuilder
from .preprocessing import make_ml_preprocessor


class AvailabilityMasked(BaseEstimator, ClassifierMixin):
    """Wraps a scikit-learn pipeline so that its probabilities respect the
    choice set of each person.

        p'_j = a_j * max(p_j, eps) / sum_k a_k * max(p_k, eps)

    The small floor eps keeps a tree that gives an available mode exactly
    zero probability from producing an infinite log loss. Conditioning on the
    available set like this is exactly what the MNL does internally.

    balanced=True reweights training rows inversely to class frequency.
    """

    def __init__(self, estimator=None, balanced=False, eps=C.PROB_FLOOR):
        self.estimator = estimator
        self.balanced = balanced
        self.eps = eps

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        y = np.asarray(y)
        kw = {}
        if self.balanced:
            last = self.estimator_.steps[-1][0]
            kw[f"{last}__sample_weight"] = compute_sample_weight("balanced", y)
        self.estimator_.fit(X, y, **kw)
        self.classes_ = np.arange(len(C.MODES))
        return self

    def predict_proba(self, X):
        raw = self.estimator_.predict_proba(X)
        P = np.zeros((len(X), len(C.MODES)))
        P[:, self.estimator_.classes_] = raw              # in case a class was absent in training
        A = X[[C.AV_COLS[m] for m in C.MODES]].to_numpy(float)
        P = np.maximum(P, self.eps) * A
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def ml_pipeline(clf):
    return Pipeline([
        ("features", FeatureBuilder()),
        ("prep", make_ml_preprocessor(CATEGORICAL, NUMERIC)),
        ("clf", clf),
    ])


def candidates(seed=C.SEED):
    """name -> (estimator, search space, search type, n_iter)."""
    p = "estimator__clf__"
    return {
        "MNL": (MultinomialLogit(), {"spec": SPEC_NAMES}, "grid", None),
        "Logistic regression": (
            AvailabilityMasked(ml_pipeline(LogisticRegression(max_iter=5000))),
            {p + "C": list(np.logspace(-3, 2, 11)), "balanced": [False, True]}, "grid", None),
        "Decision tree": (
            AvailabilityMasked(ml_pipeline(DecisionTreeClassifier(random_state=seed))),
            {p + "max_depth": [2, 3, 4, 5, 6, 8, 10, None], p + "min_samples_split": [2, 5, 10, 20],
             p + "min_samples_leaf": [1, 2, 5, 10, 20], p + "criterion": ["gini", "entropy"],
             "balanced": [False, True]}, "grid", None),
        "Random forest": (
            AvailabilityMasked(ml_pipeline(RandomForestClassifier(random_state=seed, n_jobs=1))),
            {p + "n_estimators": [200, 300, 500], p + "max_depth": [3, 5, 8, 12, None],
             p + "min_samples_split": [2, 5, 10], p + "min_samples_leaf": [1, 2, 4, 8],
             p + "max_features": ["sqrt", "log2", 0.3, 0.5], "balanced": [False, True]}, "random", 60),
        "XGBoost": (
            AvailabilityMasked(ml_pipeline(XGBClassifier(objective="multi:softprob", tree_method="hist",
                                                         random_state=seed, n_jobs=1, verbosity=0))),
            {p + "n_estimators": [100, 200, 400, 600], p + "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
             p + "max_depth": [2, 3, 4, 6], p + "subsample": [0.6, 0.8, 1.0],
             p + "colsample_bytree": [0.5, 0.7, 1.0], p + "min_child_weight": [1, 3, 5],
             p + "reg_lambda": [0.5, 1.0, 5.0, 10.0], p + "reg_alpha": [0.0, 0.1, 1.0],
             "balanced": [False, True]}, "random", 60),
    }


def tuning_cv(seed=C.SEED, splits=C.TUNE_SPLITS, repeats=C.TUNE_REPEATS):
    return RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)


def tune(name, estimator, space, kind, n_iter, X, y, cv, n_jobs=-1, seed=C.SEED):
    if kind == "grid":
        search = GridSearchCV(estimator, space, scoring="f1_macro", cv=cv, n_jobs=n_jobs, refit=True)
    else:
        search = RandomizedSearchCV(estimator, space, n_iter=n_iter, scoring="f1_macro", cv=cv,
                                    n_jobs=n_jobs, refit=True, random_state=seed)
    search.fit(X, y)
    return search


def nested_logit_from(mnl_spec, structure, lambda_bounds=(0.05, 1.0)):
    return NestedLogit(spec=mnl_spec, nests=structure, lambda_bounds=lambda_bounds)


def select_nest_structure(train, y, spec, structures, alpha=0.05):
    """Rule written before looking at any test data: a structure is usable if
    every lambda is strictly inside its bounds and significantly below one
    (one-sided Wald test); among usable structures take the highest
    log-likelihood. Returns (chosen name or None, table of all structures)."""
    from scipy import stats
    from .choice_models import likelihood_ratio_test
    mnl = MultinomialLogit(spec).fit(train, y)
    rows, fitted = [], {}
    for s in structures:
        m = NestedLogit(spec, s).fit(train, y)
        fitted[s] = m
        lam = m.lambdas_
        se = m.summary_.loc[list(lam), "std_error"]
        z = (1 - np.array(list(lam.values()))) / se.to_numpy()
        p_one_sided = stats.norm.sf(z)
        usable = (not any(m.at_bound_)) and bool(np.all(p_one_sided < alpha))
        lr = likelihood_ratio_test(mnl.loglik_, m.loglik_, len(lam))
        rows.append({"structure": s, "log_likelihood": round(m.loglik_, 3),
                     "lambdas": {k.replace("lambda_", ""): round(v, 3) for k, v in lam.items()},
                     "lambda_se": [round(v, 3) for v in se], "lambda_at_bound": m.at_bound_,
                     "p_lambda_below_1": [round(float(v), 4) for v in p_one_sided],
                     "LR_vs_MNL": lr["LR_statistic"], "LR_p_value": round(lr["p_value"], 4), "usable": usable})
    import pandas as pd
    table = pd.DataFrame(rows)
    ok = table[table["usable"]]
    chosen = None if ok.empty else ok.sort_values("log_likelihood", ascending=False)["structure"].iloc[0]
    return chosen, table, fitted, mnl
