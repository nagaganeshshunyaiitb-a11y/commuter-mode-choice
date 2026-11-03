"""Test 6 (synthetic data) and Test 7 (synthetic predictions)."""
import numpy as np

from src import config as C
from src import data_validation as V
from src import synthetic_generation as SG
from src.prediction import predict_probabilities


def test_generate_5000_valid_records(split, cleaned):
    train, test = split
    rules = cleaned[2]
    pop, _ = SG.generate("cart", train, rules, n=5000)
    assert len(pop) == 5000
    assert "Choice" not in pop.columns
    d0, d1 = train["Distance_km"].min(), train["Distance_km"].max()
    assert all(v == 0 for v in SG.constraint_violations(pop, d0, d1).values())
    assert (pop["Adults"] >= 1).all() and (pop["Adults"] + pop["Children"] == pop["HH_Size"]).all()
    for col in SG.DISCRETE:
        assert set(pop[col].unique()) <= set(train[col].unique()), col
    assert not V.time_inconsistency_mask(pop, rules.time_rate).any()
    assert (pop["LocalTrain_AV"] == 1).all()


def test_synthetic_distributions_are_close(split, cleaned):
    train, test = split
    pop, _ = SG.generate("cart", train, cleaned[2], n=5000)
    marg = SG.marginal_fidelity(train, pop)
    assert marg["distance"].mean() < 0.05
    assert marg["distance"].max() < 0.10
    mem = SG.memorisation(train, test, pop)
    assert mem["exact_copy_share_synthetic"] < 0.02


def test_saved_population_has_configured_size(population):
    assert len(population) == C.SYNTH_N


def test_synthetic_predictions_are_valid(frozen, population):
    model, _ = frozen
    P = predict_probabilities(model, population).to_numpy()
    A = population[[C.AV_COLS[m] for m in C.MODES]].to_numpy()
    assert np.isfinite(P).all()
    assert ((P >= 0) & (P <= 1)).all()
    assert np.allclose(P.sum(axis=1), 1)
    assert (P[A == 0] == 0).all()
