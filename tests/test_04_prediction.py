"""Test 4 (individual prediction) and Test 5 (availability)."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src import config as C
from src import model_training as MT
from src import preprocessing as PP
from src.data_validation import LOSRules
from src.prediction import predict_probabilities


def make_profile(rules, distance=6.0, **overrides):
    p = {"ID": 1, "Distance_km": distance, "Trip_Purpose": 2, "Income": 2, "Gender": 1, "Age": 2,
         "Education": 1, "Occup": 3, "ParkingSpace": 1, "Area_Type": 0, "Dwelling": 1, "HH_Size": 4,
         "Adults": 2, "Children": 2, "Working_Adults": 2, "SAPT": 1, "Car_Ownership": 1, "TwoW_Ownership": 1,
         "Walk_AV": 1, "TwoW_AV": 1, "Auto_AV": 1, "Car_AV": 1, "Bus_AV": 1, "LocalTrain_AV": 1}
    p.update(overrides)
    df = pd.DataFrame([p])
    return pd.concat([df, rules.times(df["Distance_km"]), rules.costs(df["Distance_km"])], axis=1)


def test_individual_profile_gives_six_probabilities(frozen, raw):
    model, _ = frozen
    person = make_profile(LOSRules.fit(raw))
    P = predict_probabilities(model, person)
    assert P.shape == (1, 6)
    assert np.isclose(P.to_numpy().sum(), 1.0)
    assert (P.to_numpy() > 0).all()


def test_unavailable_modes_get_exactly_zero(frozen, raw):
    model, _ = frozen
    person = make_profile(LOSRules.fit(raw), Bus_AV=0, Car_AV=0, Walk_AV=0)
    P = predict_probabilities(model, person).iloc[0]
    assert P["P_bus"] == 0 and P["P_car"] == 0 and P["P_walk"] == 0
    assert np.isclose(P.sum(), 1.0)


def test_masking_wrapper_renormalises_over_available_modes(split):
    """For an ML model the masked probability must equal p_j a_j / sum_k p_k a_k."""
    train, test = split
    y = PP.target(train)
    est = MT.AvailabilityMasked(MT.ml_pipeline(DecisionTreeClassifier(max_depth=4, random_state=0))).fit(train, y)
    raw_p = est.estimator_.predict_proba(test)
    A = PP.availability(test)
    expected = np.maximum(raw_p, C.PROB_FLOOR) * A
    expected /= expected.sum(axis=1, keepdims=True)
    got = est.predict_proba(test)
    assert np.allclose(got, expected)
    assert (got[A == 0] == 0).all()


def test_single_available_mode_gets_probability_one(frozen, raw):
    model, _ = frozen
    person = make_profile(LOSRules.fit(raw), Walk_AV=0, TwoW_AV=0, Auto_AV=0, Car_AV=0, Bus_AV=0)
    P = predict_probabilities(model, person).iloc[0]
    assert P["P_train"] == 1.0
