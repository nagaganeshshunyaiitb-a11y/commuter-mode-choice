"""Test 1: data validation."""
import numpy as np

from src import config as C
from src import data_validation as V
from src import preprocessing as PP


def test_raw_workbook_shape(raw):
    assert raw.shape == (531, 38)
    assert list(raw.columns) == C.EXPECTED_COLUMNS


def test_audit_finds_the_known_problems(raw):
    a = V.run_audit(raw)
    assert a["missing_values"] == 0
    assert a["duplicate_rows"] == 0
    assert ("DL_TwoW", "TwoW_AV") in a["duplicate_columns"]
    assert ("DL_Car", "Car_AV") in a["duplicate_columns"]
    assert a["constant_columns"] == ["LocalTrain_AV"]
    assert len(a["los_inconsistent"]) == 11
    bad_codes = a["code_check"].set_index("variable")
    assert bad_codes.loc["Occup", "undocumented_codes_in_data"] == [5]
    assert bad_codes.loc["Dwelling", "undocumented_codes_in_data"] == [4]
    assert bad_codes.loc["Trip_Purpose", "documented_but_unused"] == [0]
    assert a["availability"]["chosen_alternative_unavailable"] == 0


def test_cleaned_data_is_logically_consistent(cleaned):
    clean, corrections, rules = cleaned
    assert PP.validate_clean(clean, rules) == []
    assert clean["los_time_corrected"].sum() == 11
    assert set(corrections["ID"]) == {179, 273, 275, 278, 283, 326, 403, 482, 525, 533, 560}
    assert "DL_TwoW" not in clean and "DL_Car" not in clean


def test_category_codes_are_not_remapped(raw, cleaned):
    clean = cleaned[0]
    for col in ["Trip_Purpose", "Occup", "Dwelling", "Gender", "Income", "Age", "Education", "Choice"]:
        assert set(clean[col].unique()) == set(raw[col].unique()), col


def test_recovered_rules_reproduce_costs(raw):
    rules = V.LOSRules.fit(raw)
    match = V.cost_rule_match(raw, rules).set_index("cost_column")
    assert (match["within_1_inr_share"] >= 0.99).all()
    # corrected times agree with distance for every record
    clean, _, _ = PP.clean_data(raw)
    assert not V.time_inconsistency_mask(clean, rules.time_rate).any()


def test_only_time_columns_change_during_cleaning(raw, cleaned):
    clean = cleaned[0]
    keep = [c for c in raw.columns if c not in PP.DROP_COLUMNS and c not in C.TIME_COLS.values()]
    assert np.array_equal(raw[keep].to_numpy(float), clean[keep].to_numpy(float))
