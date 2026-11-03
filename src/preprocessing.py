"""Cleaning, train/test split and the shared ML preprocessing pipeline.

Cleaning decisions (all logged, none silent):
  * the 11 records whose travel times disagree with their own distance get
    their six times recomputed from distance with the recovered speed rules;
  * DL_TwoW and DL_Car are dropped because they duplicate TwoW_AV and Car_AV;
  * category codes are left exactly as recorded (no remapping);
  * household inconsistencies are flagged but not edited.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .data_validation import LOSRules, household_flags, time_inconsistency_mask

DROP_COLUMNS = ["DL_TwoW", "DL_Car"]


def clean_data(raw):
    """Return (clean dataframe, correction log, fitted LOS rules)."""
    df = raw.copy()
    rules = LOSRules.fit(df)

    bad = time_inconsistency_mask(df, rules.time_rate)
    fixed_times = rules.times(df.loc[bad, "Distance_km"])
    log = []
    for (idx, row), (_, new) in zip(df.loc[bad].iterrows(), fixed_times.iterrows()):
        for col in C.TIME_COLS.values():
            if row[col] != new[col]:
                log.append({"ID": int(row["ID"]), "column": col, "original": row[col], "corrected": new[col]})
        df.loc[idx, list(C.TIME_COLS.values())] = new.to_numpy()
    df["los_time_corrected"] = bad.astype(int)

    df = df.drop(columns=DROP_COLUMNS)

    flags = household_flags(df).drop(columns="ID").astype(int)
    df = pd.concat([df, flags], axis=1)

    int_cols = [c for c in df.columns if c not in list(C.TIME_COLS.values()) + ["Distance_km"]]
    df[int_cols] = df[int_cols].astype(int)
    return df.reset_index(drop=True), pd.DataFrame(log), rules


def validate_clean(df, rules=None):
    """Logical checks on the cleaned data. Returns a list of problems found.

    Household flags are not problems here: they were investigated and kept on
    purpose (see the audit), so they are reported separately.
    """
    problems = []
    if df.isna().any().any():
        problems.append("missing values present")
    if df["ID"].duplicated().any():
        problems.append("duplicate IDs")
    for col in DROP_COLUMNS:
        if col in df.columns:
            problems.append(f"duplicate column {col} still present")
    if (df["Distance_km"] <= 0).any():
        problems.append("non-positive distance")
    for col in list(C.TIME_COLS.values()) + list(C.COST_COLS.values()):
        if (df[col] < 0).any():
            problems.append(f"negative values in {col}")
    if rules is not None and time_inconsistency_mask(df, rules.time_rate).any():
        problems.append("travel times still inconsistent with distance")
    av = df[[C.AV_COLS[m] for m in C.MODES]].to_numpy()
    if not np.isin(av, [0, 1]).all():
        problems.append("availability not binary")
    chosen = av[np.arange(len(df)), df["Choice"].to_numpy() - 1]
    if (chosen == 0).any():
        problems.append("chosen alternative unavailable")
    if not df["Choice"].isin(range(1, 7)).all():
        problems.append("choice code outside 1-6")
    if ((df["Car_AV"] == 1) & (df["Car_Ownership"] == 0)).any():
        problems.append("car available without a household car")
    if ((df["TwoW_AV"] == 1) & (df["TwoW_Ownership"] == 0)).any():
        problems.append("two-wheeler available without a household two-wheeler")
    return problems


def split_train_test(df, test_size=C.TEST_SIZE, seed=C.SEED):
    train, test = train_test_split(df, test_size=test_size, stratify=df["Choice"], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def target(df):
    """Class index 0..5 (walk..train). sklearn and xgboost both want 0-based labels."""
    return df["Choice"].to_numpy() - 1


def availability(df):
    return df[[C.AV_COLS[m] for m in C.MODES]].to_numpy(dtype=float)


def make_ml_preprocessor(categorical, numeric):
    """One-hot encode the categorical codes and standardise the numeric columns.

    The same transformer (refitted inside each CV fold) is used for every
    machine-learning model, so differences between models are not caused by
    different preprocessing.
    """
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
         ("num", StandardScaler(), numeric)],
        remainder="drop", verbose_feature_names_out=False,
    )
