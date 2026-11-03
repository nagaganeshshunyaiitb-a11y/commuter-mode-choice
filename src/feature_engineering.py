"""Derived variables and the feature lists used by the models.

All derived variables are row-wise functions of columns that are known
before the trip is made, so there is nothing to fit and nothing that can leak
information from the target. Because they are recomputed inside the model
pipeline, a policy scenario that changes a raw travel time or cost
automatically flows through to every derived variable.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C

RARE_OCCUPATION_CODES = (1, 4)    # 3 and 4 records respectively


def add_features(df):
    out = df.copy()
    out["log_distance"] = np.log(out["Distance_km"].astype(float))
    car = out["Car_Ownership"] > 0
    twow = out["TwoW_Ownership"] > 0
    # 0 = no vehicle, 1 = two-wheeler only, 2 = car only, 3 = both
    out["veh_own_cat"] = (2 * car + twow).astype(int)
    out["private_av"] = ((out["Car_AV"] == 1) | (out["TwoW_AV"] == 1)).astype(int)
    hh = out["HH_Size"].clip(lower=1)
    out["workers_share"] = np.minimum(out["Working_Adults"], hh) / hh
    out["has_children"] = (out["Children"] > 0).astype(int)
    out["occ_group"] = out["Occup"].where(~out["Occup"].isin(RARE_OCCUPATION_CODES), 0).astype(int)
    return out


CATEGORICAL = ["Trip_Purpose", "Income", "Age", "Education", "occ_group", "Dwelling",
               "Gender", "ParkingSpace", "Area_Type", "SAPT", "veh_own_cat"]
NUMERIC = (["Distance_km", "log_distance"]
           + list(C.TIME_COLS.values()) + list(C.COST_COLS.values())
           + ["HH_Size", "Children", "Working_Adults", "workers_share", "has_children",
              "Car_Ownership", "TwoW_Ownership", "private_av",
              "Walk_AV", "TwoW_AV", "Auto_AV", "Car_AV", "Bus_AV"])
# LocalTrain_AV is constant and Adults = HH_Size - Children (528/531 rows), so
# neither is used as a predictor.

# Raw columns a record must have for prediction (used by the synthetic
# generator and the dashboard to check inputs).
RAW_PREDICTORS = ["Distance_km", "Trip_Purpose", "Income", "Gender", "Age", "Education", "Occup",
                  "ParkingSpace", "Area_Type", "Dwelling", "HH_Size", "Adults", "Children",
                  "Working_Adults", "SAPT", "Car_Ownership", "TwoW_Ownership"] \
    + list(C.TIME_COLS.values()) + list(C.COST_COLS.values()) + [C.AV_COLS[m] for m in C.MODES]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds the derived variables and returns the model columns as a DataFrame."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = self.columns or (CATEGORICAL + NUMERIC)
        return add_features(X)[cols]


def candidate_feature_review(df):
    """Assess every candidate feature from the brief before deciding to use it."""
    f = add_features(df)
    d = f["Distance_km"]
    cand = {
        "total_household_vehicles": f["Car_Ownership"] + f["TwoW_Ownership"],
        "vehicle_ownership_category": f["veh_own_cat"],
        "workers_share": f["workers_share"],
        "dependency_ratio": (f["HH_Size"] - np.minimum(f["Working_Adults"], f["HH_Size"])) / f["Working_Adults"],
        "available_mode_count": f[[C.AV_COLS[m] for m in C.MODES]].sum(axis=1),
        "public_transport_available": ((f["Bus_AV"] == 1) | (f["LocalTrain_AV"] == 1)).astype(int),
        "private_vehicle_available": f["private_av"],
        "time_advantage_bus_vs_car": f["car_time_min"] - f["bus_time_min"],
        "time_advantage_train_vs_car": f["car_time_min"] - f["local_train_time_min"],
        "cost_advantage_bus_vs_car": f["TC_Car"] - f["TC_Bus"],
        "car_cost_per_km": f["TC_Car"] / d,
        "twow_cost_per_km": f["TC_2W"] / d,
        "has_children": f["has_children"],
        "log_distance": f["log_distance"],
    }
    decisions = {
        "total_household_vehicles": ("rejected", "exact sum of Car_Ownership and TwoW_Ownership"),
        "vehicle_ownership_category": ("used", "captures the none / 2W only / car only / both pattern"),
        "workers_share": ("used", "bounded share of working members (capped at HH_Size for 3 records)"),
        "dependency_ratio": ("rejected", "monotone transform of workers_share, so it duplicates it"),
        "available_mode_count": ("rejected", "linear sum of the availability indicators already used"),
        "public_transport_available": ("rejected", "constant: suburban rail is available to everyone"),
        "private_vehicle_available": ("used", "not a linear combination (logical OR of two indicators)"),
        "time_advantage_bus_vs_car": ("rejected", "fixed multiple of distance because times are computed from it"),
        "time_advantage_train_vs_car": ("rejected", "fixed multiple of distance because times are computed from it"),
        "cost_advantage_bus_vs_car": ("rejected", "near-deterministic function of distance"),
        "car_cost_per_km": ("rejected", "near-constant by construction; variation is only rounding"),
        "twow_cost_per_km": ("rejected", "near-constant by construction; variation is only rounding"),
        "has_children": ("used", "household composition indicator"),
        "log_distance": ("used", "concave distance effect for the linear models"),
    }
    rows = []
    for name, x in cand.items():
        x = pd.Series(x, dtype=float)
        sd, mu = x.std(), x.mean()
        rho = x.corr(d, method="spearman") if sd > 0 else np.nan
        rows.append({"feature": name, "mean": round(mu, 3), "std": round(sd, 3),
                     "coef_of_variation": round(sd / mu, 3) if mu else np.nan,
                     "n_unique": int(x.nunique()),
                     "spearman_with_distance": round(rho, 3) if pd.notna(rho) else np.nan,
                     "decision": decisions[name][0], "reason": decisions[name][1]})
    return pd.DataFrame(rows)
