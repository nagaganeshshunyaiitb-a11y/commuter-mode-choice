"""Loading and auditing the raw survey workbook.

Nothing in this module changes the data. It only reads the workbook, runs the
checks listed in the brief and returns tables that describe what was found.
The corrections themselves are made (and logged) in preprocessing.py.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


def load_raw(path=C.RAW_XLSX, sheet=C.RAW_SHEET):
    df = pd.read_excel(path, sheet_name=sheet)
    missing = [c for c in C.EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in C.EXPECTED_COLUMNS]
    if missing or extra:
        raise ValueError(f"Unexpected schema. Missing: {missing}  Extra: {extra}")
    return df[C.EXPECTED_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Level-of-service (LOS) rules
# ---------------------------------------------------------------------------
# While auditing we found that every travel time is (to rounding) a fixed
# minutes-per-km rate times Distance_km, and every cost follows a fixed
# per-km rate or a stepped fare table. The class below recovers those rules
# from the rows that obey them. It is used to (a) detect the rows whose times
# are inconsistent with their distance and (b) build LOS attributes for the
# synthetic population.

def _step_table(dist, fare):
    """Turn observed (distance, fare) pairs into a step function.

    Boundaries are put halfway between the longest trip charged one fare and
    the shortest trip charged the next fare.
    """
    g = pd.DataFrame({"d": dist, "f": fare}).groupby("f")["d"].agg(["min", "max"]).sort_index()
    fares = g.index.to_numpy(dtype=float)
    uppers = []
    for i in range(len(fares) - 1):
        uppers.append((g["max"].iloc[i] + g["min"].iloc[i + 1]) / 2.0)
    uppers.append(np.inf)
    return list(zip(uppers, fares))


@dataclass
class LOSRules:
    time_rate: dict = field(default_factory=dict)   # minutes per km by mode
    car_rate: float = 0.0
    twow_rate: float = 0.0
    auto_min_fare: float = 15.0
    auto_slope: float = 0.0
    auto_intercept: float = 0.0
    bus_steps: list = field(default_factory=list)
    train_steps: list = field(default_factory=list)

    @classmethod
    def fit(cls, df):
        d = df["Distance_km"].to_numpy(float)
        # medians are not disturbed by the handful of corrupted time records
        rates = {m: float(np.median(df[c] / d)) for m, c in C.TIME_COLS.items()}
        # costs agree with distance in every record, so all rows are used here
        car_rate = float(np.sum(df["TC_Car"] * d) / np.sum(d ** 2))
        twow_rate = float(np.sum(df["TC_2W"] * d) / np.sum(d ** 2))
        auto = df["TC_Auto"].to_numpy(float)
        auto_min = float(auto.min())
        above = auto > auto_min
        slope, intercept = np.polyfit(d[above], auto[above], 1)
        return cls(
            time_rate=rates, car_rate=car_rate, twow_rate=twow_rate,
            auto_min_fare=auto_min, auto_slope=float(slope), auto_intercept=float(intercept),
            bus_steps=_step_table(df["Distance_km"], df["TC_Bus"]),
            train_steps=_step_table(df["Distance_km"], df["TC_LocalTrain"]),
        )

    @staticmethod
    def _lookup(d, steps):
        uppers = np.array([u for u, _ in steps])
        fares = np.array([f for _, f in steps])
        return fares[np.searchsorted(uppers, d, side="left")]

    def times(self, distance):
        d = np.asarray(distance, float)
        return pd.DataFrame({C.TIME_COLS[m]: np.round(r * d, 1) for m, r in self.time_rate.items()})

    def costs(self, distance):
        d = np.asarray(distance, float)
        auto = np.maximum(self.auto_min_fare, np.round(self.auto_slope * d + self.auto_intercept))
        return pd.DataFrame({
            "TC_Car": np.round(self.car_rate * d),
            "TC_2W": np.round(self.twow_rate * d),
            "TC_Auto": auto,
            "TC_Bus": self._lookup(d, self.bus_steps),
            "TC_LocalTrain": self._lookup(d, self.train_steps),
        })

    def to_dict(self):
        return {
            "time_rate_min_per_km": {m: round(v, 4) for m, v in self.time_rate.items()},
            "implied_speed_kmh": {m: round(60.0 / v, 2) for m, v in self.time_rate.items()},
            "car_rate_inr_per_km": round(self.car_rate, 4),
            "twow_rate_inr_per_km": round(self.twow_rate, 4),
            "auto_fare": {"minimum": self.auto_min_fare, "slope": round(self.auto_slope, 4),
                          "intercept": round(self.auto_intercept, 4)},
            "bus_fare_steps": [(round(u, 2) if np.isfinite(u) else "inf", f) for u, f in self.bus_steps],
            "train_fare_steps": [(round(u, 2) if np.isfinite(u) else "inf", f) for u, f in self.train_steps],
        }


def time_inconsistency_mask(df, rates):
    """True where at least one mode time disagrees with rate x distance.

    Distance is stored to 0.1 km and times to 0.1 min, so we allow the error
    that rounding alone can produce (half a unit of each) plus a small margin.
    """
    d = df["Distance_km"].to_numpy(float)
    bad = np.zeros(len(df), dtype=bool)
    for m, col in C.TIME_COLS.items():
        r = rates[m]
        tol = 0.06 * r + 0.06
        bad |= np.abs(df[col].to_numpy(float) - r * d) > tol
    return bad


def los_consistency_table(df, rules):
    """One row per inconsistent record: implied distances from each time and cost."""
    bad = time_inconsistency_mask(df, rules.time_rate)
    rows = df.loc[bad]
    out = pd.DataFrame({"ID": rows["ID"], "Distance_km": rows["Distance_km"]})
    for m, col in C.TIME_COLS.items():
        out[f"km_implied_by_{m}_time"] = (rows[col] / rules.time_rate[m]).round(2)
    out["km_implied_by_car_cost"] = (rows["TC_Car"] / rules.car_rate).round(2)
    out["km_implied_by_2w_cost"] = (rows["TC_2W"] / rules.twow_rate).round(2)
    return out.reset_index(drop=True)


def cost_rule_match(df, rules):
    """How well the recovered cost rules reproduce the observed costs."""
    pred = rules.costs(df["Distance_km"])
    res = []
    for col in pred.columns:
        diff = (pred[col].to_numpy() - df[col].to_numpy())
        res.append({"cost_column": col,
                    "exact_match_share": round(float(np.mean(diff == 0)), 3),
                    "within_1_inr_share": round(float(np.mean(np.abs(diff) <= 1)), 3),
                    "max_abs_diff_inr": float(np.max(np.abs(diff)))})
    return pd.DataFrame(res)


# ---------------------------------------------------------------------------
# Generic checks
# ---------------------------------------------------------------------------

def duplicate_columns(df):
    cols = list(df.columns)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if df[a].equals(df[b]):
                pairs.append((a, b))
    return pairs


def code_check(df):
    """Compare observed codes with the coding document, variable by variable."""
    rows = []
    for var, doc in C.DOC_CODING.items():
        counts = df[var].value_counts().sort_index()
        observed = set(counts.index.astype(int))
        documented = set(doc)
        rows.append({
            "variable": var,
            "documented_codes": ", ".join(f"{k}={v}" for k, v in doc.items()),
            "observed_counts": ", ".join(f"{int(k)}:{int(v)}" for k, v in counts.items()),
            "undocumented_codes_in_data": sorted(observed - documented),
            "documented_but_unused": sorted(documented - observed),
            "consistent": observed == documented,
        })
    return pd.DataFrame(rows)


def outlier_table(df, cols):
    rows = []
    for c in cols:
        x = df[c].astype(float)
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        hi = q3 + 3 * iqr
        rows.append({"column": c, "min": x.min(), "median": x.median(), "max": x.max(),
                     "q3_plus_3iqr": round(hi, 2), "n_above_q3_plus_3iqr": int((x > hi).sum()),
                     "n_zero": int((x == 0).sum())})
    return pd.DataFrame(rows)


def availability_checks(df):
    av = df[[C.AV_COLS[m] for m in C.MODES]].to_numpy()
    choice_idx = df["Choice"].to_numpy() - 1
    chosen_available = av[np.arange(len(df)), choice_idx]
    n_av = av.sum(axis=1)
    return {
        "chosen_alternative_unavailable": int((chosen_available == 0).sum()),
        "availability_share": {m: round(float(df[C.AV_COLS[m]].mean()), 3) for m in C.MODES},
        "n_available_alternatives": {int(k): int(v) for k, v in pd.Series(n_av).value_counts().sort_index().items()},
        "single_alternative_records": int((n_av == 1).sum()),
        "car_av_without_car": int(((df["Car_AV"] == 1) & (df["Car_Ownership"] == 0)).sum()),
        "twow_av_without_twow": int(((df["TwoW_AV"] == 1) & (df["TwoW_Ownership"] == 0)).sum()),
        "car_owned_but_unavailable": int(((df["Car_AV"] == 0) & (df["Car_Ownership"] > 0)).sum()),
        "twow_owned_but_unavailable": int(((df["TwoW_AV"] == 0) & (df["TwoW_Ownership"] > 0)).sum()),
        "walk_av_share_up_to_1.5km": round(float(df.loc[df["Distance_km"] <= 1.5, "Walk_AV"].mean()), 3),
        "walk_av_share_above_3km": round(float(df.loc[df["Distance_km"] > 3, "Walk_AV"].mean()), 3),
        "max_walk_distance_chosen_km": float(df.loc[df["Choice"] == 1, "Distance_km"].max()),
    }


def household_flags(df):
    return pd.DataFrame({
        "ID": df["ID"],
        "hh_size_not_adults_plus_children": df["HH_Size"] != df["Adults"] + df["Children"],
        "zero_adults": df["Adults"] == 0,
        "workers_exceed_adults": df["Working_Adults"] > df["Adults"],
        "workers_exceed_hh_size": df["Working_Adults"] > df["HH_Size"],
    })


def category_evidence(df):
    """Cross-tabs that bear on the undocumented / shifted codes."""
    return {
        "purpose_by_age": pd.crosstab(df["Age"], df["Trip_Purpose"]),
        "purpose_by_occupation": pd.crosstab(df["Occup"], df["Trip_Purpose"]),
        "occupation_by_age": pd.crosstab(df["Occup"], df["Age"]),
        "dwelling_by_income": pd.crosstab(df["Dwelling"], df["Income"]),
        "dwelling_by_area_type": pd.crosstab(df["Dwelling"], df["Area_Type"]),
    }


def cramers_v(x, y):
    t = pd.crosstab(x, y).to_numpy(float)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan
    n = t.sum()
    expected = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / n
    chi2 = ((t - expected) ** 2 / expected).sum()
    r, k = t.shape
    # Bergsma (2013) bias correction
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else np.nan


def run_audit(df):
    """Run every check and return a dictionary of results."""
    rules = LOSRules.fit(df)
    numeric = ["Distance_km"] + list(C.TIME_COLS.values()) + list(C.COST_COLS.values()) + \
              ["HH_Size", "Adults", "Children", "Working_Adults", "Car_Ownership", "TwoW_Ownership"]
    target = df["Choice"].value_counts().sort_index()
    hh = household_flags(df)
    return {
        "shape": df.shape,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_ignoring_id": int(df.drop(columns="ID").duplicated().sum()),
        "id_unique": bool(df["ID"].is_unique),
        "id_range": (int(df["ID"].min()), int(df["ID"].max())),
        "duplicate_columns": duplicate_columns(df),
        "constant_columns": [c for c in df.columns if df[c].nunique() == 1],
        "code_check": code_check(df),
        "outliers": outlier_table(df, numeric),
        "los_rules": rules,
        "los_inconsistent": los_consistency_table(df, rules),
        "cost_rule_match": cost_rule_match(df, rules),
        "availability": availability_checks(df),
        "household_flags": hh,
        "household_flag_counts": hh.drop(columns="ID").sum().astype(int).to_dict(),
        "target_counts": {C.MODE_LABELS[C.MODES[int(k) - 1]]: int(v) for k, v in target.items()},
        "category_evidence": category_evidence(df),
    }


# ---------------------------------------------------------------------------
# Verified data dictionary
# ---------------------------------------------------------------------------
_DESCRIPTIONS = {
    "ID": ("identifier", "Respondent identifier (not contiguous: 531 records with IDs between 1 and 654)."),
    "Distance_km": ("continuous", "Trip distance in km (one decimal)."),
    "Trip_Purpose": ("binary", "Trip purpose."),
    "Income": ("ordinal", "Monthly household income band (INR)."),
    "Gender": ("binary", "Gender of respondent."),
    "Age": ("ordinal", "Age band (years)."),
    "Education": ("ordinal", "Highest education level."),
    "Occup": ("nominal", "Occupation."),
    "ParkingSpace": ("binary", "Parking space indicator."),
    "walk_time_min": ("continuous", "Walking time (min)."),
    "TwoW_time_min": ("continuous", "Two-wheeler travel time (min)."),
    "auto_time_min": ("continuous", "Auto-rickshaw travel time (min)."),
    "car_time_min": ("continuous", "Car travel time (min)."),
    "local_train_time_min": ("continuous", "Suburban rail travel time (min)."),
    "bus_time_min": ("continuous", "Bus travel time (min)."),
    "Area_Type": ("binary", "Area type of residence."),
    "Dwelling": ("nominal", "Dwelling type."),
    "HH_Size": ("count", "Household size (persons)."),
    "Adults": ("count", "Number of adults in household."),
    "Children": ("count", "Number of children in household."),
    "Working_Adults": ("count", "Number of working members in household."),
    "SAPT": ("binary", "Satisfaction with public transport."),
    "Choice": ("target", "Chosen mode."),
    "DL_TwoW": ("binary", "Two-wheeler driving licence."),
    "DL_Car": ("binary", "Car driving licence."),
    "Car_AV": ("availability", "Car available for the trip."),
    "TwoW_AV": ("availability", "Two-wheeler available for the trip."),
    "Walk_AV": ("availability", "Walking available for the trip."),
    "Auto_AV": ("availability", "Auto-rickshaw available for the trip."),
    "Bus_AV": ("availability", "Bus available for the trip."),
    "LocalTrain_AV": ("availability", "Suburban rail available for the trip."),
    "Car_Ownership": ("count", "Number of cars owned by household."),
    "TwoW_Ownership": ("count", "Number of two-wheelers owned by household."),
    "TC_Car": ("continuous", "Car travel cost (INR)."),
    "TC_2W": ("continuous", "Two-wheeler travel cost (INR)."),
    "TC_Auto": ("continuous", "Auto-rickshaw fare (INR)."),
    "TC_Bus": ("continuous", "Bus fare (INR)."),
    "TC_LocalTrain": ("continuous", "Suburban rail fare (INR)."),
}

_STATUS = {
    "Trip_Purpose": ("DISCREPANCY", "Document says 0/1; data contain 1/2. Cross-tabs suggest 2 = work "
                     "(all 18-30 year olds and 94% of occupation code 3 report 2) but this is not confirmed.",
                     "Kept as code 1 / code 2 (neutral labels). Not remapped."),
    "Gender": ("UNDOCUMENTED", "Coding not given. 0: 16 records, 1: 515 records.",
               "Kept as binary code; too few code-0 records to estimate a gender effect in the choice models."),
    "Occup": ("DISCREPANCY", "Document lists codes 1-4; data contain 1-5 (code 5 = 155 records, undocumented). "
              "Code 2 ('Government employee') is 64% aged >55 with 53% purpose-1 trips and code 1 "
              "('Unemployed') only makes purpose-2 trips, so the document's labels may not apply.",
              "Kept as nominal codes; codes 1 and 4 (3 and 4 records) pooled as 'rare' for modelling."),
    "Dwelling": ("DISCREPANCY", "Document lists codes 1-3; data contain 1-4 (code 4 = 191 records, undocumented). "
                 "Code 2 has only 10 records.", "Kept as nominal codes, no relabelling."),
    "ParkingSpace": ("UNDOCUMENTED", "Binary 0/1; whether it refers to home or destination parking is not stated.",
                     "Kept as binary code."),
    "Area_Type": ("UNDOCUMENTED", "Binary 0/1; meaning not stated.", "Kept as binary code."),
    "Adults": ("CHECK", "One record has Adults = 0; 15 records have Working_Adults > Adults; "
               "3 records have HH_Size != Adults + Children.", "Flagged, not altered."),
    "Working_Adults": ("CHECK", "15 records exceed Adults; 3 records (IDs 288, 486, 533) exceed HH_Size.",
                       "Flagged, not altered; capped at HH_Size only inside the workers-share feature."),
    "DL_TwoW": ("DUPLICATE", "Identical to TwoW_AV in all 531 records.", "Dropped (duplicate column)."),
    "DL_Car": ("DUPLICATE", "Identical to Car_AV in all 531 records.", "Dropped (duplicate column)."),
    "LocalTrain_AV": ("CONSTANT", "Equal to 1 in all records.",
                      "Kept to define the choice set; unusable as a predictor; no availability effect can be learned."),
    "SAPT": ("OK (imbalanced)", "0: 29 records, 1: 502 records.", "Kept as binary."),
    "Choice": ("OK", "Matches document codes 1-6.", "Target."),
}

_LOS_NOTE = ("Computed, not observed: equals a fixed rate x Distance_km to rounding in 520/531 records. "
             "11 records are inconsistent with their distance (costs agree with distance).")


def build_data_dictionary(df):
    rows = []
    for col in C.EXPECTED_COLUMNS:
        vtype, desc = _DESCRIPTIONS[col]
        x = df[col]
        if x.nunique() <= 12:
            observed = ", ".join(f"{int(k)}:{int(v)}" for k, v in x.value_counts().sort_index().items())
        else:
            observed = f"min {x.min():g}, median {x.median():g}, max {x.max():g}"
        doc = C.DOC_CODING.get(col)
        doc_txt = ", ".join(f"{k}={v}" for k, v in doc.items()) if doc else "not in coding document"
        if col in _STATUS:
            status, note, treatment = _STATUS[col]
        elif col in C.TIME_COLS.values():
            status, note, treatment = "DERIVED + ERRORS", _LOS_NOTE, "11 records recomputed from distance (logged)."
        elif col in C.COST_COLS.values():
            status, note, treatment = ("DERIVED", "Computed from Distance_km with a per-km rate or a stepped fare table.",
                                       "Kept as is.")
        elif doc is not None:
            status, note, treatment = "OK", "Observed codes match the document.", "Kept as is."
        else:
            status, note, treatment = "OK", "", "Kept as is."
        rows.append({"variable": col, "type": vtype, "description": desc, "documented_coding": doc_txt,
                     "observed_values": observed, "status": status, "audit_note": note, "treatment": treatment})
    return pd.DataFrame(rows)
