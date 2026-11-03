"""Synthetic commuter population.

Only predictor variables are generated, never a Choice label. Three methods
are compared:

  bootstrap   stratified resampling of real rows (reference: perfect
              marginals but every record is a copy of a real person)
  copula      Gaussian copula for mixed data: each variable is mapped to a
              normal score, the correlation of the scores is estimated and new
              scores are drawn and mapped back
  cart        sequential conditional synthesis (as in the synthpop package):
              each variable is drawn from the leaf of a CART fitted on the
              variables generated before it

Travel times and costs are not synthesised. In the survey they are fixed
functions of distance, so they are rebuilt from the synthetic distance with
the rules recovered in data_validation.LOSRules. Logical constraints are then
enforced and the number of records that needed repair is reported.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config as C
from .data_validation import cramers_v

DISCRETE = ["Income", "Age", "Education", "Occup", "Gender", "Area_Type", "Dwelling", "HH_Size", "Children",
            "Working_Adults", "Car_Ownership", "TwoW_Ownership", "ParkingSpace", "Car_AV", "TwoW_AV",
            "Trip_Purpose", "SAPT", "Walk_AV", "Auto_AV", "Bus_AV"]
CONTINUOUS = ["Distance_km"]
SYNTH_VARS = DISCRETE[:17] + ["Distance_km"] + DISCRETE[17:]      # also the CART visit order
WALK_ALWAYS_KM = 1.5     # every surveyed trip up to 1.5 km has walking available (126/126)


# ---------------------------------------------------------------------------
# Constraints and derived columns
# ---------------------------------------------------------------------------

def constraint_violations(df, d_min, d_max):
    return {
        "children_leave_no_adult": int((df["Children"] > df["HH_Size"] - 1).sum()),
        "workers_exceed_household": int((df["Working_Adults"] > df["HH_Size"]).sum()),
        "car_available_without_car": int(((df["Car_AV"] == 1) & (df["Car_Ownership"] == 0)).sum()),
        "twow_available_without_twow": int(((df["TwoW_AV"] == 1) & (df["TwoW_Ownership"] == 0)).sum()),
        "short_trip_without_walk_option": int(((df["Distance_km"] <= WALK_ALWAYS_KM) & (df["Walk_AV"] == 0)).sum()),
        "distance_outside_observed_range": int(((df["Distance_km"] < d_min) | (df["Distance_km"] > d_max)).sum()),
    }


def repair(df, d_min, d_max):
    out = df.copy()
    out["Distance_km"] = out["Distance_km"].clip(d_min, d_max).round(1).clip(lower=0.1)
    out["Children"] = np.minimum(out["Children"], out["HH_Size"] - 1).clip(lower=0)
    out["Working_Adults"] = out["Working_Adults"].clip(lower=1)
    out["Working_Adults"] = np.minimum(out["Working_Adults"], out["HH_Size"])
    out.loc[out["Car_Ownership"] == 0, "Car_AV"] = 0
    out.loc[out["TwoW_Ownership"] == 0, "TwoW_AV"] = 0
    out.loc[out["Distance_km"] <= WALK_ALWAYS_KM, "Walk_AV"] = 1
    return out


def add_derived(df, rules):
    out = df.copy()
    out["Adults"] = out["HH_Size"] - out["Children"]
    out["LocalTrain_AV"] = 1
    out = pd.concat([out.reset_index(drop=True), rules.times(out["Distance_km"]),
                     rules.costs(out["Distance_km"])], axis=1)
    out.insert(0, "ID", np.arange(1, len(out) + 1))
    return out


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

class BootstrapSynthesizer:
    name = "bootstrap"

    def fit(self, df):
        self.data_ = df[SYNTH_VARS].reset_index(drop=True)
        self.strata_ = pd.cut(df["Distance_km"], [0, 1.5, 3, 5, 10, 20, np.inf]).astype(str) + "|" + \
            df["Income"].astype(str).to_numpy()
        return self

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        strata = self.strata_.reset_index(drop=True)
        share = strata.value_counts(normalize=True)
        counts = rng.multinomial(n, share.to_numpy())
        idx = []
        for s, k in zip(share.index, counts):
            pool = np.flatnonzero(strata.to_numpy() == s)
            idx.extend(rng.choice(pool, size=k, replace=True))
        return self.data_.iloc[rng.permutation(idx)].reset_index(drop=True)


class GaussianCopulaSynthesizer:
    """Discrete variables use a randomised probability integral transform;
    categories are ordered by frequency before the transform (as in SDV's
    frequency encoder) because the copula only captures monotone association."""
    name = "copula"

    def fit(self, df, seed=0):
        rng = np.random.default_rng(seed)
        d = df[SYNTH_VARS].reset_index(drop=True)
        self.levels_, self.cdf_ = {}, {}
        Z = np.zeros((len(d), len(SYNTH_VARS)))
        for k, v in enumerate(SYNTH_VARS):
            x = d[v].to_numpy()
            if v in CONTINUOUS:
                self.sorted_ = np.sort(x)
                u = (stats.rankdata(x) - 0.5) / len(x)
            else:
                ordinal = v in ("Income", "Age", "Education", "HH_Size", "Children", "Working_Adults",
                                "Car_Ownership", "TwoW_Ownership")
                vc = pd.Series(x).value_counts()
                levels = np.sort(vc.index.to_numpy()) if ordinal else vc.index.to_numpy()
                p = vc.reindex(levels).to_numpy() / len(x)
                cdf = np.cumsum(p)
                pos = {lv: i for i, lv in enumerate(levels)}
                i = np.array([pos[val] for val in x])
                lo = np.where(i > 0, cdf[i - 1], 0.0)
                u = lo + rng.random(len(x)) * (cdf[i] - lo)
                self.levels_[v], self.cdf_[v] = levels, cdf
            Z[:, k] = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        R = np.corrcoef(Z, rowvar=False)
        w, Q = np.linalg.eigh(R)                      # make sure R is positive definite
        R = Q @ np.diag(np.clip(w, 1e-6, None)) @ Q.T
        dsd = np.sqrt(np.diag(R))
        self.corr_ = R / np.outer(dsd, dsd)
        return self

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        Z = rng.multivariate_normal(np.zeros(len(SYNTH_VARS)), self.corr_, size=n)
        U = stats.norm.cdf(Z)
        out = {}
        for k, v in enumerate(SYNTH_VARS):
            u = U[:, k]
            if v in CONTINUOUS:
                grid = (np.arange(len(self.sorted_)) + 0.5) / len(self.sorted_)
                out[v] = np.exp(np.interp(u, grid, np.log(self.sorted_)))
            else:
                idx = np.searchsorted(self.cdf_[v], u, side="left").clip(0, len(self.levels_[v]) - 1)
                out[v] = self.levels_[v][idx]
        return pd.DataFrame(out)[SYNTH_VARS]


class SequentialCARTSynthesizer:
    """Each variable after the first is drawn from the training values that
    fall in the same CART leaf as the synthetic record. Distance is smoothed
    with a kernel inside each leaf so that records are not exact copies."""
    name = "cart"

    def __init__(self, min_leaf=10):
        # 10 rather than synthpop's default of 5: with 5 leaves, 15% of the
        # records sat closer to a training record than almost any real
        # held-out person does (see cart_leaf_sensitivity)
        self.min_leaf = min_leaf

    def fit(self, df, seed=0):
        self.data_ = df[SYNTH_VARS].reset_index(drop=True)
        self.trees_ = {}
        for k, v in enumerate(SYNTH_VARS[1:], start=1):
            X = self.data_[SYNTH_VARS[:k]].to_numpy(float)
            y = self.data_[v].to_numpy()
            Tree = DecisionTreeRegressor if v in CONTINUOUS else DecisionTreeClassifier
            tree = Tree(min_samples_leaf=self.min_leaf, random_state=seed).fit(X, np.log(y) if v in CONTINUOUS else y)
            leaves = tree.apply(X)
            self.trees_[v] = (tree, leaves)
        return self

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        first = SYNTH_VARS[0]
        out = pd.DataFrame({first: rng.choice(self.data_[first].to_numpy(), size=n, replace=True)})
        for k, v in enumerate(SYNTH_VARS[1:], start=1):
            tree, train_leaves = self.trees_[v]
            leaves = tree.apply(out[SYNTH_VARS[:k]].to_numpy(float))
            values = np.empty(n)
            y = self.data_[v].to_numpy(float)
            for leaf in np.unique(leaves):
                rows = np.flatnonzero(leaves == leaf)
                donors = y[train_leaves == leaf]
                draw = rng.choice(donors, size=len(rows), replace=True)
                if v in CONTINUOUS:
                    ld = np.log(donors)
                    sd = ld.std(ddof=1) if len(ld) > 1 else 0.0
                    iqr = np.subtract(*np.percentile(ld, [75, 25])) / 1.34
                    spread = min(sd, iqr) if iqr > 0 else sd
                    h = 0.9 * spread * len(ld) ** (-0.2)          # Silverman's rule
                    draw = np.exp(np.log(draw) + rng.normal(0, h, len(rows)))
                values[rows] = draw
            out[v] = values if v in CONTINUOUS else values.astype(int)
        return out[SYNTH_VARS]


METHODS = {"bootstrap": BootstrapSynthesizer, "copula": GaussianCopulaSynthesizer,
           "cart": SequentialCARTSynthesizer}


def generate(method, train, rules, n=C.SYNTH_N, seed=C.SYNTH_SEED):
    """Fit one generator on the training rows and return (population, violations before repair)."""
    gen = METHODS[method]()
    gen = gen.fit(train) if method == "bootstrap" else gen.fit(train, seed=seed)
    raw = gen.sample(n, seed)
    d_min, d_max = train["Distance_km"].min(), train["Distance_km"].max()
    if "Distance_km" in raw:
        raw["Distance_km"] = raw["Distance_km"].round(1)
    viol = constraint_violations(raw, d_min, d_max)
    pop = add_derived(repair(raw, d_min, d_max), rules)
    return pop, viol


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _tvd(a, b):
    pa = pd.Series(a).value_counts(normalize=True)
    pb = pd.Series(b).value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    return 0.5 * float(np.abs(pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).sum())


def marginal_fidelity(real, synth):
    rows = []
    for v in SYNTH_VARS:
        if v in CONTINUOUS:
            ks = stats.ks_2samp(real[v], synth[v]).statistic
            rows.append({"variable": v, "metric": "KS statistic", "distance": ks})
        else:
            rows.append({"variable": v, "metric": "total variation distance", "distance": _tvd(real[v], synth[v])})
    return pd.DataFrame(rows)


def association_matrix(df):
    d = df[SYNTH_VARS].copy()
    d["Distance_km"] = pd.cut(d["Distance_km"], [0, 1.5, 3, 5, 10, 20, np.inf])
    M = pd.DataFrame(np.eye(len(SYNTH_VARS)), index=SYNTH_VARS, columns=SYNTH_VARS)
    for i, a in enumerate(SYNTH_VARS):
        for b in SYNTH_VARS[i + 1:]:
            v = cramers_v(d[a], d[b])
            M.loc[a, b] = M.loc[b, a] = 0.0 if np.isnan(v) else v
    return M


def moments_table(real, synth, cols=None):
    cols = cols or ["Distance_km", "HH_Size", "Children", "Working_Adults", "Car_Ownership", "TwoW_Ownership",
                    "bus_time_min", "TC_Bus", "TC_Car"]
    rows = []
    for c in cols:
        for name, d in (("real", real), ("synthetic", synth)):
            q = d[c].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy()
            rows.append({"variable": c, "data": name, "mean": d[c].mean(), "sd": d[c].std(),
                         "p5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4]})
    return pd.DataFrame(rows)


def discriminator_auc(real, synth, seed=C.SEED):
    """Cross-validated AUC of a random forest asked to tell real from synthetic
    records (0.5 = indistinguishable)."""
    rng = np.random.default_rng(seed)
    s = synth[SYNTH_VARS].iloc[rng.choice(len(synth), size=len(real), replace=False)]
    X = pd.concat([real[SYNTH_VARS], s], ignore_index=True).to_numpy(float)
    y = np.r_[np.zeros(len(real)), np.ones(len(s))]
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(rf, X, y, cv=cv, scoring="roc_auc").mean())


def _gower(a, b, ranges):
    """Gower distance from every row of a to its nearest row in b."""
    out = np.empty(len(a))
    disc = [i for i, v in enumerate(SYNTH_VARS) if v not in CONTINUOUS]
    cont = [i for i, v in enumerate(SYNTH_VARS) if v in CONTINUOUS]
    for start in range(0, len(a), 500):
        A = a[start:start + 500]
        d = (A[:, None, disc] != b[None, :, disc]).sum(axis=2).astype(float)
        d += (np.abs(A[:, None, cont] - b[None, :, cont]) / ranges[cont]).sum(axis=2)
        out[start:start + 500] = d.min(axis=1) / len(SYNTH_VARS)
    return out


def memorisation(train, test, synth):
    """Exact copies of training records and distance to the closest training
    record (DCR), compared with real held-out records as a yardstick."""
    tr = train[SYNTH_VARS].to_numpy(float)
    te = test[SYNTH_VARS].to_numpy(float)
    sy = synth[SYNTH_VARS].to_numpy(float)
    ranges = np.ptp(tr, axis=0)
    ranges[ranges == 0] = 1.0
    keys = set(map(tuple, tr))
    dcr_syn, dcr_test = _gower(sy, tr, ranges), _gower(te, tr, ranges)
    return {
        "exact_copy_share_synthetic": float(np.mean([tuple(r) in keys for r in sy])),
        "exact_copy_share_holdout": float(np.mean([tuple(r) in keys for r in te])),
        "median_dcr_synthetic": float(np.median(dcr_syn)),
        "median_dcr_holdout": float(np.median(dcr_test)),
        "share_synthetic_closer_than_holdout_p5": float(np.mean(dcr_syn < np.percentile(dcr_test, 5))),
    }


def compare_methods(train, test, rules, n=C.SYNTH_N, seed=C.SYNTH_SEED):
    real_assoc = association_matrix(train)
    iu = np.triu_indices(len(SYNTH_VARS), 1)
    rows, pops = [], {}
    for method in METHODS:
        pop, viol = generate(method, train, rules, n=n, seed=seed)
        pops[method] = pop
        marg = marginal_fidelity(train, pop)
        assoc = association_matrix(pop)
        mem = memorisation(train, test, pop)
        rows.append({
            "method": method,
            "mean_marginal_distance": marg["distance"].mean(),
            "max_marginal_distance": marg["distance"].max(),
            "association_mad": float(np.abs(assoc.to_numpy()[iu] - real_assoc.to_numpy()[iu]).mean()),
            "discriminator_auc": discriminator_auc(train, pop, seed),
            "records_needing_repair_pct": 100 * sum(viol.values()) / n,
            **{f"viol_{k}": v for k, v in viol.items()},
            **mem,
        })
    return pd.DataFrame(rows), pops


def choose_method(table, tolerance_pp=1.0):
    """Pre-declared rule: a method is eligible if its exact-copy share is no
    more than twice the share of held-out real records that coincide with a
    training record (plus one percentage point). Among eligible methods the
    discriminator AUC closest to 0.5 wins; association MAD breaks ties.
    (AUC below 0.5 is not better than 0.5: it happens when near-copies of
    training records end up on the other side of a CV split.)"""
    limit = 2 * table["exact_copy_share_holdout"].iloc[0] + tolerance_pp / 100
    ok = table[table["exact_copy_share_synthetic"] <= limit].copy()
    ok["auc_gap"] = (ok["discriminator_auc"] - 0.5).abs()
    best = ok.sort_values(["auc_gap", "association_mad"]).iloc[0]
    return best["method"], limit


def cart_leaf_sensitivity(train, test, rules, leaves=(5, 10, 15, 20), n=C.SYNTH_N, seed=C.SYNTH_SEED):
    """Fidelity against memorisation for different minimum leaf sizes."""
    real_assoc = association_matrix(train)
    iu = np.triu_indices(len(SYNTH_VARS), 1)
    d_min, d_max = train["Distance_km"].min(), train["Distance_km"].max()
    rows = []
    for leaf in leaves:
        raw = SequentialCARTSynthesizer(min_leaf=leaf).fit(train, seed=seed).sample(n, seed)
        raw["Distance_km"] = raw["Distance_km"].round(1)
        viol = constraint_violations(raw, d_min, d_max)
        pop = add_derived(repair(raw, d_min, d_max), rules)
        mem = memorisation(train, test, pop)
        rows.append({"min_leaf": leaf,
                     "mean_marginal_distance": marginal_fidelity(train, pop)["distance"].mean(),
                     "association_mad": float(np.abs(association_matrix(pop).to_numpy()[iu]
                                                     - real_assoc.to_numpy()[iu]).mean()),
                     "discriminator_auc": discriminator_auc(train, pop, seed),
                     "records_needing_repair_pct": 100 * sum(viol.values()) / n, **mem})
    return pd.DataFrame(rows)
