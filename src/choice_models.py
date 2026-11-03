"""Random-utility discrete choice models estimated by maximum likelihood.

Both models work on the "wide" survey table but internally build a
person x alternative x parameter design array, i.e. the long format used in
discrete choice analysis. Availability enters the likelihood directly: an
unavailable alternative is simply not part of that person's choice set.

    V_nj = sum_k beta_k * x_njk
    MNL : P_nj = a_nj exp(V_nj) / sum_i a_ni exp(V_ni)
    NL  : P_nj = P(j | m) P(m),  P(j | m) = exp(V_nj / l_m) / exp(I_nm),
          I_nm = log sum_{i in m} a_ni exp(V_ni / l_m),
          P(m) = exp(l_m I_nm) / sum_k exp(l_k I_nk)

The classes follow the scikit-learn estimator interface (fit / predict_proba)
so that they can sit in the same comparison loop as the ML models.
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin

from . import config as C
from .feature_engineering import add_features

J = len(C.MODES)
MOTORISED = ["twow", "auto", "car", "bus", "train"]


def design_frame(df):
    """Columns referred to by the utility specifications."""
    d = add_features(df)
    d["one"] = 1.0
    d["inc2"] = (d["Income"] == 2).astype(float)
    d["inc3"] = (d["Income"] == 3).astype(float)
    d["inc23"] = (d["Income"] >= 2).astype(float)
    d["age1"] = (d["Age"] == 1).astype(float)
    d["age3"] = (d["Age"] == 3).astype(float)
    d["purp1"] = (d["Trip_Purpose"] == 1).astype(float)
    d["area1"] = (d["Area_Type"] == 1).astype(float)
    d["grad"] = (d["Education"] == 1).astype(float)
    d["walk_time_weighted"] = d[C.TIME_COLS["walk"]] * C.WALK_TIME_WEIGHT
    return d


ASCS = [(f"ASC_{m}", {m: "one"}) for m in ["walk", "twow", "auto", "car", "bus"]]   # rail = reference
COST = [("B_cost", {m: C.COST_COLS[m] for m in MOTORISED})]
# one time coefficient; a walking minute counts WALK_TIME_WEIGHT in-vehicle minutes
GENERIC_TIME = [("B_time", {**{"walk": "walk_time_weighted"}, **{m: C.TIME_COLS[m] for m in MOTORISED}})]
# unrestricted alternative: separate walking and in-vehicle time coefficients
FREE_TIME = [("B_walk_time", {"walk": C.TIME_COLS["walk"]}),
             ("B_time", {m: C.TIME_COLS[m] for m in MOTORISED})]

CORE_TERMS = [
    ("B_car_inc2", {"car": "inc2"}),
    ("B_car_inc3", {"car": "inc3"}),
    ("B_car_owned", {"car": "Car_Ownership"}),
    ("B_walk_purp1", {"walk": "purp1"}),
    ("B_bus_purp1", {"bus": "purp1"}),
]
EXTRA_SOCIO_TERMS = [
    ("B_twow_inc23", {"twow": "inc23"}),
    ("B_twow_owned", {"twow": "TwoW_Ownership"}),
    ("B_parking_private", {"car": "ParkingSpace", "twow": "ParkingSpace"}),
    ("B_walk_age3", {"walk": "age3"}),
    ("B_bus_age3", {"bus": "age3"}),
    ("B_twow_age1", {"twow": "age1"}),
    ("B_pt_sapt", {"bus": "SAPT", "train": "SAPT"}),
]

_SPECS = {
    # constants only = availability-masked market shares (a yardstick, not a candidate)
    "ASC": ASCS,
    # behavioural specifications (candidates for the comparison)
    "GC": ASCS + GENERIC_TIME + COST,
    "GC+CORE": ASCS + GENERIC_TIME + COST + CORE_TERMS,
    "GC+SOCIO": ASCS + GENERIC_TIME + COST + CORE_TERMS + EXTRA_SOCIO_TERMS,
    # unrestricted time coefficients, reported for reference only
    "FREE": ASCS + FREE_TIME + COST,
    "FREE+SOCIO": ASCS + FREE_TIME + COST + CORE_TERMS + EXTRA_SOCIO_TERMS,
}
SPEC_NAMES = ["GC", "GC+CORE", "GC+SOCIO"]
REFERENCE_SPECS = ["FREE", "FREE+SOCIO"]


def specification(name):
    """A priori utility specifications (see config.WALK_TIME_WEIGHT for why the
    behavioural ones use a single time coefficient)."""
    if name not in _SPECS:
        raise ValueError(f"unknown specification {name!r}")
    return list(_SPECS[name])


def signs_ok(model):
    """Time and cost coefficients must be negative for a model used in forecasting."""
    s = model.summary_["estimate"]
    keys = [k for k in s.index if k in ("B_time", "B_walk_time", "B_cost")]
    return all(s[k] < 0 for k in keys)


def build_design(df, terms):
    """Return X with shape (N, J, K) and the availability matrix A (N, J)."""
    d = design_frame(df)
    X = np.zeros((len(d), J, len(terms)))
    for k, (_, alts) in enumerate(terms):
        for m, col in alts.items():
            X[:, C.MODES.index(m), k] = d[col].to_numpy(float)
    A = d[[C.AV_COLS[m] for m in C.MODES]].to_numpy(float)
    return X, A


def to_long(df):
    """Person-alternative long format (one row per available alternative)."""
    rows = []
    d = add_features(df)
    for _, r in d.iterrows():
        for i, m in enumerate(C.MODES):
            if r[C.AV_COLS[m]] != 1:
                continue
            rows.append({
                "ID": int(r["ID"]), "alternative": m, "alt_code": i + 1,
                "chosen": int(r["Choice"] == i + 1) if "Choice" in r else np.nan,
                "time_min": r[C.TIME_COLS[m]],
                "cost_inr": r[C.COST_COLS[m]] if m in C.COST_COLS else 0.0,
                "Distance_km": r["Distance_km"], "Income": int(r["Income"]), "Age": int(r["Age"]),
                "Trip_Purpose": int(r["Trip_Purpose"]), "SAPT": int(r["SAPT"]),
                "ParkingSpace": int(r["ParkingSpace"]), "Car_Ownership": int(r["Car_Ownership"]),
                "TwoW_Ownership": int(r["TwoW_Ownership"]),
            })
    return pd.DataFrame(rows)


def _numerical_hessian(f, x, eps=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    h = eps * np.maximum(1.0, np.abs(x))
    for i in range(k):
        for j in range(i, k):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
    return H


class _ChoiceModelBase(BaseEstimator, ClassifierMixin):
    """Shared bookkeeping: design matrices, summary table, prediction."""

    def _prepare(self, X):
        self.terms_ = specification(self.spec)
        return build_design(X, self.terms_)

    @property
    def param_names_(self):
        return [t[0] for t in self.terms_]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def _finish(self, theta, ll, n, names, H):
        self.coef_ = theta
        self.loglik_ = float(ll)
        self.n_obs_ = int(n)
        k = len(theta)
        self.aic_ = 2 * k - 2 * ll
        self.bic_ = k * np.log(n) - 2 * ll
        try:
            cov = np.linalg.inv(-H)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            cov, se = None, np.full(k, np.nan)
        self.cov_ = cov
        self.summary_ = pd.DataFrame({"estimate": theta, "std_error": se}, index=names)
        self.summary_["t_stat"] = self.summary_["estimate"] / self.summary_["std_error"]
        self.summary_["p_value"] = 2 * stats.norm.sf(np.abs(self.summary_["t_stat"]))
        self.classes_ = np.arange(J)

    def fit_stats(self):
        return {"log_likelihood": round(self.loglik_, 3), "null_log_likelihood": round(self.ll_null_, 3),
                "rho_squared": round(1 - self.loglik_ / self.ll_null_, 4),
                "adj_rho_squared": round(1 - (self.loglik_ - len(self.coef_)) / self.ll_null_, 4),
                "n_parameters": len(self.coef_), "n_obs": self.n_obs_,
                "AIC": round(self.aic_, 2), "BIC": round(self.bic_, 2), "converged": bool(self.converged_)}


class MultinomialLogit(_ChoiceModelBase):
    """MNL with analytical gradient and Hessian (the log-likelihood is concave)."""

    def __init__(self, spec="GC+SOCIO"):
        self.spec = spec

    @staticmethod
    def _probs(theta, X, A):
        V = X @ theta
        V = np.where(A > 0, V, -np.inf)
        return np.exp(V - logsumexp(V, axis=1, keepdims=True))

    def fit(self, X, y):
        Xd, A = self._prepare(X)
        y = np.asarray(y)
        n = len(y)
        idx = np.arange(n)
        x_chosen = Xd[idx, y, :]

        def negll(theta):
            V = np.where(A > 0, Xd @ theta, -np.inf)
            return -(V[idx, y] - logsumexp(V, axis=1)).sum()

        def grad(theta):
            P = self._probs(theta, Xd, A)
            return -(x_chosen - np.einsum("nj,njk->nk", P, Xd)).sum(axis=0)

        def hess(theta):
            P = self._probs(theta, Xd, A)
            xbar = np.einsum("nj,njk->nk", P, Xd)
            dev = Xd - xbar[:, None, :]
            return np.einsum("nj,njk,njl->kl", P, dev, dev)

        theta0 = np.zeros(Xd.shape[2])
        res = optimize.minimize(negll, theta0, jac=grad, hess=hess, method="trust-exact",
                                options={"gtol": 1e-7, "maxiter": 500})
        self.converged_ = res.success
        self.ll_null_ = -float(np.log(A.sum(axis=1)).sum())    # equal shares among available modes
        self._finish(res.x, -res.fun, n, self.param_names_, -hess(res.x))
        return self

    def predict_proba(self, X):
        Xd, A = build_design(X, self.terms_)
        return self._probs(self.coef_, Xd, A)


NEST_STRUCTURES = {
    # name: list of nests (each a list of modes); modes not listed are singletons.
    # Each grouping is based on a shared unobserved attribute: public transport
    # (fixed routes, waiting, crowding), shared/hired vehicles (no ownership or
    # parking needed), private vehicles (ownership, parking), motorised travel.
    "PT{bus,train}": [["bus", "train"]],
    "PT{bus,train} + PRIVATE{car,twow}": [["bus", "train"], ["car", "twow"]],
    "SHARED{auto,bus,train}": [["auto", "bus", "train"]],
    "SHARED{auto,bus,train} + PRIVATE{car,twow}": [["auto", "bus", "train"], ["car", "twow"]],
    "MOTORISED{twow,auto,car,bus,train}": [["twow", "auto", "car", "bus", "train"]],
}


class NestedLogit(_ChoiceModelBase):
    """Two-level nested logit. One scale parameter per non-degenerate nest.

    lambda_bounds controls the search box for the nest parameters. Values
    in (0, 1] are consistent with random utility maximisation for every
    individual; lambda = 1 for all nests collapses the model to the MNL.
    """

    def __init__(self, spec="GC+SOCIO", nests="PT{bus,train} + PRIVATE{car,twow}", lambda_bounds=(0.05, 1.0),
                 start=None):
        self.spec = spec
        self.nests = nests
        self.lambda_bounds = lambda_bounds
        self.start = start

    def _nest_matrix(self):
        groups = [list(g) for g in NEST_STRUCTURES[self.nests]]
        listed = {m for g in groups for m in g}
        groups += [[m] for m in C.MODES if m not in listed]
        M = np.zeros((J, len(groups)))
        for g, members in enumerate(groups):
            for m in members:
                M[C.MODES.index(m), g] = 1
        free = [g for g, members in enumerate(groups) if len(members) > 1]
        return M, groups, free

    def _parts(self, theta, Xd, A):
        """Everything the likelihood and its gradient need, computed once."""
        K = Xd.shape[2]
        M, groups, free = self.nest_matrix_, self.groups_, self.free_
        lam = np.ones(len(groups))
        lam[free] = theta[K:]
        V = Xd @ theta[:K]
        nest_of = M.argmax(axis=1)
        S = np.where(A > 0, V / lam[nest_of], -np.inf)     # scaled utilities
        I = np.full((len(V), len(groups)), -np.inf)        # inclusive values
        for g in range(len(groups)):
            I[:, g] = logsumexp(S[:, M[:, g] > 0], axis=1)
        top = np.where(np.isfinite(I), lam * I, -np.inf)
        log_pm = top - logsumexp(top, axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):            # -inf - (-inf) for empty nests, masked below
            log_cond = np.where(A > 0, S - I[:, nest_of], -np.inf)
        return V, lam, nest_of, I, log_pm, log_cond

    def _log_probs(self, theta, Xd, A):
        _, _, nest_of, _, log_pm, log_cond = self._parts(theta, Xd, A)
        return log_cond + log_pm[:, nest_of]

    def _negll_grad(self, theta, Xd, A, y):
        """Negative log-likelihood and its analytical gradient.

        With m = nest of the chosen alternative i:
          d logP / dV_j = 1[j=i]/l_m + 1[j in m] (l_m - 1)/l_m P(j|m) - P_j
          d logP / dl_g = 1[g=m] (-V_i/l_m^2 - (l_m-1) Vbar_m/l_m^2 + I_m) - P(g) (I_g - Vbar_g/l_g)
        where Vbar_g = sum_{j in g} P(j|g) V_j.
        """
        n = len(y)
        idx = np.arange(n)
        K = Xd.shape[2]
        V, lam, nest_of, I, log_pm, log_cond = self._parts(theta, Xd, A)
        lp = log_cond[idx, y] + log_pm[idx, nest_of[y]]
        if not np.all(np.isfinite(lp)):
            return 1e10, np.zeros_like(theta)
        p_cond = np.exp(log_cond)                       # 0 for unavailable alternatives
        p_nest = np.exp(log_pm)                         # 0 for empty nests
        P = p_cond * p_nest[:, nest_of]
        m_n = nest_of[y]
        lam_m = lam[m_n]
        in_m = (nest_of[None, :] == m_n[:, None])
        W = -P + in_m * ((lam_m - 1) / lam_m)[:, None] * p_cond
        W[idx, y] += 1.0 / lam_m
        g_beta = np.einsum("nj,njk->k", W, Xd)
        g_lam = []
        for g in self.free_:
            cols = nest_of == g
            vbar = (p_cond[:, cols] * V[:, cols]).sum(axis=1)
            Ig = np.where(np.isfinite(I[:, g]), I[:, g], 0.0)
            chosen = (m_n == g)
            term = np.where(chosen, -V[idx, y] / lam[g] ** 2 - (lam[g] - 1) * vbar / lam[g] ** 2 + Ig, 0.0)
            term -= p_nest[:, g] * (Ig - vbar / lam[g])
            g_lam.append(term.sum())
        return -lp.sum(), -np.r_[g_beta, g_lam]

    def fit(self, X, y):
        Xd, A = self._prepare(X)
        self.nest_matrix_, self.groups_, self.free_ = self._nest_matrix()
        y = np.asarray(y)
        n = len(y)
        K = Xd.shape[2]
        n_lam = len(self.free_)
        if self.start is not None:
            beta0 = np.asarray(self.start, float)
        else:
            # start from the MNL estimates, i.e. the lambda = 1 special case
            beta0 = MultinomialLogit(self.spec).fit(X, y).coef_
        theta0 = np.r_[beta0, np.full(n_lam, min(0.9, self.lambda_bounds[1]))]
        bounds = [(None, None)] * K + [self.lambda_bounds] * n_lam
        f = lambda t: self._negll_grad(t, Xd, A, y)
        res = optimize.minimize(f, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                                options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-7})
        self.converged_ = res.success
        self.ll_null_ = -float(np.log(A.sum(axis=1)).sum())
        names = self.param_names_ + [f"lambda_{'_'.join(self.groups_[g])}" for g in self.free_]
        # Hessian by central differences of the analytical gradient
        h = 1e-5 * np.maximum(1.0, np.abs(res.x))
        H = np.zeros((len(res.x), len(res.x)))
        for i in range(len(res.x)):
            e = np.zeros_like(res.x); e[i] = h[i]
            H[:, i] = (f(res.x + e)[1] - f(res.x - e)[1]) / (2 * h[i])
        H = -(H + H.T) / 2
        self._finish(res.x, -res.fun, n, names, H)
        lo, hi = self.lambda_bounds
        self.at_bound_ = [bool(abs(l - lo) < 1e-4 or abs(l - hi) < 1e-4) for l in res.x[K:]]
        return self

    @property
    def lambdas_(self):
        K = len(self.terms_)
        return dict(zip(self.summary_.index[K:], self.coef_[K:]))

    def predict_proba(self, X):
        Xd, A = build_design(X, self.terms_)
        P = np.exp(self._log_probs(self.coef_, Xd, A))
        return np.where(A > 0, P, 0.0)


def likelihood_ratio_test(ll_restricted, ll_full, df):
    stat = 2 * (ll_full - ll_restricted)
    return {"LR_statistic": round(stat, 3), "df": df, "p_value": float(stats.chi2.sf(max(stat, 0), df))}


def value_of_time(model):
    """Implied value of travel time savings (INR per hour) from a fitted MNL/NL."""
    s = model.summary_["estimate"]
    out = {"in_vehicle_time_INR_per_hour": 60 * s["B_time"] / s["B_cost"]}
    if "B_walk_time" in s:
        out["walk_time_INR_per_hour"] = 60 * s["B_walk_time"] / s["B_cost"]
    else:
        out["walk_time_INR_per_hour"] = C.WALK_TIME_WEIGHT * out["in_vehicle_time_INR_per_hour"]
    return {k: round(float(v), 1) for k, v in out.items()}


def walk_weight_profile(train, y, weights=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0), spec="GC+SOCIO"):
    """Likelihood of the behavioural specification for a range of walk-time
    weights, tested against the unrestricted model (separate time coefficients)."""
    free = MultinomialLogit(spec.replace("GC", "FREE")).fit(train, y)
    rows = []
    for w in weights:
        t = train.copy()
        # scaling the walking times by w / WALK_TIME_WEIGHT is the same as using weight w
        t[C.TIME_COLS["walk"]] = t[C.TIME_COLS["walk"]] * (w / C.WALK_TIME_WEIGHT)
        m = MultinomialLogit(spec).fit(t, y)
        lr = likelihood_ratio_test(m.loglik_, free.loglik_, 1)
        s = m.summary_["estimate"]
        rows.append({"walk_weight": w, "log_likelihood": round(m.loglik_, 3), "LR_vs_free": lr["LR_statistic"],
                     "p_value": round(lr["p_value"], 4), "B_time": round(s["B_time"], 4),
                     "B_cost": round(s["B_cost"], 4), "VOT_INR_per_hour": round(60 * s["B_time"] / s["B_cost"], 1)})
    return pd.DataFrame(rows), free
