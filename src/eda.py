"""Exploratory figures. Each function saves one PNG and returns the table behind it
so that the written interpretation can quote real numbers."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .data_validation import cramers_v
from .feature_engineering import add_features
from .plot_style import (AXIS, BLUES, INK, INK_2, MODE_COLORS, MUTED, SERIES, SURFACE,
                         apply_style, mode_legend, plain_log, save)

VEH_LABELS = {0: "No vehicle", 1: "Two-wheeler only", 2: "Car only", 3: "Car and two-wheeler"}


def _with_mode(df):
    out = add_features(df)
    out["mode"] = pd.Categorical([C.MODES[c - 1] for c in out["Choice"]], categories=C.MODES)
    return out


def _segment_label_color(hex_color):
    return INK if hex_color.lower() == "#eda100" else "white"


def stacked_share(df, var, labels, title, path, min_label=9.0):
    d = _with_mode(df)
    tab = pd.crosstab(d[var], d["mode"], normalize="index").reindex(columns=C.MODES, fill_value=0) * 100
    counts = d[var].value_counts()
    tab = tab.loc[[k for k in labels if k in tab.index]]
    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(tab) + 1.5))
    y = np.arange(len(tab))[::-1]
    left = np.zeros(len(tab))
    for m in C.MODES:
        vals = tab[m].to_numpy()
        ax.barh(y, vals, left=left, height=0.62, color=MODE_COLORS[m], edgecolor=SURFACE, linewidth=1.5)
        for yi, l, v in zip(y, left, vals):
            if v >= min_label:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                        color=_segment_label_color(MODE_COLORS[m]))
        left += vals
    ax.set_yticks(y, [f"{labels[k]}  (n={counts[k]})" for k in tab.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of respondents in the group (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_title(title)
    mode_legend(ax, ncol=6)
    save(fig, path)
    return tab.round(1).assign(n=[counts[k] for k in tab.index])


def mode_distribution(df, path):
    counts = df["Choice"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(6)
    bars = ax.bar(x, counts.to_numpy(), width=0.55, color=[MODE_COLORS[m] for m in C.MODES])
    total = counts.sum()
    for b, v in zip(bars, counts.to_numpy()):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}  ({100 * v / total:.1f}%)",
                ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, [C.MODE_LABELS[m] for m in C.MODES])
    ax.set_ylabel("Respondents")
    ax.set_title(f"Observed mode choice (n = {total})")
    ax.set_ylim(0, counts.max() * 1.18)
    save(fig, path)
    return pd.DataFrame({"mode": [C.MODE_LABELS[m] for m in C.MODES], "count": counts.to_numpy(),
                         "share_pct": (100 * counts / total).round(1).to_numpy()})


def los_vs_distance(raw, flagged_ids, path):
    """Small multiples: each mode's travel time against distance, raw data."""
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6), sharex=True)
    bad = raw["ID"].isin(flagged_ids)
    for ax, m in zip(axes.ravel(), C.MODES):
        col = C.TIME_COLS[m]
        ax.scatter(raw.loc[~bad, "Distance_km"], raw.loc[~bad, col], s=10, color=SERIES[0], alpha=0.55,
                   linewidths=0, label="Consistent with distance")
        ax.scatter(raw.loc[bad, "Distance_km"], raw.loc[bad, col].clip(lower=0.05), s=26, color=SERIES[1],
                   edgecolor=SURFACE, linewidths=1.2, label="Inconsistent (11 records)")
        plain_log(ax, "x"); plain_log(ax, "y")
        ax.set_title(C.MODE_LABELS[m], fontsize=9.5)
        ax.grid(axis="x", visible=True)
    for ax in axes[1]:
        ax.set_xlabel("Distance (km, log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Travel time (min, log)")
    axes[0, 0].legend(loc="upper left", fontsize=7.5, handletextpad=0.3)
    fig.suptitle("Recorded travel times against trip distance (raw workbook)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def _box_by_mode(values_by_mode, xlabel, title, path, log=True):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    modes = list(values_by_mode)
    data = [values_by_mode[m] for m in modes]
    pos = np.arange(len(modes))[::-1]
    bp = ax.boxplot(data, positions=pos, orientation="horizontal", widths=0.5, patch_artist=True,
                    showfliers=True, flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                                     markeredgewidth=0, alpha=0.6),
                    medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=AXIS),
                    capprops=dict(color=AXIS))
    for patch, m in zip(bp["boxes"], modes):
        patch.set_facecolor(MODE_COLORS[m]); patch.set_alpha(0.85); patch.set_edgecolor(SURFACE)
    ax.set_yticks(pos, [C.MODE_LABELS[m] for m in modes])
    if log:
        plain_log(ax, "x")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_title(title)
    save(fig, path)


def travel_time_distribution(df, path):
    vals = {m: df[C.TIME_COLS[m]].clip(lower=0.1).to_numpy() for m in C.MODES}
    _box_by_mode(vals, "Travel time (min, log scale)", "Travel time by mode, all respondents (cleaned)", path)
    return pd.DataFrame({C.MODE_LABELS[m]: df[C.TIME_COLS[m]].describe() for m in C.MODES}).T.round(1)


def travel_cost_distribution(df, path):
    vals = {m: df[C.COST_COLS[m]].clip(lower=0.5).to_numpy() for m in C.MODES if m in C.COST_COLS}
    _box_by_mode(vals, "Travel cost (INR, log scale)", "Travel cost by mode, all respondents", path)
    return pd.DataFrame({C.MODE_LABELS[m]: df[C.COST_COLS[m]].describe() for m in C.COST_COLS}).T.round(1)


def distance_by_choice(df, path, seed=C.SEED):
    d = _with_mode(df)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    pos = np.arange(6)[::-1]
    for p, m in zip(pos, C.MODES):
        x = d.loc[d["mode"] == m, "Distance_km"].to_numpy()
        ax.scatter(x, p + rng.uniform(-0.18, 0.18, len(x)), s=9, color=MODE_COLORS[m], alpha=0.6, linewidths=0)
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        ax.plot([q1, q3], [p, p], color=INK, linewidth=2.5, solid_capstyle="butt")
        ax.plot(med, p, marker="o", markersize=6, color=SURFACE, markeredgecolor=INK, markeredgewidth=1.5)
    ax.set_yticks(pos, [C.MODE_LABELS[m] for m in C.MODES])
    plain_log(ax, "x")
    ax.set_xlabel("Trip distance (km, log scale). Dot = median, bar = interquartile range")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_title("Trip distance by chosen mode")
    save(fig, path)
    return d.groupby("mode", observed=True)["Distance_km"].describe()[["count", "25%", "50%", "75%", "max"]].round(2)


def availability_vs_choice(df, path):
    rows = []
    for i, m in enumerate(C.MODES):
        av = df[C.AV_COLS[m]] == 1
        chose = df["Choice"] == i + 1
        rows.append({"mode": C.MODE_LABELS[m], "available_pct": 100 * av.mean(),
                     "chosen_pct": 100 * chose.mean(), "chosen_when_available_pct": 100 * chose[av].mean()})
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = np.arange(6)
    w = 0.26
    series = [("available_pct", "Available to respondent", SERIES[0]),
              ("chosen_when_available_pct", "Chosen, among those with it available", SERIES[1]),
              ("chosen_pct", "Chosen, all respondents", SERIES[2])]
    for k, (col, lab, colr) in enumerate(series):
        ax.bar(x + (k - 1) * (w + 0.02), tab[col], width=w, color=colr, label=lab)
    for xi, v in zip(x, tab["available_pct"]):
        ax.text(xi - w - 0.02, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, tab["mode"])
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 112)
    ax.set_title("Availability of each mode and how often it is chosen")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, path)
    return tab.round(1)


def association_with_choice(df, path):
    """Bias-corrected Cramer's V between each predictor and the chosen mode.
    Continuous variables are cut into quintiles first."""
    d = add_features(df)
    cands = ["Trip_Purpose", "Income", "Gender", "Age", "Education", "Occup", "ParkingSpace", "Area_Type",
             "Dwelling", "SAPT", "veh_own_cat", "Car_AV", "TwoW_AV", "Walk_AV", "Auto_AV", "Bus_AV",
             "Car_Ownership", "TwoW_Ownership", "HH_Size", "Working_Adults", "Distance_km"]
    rows = []
    for c in cands:
        x = d[c]
        if x.nunique() > 8:
            x = pd.qcut(x, 5, duplicates="drop")
        rows.append({"variable": c, "cramers_v": cramers_v(x, d["Choice"])})
    tab = pd.DataFrame(rows).sort_values("cramers_v")
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.barh(tab["variable"], tab["cramers_v"], height=0.6, color=SERIES[0])
    for yi, v in enumerate(tab["cramers_v"]):
        ax.text(v + 0.006, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_xlabel("Bias-corrected Cramér's V with chosen mode")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_title("Strength of association with mode choice")
    ax.set_xlim(0, tab["cramers_v"].max() * 1.15)
    save(fig, path)
    return tab.sort_values("cramers_v", ascending=False).round(3)


def numeric_correlation(df, path):
    cols = ["Distance_km"] + list(C.TIME_COLS.values()) + list(C.COST_COLS.values()) + \
           ["HH_Size", "Working_Adults", "Car_Ownership", "TwoW_Ownership"]
    corr = df[cols].corr(method="spearman")
    short = {"Distance_km": "Distance", "walk_time_min": "Walk time", "TwoW_time_min": "2W time",
             "auto_time_min": "Auto time", "car_time_min": "Car time", "bus_time_min": "Bus time",
             "local_train_time_min": "Rail time", "TC_2W": "2W cost", "TC_Auto": "Auto fare",
             "TC_Car": "Car cost", "TC_Bus": "Bus fare", "TC_LocalTrain": "Rail fare", "HH_Size": "HH size",
             "Working_Adults": "Workers", "Car_Ownership": "Cars owned", "TwoW_Ownership": "2Ws owned"}
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(corr.to_numpy(), cmap=BLUES, vmin=0, vmax=1)
    labels = [short[c] for c in cols]
    ax.set_xticks(range(len(cols)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 else INK_2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Spearman correlation (negative values shown on the same light end)")
    ax.set_title("Rank correlations among numeric variables (cleaned data)")
    save(fig, path)
    return corr.round(3)


def make_all(raw, clean, flagged_ids, figdir):
    """Produce every EDA figure. Returns a dict of the tables behind them."""
    apply_style()
    figdir.mkdir(parents=True, exist_ok=True)
    T = {}
    T["mode_distribution"] = mode_distribution(clean, figdir / "eda_mode_distribution.png")
    los_vs_distance(raw, flagged_ids, figdir / "audit_time_vs_distance.png")
    T["by_income"] = stacked_share(clean, "Income", {1: "Below 30k", 2: "30k-80k", 3: "Above 80k"},
                                   "Mode share by monthly household income (INR)", figdir / "eda_mode_by_income.png")
    T["by_age"] = stacked_share(clean, "Age", {1: "18-30", 2: "30-55", 3: "Above 55"},
                                "Mode share by age group", figdir / "eda_mode_by_age.png")
    T["by_occupation"] = stacked_share(clean, "Occup", {k: f"Occupation code {k}" for k in range(1, 6)},
                                       "Mode share by occupation code (codes as recorded)",
                                       figdir / "eda_mode_by_occupation.png")
    T["by_gender"] = stacked_share(clean, "Gender", {0: "Gender code 0", 1: "Gender code 1"},
                                   "Mode share by gender code (coding not documented)",
                                   figdir / "eda_mode_by_gender.png")
    T["by_purpose"] = stacked_share(clean, "Trip_Purpose", {1: "Purpose code 1", 2: "Purpose code 2"},
                                    "Mode share by trip purpose code", figdir / "eda_mode_by_purpose.png")
    T["by_vehicle"] = stacked_share(clean, "veh_own_cat", VEH_LABELS, "Mode share by household vehicle ownership",
                                    figdir / "eda_mode_by_vehicle_ownership.png")
    T["by_sapt"] = stacked_share(clean, "SAPT", {0: "Not satisfied", 1: "Satisfied"},
                                 "Mode share by satisfaction with public transport",
                                 figdir / "eda_mode_by_sapt.png")
    d = clean.assign(dist_band=pd.cut(clean["Distance_km"], [0, 1.5, 3, 5, 10, 20, 100],
                                      labels=["0-1.5", "1.5-3", "3-5", "5-10", "10-20", "20+"]))
    T["by_distance"] = stacked_share(d, "dist_band", {k: f"{k} km" for k in ["0-1.5", "1.5-3", "3-5", "5-10",
                                                                             "10-20", "20+"]},
                                     "Mode share by trip distance band", figdir / "eda_mode_by_distance_band.png")
    T["time_summary"] = travel_time_distribution(clean, figdir / "eda_travel_time.png")
    T["cost_summary"] = travel_cost_distribution(clean, figdir / "eda_travel_cost.png")
    T["distance_by_choice"] = distance_by_choice(clean, figdir / "eda_distance_by_choice.png")
    T["availability"] = availability_vs_choice(clean, figdir / "eda_availability_vs_choice.png")
    T["association"] = association_with_choice(clean, figdir / "eda_association_with_choice.png")
    T["numeric_corr"] = numeric_correlation(clean, figdir / "eda_numeric_correlation.png")
    return T
