"""Figures for model comparison, synthetic data validation, scenarios and the
optimisation. EDA figures live in eda.py."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .plot_style import (AXIS, BLUES, DIVERGING, GRID, INK, INK_2, MODE_COLORS, MUTED, SERIES, SURFACE,
                         apply_style, mode_legend, plain_log, save)
from .synthetic_generation import SYNTH_VARS

__all__ = ["apply_style"]
SHORT = {m: C.MODE_LABELS[m] for m in C.MODES}
SHORT.update({"twow": "2W", "auto": "Auto", "train": "Rail"})


def cv_comparison(cv_sum, path):
    metrics = [("macro_f1", "Macro-F1 (higher is better)"), ("balanced_accuracy", "Balanced accuracy"),
               ("log_loss", "Log loss (lower is better)")]
    models = list(cv_sum.index)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, (m, title) in zip(axes, metrics):
        mean, se = cv_sum[f"{m}_mean"].to_numpy(), cv_sum[f"{m}_se"].to_numpy()
        ax.hlines(y, mean - 1.96 * se, mean + 1.96 * se, color=SERIES[0], linewidth=2)
        ax.plot(mean, y, "o", color=SERIES[0], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        for yi, v in zip(y, mean):
            ax.text(v, yi + 0.28, f"{v:.3f}", ha="center", fontsize=7.5, color=INK_2)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        ax.set_ylim(-0.6, len(models) - 0.3)
    axes[0].set_yticks(y, models)
    fig.suptitle("Cross-validated performance on the training partition (5 x 5 folds, mean and 95% interval)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def confusion_matrices(cms, path):
    names = list(cms)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    labels = [SHORT[m] for m in C.MODES]
    for ax, name in zip(axes.ravel(), names):
        cm = np.asarray(cms[name], float)
        rown = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rown, cmap=BLUES, vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if cm[i, j] > 0:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if rown[i, j] > 0.55 else INK_2)
        ax.set_xticks(range(6), labels, rotation=45, ha="right")
        ax.set_yticks(range(6), labels)
        ax.grid(False)
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{name}  (accuracy {acc:.2f})", fontsize=9.5)
        ax.set_xlabel("Predicted", fontsize=8.5)
        ax.set_ylabel("Observed", fontsize=8.5)
    fig.suptitle("Held-out test set confusion matrices (counts; shading = share of the observed row)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def calibration(y, oof, models, path):
    """One-vs-rest reliability curves from pooled out-of-fold predictions."""
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2), sharex=True, sharey=True)
    yy = np.tile(y, oof[models[0]].shape[0])
    for k, name in enumerate(models):
        P = oof[name].reshape(-1, len(C.MODES))
        for ax, (j, m) in zip(axes.ravel(), enumerate(C.MODES)):
            pj = P[:, j]
            edges = np.quantile(pj[pj > 0], np.linspace(0, 1, 9)) if (pj > 0).sum() > 50 else np.linspace(0, 1, 9)
            edges = np.unique(edges)
            idx = np.clip(np.searchsorted(edges, pj, side="right") - 1, 0, len(edges) - 2)
            xs, ys = [], []
            for b in range(len(edges) - 1):
                sel = (idx == b) & (pj > 0)
                if sel.sum() >= 10:
                    xs.append(pj[sel].mean()); ys.append((yy[sel] == j).mean())
            ax.plot(xs, ys, "-o", color=SERIES[k], markersize=4, linewidth=1.8, label=name)
            if k == 0:
                ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
                ax.set_title(C.MODE_LABELS[m], fontsize=9.5)
                ax.grid(axis="x", visible=True)
    for ax in axes[1]:
        ax.set_xlabel("Mean predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed frequency")
    axes[0, 0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Reliability of out-of-fold probabilities (training CV, 8 quantile bins per mode)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def readiness(ready_df, path):
    levers = list(dict.fromkeys(ready_df["lever"]))
    models = list(dict.fromkeys(ready_df["model"]))
    fig, axes = plt.subplots(1, len(levers), figsize=(11, 3.4), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, lev in zip(axes, levers):
        d = ready_df[ready_df["lever"] == lev].set_index("model").loc[models]
        vals = d["mean_change_pp"].to_numpy()
        ax.barh(y, vals, height=0.55, color=[SERIES[0] if p else SERIES[1] for p in d["passes"]])
        ax.axvline(0, color=AXIS, linewidth=1)
        lim = max(0.5, np.abs(vals).max() * 1.15)
        for yi, v, w in zip(y, vals, d["share_wrong_direction"]):
            # labels always sit right of the bar end or of zero, never over the axis labels
            ax.text(max(v, 0) + 0.03 * lim, yi + 0.3, f"{v:+.2f}" + (f" ({100 * w:.0f}% wrong)" if w > 0.005 else ""),
                    va="center", ha="left", fontsize=7, color=INK_2)
        ax.set_title(f"{lev} ({'-' if 'car' not in lev else '+'}{C.READINESS_SHOCK_PCT:.0f}%)", fontsize=9.5)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        ax.set_xlim(-lim, lim * 2.2)
    axes[0].set_yticks(y, models)
    fig.text(0.5, -0.02, "Mean change in the probability of the affected mode (percentage points, training rows "
             "where the mode is available). Blue = passes the check, orange = fails.", ha="center",
             fontsize=8.5, color=INK_2)
    fig.suptitle("Simulation-readiness check: response to a 20% change in each policy lever",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def coefficients(summary, path):
    """Time and cost are per minute / per rupee, so they get their own panel."""
    s = summary[~summary.index.str.startswith("ASC")]
    los = s[s.index.isin(["B_time", "B_walk_time", "B_cost"])]
    other = s[~s.index.isin(los.index)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.4 * max(len(other), 3) + 1.4),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    for ax, d, xl in ((axes[0], los, "per minute (time) or per INR (cost)"),
                      (axes[1], other, "utility units")):
        y = np.arange(len(d))[::-1]
        lo, hi = d["estimate"] - 1.96 * d["std_error"], d["estimate"] + 1.96 * d["std_error"]
        ax.hlines(y, lo, hi, color=SERIES[0], linewidth=2)
        ax.plot(d["estimate"], y, "o", color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=1.5)
        for yi, v in zip(y, d["estimate"]):
            ax.text(v, yi + 0.22, f"{v:.3f}", ha="center", fontsize=7.5, color=INK_2)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(y, d.index)
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        ax.set_xlabel(f"Estimate and 95% CI ({xl})")
    axes[0].set_title("Level-of-service terms", fontsize=9.5)
    axes[1].set_title("Person and trip terms", fontsize=9.5)
    fig.suptitle("Selected MNL: estimated coefficients (constants not shown)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def synthetic_marginals(real, synth, path):
    vars_ = [v for v in SYNTH_VARS if v != "Distance_km"]
    ncol = 5
    nrow = int(np.ceil(len(vars_) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.2 * nrow))
    for ax, v in zip(axes.ravel(), vars_):
        pr = real[v].value_counts(normalize=True)
        ps = synth[v].value_counts(normalize=True)
        idx = sorted(pr.index.union(ps.index))
        x = np.arange(len(idx))
        ax.bar(x - 0.2, pr.reindex(idx, fill_value=0) * 100, width=0.38, color=SERIES[0], label="Real (train)")
        ax.bar(x + 0.2, ps.reindex(idx, fill_value=0) * 100, width=0.38, color=SERIES[1], label="Synthetic")
        ax.set_xticks(x, [str(int(i)) for i in idx], fontsize=7)
        ax.set_title(v, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.ravel()[len(vars_):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Category frequencies (%): training data versus synthetic population",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(handles, labels, ncol=2, loc="upper right", bbox_to_anchor=(0.99, 1.0), fontsize=8)
    save(fig, path)


def synthetic_distance(real, synth, path):
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for d, lab, col in ((real, "Real (train)", SERIES[0]), (synth, "Synthetic", SERIES[1])):
        x = np.sort(d["Distance_km"].to_numpy())
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col, linewidth=2, label=lab)
    plain_log(ax, "x")
    ax.set_xlabel("Trip distance (km, log scale)")
    ax.set_ylabel("Cumulative share")
    ax.grid(axis="x", visible=True)
    ax.legend(loc="upper left")
    ax.set_title("Distance distribution: real versus synthetic (empirical CDF)")
    save(fig, path)


def synthetic_associations(a_real, a_syn, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    labels = list(a_real.index)
    for ax, M, title, cmap, vmin in ((axes[0], a_real, "Real (train)", BLUES, 0),
                                     (axes[1], a_syn, "Synthetic", BLUES, 0),
                                     (axes[2], a_syn - a_real, "Synthetic minus real", DIVERGING, -0.5)):
        im = ax.imshow(M.to_numpy(), cmap=cmap, vmin=vmin, vmax=1 if vmin == 0 else 0.5)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
        ax.grid(False)
        ax.set_title(title, fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Pairwise association (bias-corrected Cramér's V; distance in six bands)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def synthetic_pairs(real, synth, path):
    """Three relationships the brief asks to preserve."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    # income vs car ownership
    for k, (d, lab, col) in enumerate(((real, "Real", SERIES[0]), (synth, "Synthetic", SERIES[1]))):
        share = d.groupby("Income")["Car_Ownership"].apply(lambda s: (s > 0).mean() * 100)
        axes[0].bar(np.arange(3) + (k - 0.5) * 0.38, share.reindex([1, 2, 3]).to_numpy(), width=0.36,
                    color=col, label=lab)
        sizes = real["HH_Size"].value_counts()
        keep = sizes[sizes >= 5].index          # larger households have one or two real records
        mean_ch = d[d["HH_Size"].isin(keep)].groupby("HH_Size")["Children"].mean()
        axes[1].plot(mean_ch.index, mean_ch.to_numpy(), "-o", color=col, markersize=4, label=lab)
        walk = d.groupby(pd.cut(d["Distance_km"], [0, 1.5, 2, 3, 5, 10, 100]), observed=True)["Walk_AV"].mean() * 100
        axes[2].plot(range(len(walk)), walk.to_numpy(), "-o", color=col, markersize=4, label=lab)
    axes[0].set_xticks(range(3), ["<30k", "30k-80k", ">80k"])
    axes[0].set_ylabel("Households owning a car (%)")
    axes[0].set_title("Income and car ownership", fontsize=9.5)
    axes[1].set_xlabel("Household size"); axes[1].set_ylabel("Mean number of children")
    axes[1].set_title("Household size and children (sizes with 5+ real records)", fontsize=9.5)
    axes[2].set_xticks(range(6), ["<=1.5", "1.5-2", "2-3", "3-5", "5-10", "10+"], fontsize=7.5)
    axes[2].set_xlabel("Distance band (km)"); axes[2].set_ylabel("Walking available (%)")
    axes[2].set_title("Distance and walking availability", fontsize=9.5)
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, path)


def baseline_shares(comp, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(comp))
    series = [("observed_train_pct", "Observed (training data)", SERIES[0]),
              ("synthetic_expected_pct", "Synthetic population: expected share", SERIES[1]),
              ("synthetic_hard_pct", "Synthetic population: most-likely-mode share", SERIES[2])]
    for k, (col, lab, c) in enumerate(series):
        ax.bar(x + (k - 1) * 0.27, comp[col], width=0.25, color=c, label=lab)
    ax.set_xticks(x, comp["mode"])
    ax.set_ylabel("Share (%)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    ax.set_title("Baseline mode shares: observed sample versus predicted synthetic population")
    save(fig, path)


def sensitivity_lines(cost, time, path):
    """Change in expected sustainable share for one-lever scenarios."""
    def pick(t, prefix):
        d = t[t["scenario"].str.fullmatch(prefix.replace("+", r"\+") + r"\d+%")]
        lv = d["scenario"].str.extract(r"([+-]?\d+)%")[0].astype(float).abs()
        return np.r_[0, lv], np.r_[0, d["sustainable_change_pp"].to_numpy()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    groups = [(axes[0], cost, [("Bus fare -", "Bus fare reduction", SERIES[0]),
                               ("Rail fare -", "Rail fare reduction", SERIES[1]),
                               ("Car cost +", "Car cost increase", SERIES[2])], "Fare / cost change (%)"),
              (axes[1], time, [("Bus time -", "Bus time reduction", SERIES[0]),
                               ("Rail time -", "Rail time reduction", SERIES[1]),
                               ("Car time +", "Car time increase", SERIES[2])], "Travel time change (%)")]
    for ax, t, items, xl in groups:
        for prefix, lab, col in items:
            x, yv = pick(t, prefix)
            ax.plot(x, yv, "-o", color=col, markersize=4, label=lab)
            ax.text(x[-1], yv[-1], f"  {yv[-1]:+.2f}", va="center", fontsize=7.5, color=INK_2)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xlabel(xl)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(axis="x", visible=True)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    axes[0].set_ylabel("Change in expected sustainable share (pp)")
    fig.suptitle("One-lever sensitivity of the sustainable share (synthetic population, frozen model)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def scenario_stacks(t, path, title="Mode shares under combined policy scenarios"):
    fig, ax = plt.subplots(figsize=(8.6, 0.6 * len(t) + 1.6))
    y = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for m in C.MODES:
        v = t[f"{m}_pct"].to_numpy()
        ax.barh(y, v, left=left, height=0.62, color=MODE_COLORS[m], edgecolor=SURFACE, linewidth=1.5)
        for yi, l, vv in zip(y, left, v):
            if vv >= 8:
                ax.text(l + vv / 2, yi, f"{vv:.0f}", ha="center", va="center", fontsize=7.5,
                        color=INK if MODE_COLORS[m] == "#eda100" else "white")
        left += v
    names = [s if len(s) < 48 else s[:45] + "..." for s in t["scenario"]]
    ax.set_yticks(y, [f"{n}  [{s:.1f}%]" for n, s in zip(names, t["expected_sustainable_share_pct"])], fontsize=8)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("Expected mode share (%). Bracket = expected sustainable share")
    ax.set_title(title)
    mode_legend(ax)
    save(fig, path)


def optimisation_shares(shares, path):
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    x = np.arange(len(shares))
    ax.bar(x - 0.2, shares["baseline_expected_pct"], width=0.38, color=SERIES[0], label="Baseline")
    ax.bar(x + 0.2, shares["optimised_expected_pct"], width=0.38, color=SERIES[1], label="Optimised policy")
    for xi, b, o in zip(x, shares["baseline_expected_pct"], shares["optimised_expected_pct"]):
        ax.text(xi + 0.2, o + 0.5, f"{o - b:+.1f}", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, shares["mode"])
    ax.set_ylabel("Expected share (%)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Expected mode shares: baseline versus optimised policy (labels = change in pp)")
    save(fig, path)


def optimisation_benchmarks(bench, path):
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(bench) + 1.2))
    d = bench.copy()
    y = np.arange(len(d))[::-1]
    cols = [SERIES[1] if p == "Optimised policy" else SERIES[0] for p in d["policy"]]
    ax.barh(y, d["sustainable_share_pct"], height=0.55, color=cols)
    lo = d["sustainable_share_pct"].min()
    for yi, v in zip(y, d["sustainable_share_pct"]):
        ax.text(v + 0.1, yi, f"{v:.2f}%", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, d["policy"], fontsize=8)
    ax.set_xlim(lo - 3, d["sustainable_share_pct"].max() + 2.5)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("Expected sustainable share (%)")
    ax.set_title("Optimised policy against benchmark policies")
    save(fig, path)


def effort_frontier(budget_df, path):
    from .policy_optimisation import LEVERS
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    axes[0].plot(budget_df["effort_budget"], budget_df["sustainable_share_pct"], "-o", color=SERIES[0],
                 markersize=5)
    axes[0].set_xlabel("Effort budget (sum of levers as a share of their bound)")
    axes[0].set_ylabel("Best expected sustainable share (%)")
    axes[0].set_title("Best achievable share for each effort budget", fontsize=9.5)
    axes[0].grid(axis="x", visible=True)
    x = np.arange(len(budget_df))
    left = np.zeros(len(budget_df))
    for k, lev in enumerate(LEVERS):
        share = budget_df[lev] / C.POLICY_BOUNDS[lev][1]
        axes[1].bar(x, share, bottom=left, color=SERIES[k], width=0.6, edgecolor=SURFACE, linewidth=1,
                    label=C.POLICY_LABELS[lev].replace(" (%)", ""))
        left += share
    axes[1].set_xticks(x, [f"{b:g}" for b in budget_df["effort_budget"]])
    axes[1].set_xlabel("Effort budget")
    axes[1].set_ylabel("Effort used per lever")
    axes[1].set_title("How the optimiser spends the budget", fontsize=9.5)
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, path)
