"""Run the whole project from the raw workbook to the optimised policy.

    python run_pipeline.py            # full run (results used in the report)
    python run_pipeline.py --fast     # quick smoke run with smaller searches
    python run_pipeline.py --root D   # write every output under folder D

Phases follow the brief: audit -> cleaning/EDA -> model comparison -> model
selection and freezing -> synthetic population -> baseline prediction ->
scenarios -> optimisation.
"""
import argparse
import json
import platform
import shutil
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from src import config as C
from src import choice_models as CM
from src import data_validation as V
from src import eda
from src import feature_engineering as FE
from src import figures as FIG
from src import model_evaluation as ME
from src import model_training as MT
from src import policy_optimisation as PO
from src import prediction as PR
from src import preprocessing as PP
from src import scenario_simulation as SS
from src import synthetic_generation as SG

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _json(obj, path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.DataFrame, pd.Series)):
            return o.to_dict()
        if isinstance(o, Path):
            return str(o)
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, default=default))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(root=C.ROOT, fast=False, n_synth=None):
    paths = C.Paths(Path(root))
    paths.make_dirs()
    R, F = paths.results, paths.figures
    summary = {"fast_mode": fast, "seed": C.SEED, "cv_seed": C.CV_SEED, "synth_seed": C.SYNTH_SEED}
    t_start = time.time()

    # ------------------------------------------------------------------ phase 1
    log("Phase 1: audit")
    raw = V.load_raw()
    audit = V.run_audit(raw)
    rules = audit["los_rules"]
    _json({k: audit[k] for k in ["shape", "missing_values", "duplicate_rows", "duplicate_rows_ignoring_id",
                                 "id_unique", "id_range", "duplicate_columns", "constant_columns", "availability",
                                 "household_flag_counts", "target_counts"]} | {"los_rules": rules.to_dict()},
          R / "audit_summary.json")
    audit["code_check"].to_csv(R / "audit_code_check.csv", index=False)
    audit["outliers"].to_csv(R / "audit_outliers.csv", index=False)
    audit["los_inconsistent"].to_csv(R / "audit_los_inconsistent_records.csv", index=False)
    audit["cost_rule_match"].to_csv(R / "audit_cost_rule_match.csv", index=False)
    hh = audit["household_flags"]
    hh[hh.drop(columns="ID").any(axis=1)].to_csv(R / "audit_household_flags.csv", index=False)
    for k, t in audit["category_evidence"].items():
        t.to_csv(R / f"audit_evidence_{k}.csv")
    ddict = V.build_data_dictionary(raw)
    ddict.to_csv(paths.processed / "data_dictionary.csv", index=False)
    summary["audit"] = {"n_time_inconsistent": int(len(audit["los_inconsistent"])),
                        "household_flags": audit["household_flag_counts"]}

    # ------------------------------------------------------------------ phase 2
    log("Phase 2: cleaning, features and EDA")
    clean, corrections, rules = PP.clean_data(raw)
    problems = PP.validate_clean(clean, rules)
    if problems:
        raise RuntimeError(f"cleaned data failed validation: {problems}")
    clean.to_csv(paths.processed / "clean_data.csv", index=False)
    corrections.to_csv(R / "cleaning_time_corrections.csv", index=False)
    FE.candidate_feature_review(clean).to_csv(R / "feature_review.csv", index=False)
    flagged = clean.loc[clean["los_time_corrected"] == 1, "ID"].tolist()
    tables = eda.make_all(raw, clean, flagged, F)
    for k, t in tables.items():
        t.to_csv(R / f"eda_{k}.csv")
    train, test = PP.split_train_test(clean)
    train.to_csv(paths.processed / "train.csv", index=False)
    test.to_csv(paths.processed / "test.csv", index=False)
    CM.to_long(train).to_csv(paths.processed / "train_long_format.csv", index=False)
    y_tr, y_te = PP.target(train), PP.target(test)
    summary["split"] = {"train": len(train), "test": len(test),
                        "train_counts": np.bincount(y_tr).tolist(), "test_counts": np.bincount(y_te).tolist()}

    # ------------------------------------------------------------------ phase 3
    log("Phase 3: discrete choice estimation and hyperparameter tuning")
    spec_rows = []
    for spec in CM.SPEC_NAMES + CM.REFERENCE_SPECS:
        m = CM.MultinomialLogit(spec).fit(train, y_tr)
        spec_rows.append({"specification": spec, "behavioural_candidate": spec in CM.SPEC_NAMES,
                          "signs_ok": CM.signs_ok(m), **m.fit_stats(), **CM.value_of_time(m)})
        m.summary_.to_csv(R / f"mnl_estimates_{spec.replace('+', '_')}.csv")
    pd.DataFrame(spec_rows).to_csv(R / "mnl_specifications.csv", index=False)
    profile, _ = CM.walk_weight_profile(train, y_tr)
    profile.to_csv(R / "mnl_walk_weight_profile.csv", index=False)

    cv_tune = MT.tuning_cv(repeats=1 if fast else C.TUNE_REPEATS)
    cands = MT.candidates()
    if fast:   # shrink the searches for the smoke run
        cands["Decision tree"] = (cands["Decision tree"][0], {"estimator__clf__max_depth": [3, 5],
                                                              "balanced": [False, True]}, "grid", None)
        for k in ("Random forest", "XGBoost"):
            est, space, _, _ = cands[k]
            cands[k] = (est, space, "random", 4)
    tuned, tune_rows = {}, []
    for name, (est, space, kind, n_iter) in cands.items():
        t0 = time.time()
        s = MT.tune(name, est, space, kind, n_iter, train, y_tr, cv_tune)
        tuned[name] = s.best_estimator_
        tune_rows.append({"model": name, "search": kind, "configurations": len(s.cv_results_["params"]),
                          "best_cv_macro_f1": s.best_score_, "best_params": json.dumps(s.best_params_, default=str),
                          "seconds": round(time.time() - t0, 1)})
        log(f"  tuned {name}: macro-F1 {s.best_score_:.4f}  {s.best_params_}")
    spec = tuned["MNL"].spec
    nest_choice, nest_table, nest_fits, _ = MT.select_nest_structure(train, y_tr, spec, list(CM.NEST_STRUCTURES))
    nest_table.to_csv(R / "nested_logit_structures.csv", index=False)
    if nest_choice is None:
        nest_choice = nest_table.sort_values("log_likelihood", ascending=False)["structure"].iloc[0]
        summary["nested_logit_note"] = "no structure met the usability rule; best-LL structure kept for comparison"
    tuned["Nested logit"] = CM.NestedLogit(spec=spec, nests=nest_choice).fit(train, y_tr)
    tuned["Nested logit"].summary_.to_csv(R / "nested_logit_estimates.csv")
    tuned["MNL"].summary_.to_csv(R / "mnl_estimates_selected.csv")
    tune_rows.append({"model": "Nested logit", "search": "rule", "configurations": len(CM.NEST_STRUCTURES),
                      "best_cv_macro_f1": np.nan, "best_params": json.dumps({"spec": spec, "nests": nest_choice}),
                      "seconds": np.nan})
    pd.DataFrame(tune_rows).to_csv(R / "tuning_results.csv", index=False)
    summary["mnl_spec"] = spec
    summary["nest_structure"] = nest_choice
    summary["mnl_fit"] = tuned["MNL"].fit_stats()
    summary["nl_fit"] = tuned["Nested logit"].fit_stats()
    summary["vot"] = CM.value_of_time(tuned["MNL"])

    log("Phase 3: repeated cross-validated comparison")
    order = ["MNL", "Nested logit", "Logistic regression", "Decision tree", "Random forest", "XGBoost"]
    REF = "Reference: constants only"      # market shares within each choice set; not a candidate
    models = {k: tuned[k] for k in order}
    ref_model = CM.MultinomialLogit("ASC").fit(train, y_tr)
    folds, oof = ME.cross_validate_models({**models, REF: ref_model}, train, y_tr,
                                          repeats=1 if fast else C.CV_REPEATS)
    folds.to_csv(R / "cv_fold_metrics.csv", index=False)
    cv_all = ME.summarise_cv(folds).loc[order + [REF]]
    cv_all.to_csv(R / "cv_summary.csv")
    cv_sum = cv_all.loc[order]

    test_rows, per_class, cms, test_probs, cis = [], {}, {}, {}, {}
    for name, m in {**models, REF: ref_model}.items():
        res, ci, pc, cm, P = ME.evaluate_on_test(m, test, y_te, n_boot=200 if fast else 1000)
        test_rows.append({"model": name, **res})
        per_class[name], cms[name], test_probs[name], cis[name] = pc, cm, P, ci
        pc.to_csv(R / f"test_per_class_{name.replace(' ', '_').lower()}.csv", index=False)
    test_df = pd.DataFrame(test_rows).set_index("model").loc[order + [REF]]
    test_df.to_csv(R / "test_metrics.csv")
    cms.pop(REF)
    _json(cis, R / "test_bootstrap_ci.json")
    _json({k: v.tolist() for k, v in cms.items()}, R / "test_confusion_matrices.json")

    readiness, ready_rows = {}, []
    for name, m in models.items():
        ok, tab = ME.simulation_readiness(m, train)
        readiness[name] = ok
        ready_rows.append(tab.assign(model=name))
    ready_df = pd.concat(ready_rows, ignore_index=True)
    ready_df.to_csv(R / "simulation_readiness.csv", index=False)

    # ------------------------------------------------------------------ phase 4
    log("Phase 4: selection and freezing")
    winner, sel_log = ME.select_model(cv_sum, readiness)
    sel_log["readiness"] = readiness
    _json(sel_log, R / "model_selection.json")
    log(f"  selected model: {winner}")
    best = models[winner]
    for d in (paths.best_model, paths.preprocessing):
        for f in d.glob("*"):
            f.unlink()
    joblib.dump(best, paths.best_model / "model.joblib")
    cand_dir = paths.root / "models" / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        joblib.dump(m, cand_dir / f"{name.replace(' ', '_').lower()}.joblib")
    card = {
        "selected_model": winner,
        "trained_on": "training partition only (80%, stratified, seed 42); never refitted on test or synthetic data",
        "n_train": len(train), "n_test": len(test), "seed": C.SEED,
        "hyperparameters": {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
                            for k, v in best.get_params().items() if "__" not in k},
        "cv_summary": cv_sum.loc[winner].to_dict(), "test_metrics": test_df.loc[winner].to_dict(),
        "selection_log": sel_log,
        "classes": {i: C.MODE_LABELS[m] for i, m in enumerate(C.MODES)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    if winner in ("MNL", "Nested logit"):
        card["utility_specification"] = {n: alts for n, alts in best.terms_}
        card["walk_time_weight"] = C.WALK_TIME_WEIGHT
        card["estimates"] = best.summary_.round(6).to_dict(orient="index")
    _json(card, paths.best_model / "model_card.json")
    _json({"dropped_columns": PP.DROP_COLUMNS, "los_rules": rules.to_dict(),
           "time_corrections": corrections.to_dict(orient="records"),
           "categorical_features": FE.CATEGORICAL, "numeric_features": FE.NUMERIC,
           "raw_predictors": FE.RAW_PREDICTORS, "rare_occupation_codes_pooled": list(FE.RARE_OCCUPATION_CODES)},
          paths.preprocessing / "preprocessing_config.json")
    joblib.dump({"feature_builder": FE.FeatureBuilder(), "los_rules": rules}, paths.preprocessing / "pipeline.joblib")
    summary["selected_model"] = winner
    summary["selection"] = sel_log

    FIG.apply_style()
    FIG.cv_comparison(cv_sum, F / "model_cv_comparison.png")
    FIG.confusion_matrices(cms, F / "model_confusion_matrices.png")
    rival = cv_sum.drop(index=winner)["log_loss_mean"].idxmin()     # best-calibrated other model
    FIG.calibration(y_tr, oof, [winner, rival], F / "model_calibration.png")
    FIG.readiness(ready_df, F / "model_simulation_readiness.png")
    FIG.coefficients(tuned["MNL"].summary_, F / "mnl_coefficients.png")
    rel = []
    for name in order:
        rt = ME.reliability_table(np.tile(y_tr, oof[name].shape[0]), oof[name].reshape(-1, len(C.MODES)))
        rel.append(rt.assign(model=name))
    pd.concat(rel).to_csv(R / "cv_reliability.csv", index=False)

    # ------------------------------------------------------------------ phase 5
    log("Phase 5: synthetic population")
    n = n_synth or (1000 if fast else C.SYNTH_N)
    SG.cart_leaf_sensitivity(train, test, rules, leaves=(5, 10) if fast else (5, 10, 15, 20),
                             n=n).to_csv(R / "synthetic_cart_leaf_sensitivity.csv", index=False)
    method_table, pops = SG.compare_methods(train, test, rules, n=n)
    method, limit = SG.choose_method(method_table)
    method_table.to_csv(R / "synthetic_method_comparison.csv", index=False)
    pop = pops[method]
    pop.to_csv(paths.synthetic / "synthetic_population.csv", index=False)
    marg = SG.marginal_fidelity(train, pop)
    marg.to_csv(R / "synthetic_marginal_fidelity.csv", index=False)
    mom = SG.moments_table(train, pop)
    mom.to_csv(R / "synthetic_moments.csv", index=False)
    a_real, a_syn = SG.association_matrix(train), SG.association_matrix(pop)
    a_real.to_csv(R / "synthetic_assoc_real.csv")
    a_syn.to_csv(R / "synthetic_assoc_synthetic.csv")
    holdout_marg = SG.marginal_fidelity(test, pop)
    holdout_marg.to_csv(R / "synthetic_marginal_fidelity_vs_test.csv", index=False)
    summary["synthetic"] = {"method": method, "n": n, "copy_limit": limit,
                            "table": method_table.set_index("method").to_dict(orient="index")}
    FIG.synthetic_marginals(train, pop, F / "synthetic_marginals.png")
    FIG.synthetic_distance(train, pop, F / "synthetic_distance.png")
    FIG.synthetic_associations(a_real, a_syn, F / "synthetic_associations.png")
    FIG.synthetic_pairs(train, pop, F / "synthetic_pairwise.png")

    # ------------------------------------------------------------------ phase 6
    log("Phase 6: baseline prediction for the synthetic population")
    frozen, _ = PR.load_frozen(paths)
    P = PR.predict_probabilities(frozen, pop)
    out = pd.concat([pop[["ID"]], P], axis=1)
    out["P_sustainable"] = PR.sustainable_probability(P)
    out["predicted_mode"] = [C.MODE_LABELS[C.MODES[i]] for i in P.to_numpy().argmax(axis=1)]
    out.to_csv(paths.synthetic / "synthetic_predictions.csv", index=False)
    base_table, base_sus = PR.summarise(P)
    base_table.to_csv(R / "baseline_shares_synthetic.csv", index=False)
    obs = pd.Series(y_tr).value_counts(normalize=True).sort_index().to_numpy() * 100
    train_pred = PR.summarise(PR.predict_probabilities(frozen, train))[0]
    comp = pd.DataFrame({"mode": base_table["mode"], "observed_train_pct": obs,
                         "model_on_train_pct": train_pred["expected_share_pct"],
                         "synthetic_expected_pct": base_table["expected_share_pct"],
                         "synthetic_hard_pct": base_table["hard_share_pct"]})
    comp.to_csv(R / "baseline_share_comparison.csv", index=False)
    summary["baseline"] = base_sus | {"observed_train_sustainable_pct": float(obs[[0, 4, 5]].sum())}
    FIG.baseline_shares(comp, F / "baseline_shares.png")

    # ------------------------------------------------------------------ phase 7
    log("Phase 7: scenario experiments")
    exps = {"cost": SS.experiment_cost, "time": SS.experiment_time,
            "availability": SS.experiment_availability, "combined": SS.experiment_combined}
    exp_tables = {}
    for k, fn in exps.items():
        t = fn(frozen, pop)
        t.to_csv(R / f"scenario_{k}.csv", index=False)
        exp_tables[k] = t
    nochange = SS.simulate(frozen, pop)[0]
    summary["no_change_max_abs_diff"] = float(np.abs(nochange.to_numpy() - P.to_numpy()).max())
    FIG.sensitivity_lines(exp_tables["cost"], exp_tables["time"], F / "scenario_sensitivity.png")
    FIG.scenario_stacks(exp_tables["combined"], F / "scenario_combined.png")
    FIG.scenario_stacks(exp_tables["availability"], F / "scenario_availability.png",
                        title="Mode shares under availability scenarios (extrapolations)")

    # ------------------------------------------------------------------ phase 8
    log("Phase 8: policy optimisation")
    obj = PO.Objective(frozen, pop)
    step = 10.0 if fast else C.GRID_STEP
    g_best, g_table, g_secs = PO.grid_search(obj, step=step)
    g_table.to_csv(R / "optimisation_grid.csv", index=False)
    de_x, de_val, de_res, de_secs = PO.de_search(obj, maxiter=30 if fast else 200)
    grid_x = g_best[PO.LEVERS].to_numpy(float)
    best_x, best_val, source = (de_x, de_val, "differential evolution") if de_val >= g_best["objective"] - 1e-12 \
        else (grid_x, float(g_best["objective"]), "grid search")
    levers = dict(zip(PO.LEVERS, map(float, best_x)))
    fresh_model, _ = PR.load_frozen(paths)
    fresh_pop = pd.read_csv(paths.synthetic / "synthetic_population.csv")
    verification = PO.verify_solution(levers, best_val, fresh_model, fresh_pop)
    bench = PO.compare_with_benchmarks(obj, best_x)
    bench.to_csv(R / "optimisation_benchmarks.csv", index=False)
    opt_table, opt_sus = PO.shares_for(obj, best_x)
    shares = pd.DataFrame({"mode": base_table["mode"], "baseline_expected_pct": base_table["expected_share_pct"],
                           "optimised_expected_pct": opt_table["expected_share_pct"],
                           "baseline_hard_pct": base_table["hard_share_pct"],
                           "optimised_hard_pct": opt_table["hard_share_pct"]})
    shares["change_expected_pp"] = shares["optimised_expected_pct"] - shares["baseline_expected_pct"]
    shares.to_csv(R / "optimisation_mode_shares.csv", index=False)

    budget_df = PO.budget_frontier(g_table, (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0))
    budget_df.to_csv(R / "optimisation_effort_budget.csv", index=False)
    budget_check = []
    for B in (1.0, 2.0):      # differential evolution as an independent check of the grid under a budget
        xb, vb, _, _ = PO.de_search(obj, budget=B, maxiter=30 if fast else 200)
        gb = budget_df.loc[budget_df["effort_budget"] == B].iloc[0]
        budget_check.append({"effort_budget": B, "grid_best_pct": gb["sustainable_share_pct"],
                             "de_best_pct": 100 * vb, "de_effort": PO.effort(xb),
                             **{f"de_{k}": float(v) for k, v in zip(PO.LEVERS, xb)}})
    pd.DataFrame(budget_check).to_csv(R / "optimisation_budget_de_check.csv", index=False)
    if winner in ("MNL", "Nested logit"):
        PO.walk_weight_sensitivity(best, train, y_tr, pop).to_csv(R / "sensitivity_walk_weight.csv", index=False)

    opt = {"optimal_levers": levers, "objective_sustainable_share_pct": 100 * best_val, "found_by": source,
           "grid": {"step": step, "evaluations": len(g_table), "seconds": round(g_secs, 1),
                    "best_levers": {k: float(g_best[k]) for k in PO.LEVERS},
                    "best_objective_pct": 100 * float(g_best["objective"])},
           "differential_evolution": {"levers": dict(zip(PO.LEVERS, map(float, de_x))), "objective_pct": 100 * de_val,
                                      "evaluations": int(de_res.nfev), "seconds": round(de_secs, 1)},
           "baseline_sustainable_pct": base_sus["expected_sustainable_share_pct"],
           "improvement_pp": 100 * best_val - base_sus["expected_sustainable_share_pct"],
           "hard_sustainable": {"baseline_pct": base_sus["hard_sustainable_share_pct"],
                                "optimised_pct": opt_sus["hard_sustainable_share_pct"]},
           "verification": verification, "bounds": C.POLICY_BOUNDS, "fast_evaluator": obj.fast}
    _json(opt, R / "optimisation_result.json")
    summary["optimisation"] = opt
    FIG.optimisation_shares(shares, F / "optimisation_mode_shares.png")
    FIG.optimisation_benchmarks(bench, F / "optimisation_benchmarks.png")
    FIG.effort_frontier(budget_df, F / "optimisation_effort_budget.png")

    summary["runtime_minutes"] = round((time.time() - t_start) / 60, 2)
    _json(summary, R / "pipeline_summary.json")
    log(f"Done in {summary['runtime_minutes']} min. Selected model: {winner}. "
        f"Sustainable share {base_sus['expected_sustainable_share_pct']:.2f}% -> {100 * best_val:.2f}%")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--n-synth", type=int, default=None)
    a = ap.parse_args()
    run(root=a.root, fast=a.fast, n_synth=a.n_synth)
