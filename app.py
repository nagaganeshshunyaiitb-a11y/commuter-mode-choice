"""Streamlit dashboard.

    streamlit run app.py

Reads the artefacts written by run_pipeline.py; nothing is retrained here.
Every scenario uses the frozen model and the saved synthetic population.
"""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src import policy_optimisation as PO
from src.data_validation import LOSRules
from src.plot_style import MODE_COLORS, SERIES
from src.prediction import load_frozen, predict_probabilities, summarise
from src.scenario_simulation import apply_policy, levers_to_policy

st.set_page_config(page_title="Commuter mode choice and policy optimisation", layout="wide")
PATHS = C.Paths()
RES, FIG = PATHS.results, PATHS.figures
MODE_NAMES = [C.MODE_LABELS[m] for m in C.MODES]
COLOR_SCALE = alt.Scale(domain=MODE_NAMES, range=[MODE_COLORS[m] for m in C.MODES])

if not (PATHS.best_model / "model.joblib").exists():
    st.error("No frozen model found. Run `python run_pipeline.py` first.")
    st.stop()


@st.cache_data
def csv(name, **kw):
    return pd.read_csv(RES / name, **kw)


@st.cache_data
def js(name):
    return json.loads((RES / name).read_text())


@st.cache_resource
def frozen():
    return load_frozen(PATHS)


@st.cache_data
def population():
    return pd.read_csv(PATHS.synthetic / "synthetic_population.csv")


@st.cache_resource
def los_rules():
    return LOSRules.fit(pd.read_csv(PATHS.processed / "clean_data.csv"))


@st.cache_data
def baseline():
    model, _ = frozen()
    P = predict_probabilities(model, population())
    return summarise(P)


def figure(name, caption=None):
    path = FIG / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


def share_chart(df, value_cols, labels, title, height=320):
    long = df.melt(id_vars="mode", value_vars=value_cols, var_name="series", value_name="share")
    long["series"] = long["series"].map(dict(zip(value_cols, labels)))
    return alt.Chart(long, title=title, height=height).mark_bar(cornerRadiusEnd=3, size=14).encode(
        x=alt.X("mode:N", sort=MODE_NAMES, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("series:N", sort=labels),
        y=alt.Y("share:Q", title="Share (%)"),
        color=alt.Color("series:N", sort=labels, scale=alt.Scale(range=SERIES[:len(labels)]), title=None,
                        legend=alt.Legend(orient="bottom")),
        tooltip=["mode", "series", alt.Tooltip("share:Q", format=".2f")])


st.sidebar.title("Commuter mode choice")
st.sidebar.caption("Prediction, simulation and policy optimisation with a frozen model and a synthetic population")
page = st.sidebar.radio("Page", ["Dataset overview", "Model comparison", "Synthetic population",
                                 "Policy scenario simulator", "Policy optimisation"])
st.sidebar.markdown("---")
st.sidebar.caption("All policy results are model-based counterfactuals, not causal estimates. "
                   "Bounds are scenario assumptions; no intervention cost data were available.")

# =============================================================================
if page == "Dataset overview":
    st.header("Dataset overview")
    audit = js("audit_summary.json")
    clean = pd.read_csv(PATHS.processed / "clean_data.csv")
    a, b, c, d = st.columns(4)
    a.metric("Respondents", audit["shape"][0])
    b.metric("Columns in workbook", audit["shape"][1])
    c.metric("Alternatives", 6)
    d.metric("Records with corrected travel times", int(clean["los_time_corrected"].sum()))

    counts = pd.DataFrame({"mode": list(audit["target_counts"]), "count": list(audit["target_counts"].values())})
    counts["share"] = 100 * counts["count"] / counts["count"].sum()
    st.altair_chart(alt.Chart(counts, title="Observed mode choice", height=280).mark_bar(cornerRadiusEnd=3, size=28)
                    .encode(x=alt.X("mode:N", sort=MODE_NAMES, title=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("count:Q", title="Respondents"),
                            color=alt.Color("mode:N", scale=COLOR_SCALE, legend=None),
                            tooltip=["mode", "count", alt.Tooltip("share:Q", format=".1f")]), width="stretch")

    st.subheader("Data-quality findings")
    st.markdown(f"""
- No missing values and no duplicate records. IDs run from {audit['id_range'][0]} to {audit['id_range'][1]} with gaps (531 records).
- `DL_TwoW` and `DL_Car` are identical to `TwoW_AV` and `Car_AV` and were dropped. `LocalTrain_AV` is 1 for everybody, so no rail-availability effect can be learned.
- **Travel times and costs are computed from distance**, not measured: each time equals distance x a fixed speed (walk 4, two-wheeler 25, auto 20, car 22, bus 17, rail 35 km/h). Eleven records had times inconsistent with their own distance (factors up to 1,000x); their costs agree with distance, so the times were recomputed.
- Coding discrepancies against `Data Coding.docx`: Trip_Purpose is 1/2 (document: 0/1), Occupation has an undocumented code 5 (155 records), Dwelling has an undocumented code 4 (191 records); Gender, Area_Type and ParkingSpace are not documented. Codes were kept as recorded.
- Household checks: {audit['household_flag_counts']['workers_exceed_adults']} records have more workers than adults, {audit['household_flag_counts']['hh_size_not_adults_plus_children']} have HH_Size not equal to Adults + Children, one has zero adults. Flagged, not altered.
""")
    with st.expander("Verified data dictionary"):
        st.dataframe(pd.read_csv(PATHS.processed / "data_dictionary.csv"), hide_index=True)
    with st.expander("Records with inconsistent travel times (implied distances)"):
        st.dataframe(csv("audit_los_inconsistent_records.csv"), hide_index=True)

    st.subheader("Exploratory figures")
    eda = {
        "Mode share by distance band": ("eda_mode_by_distance_band.png",
            "Distance organises the choice set: walking disappears beyond 3.6 km, rail rises from under 1% of "
            "trips below 1.5 km to 75% of trips above 20 km, and car and two-wheeler dominate short and medium trips."),
        "Mode share by income": ("eda_mode_by_income.png",
            "Car use climbs steeply with income (3% below 30k, 33% at 30-80k, 59% above 80k) while bus, rail and "
            "auto shares fall."),
        "Mode share by age": ("eda_mode_by_age.png",
            "Respondents over 55 walk and take the bus more; the 18-30 group is small (33) and leans to two-wheeler and rail."),
        "Mode share by occupation code": ("eda_mode_by_occupation.png",
            "Codes 1 and 4 have only 3 and 4 records. Code 3 has the highest rail share (33%)."),
        "Mode share by gender code": ("eda_mode_by_gender.png",
            "Only 16 records carry code 0, so the differences cannot be interpreted with any confidence."),
        "Mode share by trip purpose code": ("eda_mode_by_purpose.png",
            "Purpose code 1 has much more walking (20% vs 6%) and bus use; it behaves like non-work travel."),
        "Vehicle ownership and mode choice": ("eda_mode_by_vehicle_ownership.png",
            "Households without vehicles split between rail (41%), bus (24%), walking (19%) and auto (17%)."),
        "Satisfaction with public transport": ("eda_mode_by_sapt.png",
            "Only 29 respondents are dissatisfied; they use two-wheelers far more (48%)."),
        "Availability versus choice": ("eda_availability_vs_choice.png",
            "Bus is available to only 18% of respondents but chosen by 59% of them; rail is available to all."),
        "Distance by chosen mode": ("eda_distance_by_choice.png",
            "Median distances: walk 1.3 km, auto 2.0, car 2.5, two-wheeler 2.8, bus 8.9, rail 13.7 km."),
        "Travel time and cost distributions": ("eda_travel_time.png", "Cleaned travel times by mode (log scale)."),
        "Travel times against distance (raw)": ("audit_time_vs_distance.png",
            "All but 11 records lie exactly on a fixed-speed line, which shows the times were computed from distance."),
        "Association with choice": ("eda_association_with_choice.png",
            "Availability indicators, distance and income carry the strongest association with the chosen mode."),
        "Correlation among numeric variables": ("eda_numeric_correlation.png",
            "Distance, all times and most costs are almost perfectly rank-correlated."),
    }
    choice = st.selectbox("Figure", list(eda))
    f, text = eda[choice]
    figure(f)
    st.caption(text)
    if choice == "Travel time and cost distributions":
        figure("eda_travel_cost.png")

# =============================================================================
elif page == "Model comparison":
    st.header("Model comparison and selection")
    cv = csv("cv_summary.csv", index_col=0)
    test = csv("test_metrics.csv", index_col=0)
    sel = js("model_selection.json")
    _, card = frozen()
    st.success(f"Selected and frozen model: **{sel['winner']}**")
    st.markdown("Selection rule (declared before fitting): shortlist on cross-validated macro-F1 (within one "
                "standard error of the best), then balanced accuracy, then log loss, then simplicity. The chosen "
                "model must also pass a simulation-readiness check, because it is used for policy simulation.")
    cols = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "log_loss", "brier", "ece", "share_mae_pp"]
    show = pd.DataFrame({m: cv[f"{m}_mean"] for m in cols}).round(3)
    show.insert(0, "macro_f1_sd", cv["macro_f1_sd"].round(3))
    st.subheader("Cross-validation on the training partition (5 folds x 5 repeats)")
    st.dataframe(show, width="stretch")
    st.subheader("Held-out test set (107 records)")
    st.dataframe(test[cols].round(3), width="stretch")
    figure("model_cv_comparison.png")
    c1, c2 = st.columns(2)
    with c1:
        figure("model_confusion_matrices.png")
    with c2:
        figure("model_calibration.png")
    st.subheader("Simulation-readiness check")
    figure("model_simulation_readiness.png")
    st.dataframe(csv("simulation_readiness.csv").round(4), hide_index=True)
    st.markdown(f"Ranking after the predictive criteria: {', '.join(sel['ranking'])}.  \n"
                f"Skipped for failing the readiness check: {', '.join(sel['skipped']) or 'none'}.")
    if "estimates" in card:
        with st.expander("Estimated utility coefficients of the frozen model"):
            st.dataframe(pd.DataFrame(card["estimates"]).T.round(4))
        figure("mnl_coefficients.png")

# =============================================================================
elif page == "Synthetic population":
    st.header("Synthetic population")
    pop = population()
    summ = js("pipeline_summary.json")["synthetic"]
    a, b, c = st.columns(3)
    a.metric("Synthetic individuals", f"{len(pop):,}")
    b.metric("Generator", summ["method"].upper())
    c.metric("Discriminator AUC", f"{summ['table'][summ['method']]['discriminator_auc']:.3f}",
             help="Cross-validated AUC of a random forest asked to separate real from synthetic records. 0.5 = indistinguishable.")
    st.markdown("The generator was fitted on the 424 training records only. Choice labels are never generated: "
                "the frozen model predicts them. Travel times and costs are rebuilt from the synthetic distance with "
                "the speed and fare rules found in the survey. The population mirrors the survey sample; it is not "
                "calibrated to any city's census.")
    st.subheader("Comparison of generation methods")
    st.dataframe(csv("synthetic_method_comparison.csv").round(4), hide_index=True)
    with st.expander("CART leaf-size sensitivity (fidelity against memorisation)"):
        st.dataframe(csv("synthetic_cart_leaf_sensitivity.csv").round(4), hide_index=True)
    figure("synthetic_marginals.png")
    c1, c2 = st.columns(2)
    with c1:
        figure("synthetic_distance.png")
    with c2:
        figure("synthetic_pairwise.png")
    figure("synthetic_associations.png")
    st.subheader("Baseline predicted mode shares")
    comp = csv("baseline_share_comparison.csv")
    st.altair_chart(share_chart(comp, ["observed_train_pct", "synthetic_expected_pct", "synthetic_hard_pct"],
                                ["Observed (training)", "Synthetic: expected", "Synthetic: most likely mode"],
                                "Observed sample versus predicted synthetic population"), width="stretch")
    st.dataframe(comp.round(2), hide_index=True)

# =============================================================================
elif page == "Policy scenario simulator":
    st.header("Policy scenario simulator")
    model, card = frozen()
    tab_pop, tab_ind = st.tabs(["Synthetic population", "Individual traveller"])
    with tab_pop:
        st.caption("Percentage changes applied to every synthetic individual; the frozen model re-predicts.")
        c1, c2, c3 = st.columns(3)
        with c1:
            bus_fare = st.slider("Bus fare change (%)", -50, 50, 0, 5)
            bus_time = st.slider("Bus travel-time change (%)", -50, 30, 0, 5)
        with c2:
            rail_fare = st.slider("Suburban rail fare change (%)", -50, 50, 0, 5)
            rail_time = st.slider("Suburban rail travel-time change (%)", -40, 30, 0, 5)
        with c3:
            car_cost = st.slider("Car out-of-pocket cost change (%)", -20, 100, 0, 5,
                                 help="The data's car cost is a per-km money cost; an increase is a proxy for "
                                      "pricing measures such as fuel taxes or parking charges.")
            bus_ext = st.slider("Extend bus availability to this share of people without it (%)", 0, 100, 0, 5,
                                help="Extrapolation: puts people into choice sets the survey never observed for them.")
        pol = {"time_pct": {"bus": bus_time, "train": rail_time}, "cost_pct": {"bus": bus_fare, "train": rail_fare,
                                                                                "car": car_cost},
               "bus_av_extension": bus_ext / 100}
        base_table, base_sus = baseline()
        P = predict_probabilities(model, apply_policy(population(), **pol))
        table, sus = summarise(P)
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected sustainable share", f"{sus['expected_sustainable_share_pct']:.2f}%",
                  f"{sus['expected_sustainable_share_pct'] - base_sus['expected_sustainable_share_pct']:+.2f} pp")
        m2.metric("Most-likely-mode sustainable share", f"{sus['hard_sustainable_share_pct']:.2f}%",
                  f"{sus['hard_sustainable_share_pct'] - base_sus['hard_sustainable_share_pct']:+.2f} pp")
        m3.metric("Sustainable modes", "Walk + Bus + Suburban rail")
        comp = pd.DataFrame({"mode": MODE_NAMES, "Baseline": base_table["expected_share_pct"],
                             "Scenario": table["expected_share_pct"]})
        st.altair_chart(share_chart(comp, ["Baseline", "Scenario"], ["Baseline", "Scenario"],
                                    "Expected mode shares"), width="stretch")
        comp["Change (pp)"] = comp["Scenario"] - comp["Baseline"]
        st.dataframe(comp.round(3), hide_index=True)
        if bus_ext:
            st.warning("Bus availability extension is an extrapolation beyond the observed choice sets.")

    with tab_ind:
        st.caption("Describe one commuter. Travel times and costs follow the survey's speed and fare rules for the "
                   "chosen distance; tick the modes that are available.")
        rules = los_rules()
        c1, c2, c3 = st.columns(3)
        with c1:
            dist = st.number_input("Trip distance (km)", 0.1, 90.0, 6.0, 0.1)
            income = st.selectbox("Monthly household income", [1, 2, 3], index=1,
                                  format_func={1: "<30,000", 2: "30,000-80,000", 3: ">80,000"}.get)
            age = st.selectbox("Age", [1, 2, 3], index=1, format_func={1: "18-30", 2: "30-55", 3: ">55"}.get)
            purpose = st.selectbox("Trip purpose code (as recorded)", [1, 2], index=1)
            gender = st.selectbox("Gender code (as recorded)", [0, 1], index=1)
        with c2:
            edu = st.selectbox("Education", [1, 2, 3], format_func={1: "Graduation", 2: "SSC/Diploma", 3: "Below SSC"}.get)
            occ = st.selectbox("Occupation code (as recorded)", [1, 2, 3, 4, 5], index=2)
            dwell = st.selectbox("Dwelling code (as recorded)", [1, 2, 3, 4])
            area = st.selectbox("Area type code", [0, 1])
            parking = st.selectbox("Parking space", [0, 1], index=1)
            sapt = st.selectbox("Satisfied with public transport", [0, 1], index=1)
        with c3:
            hh = st.number_input("Household size", 1, 12, 4)
            children = st.number_input("Children", 0, 11, 1)
            workers = st.number_input("Working members", 1, 12, 1)
            cars = st.number_input("Cars owned", 0, 4, 1)
            twows = st.number_input("Two-wheelers owned", 0, 5, 1)
        st.markdown("**Available modes**")
        av_cols = st.columns(6)
        av = {}
        for col, m in zip(av_cols, C.MODES):
            default = not (m == "car" and cars == 0) and not (m == "twow" and twows == 0)
            av[m] = col.checkbox(C.MODE_LABELS[m], value=default if m != "train" else True, disabled=(m == "train"),
                                 key=f"av_{m}")
        problems = []
        if children > hh - 1:
            problems.append("children must leave at least one adult in the household")
        if workers > hh:
            problems.append("working members cannot exceed household size")
        if av["car"] and cars == 0:
            problems.append("car marked available but the household owns no car")
        if av["twow"] and twows == 0:
            problems.append("two-wheeler marked available but the household owns none")
        if problems:
            st.error("Please fix: " + "; ".join(problems))
        else:
            person = pd.DataFrame([{
                "ID": 1, "Distance_km": dist, "Trip_Purpose": purpose, "Income": income, "Gender": gender, "Age": age,
                "Education": edu, "Occup": occ, "ParkingSpace": parking, "Area_Type": area, "Dwelling": dwell,
                "HH_Size": hh, "Adults": hh - children, "Children": children, "Working_Adults": workers, "SAPT": sapt,
                "Car_Ownership": cars, "TwoW_Ownership": twows,
                **{C.AV_COLS[m]: int(av[m]) for m in C.MODES}}])
            person = pd.concat([person, rules.times(person["Distance_km"]), rules.costs(person["Distance_km"])], axis=1)
            st.markdown("**What-if changes for this traveller**")
            w1, w2, w3, w4, w5 = st.columns(5)
            ib_t = w1.slider("Bus time (%)", -50, 30, 0, 5, key="ibt")
            ib_f = w2.slider("Bus fare (%)", -50, 50, 0, 5, key="ibf")
            ir_t = w3.slider("Rail time (%)", -40, 30, 0, 5, key="irt")
            ir_f = w4.slider("Rail fare (%)", -50, 50, 0, 5, key="irf")
            ic = w5.slider("Car cost (%)", -20, 100, 0, 5, key="ic")
            p0 = predict_probabilities(model, person).iloc[0].to_numpy()
            changed = apply_policy(person, time_pct={"bus": ib_t, "train": ir_t},
                                   cost_pct={"bus": ib_f, "train": ir_f, "car": ic})
            p1 = predict_probabilities(model, changed).iloc[0].to_numpy()
            out = pd.DataFrame({"mode": MODE_NAMES, "Current": 100 * p0, "With changes": 100 * p1})
            st.altair_chart(share_chart(out, ["Current", "With changes"], ["Current", "With changes"],
                                        "Choice probabilities for this traveller (%)"), width="stretch")
            out["Travel time (min)"] = [person[C.TIME_COLS[m]].iloc[0] for m in C.MODES]
            out["Cost (INR)"] = [person[C.COST_COLS[m]].iloc[0] if m in C.COST_COLS else 0 for m in C.MODES]
            st.dataframe(out.round(2), hide_index=True)

# =============================================================================
elif page == "Policy optimisation":
    st.header("Policy optimisation")
    st.markdown("Maximise the expected sustainable share  S(x) = mean over individuals of "
                "P(walk) + P(bus) + P(rail), within the bounds below. There is no intervention cost data, so this "
                "is bounded optimisation. The optional effort budget limits the sum of levers, each measured as a "
                "share of its upper bound; it is dimensionless, not money.")
    saved = js("optimisation_result.json")
    with st.expander("Result of the main run (pipeline)", expanded=True):
        lev = pd.DataFrame({"Lever": [C.POLICY_LABELS[k] for k in PO.LEVERS],
                            "Optimised value (%)": [saved["optimal_levers"][k] for k in PO.LEVERS],
                            "Upper bound (%)": [C.POLICY_BOUNDS[k][1] for k in PO.LEVERS]})
        st.dataframe(lev, hide_index=True)
        a, b, c = st.columns(3)
        a.metric("Baseline sustainable share", f"{saved['baseline_sustainable_pct']:.2f}%")
        b.metric("Optimised sustainable share", f"{saved['objective_sustainable_share_pct']:.2f}%",
                 f"{saved['improvement_pp']:+.2f} pp")
        c.metric("Independent verification", "passed" if saved["verification"]["matches"] else "FAILED")
        figure("optimisation_mode_shares.png")
        figure("optimisation_effort_budget.png")

    st.subheader("Run your own optimisation")
    model, _ = frozen()
    bounds = {}
    cols = st.columns(len(PO.LEVERS))
    for col, k in zip(cols, PO.LEVERS):
        lo, hi = C.POLICY_BOUNDS[k]
        bounds[k] = col.slider(C.POLICY_LABELS[k], 0.0, 100.0, (lo, hi), 5.0, key=f"b_{k}")
    c1, c2, c3 = st.columns(3)
    step = c1.selectbox("Grid step (%)", [10.0, 5.0], help="5% gives a finer grid but takes longer.")
    use_budget = c2.checkbox("Limit total effort")
    budget = c3.slider("Effort budget", 0.25, 5.0, 1.5, 0.25, disabled=not use_budget)
    if st.button("Run grid search", type="primary"):
        obj = PO.Objective(model, population())
        with st.spinner("Evaluating policy combinations..."):
            best, table, secs = PO.grid_search(obj, bounds=bounds, step=step, budget=budget if use_budget else None)
        x = best[PO.LEVERS].to_numpy(float)
        base_table, base_sus = baseline()
        opt_table, opt_sus = PO.shares_for(obj, x)
        st.success(f"Evaluated {len(table):,} combinations in {secs:.1f} s.")
        st.dataframe(pd.DataFrame({"Lever": [C.POLICY_LABELS[k] for k in PO.LEVERS],
                                   "Optimal value (%)": x}), hide_index=True)
        a, b = st.columns(2)
        a.metric("Expected sustainable share", f"{opt_sus['expected_sustainable_share_pct']:.2f}%",
                 f"{opt_sus['expected_sustainable_share_pct'] - base_sus['expected_sustainable_share_pct']:+.2f} pp")
        b.metric("Most-likely-mode sustainable share", f"{opt_sus['hard_sustainable_share_pct']:.2f}%",
                 f"{opt_sus['hard_sustainable_share_pct'] - base_sus['hard_sustainable_share_pct']:+.2f} pp")
        comp = pd.DataFrame({"mode": MODE_NAMES, "Baseline": base_table["expected_share_pct"],
                             "Optimised": opt_table["expected_share_pct"]})
        st.altair_chart(share_chart(comp, ["Baseline", "Optimised"], ["Baseline", "Optimised"],
                                    "Expected mode shares"), width="stretch")
        problems = PO.check_constraints(dict(zip(PO.LEVERS, x)), bounds, budget if use_budget else None)
        st.caption("Constraint check: " + ("all satisfied" if not problems else "; ".join(problems)))
