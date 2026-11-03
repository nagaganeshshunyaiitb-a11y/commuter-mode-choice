"""Write the seven analysis notebooks and execute them in place.

    python notebooks/build_notebooks.py

The notebooks call the functions in src/ and read the artefacts written by
run_pipeline.py, so run the pipeline first.
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent

SETUP = """import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)
from src import config as C
RES, FIG = C.Paths().results, C.Paths().figures
def fig(name, width=900):
    display(Image(filename=str(FIG / name), width=width))"""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


NOTEBOOKS = {}

NOTEBOOKS["01_data_audit"] = [
    md("""# 01 - Data audit

We inspect the workbook before touching it. The checks follow the brief: missing values, duplicates,
duplicate columns, outliers, invalid codes, class balance, logical consistency and availability.
Nothing is changed in this notebook; corrections happen in `src/preprocessing.py` and are logged."""),
    code(SETUP + """
from src import data_validation as V
raw = V.load_raw()
print(raw.shape)
raw.head()"""),
    code("""audit = V.run_audit(raw)
for k in ["missing_values", "duplicate_rows", "duplicate_rows_ignoring_id", "id_unique", "id_range",
          "duplicate_columns", "constant_columns", "target_counts"]:
    print(f"{k:28s}", audit[k])"""),
    md("""No values are missing and no record is duplicated. IDs are unique but run from 1 to 654, so 123
IDs are absent; presumably records were removed before the file reached us. `DL_TwoW` and `DL_Car`
are exact copies of `TwoW_AV` and `Car_AV`, and `LocalTrain_AV` equals 1 for all 531 people.

The target is imbalanced: auto-rickshaw (34) and walking (47) are the two smallest classes."""),
    md("## Coding scheme against `Data Coding.docx`"),
    code("""audit["code_check"]"""),
    md("""Three variables disagree with the coding document:

* **Trip_Purpose** is coded 1/2 in the data but 0/1 in the document.
* **Occup** has a fifth code (5, 155 records) that the document does not list.
* **Dwelling** has a fourth code (4, 191 records) that the document does not list.

Gender, Area_Type and ParkingSpace are not in the document at all. We do not relabel anything. The
cross-tabulations below are the evidence we have; they suggest, but do not prove, how some codes
should be read."""),
    code("""ev = audit["category_evidence"]
display(ev["purpose_by_age"]); display(ev["purpose_by_occupation"]); display(ev["occupation_by_age"])"""),
    md("""All 33 respondents aged 18-30 report purpose code 2, as do 94% of occupation code 3, while purpose
code 1 is concentrated among people over 55. That pattern fits "2 = work trip, 1 = non-work trip",
the document's 0/1 shifted by one, but nobody has confirmed it. The occupation evidence is more
worrying: code 2, labelled "Government employee", is 64% aged over 55 and makes mostly purpose-1
trips, and code 1 ("Unemployed") only makes purpose-2 trips. The document's occupation labels may
simply not apply to this file. We keep neutral labels ("occupation code 2") everywhere."""),
    md("## Travel times and costs are computed from distance"),
    code("""rules = audit["los_rules"]
print(json.dumps({k: v for k, v in rules.to_dict().items() if k not in ("bus_fare_steps",)}, indent=1))"""),
    code("""fig("audit_time_vs_distance.png")"""),
    md("""Every travel time equals distance multiplied by a fixed rate, to rounding: walking at 4 km/h,
two-wheeler 25, auto 20, car 22, bus 17 and rail 35 km/h. Costs follow a per-km rate (car INR 4.45,
two-wheeler INR 1.60), an auto fare of max(15, 9.87 d - 0.74), and stepped bus and rail fare tables.
So the level-of-service attributes were generated from distance, not observed for each trip. This
matters a lot later: no model can learn a separate effect of bus travel time, because in these data
bus time is just distance times 3.53.

Eleven records break the rule:"""),
    code("""display(audit["los_inconsistent"])
display(audit["cost_rule_match"])"""),
    md("""For all eleven, the car and two-wheeler costs imply the recorded distance, while the times imply
distances that are off by large factors (ID 179's motorised times are exactly 1,000 times too large,
the pattern of metres typed as kilometres). The distance is therefore right and the times are wrong.
`preprocessing.clean_data` recomputes the six times of these records from distance and logs every
change. Extreme distances themselves (up to 89.7 km) agree with their costs and are kept."""),
    md("## Availability"),
    code("""audit["availability"]"""),
    md("""No respondent chose an unavailable mode. Bus is available to only 18% of respondents but chosen by
59% of them, which hints that bus availability may have been recorded with the chosen mode in mind;
we note this as a limitation. Car availability equals licence holding among car-owning households;
60 people live in car-owning households but have no car available."""),
    md("## Household consistency and outliers"),
    code("""print(audit["household_flag_counts"])
hh = audit["household_flags"]
raw.loc[hh.drop(columns="ID").any(axis=1).to_numpy(), ["ID", "HH_Size", "Adults", "Children", "Working_Adults"]]"""),
    md("""Fifteen households report more workers than adults. That is possible if "Children" counts sons
and daughters of any age rather than minors, so we flag these records rather than editing them.
Three records have HH_Size different from Adults + Children and three have more workers than
members; the derived workers-share feature caps workers at household size."""),
    code("""audit["outliers"]"""),
    md("## Verified data dictionary"),
    code("""dd = V.build_data_dictionary(raw)
dd[["variable", "type", "documented_coding", "observed_values", "status", "treatment"]]"""),
    md("""**Open questions for the data owner:** (1) the meaning of Trip_Purpose codes 1 and 2; (2) the labels
of occupation codes 1-5, in particular code 5; (3) the label of dwelling code 4; (4) the coding of
Gender, Area_Type and ParkingSpace (home or destination parking?); (5) whether "Children" means minors;
(6) how mode availability was determined."""),
]

NOTEBOOKS["02_eda"] = [
    md("""# 02 - Exploratory data analysis

All figures use the cleaned data (531 records). Each mode keeps the same colour in every chart."""),
    code(SETUP + """
clean = pd.read_csv(C.Paths().processed / "clean_data.csv")
print(clean.shape)"""),
    code("""fig("eda_mode_distribution.png", 700)
pd.read_csv(RES / "eda_mode_distribution.csv", index_col=0)"""),
    md("""Two-wheeler (26.7%), car (24.7%) and suburban rail (22.8%) dominate; bus (10.5%), walking (8.9%)
and auto-rickshaw (6.4%) are minorities. With 34 auto users, any per-class metric for auto will be
noisy, which is why macro-averaged metrics matter."""),
    code("""fig("eda_mode_by_distance_band.png", 800)"""),
    md("""Distance is the main organising variable. Walking is only chosen up to 3.6 km. Rail grows from
under 1% of trips below 1.5 km to 55% at 10-20 km and 75% beyond 20 km, and bus peaks in the 5-10 km
band (19%). Car is strongest for short trips (35% below 1.5 km)."""),
    code("""fig("eda_mode_by_income.png", 800)
pd.read_csv(RES / "eda_by_income.csv", index_col=0)"""),
    md("""Car use rises steeply with income: 3% in the lowest band, 33% in the middle and 59% in the highest.
Rail, bus and auto shares fall as income rises, so income should enter the car utility."""),
    code("""fig("eda_mode_by_age.png", 800)
fig("eda_mode_by_purpose.png", 800)"""),
    md("""People over 55 walk (13.5%) and use the bus (15.0%) more than the 30-55 group. Purpose code 1
trips have much more walking (19.8% vs 6.1%) and bus (15.1% vs 9.4%), consistent with non-work
travel. The 18-30 group has only 33 people."""),
    code("""fig("eda_mode_by_occupation.png", 800)
fig("eda_mode_by_gender.png", 800)"""),
    md("""Occupation codes 1 and 4 have 3 and 4 people, so their bars mean nothing. Code 3 has the highest rail
share (32.5%). Gender code 0 has only 16 people; the higher walking share there (25%) rests on four
individuals, so gender is left out of the choice model."""),
    code("""fig("eda_mode_by_vehicle_ownership.png", 800)
fig("eda_mode_by_sapt.png", 800)"""),
    md("""Households with no vehicle split between rail (40.7%), bus (23.7%), walking (18.6%) and auto
(16.9%). Two-wheeler-only households use their two-wheeler for half of trips. Only 29 people are
dissatisfied with public transport, and they use two-wheelers far more (48% vs 25%)."""),
    code("""fig("eda_travel_time.png", 750)
fig("eda_travel_cost.png", 750)
fig("eda_distance_by_choice.png", 750)
pd.read_csv(RES / "eda_distance_by_choice.csv", index_col=0)"""),
    md("""Median distances by chosen mode: walk 1.3 km, auto 2.0, car 2.5, two-wheeler 2.8, bus 8.9 and rail
13.7 km. The time and cost box plots look like rescaled copies of each other because they are
rescaled copies of distance. Rail fares barely vary (INR 5 for 95% of trips)."""),
    code("""fig("eda_availability_vs_choice.png", 800)
pd.read_csv(RES / "eda_availability.csv", index_col=0)"""),
    code("""fig("eda_association_with_choice.png", 650)
fig("eda_numeric_correlation.png", 650)"""),
    md("""Bus availability has the strongest association with the chosen mode (Cramér's V 0.75), followed by
car and two-wheeler availability, distance, income and vehicle ownership. Household size and the
number of workers show no association at all. The correlation heatmap shows distance, all six times
and most costs at |rho| close to 1, the collinearity found in the audit."""),
]

NOTEBOOKS["03_feature_engineering"] = [
    md("""# 03 - Feature engineering

Every candidate feature listed in the brief is checked for (a) a defensible definition,
(b) availability at prediction time and (c) whether it simply duplicates information already present."""),
    code(SETUP + """
from src import feature_engineering as FE
clean = pd.read_csv(C.Paths().processed / "clean_data.csv")
FE.candidate_feature_review(clean)"""),
    md("""Most "relative advantage" features fail check (c). Because times and costs are computed from distance,
the bus-versus-car time difference is just distance multiplied by a constant (Spearman rho = -0.999
with distance), and cost per km is constant apart from rounding. Total vehicles and the count of
available modes are exact sums of columns already in the model. Public-transport availability is
constant because rail is available to everyone.

Five derived variables are kept: log distance, vehicle-ownership category, private vehicle
available, workers share (workers per household member, capped at 1) and a has-children flag.
Rare occupation codes 1 and 4 (7 people) are pooled for estimation."""),
    code("""print("categorical:", FE.CATEGORICAL)
print("numeric:", FE.NUMERIC)
FE.FeatureBuilder().transform(clean).head()"""),
    md("""The features are built inside the model pipeline (`FeatureBuilder`), not stored in the data. A policy
scenario that changes a raw time or cost therefore updates every derived feature automatically, and
nothing derived from the target or from other records can leak into a prediction.

For the discrete choice models the same data are reshaped into person-alternative long format, one
row per available alternative:"""),
    code("""long = pd.read_csv(C.Paths().processed / "train_long_format.csv")
print(long.shape, "rows for", long["ID"].nunique(), "people")
long.head(12)"""),
]

NOTEBOOKS["04_model_comparison"] = [
    md("""# 04 - Model development, comparison and selection

Candidates: multinomial logit (random-utility MNL, estimated by maximum likelihood on long-format data),
nested logit, regularised multinomial logistic regression, decision tree, random forest and XGBoost.
The 80/20 split is stratified (seed 42). Tuning uses repeated stratified 5-fold CV (2 repeats) on the
training partition only; the comparison uses a different set of 5 x 5 folds; the test set is used once."""),
    code(SETUP + """
from src import choice_models as CM
pd.read_csv(RES / "mnl_specifications.csv")"""),
    md("""The two "FREE" specifications let walking time and in-vehicle time have separate coefficients. Their
in-vehicle time coefficient comes out **positive** (insignificant), which is impossible behaviourally:
since every time is a fixed multiple of distance, the data cannot separate time from distance. The
behavioural specifications ("GC") use one time coefficient in which a walking minute counts as two
in-vehicle minutes. The profile likelihood below shows that the data do not reject a weight of 2:"""),
    code("""pd.read_csv(RES / "mnl_walk_weight_profile.csv")"""),
    md("""A weight of 1 (walking and riding minutes valued equally) is rejected (p = 0.008), a weight of 2 is
not (p = 0.087). Among the behavioural specifications, GC+CORE has the best AIC and BIC and the
best tuning-fold macro-F1. Its estimates:"""),
    code("""est = pd.read_csv(RES / "mnl_estimates_selected.csv", index_col=0)
est.round(4)"""),
    code("""fig("mnl_coefficients.png", 850)"""),
    md("""Time (-0.024 per minute) and cost (-0.050 per rupee) are negative and significant, giving a value of
in-vehicle time of about INR 30 per hour. Higher income and more cars raise car utility; purpose code 1
raises walking and bus utility.

## Nested logit"""),
    code("""pd.read_csv(RES / "nested_logit_structures.csv")"""),
    md("""Five nesting structures were estimated. A {bus, rail} nest drives its scale parameter to the lower
bound, a sign of separation in the small bus-versus-rail sub-sample, so it is not usable. A private
{car, two-wheeler} nest pushes lambda to 1 or above (no nesting, or inconsistent with random utility).
The only usable structure is a "shared/public" nest {auto, bus, rail}: lambda = 0.537 (s.e. 0.119),
significantly below 1, likelihood-ratio test against the MNL p = 0.003."""),
    md("## Tuning results"),
    code("""pd.read_csv(RES / "tuning_results.csv")[["model", "configurations", "best_cv_macro_f1", "best_params"]]"""),
    md("## Cross-validated comparison (training partition, 25 folds)"),
    code("""cv = pd.read_csv(RES / "cv_summary.csv", index_col=0)
cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "balanced_accuracy",
        "log_loss", "brier", "ece", "share_mae_pp"]
display(pd.DataFrame({c: cv[f"{c}_mean"] for c in cols}).round(3))
pd.DataFrame({c: cv[f"{c}_sd"] for c in ["macro_f1", "balanced_accuracy", "log_loss"]}).round(3)"""),
    code("""fig("model_cv_comparison.png", 950)
fig("model_calibration.png", 850)"""),
    md("""XGBoost has the best cross-validated macro-F1 (0.620) and the lowest log loss (0.784). The MNL is
clearly behind on classification (macro-F1 0.562) but still far better than the constants-only
reference (0.396). Logistic regression with balanced class weights reaches a high balanced
accuracy, at the price of distorted aggregate shares (share error 5.0 pp against 1.6 for the MNL).

## Simulation readiness

The frozen model's job is to answer "what if bus fares fall by 20%?". A model that gives the wrong
sign, or no answer at all, cannot do that job however well it classifies."""),
    code("""r = pd.read_csv(RES / "simulation_readiness.csv")
display(r.pivot(index="model", columns="lever", values="mean_change_pp").round(3))
display(r.pivot(index="model", columns="lever", values="share_wrong_direction").round(3))
fig("model_simulation_readiness.png", 1000)"""),
    md("""Only the two logit models respond to every lever in the direction theory requires. The tuned XGBoost
never splits on any policy variable (distance carries the same information), so its predictions do
not move at all. Logistic regression and the random forest learned that longer bus or rail times go
with more bus or rail use, because both rise with distance, so a faster bus or train makes that mode
*less* likely for 36-99% of the people who have it. The decision tree also lowers bus use when bus
time falls.

## Selection"""),
    code("""print(json.dumps(json.load(open(RES / "model_selection.json")), indent=1))"""),
    md("""Following the rule written down before fitting, XGBoost and the decision tree form the predictive
shortlist, but every machine-learning model fails the readiness check, so the MNL is selected and
frozen. Had the readiness check been left out, XGBoost would have been chosen.

## Held-out test set"""),
    code("""test = pd.read_csv(RES / "test_metrics.csv", index_col=0)
display(test.round(3))
print(json.load(open(RES / "test_bootstrap_ci.json"))["MNL"])
fig("model_confusion_matrices.png", 950)"""),
    md("""On the 107 test records the MNL does worse than its cross-validation suggested (macro-F1 0.451,
bootstrap 95% CI 0.35-0.56), and its accuracy equals that of the constants-only reference, although
its log loss (0.949 vs 1.094) is clearly better. XGBoost and the decision tree classify the test set
much better. We report this as it is: the frozen model is the best model for *simulation*, not the
best classifier.

Refitting the frozen configuration on the training rows reproduces the frozen model exactly, which
confirms that nothing from the test set was used in fitting:"""),
    code("""from sklearn.base import clone
from src.prediction import load_frozen
from src import preprocessing as PP
model, card = load_frozen()
train = pd.read_csv(C.Paths().processed / "train.csv"); test_df = pd.read_csv(C.Paths().processed / "test.csv")
refit = clone(model).fit(train, PP.target(train))
print("max abs difference in test probabilities:", np.abs(refit.predict_proba(test_df) - model.predict_proba(test_df)).max())"""),
]

NOTEBOOKS["05_synthetic_population"] = [
    md("""# 05 - Synthetic population

The generator is fitted on the 424 training records only. It produces predictor variables, never a
Choice label. Travel times and costs are not synthesised; they are rebuilt from the synthetic distance
with the speed and fare rules recovered in the audit. Three methods are compared."""),
    code(SETUP + """
from src import synthetic_generation as SG
comp = pd.read_csv(RES / "synthetic_method_comparison.csv")
comp.set_index("method").T"""),
    md("""* **Bootstrap** reproduces the marginals best but 99.3% of its records are copies of real people;
  it fails the memorisation rule and serves only as a reference.
* **Gaussian copula** creates new records, but a random forest separates them from real ones easily
  (AUC 0.747), associations are distorted the most, and 31% of records break a logical rule before
  repair (for example, car available in a household without a car).
* **Sequential CART** keeps the associations best (mean absolute difference in Cramér's V 0.042),
  its records are indistinguishable from real ones (AUC 0.492), only 1.5% need repair, and only 0.38%
  coincide with a training record.

The rule chosen in advance (exact copies no more than twice the holdout rate plus one point, then the
AUC closest to 0.5) selects CART. Its leaf size was set to 10 after this check:"""),
    code("""pd.read_csv(RES / "synthetic_cart_leaf_sensitivity.csv")"""),
    md("""With leaf size 5, 15% of synthetic people sit closer to a training record than 95% of real held-out
people do (memorisation). With 10, the median distance to the closest training record equals the
holdout's (0.143) and only 9% fall below the holdout's 5th percentile, at almost no cost in fidelity."""),
    code("""pop = pd.read_csv(C.Paths().synthetic / "synthetic_population.csv")
print(pop.shape); pop.head()"""),
    code("""pd.read_csv(RES / "synthetic_marginal_fidelity.csv").round(4)"""),
    code("""pd.read_csv(RES / "synthetic_moments.csv").round(2)"""),
    code("""fig("synthetic_marginals.png", 1000)
fig("synthetic_distance.png", 650)"""),
    code("""fig("synthetic_associations.png", 1000)
fig("synthetic_pairwise.png", 1000)"""),
    md("""Category frequencies match to within about 1.3 percentage points for every variable except bus
availability (21.5% of synthetic commuters against 17.0% in the training data). The CART synthesis
slightly weakens the association between car and two-wheeler ownership
and between household size and children, and slightly strengthens the link between distance and bus
availability. Income still drives car ownership, children grow with household size and walking
availability falls with distance as in the real data.

Logical constraints on the saved population:"""),
    code("""train = pd.read_csv(C.Paths().processed / "train.csv")
print(SG.constraint_violations(pop, train["Distance_km"].min(), train["Distance_km"].max()))
print("adults >= 1:", (pop["Adults"] >= 1).all(), "| adults + children = size:", (pop["Adults"] + pop["Children"] == pop["HH_Size"]).all())
print("Choice column present:", "Choice" in pop.columns)"""),
    md("""The population mirrors the survey sample. It is not calibrated to census totals for any city, so it
should be read as "a larger population that looks like our respondents", not as Mumbai."""),
]

NOTEBOOKS["06_policy_simulation"] = [
    md("""# 06 - Baseline prediction and policy scenarios

The frozen MNL predicts six probabilities for each of the 5,000 synthetic people. Every scenario
changes travel attributes and re-predicts with the same frozen model. These are model-based
counterfactuals: they show what the fitted model implies, not what an intervention would cause."""),
    code(SETUP + """
from src.prediction import load_frozen, predict_probabilities, summarise
from src import scenario_simulation as SS
model, card = load_frozen()
pop = pd.read_csv(C.Paths().synthetic / "synthetic_population.csv")
P = predict_probabilities(model, pop)
table, sus = summarise(P)
display(table.round(2)); print(sus)"""),
    code("""pd.read_csv(RES / "baseline_share_comparison.csv").round(2)"""),
    code("""fig("baseline_shares.png", 800)"""),
    md("""Expected shares in the synthetic population are close to the observed training shares (bus is 2 points
higher because the synthetic population has slightly more bus availability). The expected sustainable
share is 43.9%. Assigning everybody to their most likely mode gives a very different picture (car
36%, walking 3%, sustainable 35%): hard assignment exaggerates the majority mode in each choice set,
so the expected share is the quantity we optimise.

Availability is respected exactly:"""),
    code("""A = pop[[C.AV_COLS[m] for m in C.MODES]].to_numpy()
print("max probability on an unavailable mode:", P.to_numpy()[A == 0].max())
print("rows summing to one:", np.allclose(P.sum(axis=1), 1))"""),
    md("## Experiment 1: travel cost"),
    code("""t = SS.experiment_cost(model, pop)
t[["scenario", "expected_sustainable_share_pct", "sustainable_change_pp", "car_pct", "bus_pct", "train_pct"]].round(2)"""),
    md("""A 50% higher car cost adds 2.3 points of sustainable share, a 50% lower rail fare 1.3 points and a 50%
lower bus fare 0.5 points. Bus measures are weak at population level because only about one person
in five has a bus option at all. Part of the car users who leave move to two-wheelers, which are not
sustainable under our definition.

## Experiment 2: travel time"""),
    code("""t = SS.experiment_time(model, pop)
t[["scenario", "expected_sustainable_share_pct", "sustainable_change_pp", "car_pct", "bus_pct", "train_pct"]].round(2)
"""),
    code("""fig("scenario_sensitivity.png", 950)"""),
    md("""Time effects are small (a 20% faster rail adds 0.8 points, a 30% faster bus 0.2 points) and, unlike
the cost effects, they depend on the assumed walk-time weight (see notebook 07).

## Experiment 3: availability (extrapolation)"""),
    code("""t = SS.experiment_availability(model, pop)
display(t[["scenario", "expected_sustainable_share_pct", "sustainable_change_pp", "bus_pct", "walk_pct"]].round(2))
fig("scenario_availability.png", 850)"""),
    md("""Offering a bus to people who currently have none produces very large changes (up to +29 points). These
numbers should not be taken at face value: they put people into choice sets the survey never
observed for them, assume they value the bus exactly like current bus users (who choose it 59% of the
time and may have been recorded as having a bus option *because* they use it) and ignore service
quality. Rail availability cannot be varied at all, since rail is already available to everyone.

## Experiment 4: combined packages"""),
    code("""t = SS.experiment_combined(model, pop)
display(t[["scenario", "expected_sustainable_share_pct", "hard_sustainable_share_pct", "sustainable_change_pp"]].round(2))
fig("scenario_combined.png", 900)"""),
    md("## No-change check"),
    code("""same, _, _ = SS.simulate(model, pop, **SS.levers_to_policy({k: 0 for k in C.POLICY_BOUNDS}))
print("max |difference| from baseline:", np.abs(same.to_numpy() - P.to_numpy()).max())"""),
]

NOTEBOOKS["07_policy_optimisation"] = [
    md("""# 07 - Policy optimisation

maximise S(x) = (1/N) * sum over people of [P(walk) + P(bus) + P(rail)]

over x = (bus time reduction, bus fare reduction, rail time reduction, rail fare reduction, car cost
increase), each within its bounds. There is no intervention cost data, so this is bounded
optimisation; an optional, dimensionless "effort budget" (the sum of levers as shares of their upper
bounds) is used as a sensitivity analysis only."""),
    code(SETUP + """
from src import policy_optimisation as PO
from src.prediction import load_frozen
res = json.load(open(RES / "optimisation_result.json"))
pd.DataFrame({"lever": [C.POLICY_LABELS[k] for k in PO.LEVERS],
              "bounds": [C.POLICY_BOUNDS[k] for k in PO.LEVERS],
              "optimised value": [res["optimal_levers"][k] for k in PO.LEVERS]})"""),
    code("""print("baseline:", round(res["baseline_sustainable_pct"], 3), "% -> optimised:", round(res["objective_sustainable_share_pct"], 3),
      "% (+", round(res["improvement_pp"], 3), "pp)")
print("most-likely-mode sustainable share:", res["hard_sustainable"])
print("grid:", res["grid"]["evaluations"], "evaluations,", res["grid"]["seconds"], "s; DE:", res["differential_evolution"]["evaluations"], "evaluations")"""),
    code("""pd.read_csv(RES / "optimisation_mode_shares.csv").round(2)"""),
    code("""fig("optimisation_mode_shares.png", 800)"""),
    md("""Under the MNL every lever moves the sustainable share in one direction, so the bounded optimum is the
corner where every lever sits at its upper bound. Grid search (46,585 points at 5% steps) and
differential evolution agree. The optimum raises the expected sustainable share from 43.91% to 48.99%
(+5.07 points), mostly by moving people out of cars (-5.4 points) into rail (+3.5) and bus (+1.6).

## Benchmarks"""),
    code("""display(pd.read_csv(RES / "optimisation_benchmarks.csv").round(3))
fig("optimisation_benchmarks.png", 850)"""),
    md("## Effort budget: which levers come first?"),
    code("""display(pd.read_csv(RES / "optimisation_effort_budget.csv").round(3))
display(pd.read_csv(RES / "optimisation_budget_de_check.csv").round(3))
fig("optimisation_effort_budget.png", 950)"""),
    md("""When the total effort is limited, the optimiser spends it in a fixed order: car pricing first, then the
rail fare, rail time, bus fare and finally bus time. Differential evolution under the same budgets
lands on the grid's answer.

## Robustness to the walk-time weight"""),
    code("""pd.read_csv(RES / "sensitivity_walk_weight.csv").round(3)"""),
    md("""Across walk weights from 1.5 to 3, the optimised share moves only between 48.8% and 49.2%, and the
ranking of the cost levers does not change. The rail-time effect is the one that depends on the
assumption (1.0 points at weight 1.5, 0.5 at weight 3).

## Independent verification

The objective is recomputed from scratch: the frozen model and the population are reloaded from disk,
the policy is applied with the general data-frame path (not the fast evaluator used by the search)
and every constraint is checked."""),
    code("""model, _ = load_frozen()
pop = pd.read_csv(C.Paths().synthetic / "synthetic_population.csv")
check = PO.verify_solution(res["optimal_levers"], res["objective_sustainable_share_pct"] / 100, model, pop)
check"""),
]


def build(execute=True):
    ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = HERE / f"{name}.ipynb"
        if execute:
            ep.preprocess(nb, {"metadata": {"path": str(HERE)}})
        nbf.write(nb, path)
        print("wrote", path.name)


if __name__ == "__main__":
    build()
