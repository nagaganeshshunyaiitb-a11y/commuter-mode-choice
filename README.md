# Commuter Mode Choice Modelling and Sustainable Transport Policy Optimisation

AI and Data Science course project. Using a survey of 531 commuters, we predict the
probability of choosing each of six modes (walk, two-wheeler, auto-rickshaw, car, bus, suburban
rail), simulate how those probabilities respond to changes in fares and travel times, and search
for the combination of feasible interventions that maximises the expected sustainable share
(walk + bus + rail) in a synthetic population of 5,000 commuters.

## What we found

* **The travel times and costs in the workbook were computed from distance, not observed.** Every
  time equals distance x a fixed speed (walk 4, two-wheeler 25, auto 20, car 22, bus 17, rail 35 km/h)
  and every cost follows a per-km rate or a stepped fare table. Eleven records had times that did not
  match their own distance (up to 1,000x off); their costs did match, so the times were recomputed.
* Because of this, no model can learn the effect of travel time separately from distance. The
  unrestricted logit gives a positive time coefficient. Our behavioural MNL uses one time coefficient
  in which a walking minute counts as two in-vehicle minutes (not rejected by a likelihood-ratio test,
  p = 0.087); the results are shown to be robust to this choice.
* XGBoost classifies best (cross-validated macro-F1 0.620 vs 0.562 for the MNL; test 0.605 vs 0.451),
  **but no machine-learning model is usable for policy simulation**: the tuned XGBoost does not react
  to any policy lever at all, and logistic regression and the random forest make a faster bus or train
  *less* attractive for 36-99% of people. Following the selection rule declared before fitting, the
  MNL was frozen. Without the simulation-readiness requirement, XGBoost would have been chosen.
* A {auto, bus, rail} nested logit is estimable (lambda = 0.54, LR test p = 0.003) but did not
  beat the MNL in cross-validation.
* Sequential CART synthesis gave the most faithful synthetic population (discriminator AUC 0.49,
  0.4% exact copies, nearest-record distances like those of unseen real people).
* With every lever at its upper bound (bus time -30%, bus fare -50%, rail time -20%, rail fare -50%,
  car cost +50%), the expected sustainable share rises from **43.9% to 49.0% (+5.1 pp)**. With a
  limited effort budget the levers are used in the order car pricing, rail fare, rail time, bus fare,
  bus time. Bus measures are weak because only about one commuter in five has a bus option.
* All policy results are model-based counterfactuals, not causal estimates.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt

python run_pipeline.py            # full pipeline, about 20 minutes on 8 cores
python run_pipeline.py --fast     # smoke run with small searches, about 3 minutes
pytest                            # 43 tests; add -m "not slow" to skip the end-to-end run
streamlit run app.py              # dashboard
python notebooks/build_notebooks.py   # rebuild and execute the seven notebooks
```

Rebuilding the report (needs Node.js):

```bash
python reports/report_builder/export_report_data.py
cd reports/report_builder && npm install && node build_report.js
```

A freshly built report asks Word to update fields when first opened; accept it to fill in the table
of contents. `reports/Project_Report.pdf` is a PDF copy of the delivered report.

Tested with Python 3.13.15 on Windows 11 (package versions pinned in `requirements.txt`).

## Project structure

```
data/raw/                 original workbook and coding document (never modified)
data/processed/           cleaned data, train/test split, long format, verified data dictionary
data/synthetic/           synthetic population and its predicted probabilities
notebooks/                01 audit ... 07 optimisation (executed, with outputs)
src/
  config.py               paths, seeds, CV design, selection rule, policy bounds
  data_validation.py      audit checks, recovered speed/fare rules, data dictionary
  preprocessing.py        cleaning (logged), stratified split, ML preprocessor
  feature_engineering.py  derived features, candidate-feature review
  eda.py, figures.py      figures (plot_style.py holds the shared style)
  choice_models.py        MNL and nested logit by maximum likelihood with availability
  model_training.py       candidates, search spaces, availability masking, tuning
  model_evaluation.py     metrics, CV comparison, calibration, readiness check, selection
  synthetic_generation.py bootstrap / Gaussian copula / sequential CART, validation
  prediction.py           frozen-model prediction and population summaries
  scenario_simulation.py  policy changes and the four experiments
  policy_optimisation.py  grid search, differential evolution, verification
models/best_model/        frozen model + model card (hyperparameters, metrics, versions, seed)
models/preprocessing_pipeline/  cleaning rules, LOS rules, feature lists
models/candidates/        every tuned candidate, for reference
reports/figures/          all figures
reports/results/          every table behind the report (CSV/JSON)
reports/Project_Report.docx   final report (66 pages); Project_Report.pdf is a copy
reports/report_builder/   scripts that generate the report from reports/results
run_pipeline.py           phases 1-8 end to end
app.py                    Streamlit dashboard
tests/                    the 12 tests of the brief and a few more
```

## Decisions worth knowing about

* Category codes are kept exactly as recorded. Where the data disagree with `Data Coding.docx`
  (Trip_Purpose 1/2, occupation code 5, dwelling code 4) or the document is silent (Gender,
  Area_Type, ParkingSpace), neutral labels such as "purpose code 2" are used.
* `DL_TwoW`/`DL_Car` were dropped (duplicates of the availability columns). `LocalTrain_AV` is
  constant, so no rail-availability effect is claimed anywhere.
* Household inconsistencies (15 households with more workers than adults, etc.) are flagged, not
  edited.
* The synthetic population is generated from the training partition only and never used for
  training. It mirrors the survey sample; it is not calibrated to census totals.
* Policy bounds are scenario assumptions (editable in the dashboard). There is no cost data, so no
  financial budget is optimised; the "effort budget" is dimensionless.
* The car lever is a change in the per-km money cost of driving: a proxy for pricing measures such
  as fuel taxes or parking charges, not a modelled parking policy.

## Open questions for the data owner

1. What do Trip_Purpose codes 1 and 2 mean? (The data suggest 2 = work trip.)
2. What are occupation codes 1-5? Code 2 behaves unlike "Government employee" (64% over 55).
3. What is dwelling code 4 (191 records)?
4. How are Gender, Area_Type and ParkingSpace coded? Is ParkingSpace at home or at the destination?
5. Does "Children" mean household members under 18?
6. How was mode availability recorded, and were travel times and costs computed from distance on
   purpose?
