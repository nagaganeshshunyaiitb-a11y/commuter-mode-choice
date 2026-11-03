"""Project-wide constants: paths, seeds, the coding scheme and modelling settings.

Everything that a reader might want to change (seeds, CV design, policy bounds,
synthetic population size) lives here so that the rest of the code does not
hide magic numbers.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "BTech Dataset.xlsx"
RAW_SHEET = "BTech Dataset"

SEED = 42            # train/test split, tuning folds, model seeds
CV_SEED = 2026       # separate seed for the model-comparison folds
SYNTH_SEED = 7       # synthetic population generator


@dataclass(frozen=True)
class Paths:
    """All output locations, relative to a root folder.

    The end-to-end test points this at a temporary folder so that a test run
    never overwrites the real results.
    """
    root: Path = ROOT

    @property
    def processed(self): return self.root / "data" / "processed"
    @property
    def synthetic(self): return self.root / "data" / "synthetic"
    @property
    def best_model(self): return self.root / "models" / "best_model"
    @property
    def preprocessing(self): return self.root / "models" / "preprocessing_pipeline"
    @property
    def figures(self): return self.root / "reports" / "figures"
    @property
    def results(self): return self.root / "reports" / "results"

    def make_dirs(self):
        for p in (self.processed, self.synthetic, self.best_model,
                  self.preprocessing, self.figures, self.results):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Alternatives. The order matches the Choice codes 1..6 in the coding document.
# ---------------------------------------------------------------------------
MODES = ["walk", "twow", "auto", "car", "bus", "train"]
MODE_LABELS = {
    "walk": "Walk", "twow": "Two-wheeler", "auto": "Auto-rickshaw",
    "car": "Car", "bus": "Bus", "train": "Suburban rail",
}
CHOICE_CODE = {m: i + 1 for i, m in enumerate(MODES)}       # walk -> 1 ... train -> 6

AV_COLS = {
    "walk": "Walk_AV", "twow": "TwoW_AV", "auto": "Auto_AV",
    "car": "Car_AV", "bus": "Bus_AV", "train": "LocalTrain_AV",
}
TIME_COLS = {
    "walk": "walk_time_min", "twow": "TwoW_time_min", "auto": "auto_time_min",
    "car": "car_time_min", "bus": "bus_time_min", "train": "local_train_time_min",
}
COST_COLS = {   # walking has no money cost in the data
    "twow": "TC_2W", "auto": "TC_Auto", "car": "TC_Car",
    "bus": "TC_Bus", "train": "TC_LocalTrain",
}

# Operational definition used throughout (Part 8 of the brief).
SUSTAINABLE = ["walk", "bus", "train"]
INTERMEDIATE = ["auto"]
PRIVATE = ["twow", "car"]

EXPECTED_COLUMNS = [
    "ID", "Distance_km", "Trip_Purpose", "Income", "Gender", "Age", "Education",
    "Occup", "ParkingSpace", "walk_time_min", "TwoW_time_min", "auto_time_min",
    "car_time_min", "local_train_time_min", "bus_time_min", "Area_Type",
    "Dwelling", "HH_Size", "Adults", "Children", "Working_Adults", "SAPT",
    "Choice", "DL_TwoW", "DL_Car", "Car_AV", "TwoW_AV", "Walk_AV", "Auto_AV",
    "Bus_AV", "LocalTrain_AV", "Car_Ownership", "TwoW_Ownership", "TC_Car",
    "TC_2W", "TC_Auto", "TC_Bus", "TC_LocalTrain",
]

# Coding scheme exactly as written in "Data Coding.docx". Variables that the
# document does not mention are simply absent from this dict.
DOC_CODING = {
    "Age": {1: "18-30", 2: "30-55", 3: ">55"},
    "Education": {1: "Graduation", 2: "SSC/Diploma", 3: "Below SSC"},
    "Occup": {1: "Unemployed", 2: "Government employee", 3: "Service", 4: "Self-employed"},
    "Income": {1: "<30,000", 2: "30,000-80,000", 3: ">80,000"},
    "Dwelling": {1: "Apartment", 2: "Independent house", 3: "Row house"},
    "SAPT": {0: "Not satisfied", 1: "Satisfied"},
    "Trip_Purpose": {0: "Non-work", 1: "Work"},
    "Choice": {1: "Walk", 2: "Two-wheeler", 3: "Auto-rickshaw", 4: "Car", 5: "Bus", 6: "Suburban rail"},
}

# ---------------------------------------------------------------------------
# Discrete choice specification
# ---------------------------------------------------------------------------
# Travel times in the workbook are distance x a fixed speed, so an in-vehicle
# time coefficient cannot be separated from distance by the data alone (the
# unrestricted estimate is positive and insignificant). The behavioural
# specifications therefore use one generic time coefficient in which a walking
# minute counts WALK_TIME_WEIGHT in-vehicle minutes. A weight of about two is
# the usual convention in transport appraisal; the likelihood-ratio test on the
# training data does not reject it (see choice_models / the report).
WALK_TIME_WEIGHT = 2.0

# ---------------------------------------------------------------------------
# Model comparison design
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
TUNE_SPLITS, TUNE_REPEATS = 5, 2        # hyperparameter search folds
CV_SPLITS, CV_REPEATS = 5, 5            # model comparison folds (different seed)
PROB_FLOOR = 1e-6                       # numerical floor before availability renormalisation

# Pre-declared selection rule (written before any model was fitted):
#  1. shortlist models whose mean CV macro-F1 is within one standard error
#     of the best mean CV macro-F1;
#  2. inside the shortlist keep models whose balanced accuracy is within one
#     standard error of the best balanced accuracy in the shortlist;
#  3. pick the lowest CV log loss; if two log losses are within one standard
#     error, pick the simpler model (order below);
#  4. the chosen model has to pass the simulation-readiness check (see
#     model_evaluation.simulation_readiness); otherwise move down the ranking.
#  Held-out test results are reported for confirmation only.
COMPLEXITY_ORDER = ["MNL", "Nested logit", "Logistic regression",
                    "Decision tree", "Random forest", "XGBoost"]
READINESS_SHOCK_PCT = 20.0              # size of the probe change for the readiness check
READINESS_MAX_WRONG_SHARE = 0.05        # max share of people moving the "wrong" way

# ---------------------------------------------------------------------------
# Synthetic population and policy optimisation
# ---------------------------------------------------------------------------
SYNTH_N = 5000

# Upper limits for each lever (percent). These are scenario assumptions, not
# engineering facts, and can be changed from the dashboard.
POLICY_BOUNDS = {
    "bus_time_red": (0.0, 30.0),
    "bus_fare_red": (0.0, 50.0),
    "train_time_red": (0.0, 20.0),
    "train_fare_red": (0.0, 50.0),
    "car_cost_inc": (0.0, 50.0),
}
POLICY_LABELS = {
    "bus_time_red": "Bus travel-time reduction (%)",
    "bus_fare_red": "Bus fare reduction (%)",
    "train_time_red": "Suburban rail travel-time reduction (%)",
    "train_fare_red": "Suburban rail fare reduction (%)",
    "car_cost_inc": "Car out-of-pocket cost increase (%)",
}
GRID_STEP = 5.0
