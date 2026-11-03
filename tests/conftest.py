import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=UserWarning)

from src import config as C                      # noqa: E402
from src import data_validation as V             # noqa: E402
from src import preprocessing as PP              # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return V.load_raw()


@pytest.fixture(scope="session")
def cleaned(raw):
    return PP.clean_data(raw)          # (clean, corrections, rules)


@pytest.fixture(scope="session")
def split(cleaned):
    return PP.split_train_test(cleaned[0])


@pytest.fixture(scope="session")
def frozen():
    """The model frozen by run_pipeline.py. Tests that need it are skipped
    until the pipeline has been run once."""
    if not (C.Paths().best_model / "model.joblib").exists():
        pytest.skip("run `python run_pipeline.py` first")
    from src.prediction import load_frozen
    return load_frozen()


@pytest.fixture(scope="session")
def population():
    path = C.Paths().synthetic / "synthetic_population.csv"
    if not path.exists():
        pytest.skip("run `python run_pipeline.py` first")
    return pd.read_csv(path)
