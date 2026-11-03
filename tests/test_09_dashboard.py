"""The Streamlit dashboard: every page renders and the interactive parts run."""
import pytest

from src import config as C

PAGES = ["Dataset overview", "Model comparison", "Synthetic population", "Policy scenario simulator",
         "Policy optimisation"]


@pytest.fixture(scope="module")
def app():
    if not (C.Paths().best_model / "model.joblib").exists():
        pytest.skip("run the pipeline first")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, app.exception


def test_simulator_reacts_to_sliders(app):
    app.sidebar.radio[0].set_value("Policy scenario simulator").run()
    before = app.metric[0].value
    app.slider[0].set_value(-30).run()          # bus fare
    app.slider[4].set_value(40).run()           # car cost
    assert not app.exception
    assert app.metric[0].value != before


def test_individual_traveller_tab(app):
    app.sidebar.radio[0].set_value("Policy scenario simulator").run()
    app.number_input[0].set_value(12.0).run()   # distance
    assert not app.exception
    tables = [df.value for df in app.dataframe if "Travel time (min)" in df.value.columns]
    assert tables, "traveller table not shown"
    person = tables[0]
    # the table is shown rounded to 2 decimals: six values can drift by up to 0.03 in total
    assert abs(person["Current"].sum() - 100) <= 0.03 + 1e-9
    assert abs(person.loc[person["mode"] == "Walk", "Travel time (min)"].iloc[0] - 12.0 * 15) < 1e-6


def test_custom_optimisation_runs(app):
    app.sidebar.radio[0].set_value("Policy optimisation").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("combinations" in s.value for s in app.success)
