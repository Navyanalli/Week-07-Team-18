"""Tests for the Streamlit app (app.py): normal inputs, missing data and invalid inputs.

The app is copied into a temporary folder with small made-up data, so these
tests do not need the real NESO files. Chronos is replaced by a tiny fake that
returns "yesterday's value" with a band of +/-50%, so no model is downloaded.
"""

import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
torch = pytest.importorskip("torch")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"
MISSING_DAY = pd.Timestamp("2025-04-08")


class FakeChronos:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def predict_quantiles(self, context, prediction_length, quantile_levels):
        last = float(context[-1])
        return torch.tensor([[[0.5 * last, last, 1.5 * last]]]), None


@pytest.fixture(autouse=True)
def fake_chronos(monkeypatch):
    monkeypatch.setitem(sys.modules, "chronos", types.SimpleNamespace(BaseChronosPipeline=FakeChronos))
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def make_project(tmp_path: Path, with_table=True, with_results=True) -> Path:
    """A small copy of the project: app.py, a daily table and the results files."""
    shutil.copy(APP, tmp_path / "app.py")
    idx = pd.date_range("2024-01-01", "2026-03-31", freq="D")
    target = pd.Series(np.arange(len(idx), dtype=float) % 50 * 1000 + 500, index=idx)
    if with_table:
        (tmp_path / "data" / "processed").mkdir(parents=True)
        t = pd.DataFrame({"target": target})
        t.loc[MISSING_DAY, "target"] = np.nan
        t.to_parquet(tmp_path / "data" / "processed" / "daily_model_table_20260101.parquet")
    if with_results:
        res = tmp_path / "results"
        res.mkdir()
        test_days = pd.date_range("2025-04-02", "2026-03-31", freq="D").drop(MISSING_DAY)
        preds = pd.DataFrame({"actual": target.loc[test_days],
                              "persistence": target.shift(1).loc[test_days],
                              "Chronos": target.loc[test_days] * 0.9}, index=test_days)
        preds.to_csv(res / "test_predictions.csv")
        pd.DataFrame({"model": ["Chronos", "persistence"], "MAE_MWh": [100.0, 120.0], "RMSE_MWh": [150.0, 170.0],
                      "MASE": [0.9, 1.0], "vs_persistence_%": [16.7, 0.0], "DM_p": [0.04, None]}
                     ).to_csv(res / "test_metrics_final.csv", index=False)
    return tmp_path / "app.py"


def run_page(app_file: Path, page: str) -> AppTest:
    at = AppTest.from_file(str(app_file), default_timeout=60)
    at.run()
    if page != "Tomorrow's forecast" and len(at.sidebar.radio):
        at.sidebar.radio[0].set_value(page).run()
    return at


# ---------- normal inputs ----------
@pytest.mark.parametrize("page", ["Tomorrow's forecast", "Check a past day", "Model comparison", "About"])
def test_every_page_opens_without_errors(tmp_path, page):
    at = run_page(make_project(tmp_path), page)
    assert not at.exception


def test_tomorrow_forecast_is_for_the_day_after_the_latest_data(tmp_path):
    at = run_page(make_project(tmp_path), "Tomorrow's forecast")
    assert len(at.metric) == 3
    assert "31 March 2026" in at.markdown[0].value and "01 April 2026" in at.markdown[0].value
    # fake Chronos returns the last value, so median = latest day
    assert at.metric[0].value == at.metric[2].value


def test_past_day_shows_actual_and_both_forecasts(tmp_path):
    at = run_page(make_project(tmp_path), "Check a past day")
    at.date_input[0].set_value(pd.Timestamp("2025-06-01").date()).run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Actual", "Chronos", "Persistence"]


def test_comparison_page_shows_the_results_table(tmp_path):
    at = run_page(make_project(tmp_path), "Model comparison")
    assert len(at.dataframe) >= 1
    assert "MASE" in at.dataframe[0].value.columns


# ---------- missing data ----------
def test_missing_day_gives_a_message_not_a_crash(tmp_path):
    at = run_page(make_project(tmp_path), "Check a past day")
    at.date_input[0].set_value(MISSING_DAY.date()).run()
    assert not at.exception
    assert any("No forecast for this day" in i.value for i in at.info)


def test_forecast_never_negative_even_with_zero_history(tmp_path):
    app = make_project(tmp_path)
    t = pd.read_parquet(next((tmp_path / "data" / "processed").glob("*.parquet")))
    t["target"] = 0.0
    t.to_parquet(next((tmp_path / "data" / "processed").glob("*.parquet")))
    at = run_page(app, "Tomorrow's forecast")
    assert not at.exception
    assert not at.metric[0].value.startswith("-")


# ---------- invalid inputs ----------
def test_no_daily_table_shows_an_error(tmp_path):
    at = run_page(make_project(tmp_path, with_table=False), "Tomorrow's forecast")
    assert not at.exception
    assert any("No daily table found" in e.value for e in at.error)


def test_no_results_files_shows_a_warning(tmp_path):
    at = run_page(make_project(tmp_path, with_results=False), "Check a past day")
    assert not at.exception
    assert any("Run notebook 08 first" in w.value for w in at.warning)


def test_comparison_page_without_results_does_not_crash(tmp_path):
    at = run_page(make_project(tmp_path, with_results=False), "Model comparison")
    assert not at.exception
