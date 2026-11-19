"""Team 18 app: next-day GB thermal constraint volume forecast.

Run from the project folder:
    streamlit run app.py

Uses the final model (Chronos, zero-shot) and the results saved by notebooks 01-08.
Data: Supported by National Energy SO Open Data (NESO Open Data Licence).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

RESULTS = ROOT / "results"
MODEL_NAME = "amazon/chronos-bolt-small"
CONTEXT_DAYS = 512

st.set_page_config(page_title="GB Constraint Forecast", page_icon="⚡", layout="wide")


# ---------- loading (cached so the app stays fast) ----------
@st.cache_data
def load_table() -> pd.DataFrame:
    tables = sorted((ROOT / "data" / "processed").glob("daily_model_table_*.parquet"))
    if not tables:
        st.error("No daily table found. Run notebook 01 first.")
        st.stop()
    return pd.read_parquet(tables[-1]).asfreq("D")


@st.cache_resource
def load_chronos():
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(MODEL_NAME, device_map="cpu", torch_dtype=torch.float32)


@st.cache_data
def load_csv(name: str, dates: bool = False) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        return None
    if dates:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return pd.read_csv(path)


def chronos_forecast(history: pd.Series) -> tuple[float, float, float]:
    """Median, 10% and 90% forecast for the day after the end of ``history`` (MWh)."""
    pipe = load_chronos()
    context = torch.tensor(history.tail(CONTEXT_DAYS).values, dtype=torch.float32)
    q, _ = pipe.predict_quantiles(context, prediction_length=1, quantile_levels=[0.1, 0.5, 0.9])
    low, mid, high = (float(v) for v in q[0, 0])
    return max(mid, 0.0), max(low, 0.0), max(high, 0.0)


def mwh(x: float) -> str:
    return f"{x:,.0f} MWh"


table = load_table()
target = table["target"]
last_day = target.last_valid_index()

# ---------- sidebar ----------
st.sidebar.title("⚡ GB Constraint Forecast")
page = st.sidebar.radio("Page", ["Tomorrow's forecast", "Check a past day", "Model comparison", "About"])
st.sidebar.caption("Team 18 · 7PAM2033\n\nSupported by National Energy SO Open Data")

# ---------- page 1: tomorrow ----------
if page == "Tomorrow's forecast":
    next_day = last_day + pd.Timedelta(days=1)
    st.title("Next-day thermal constraint volume")
    st.write(f"Latest NESO data: **{last_day:%A %d %B %Y}**. Forecast for **{next_day:%A %d %B %Y}**.")

    with st.spinner("Running Chronos..."):
        mid, low, high = chronos_forecast(target.loc[:last_day])
    today = float(target.loc[last_day])

    c1, c2, c3 = st.columns(3)
    c1.metric("Chronos forecast (median)", mwh(mid), delta=f"{mid - today:+,.0f} vs latest day", delta_color="off")
    c2.metric("Likely range (10%-90%)", f"{low:,.0f} – {high:,.0f}")
    c3.metric("Latest day (persistence forecast)", mwh(today))
    st.caption("On the test year the actual value fell inside the 10%-90% range on 83% of days.")

    days = st.slider("Days of history to show", 30, 365, 60, step=15)
    recent = target.loc[:last_day].tail(days)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(recent.index, recent.values, color="black", lw=1.2, label="actual")
    ax.errorbar([next_day], [mid], yerr=[[mid - low], [high - mid]], fmt="o", color="tab:green",
                capsize=6, lw=2, label="Chronos forecast (10%-90%)")
    ax.set_ylabel("MWh")
    ax.set_title("Thermal constraint volume: recent days and tomorrow's forecast")
    ax.legend(loc="upper left")
    st.pyplot(fig)

# ---------- page 2: past day ----------
elif page == "Check a past day":
    st.title("Check a past day (test year 2025-26)")
    preds = load_csv("test_predictions.csv", dates=True)
    if preds is None:
        st.warning("Run notebook 08 first (results/test_predictions.csv is missing).")
        st.stop()
    day = st.date_input("Pick a day", value=preds.index[len(preds) // 2].date(),
                        min_value=preds.index.min().date(), max_value=preds.index.max().date())
    day = pd.Timestamp(day)
    if day not in preds.index:
        st.info("No forecast for this day (missing data). Pick another day.")
        st.stop()
    row = preds.loc[day]
    c1, c2, c3 = st.columns(3)
    c1.metric("Actual", mwh(row["actual"]))
    c2.metric("Chronos", mwh(row["Chronos"]), delta=f"error {row['Chronos'] - row['actual']:+,.0f}", delta_color="off")
    c3.metric("Persistence", mwh(row["persistence"]),
              delta=f"error {row['persistence'] - row['actual']:+,.0f}", delta_color="off")

    window = preds.loc[day - pd.Timedelta(days=30): day + pd.Timedelta(days=30)]
    show = st.multiselect("Models to show", [c for c in preds.columns if c != "actual"], default=["Chronos", "persistence"])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(window.index, window["actual"], color="black", lw=1.4, label="actual")
    for m in show:
        ax.plot(window.index, window[m], lw=1, label=m)
    ax.axvline(day, color="grey", ls="--", lw=1)
    ax.set_ylabel("MWh")
    ax.set_title(f"30 days either side of {day:%d %b %Y}")
    ax.legend(loc="upper left")
    st.pyplot(fig)

# ---------- page 3: comparison ----------
elif page == "Model comparison":
    st.title("How the models compare")
    test = load_csv("test_metrics_final.csv")
    val = load_csv("model_comparison_val.csv")
    if test is not None:
        st.subheader("Final result: test year 2025-26 (never seen during building or tuning)")
        st.dataframe(test, hide_index=True, width="stretch")
        c = test.sort_values("MAE_MWh")
        fig, ax = plt.subplots(figsize=(8, 2.8))
        colors = ["tab:green" if m == "Chronos" else "grey" if m == "persistence" else "tab:blue" for m in c["model"]]
        ax.barh(c["model"], c["MAE_MWh"], color=colors)
        ax.axvline(c.loc[c["model"] == "persistence", "MAE_MWh"].iloc[0], color="black", ls="--", lw=1)
        ax.invert_yaxis()
        ax.set_xlabel("MAE (MWh), lower is better")
        st.pyplot(fig)
    if val is not None:
        st.subheader("Validation year 2024-25 (used to choose and tune the models)")
        st.dataframe(val, hide_index=True, width="stretch")
    st.caption("vs_persistence_%: above 0 means better than 'tomorrow = today'. "
               "DM_p below 0.05 means the difference is statistically significant.")

# ---------- page 4: about ----------
else:
    st.title("About this app")
    st.markdown(
        """
**What it forecasts:** the size of tomorrow's thermal constraint volume (MWh) in Great Britain,
i.e. how much generation NESO has to turn down or up because the network cannot carry it.

**Model:** Chronos-Bolt (small), a pre-trained time-series model (Ansari et al., 2024), used zero-shot
on the last 512 days of history. It was the only model to beat the persistence baseline
("tomorrow = today") on the unseen test year 2025-26: MAE 2.8% lower and RMSE 8.2% lower.
The improvement was not statistically significant, so treat forecasts as guidance.

**Other models tested:** LSTM, SARIMAX, SARIMAX + LSTM hybrid (all tuned with Optuna).

**Data:** NESO Constraint Breakdown, Historic GB Generation Mix and Day-Ahead Constraint Flows and Limits.
Supported by National Energy SO Open Data (NESO Open Data Licence).

**Team 18**, 7PAM2033 Data Science Professional Team Project, University of Hertfordshire.
"""
    )
