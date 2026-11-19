# Week 07 – Team 18

**Project:** Forecasting Great Britain's grid constraint volumes from wind generation and demand
**Team GitHub:** https://github.com/DS-Professional-Team-Project-Group-18
**Supervisor:** Pedro Carrilho

Data: Supported by National Energy SO Open Data (NESO Open Data Licence).

## What we did this week
- **Final test** on the unseen test year 2025-26, run once (`08_test_year.ipynb`)
- **Streamlit app** with four pages: tomorrow's forecast, check a past day, model comparison, about (`app.py`)
- **App tests**: normal inputs, missing data and invalid inputs (`test_app.py`)

## Final results (test year 2025-26, 362 days)

| Model | MAE (MWh) | RMSE (MWh) | MASE | vs persistence | DM p |
|---|---|---|---|---|---|
| **Chronos (zero-shot)** | **23,684** | **29,988** | **2.249** | **+2.8%** | 0.43 |
| Persistence | 24,377 | 32,677 | 2.315 | – | – |
| Hybrid tuned | 24,731 | 33,339 | 2.348 | −1.5% | 0.75 |
| LSTM tuned | 25,320 | 33,998 | 2.404 | −3.9% | 0.44 |

Chronos is the only model that beat persistence on both the validation and test years; its 10–90% band held the
actual value on 83% of test days. The tuned models did not keep their validation advantage (tuning overfitted the
validation year). No result is statistically significant and the 10% KPI was not met.

## Files
| File | What it is |
|---|---|
| `08_test_year.ipynb` | Final scores on the test year |
| `app.py` | Streamlit app (`streamlit run app.py`) |
| `test_app.py` | 12 app tests |
| `requirements.txt` | Python packages |
| `T18_Week07_full_project_runnable.zip` | The full project (notebooks 01–08, app, code, all 51 tests, results) |

## Code status
All code is **Development** until the code review (Weeks 7–9) and Team Leader sign-off.

## Team
Navya Nalli (Team Leader), Raghavendra Para (Scrum Master),
Mahendar Reddy Tummalapalli (Communications Leader),
Tahsan Mahmud (Test Plan Manager), Devender Karnati (Code Review Manager),
Venkatesh Parimi (GitHub Manager)
