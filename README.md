# MonsoonWatch — explainable next-day heavy-rainfall warning for Sri Lanka

Predicts whether a Sri Lankan location will get **≥ 20 mm of rain tomorrow**
(WMO ETCCDI "very heavy precipitation day") with an **Explainable Boosting
Machine** (InterpretML), explains every prediction, and serves it through a
**Streamlit** web app where users enter today's weather and see the prediction
and its explanation.

- **Live app (Azure App Service):** LIVE_URL_PLACEHOLDER
- **Repository:** <https://github.com/dilshanpga/monsoonwatch>

## Run the app on your own computer

Needs **Python 3.12 or newer** ([python.org](https://www.python.org/downloads/);
on Windows tick *Add python.exe to PATH*).

- **Windows:** double-click `run_app.bat`
- **macOS / Linux:** `bash run_app.sh`

The first run installs the packages (3–5 minutes). The app then opens at
<http://localhost:8501>. Manual equivalent:

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

On Windows with Smart App Control, always call tools as `python -m pip` and
`python -m streamlit`; the small `pip.exe` / `streamlit.exe` launchers can be
blocked by Application Control.

**Test results (Jan 2022 – Jun 2023, never used for training or tuning)**

| Model | ROC-AUC | PR-AUC | F1 | Recall |
|---|---|---|---|---|
| **EBM (proposed)** | **0.853** | **0.235** | **0.345** | 0.489 |
| LightGBM | 0.843 | 0.210 | 0.299 | 0.390 |
| Random forest | 0.844 | 0.204 | 0.304 | 0.444 |
| Logistic regression | 0.836 | 0.204 | 0.315 | 0.534 |

## Data

*Sri Lanka Weather Dataset* (rasulmah, Kaggle, **CC0**), compiled from the
Open-Meteo historical weather API: daily weather for 30 cities, 2010-01-01 to
2023-06-17, 147,480 rows. Cities sharing a weather-model grid cell have identical
rainfall, so one city per cell is kept → 15 locations, 73,530 modelling rows.
No personal data.

## Project layout

```
data/raw/            original CSV
data/processed/      train / valid / test splits (made by data_prep.py)
src/                 pipeline scripts (run in this order)
  data_prep.py       checks, de-duplication, features, chronological split
  train.py           EBM grid search + final EBM + baselines
  evaluate.py        test metrics, bootstrap CIs, plots
  explain.py         EBM shapes, SHAP, PDP, local explanations
  export_app_assets.py  copies model + data into app/assets/
models/              saved models + model_meta.json
reports/             metrics, CSVs, figures, app screenshots
notebooks/           walkthrough notebook (whole pipeline with outputs)
app/app.py           Streamlit front-end
run_app.bat / .sh    one-click local launcher
```

## Reproduce

```bash
python -m pip install -r requirements-dev.txt
python src/data_prep.py
python src/train.py              # full search, ~5 min on 2 cores (--quick skips it)
python src/evaluate.py
python src/explain.py
python src/export_app_assets.py
python -m streamlit run app/app.py
```

Python 3.12+ recommended. Seeds fixed at 42.

## Deployment

The app runs on **Azure App Service** (Linux, Python 3.12, Basic B1). Azure's
Deployment Center is connected to this repository, so every push to `main` is
built and redeployed through GitHub Actions.

*Student project — not an official forecast. Follow the Department of
Meteorology and the Disaster Management Centre for warnings.*
