# The ML Pipeline

This repository contains a three-stage machine-learning pipeline that forecasts air
pollution (ISPU) readings six days ahead for a monitoring station in Indonesia.

The pipeline is a Python port of the original notebook
[`src/ml_src_source.ipynb`](../src/ml_src_source.ipynb). The notebook is kept in the
repository untouched as the reference artifact; the three scripts are the maintained
implementation.

```
extract.py  ->  transform.py  ->  load.py
   raw CSV        tidy frames      models, metrics, forecasts
```

Every stage reads and writes only CSV/PNG files under `artifacts/`, so the stages can
be run independently, re-run, and debugged in isolation.

---

## Table of Contents

1. [Quick start](#1-quick-start)
2. [Environment variables](#2-environment-variables)
3. [Repository layout](#3-repository-layout)
4. [Stage 1 — `extract.py`](#4-stage-1--extractpy)
5. [Stage 2 — `transform.py`](#5-stage-2--transformpy)
6. [Stage 3 — `load.py`](#6-stage-3--loadpy)
7. [Artifact tree](#7-artifact-tree)
8. [Data contract](#8-data-contract)
9. [ISPU categories](#9-ispu-categories)
10. [Output charts](#10-output-charts)

---

## 1. Quick start

Install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm catboost
```

The source CSV files are not committed to this repository. Point `DATA_DIR` at a
directory that contains them, then run the three stages in order:

```bash
export DATA_DIR=/path/to/ispu-data

python src/extract.py
python src/transform.py
python src/load.py
```

Each stage prints its progress to stdout. `load.py` is by far the slowest stage
(it performs an exhaustive hyperparameter grid search, so expect a few minutes).

---

## 2. Environment variables

All three stages read the same set of environment variables. Every one of them has a
sensible default, so the pipeline runs without any configuration.

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `DATA_DIR` | `<project>/data` | `extract.py` | Directory holding the two source CSV files |
| `TRAIN_FILE` | `Data_ISPU_2022_2024.csv` | `extract.py` | Historical training file inside `DATA_DIR` |
| `TEST_FILE` | `aqms_full_jan_mar_2026.csv` | `extract.py` | Held-out test file inside `DATA_DIR` |
| `ARTIFACT_DIR` | `<project>/artifacts` | all | Root directory for every generated artifact |
| `SAVE_PLOTS` | `1` | `transform.py`, `load.py` | Write charts to `artifacts/figures/` |
| `SHOW_PLOTS` | `0` | `transform.py`, `load.py` | Also open an interactive window for each chart |
| `SAVE_MODELS` | `1` | `load.py` | Write trained models to `artifacts/models/` |

Example:

```bash
DATA_DIR=/mnt/ispu/data ARTIFACT_DIR=/mnt/ispu/out python src/extract.py
```

---

## 3. Repository layout

```
.
├── README.md                  Project overview
├── docs/
│   ├── README.md              Documentation index
│   ├── Docker.md              PostgreSQL-in-Docker tutorial (utility reference)
│   └── pipeline.md            This file
└── src/
    ├── extract.py             Stage 1 — read + normalise raw CSVs
    ├── transform.py           Stage 2 — clean, impute, engineer features, split
    ├── load.py                Stage 3 — tune, train, evaluate, forecast
    └── ml_src_source.ipynb    Original notebook (read-only reference)
```

At runtime the pipeline also creates `artifacts/` at the repository root, which
is git-ignored.

---

## 4. Stage 1 — `extract.py`

Reads the two source CSVs and writes normalised copies into `artifacts/raw/`.

What it does:

1. Resolves `DATA_DIR` / `ARTIFACT_DIR` and validates that both source files exist.
2. Reads each file with the source CSV headers as-is.
3. Parses the `Waktu` (Indonesian for *time*) date column into a real `datetime`,
   handling the Indonesian month names used in the source data
   (`Januari`, `Februari`, `Maret`, …).
4. Renames `Waktu` to `date` and the pollutant columns to their lowercase model names.
5. Coerces the pollutant columns to numeric, treating the source `-` placeholder as
   missing.
6. Sorts chronologically, de-duplicates timestamps and writes:
   - `artifacts/raw/train_raw.csv`
   - `artifacts/raw/test_raw.csv`

Input columns (`Waktu`, `PM10`, `PM2.5`, `SO2`, `O3`, `NO2`, `CO`) are part of the
**data contract** and must not be renamed in the source files.

---

## 5. Stage 2 — `transform.py`

The cleaning and feature-engineering stage. It consumes `artifacts/raw/` and produces
the modelling frames plus the exploratory analysis tables and charts.

Steps, in order:

| # | Step | What happens |
|---|---|---|
| 1 | Raw-data exploration | Shape, missing-value rate, formal descriptive statistics and per-pollutant profiles on the untouched frames |
| 2 | k sensitivity analysis | Repeat KNN imputation for a range of `k` and report how much the mean imputed value moves |
| 3 | Imputation validation | Artificially drop 10% of complete cells (30 iterations), impute them back and compare against the true values; also compare accuracy across `k` |
| 4 | KNN imputation | Missing pollutant values filled with KNN (k=5) over min-max-scaled data, then inverse-scaled |
| 5 | IQR capping | Clamp extreme outliers to `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]` on the training set, then apply the same bounds to the test set; check variance over the test window |
| 6 | ISPU classification | Bin `pm25` into the five PM2.5 ISPU categories → `pm25_category` |
| 7 | Feature engineering | Lag features (`_lag_{n}d`) and rolling means (`_rmean_{n}d`) for every pollutant, plus calendar features and cyclical time encoding |
| 8 | Chronological split | Train = 2022–2024, validation and test = the 2026 file, split 50:50 |

Models see `MODEL_LAGS = 7` lags and rolling windows of `(7, 14)` days
(`MAX_LAGS = 13` is used for the exploratory profiles).

Key constants live at the top of the file:

```python
POLLUTANT_COLS     = ["co", "no2", "o3", "so2", "pm10", "pm25"]
ISPU_BINS          = [0, 50, 100, 200, 300, np.inf]
KNN_K              = 5
VALIDATION_MISSING_RATE = 0.10
VALIDATION_N_ITER  = 30
MAX_LAGS           = 13
MODEL_LAGS         = 7
MODEL_ROLLING      = (7, 14)
```

Outputs:

- `artifacts/transformed/train_ml_input.csv`, `test_ml_input.csv` — full feature
  matrices consumed by `load.py`
- `artifacts/transformed/train_features.csv`, `test_features.csv` — pre-capping frames
- `artifacts/transformed/train_capped.csv`, `test_capped.csv` — post-capping frames
- `artifacts/tables/raw_descriptive_stats.csv`
- `artifacts/tables/imputation_validation.csv`
- `artifacts/tables/k_comparison.csv`
- `artifacts/tables/ispu_category_distribution.csv`
- `artifacts/figures/*.png`

`run_transform()` returns the same objects as an in-memory dict, so `load.py` can
call it directly instead of re-reading the CSVs.

---

## 6. Stage 3 — `load.py`

The modelling stage. In ML-pipeline terminology "load" means *load the transformed
data into the models and write the outputs to disk*.

Steps:

| # | Step | What happens |
|---|---|---|
| 1 | Hyperparameter tuning | Exhaustive grid search per (model, target) on a train/val split. `24 × 6 = 144` fits per framework, `288` total |
| 2 | Cross-validation | `TimeSeriesSplit` (5 folds) on train+val using the winning hyperparameters. Folds with near-zero target variance are skipped |
| 3 | Final training + test evaluation | Retrain on train+val, evaluate all six targets on the test set, and rank MAE / RMSE / R² per target |
| 4 | Scenario forecasting | Train and evaluate a separate model per horizon: 7, 10 and 14 days |
| 5 | Combined forecast | Stitch days 1–7 from the 7-day scenario, 8–10 from the 10-day scenario and 11–14 from the 14-day scenario into a single 18–31 March 2026 forecast, then bin it into ISPU categories |
| 6 | Persistence | Write models, tables and predictions to `artifacts/` |

Two gradient-boosting frameworks are compared side by side: **CatBoost** and
**LightGBM**. The evaluation prints a win-count summary and names the best overall
model.

Scenario definitions:

```python
SCENARIOS = {
    7:  {"n_lags": 7,  "rolling": [7]},
    10: {"n_lags": 10, "rolling": [7, 10]},
    14: {"n_lags": 14, "rolling": [7, 14]},
}
FORECAST_START = "2026-03-18"
FORECAST_END   = "2026-03-31"
N_SPLITS_CV    = 5
```

Outputs:

- `artifacts/models/catboost_*.txt`, `lightgbm_*.txt` — trained models
- `artifacts/tables/cross_validation.csv`
- `artifacts/tables/test_evaluation.csv`
- `artifacts/tables/ranking_model.csv`
- `artifacts/tables/scenario_results.csv`
- `artifacts/tables/best_params_catboost.csv`, `best_params_lightgbm.csv`
- `artifacts/tables/forecast_ispu_distribution.csv`
- `artifacts/predictions/march_predictions_per_scenario.csv`
- `artifacts/predictions/combined_forecast_18_31_march.csv`
- `artifacts/figures/*.png`

---

## 7. Artifact tree

After a complete run, `artifacts/` looks like this:

```
artifacts/
├── raw/
│   ├── train_raw.csv
│   └── test_raw.csv
├── transformed/
│   ├── train_features.csv
│   ├── test_features.csv
│   ├── train_capped.csv
│   ├── test_capped.csv
│   ├── train_ml_input.csv
│   └── test_ml_input.csv
├── tables/
│   ├── raw_descriptive_stats.csv
│   ├── imputation_validation.csv
│   ├── k_comparison.csv
│   ├── ispu_category_distribution.csv
│   ├── cross_validation.csv
│   ├── test_evaluation.csv
│   ├── ranking_model.csv
│   ├── scenario_results.csv
│   ├── best_params_catboost.csv
│   ├── best_params_lightgbm.csv
│   └── forecast_ispu_distribution.csv
├── figures/
│   ├── *.png
├── models/
│   ├── catboost_<target>.txt
│   └── lightgbm_<target>.txt
└── predictions/
    ├── march_predictions_per_scenario.csv
    └── combined_forecast_18_31_march.csv
```

Delete the directory and re-run all three stages to reproduce it from scratch.

---

## 8. Data contract

These inputs are **not** owned by this repository and must keep their original form.
The pipeline parses them; it never rewrites the source files.

| Contract item | Value |
|---|---|
| Source filenames | `Data_ISPU_2022_2024.csv`, `aqms_full_jan_mar_2026.csv` (overridable via `TRAIN_FILE` / `TEST_FILE`) |
| Date column | `Waktu` |
| Pollutant columns | `PM10`, `PM2.5`, `SO2`, `O3`, `NO2`, `CO` |
| Month names in the date strings | `Januari`, `Februari`, `Maret`, `April`, `Mei`, `Juni`, `Juli`, `Agustus`, `September`, `Oktober`, `November`, `Desember` |
| Missing-value placeholder | `-` |
| ISPU bin edges | `ISPU_BINS = [0, 50, 100, 200, 300, inf]` |

Everything else in the pipeline — column names inside `artifacts/`, table headers,
chart labels, file names of the generated outputs — is English.

---

## 9. ISPU categories

ISPU (*Indeks Standar Pencemar Udara*, the Indonesian Air Pollution Standard Index)
categorises PM2.5 concentrations into five bands.

| Band | Indonesian | English | Range (µg/m³) |
|---|---|---|---|
| 1 | Baik | Good | 0 – 50 |
| 2 | Sedang | Moderate | 51 – 100 |
| 3 | Tidak Sehat | Unhealthy | 101 – 200 |
| 4 | Sangat Tidak Sehat | Very Unhealthy | 201 – 300 |
| 5 | Berbahaya | Hazardous | > 300 |

Source: **Permen LHK No. 14/2020** (Ministerial Regulation of Environment and
Forestry No. 14 of 2020).

The pipeline uses two related label sets:

- `ISPU_LABELS` in `transform.py` — the full five-band scale, used to classify the
  PM2.5 training data into `pm25_category`.
- `FORECAST_ISPU_LABELS` in `load.py` — the first four bands, used to describe the
  14-day forecast distribution, together with per-pollutant threshold tables
  (`FORECAST_ISPU_THRESHOLDS`).

---

## 10. Output charts

Charts are written to `artifacts/figures/` as PNG when `SAVE_PLOTS=1` (the default).

| File | Produced by | Content |
|---|---|---|
| `eda_raw_profiles.png` | `transform.py` | Per-pollutant time series on the raw data |
| `eda_post_imputation.png` | `transform.py` | Distribution comparison before vs after cleaning |
| `eda_trends_correlation_trends.png` | `transform.py` | Final trend lines per pollutant |
| `eda_trends_correlation_correlation.png` | `transform.py` | Pollutant correlation matrix |
| `ispu_category_distribution.png` | `transform.py` | Raw vs final ISPU category counts |
| `model_comparison.png` | `load.py` | RMSE and R² per parameter for both models (two panels) |
| `predictions_vs_actual.png` | `load.py` | 3×2 grid of predicted vs actual test values |
| `scenario_comparison_rmse.png`, `scenario_comparison_r2.png` | `load.py` | Metrics per 7/10/14-day scenario |
| `combined_march_forecast.png` | `load.py` | Actual March values + combined 18–31 March forecast |
| `ispu_forecast_distribution.png` | `load.py` | ISPU category counts over the 14-day forecast |
