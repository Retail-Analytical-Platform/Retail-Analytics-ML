"""
LOAD — Stage 3 of the ISPU Pollution Forecast pipeline.

Responsibilities (following the cell order of the ml_src_source.ipynb notebook):
    1. Hyperparameter tuning of CatBoost & LightGBM per pollutant target.
    2. TimeSeriesSplit cross-validation (5-fold) on train+val.
    3. Final model training on train+val, evaluation on the test set.
    4. Multi-scenario 7 / 10 / 14 day forecasting.
    5. Combined 18-31 March 2026 predictions + forecast ISPU category distribution.
    6. Persistence of every table & chart as an artifact.

Note: in an ML pipeline, "load" means loading the transformed data into the
models and writing the outputs (models, metrics, predictions) to disk.

This script can be run directly:  python load.py
"""

from __future__ import annotations

import os
from itertools import product

import matplotlib

matplotlib.use("Agg")  # headless backend; plt.show() stays safe

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from extract import ARTIFACT_DIR, RANDOM_STATE  # noqa: E402
from transform import (  # noqa: E402
    POLLUTANT_COLS,
    TRANSFORM_DIR,
    UNIT_MAP,
    _ensure_dir,
    _save_table,
    _savefig,
    chronological_split,
    get_features,
    run_transform,
)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

MODEL_DIR = ARTIFACT_DIR / "models"
PREDICTION_DIR = ARTIFACT_DIR / "predictions"

SAVE_MODELS = os.environ.get("SAVE_MODELS", "1") == "1"

# Hyperparameter grids (identical to the notebook).
CATBOOST_GRID = {
    "iterations": [500, 1000],
    "learning_rate": [0.01, 0.05, 0.10],
    "depth": [4, 6],
    "l2_leaf_reg": [1, 3],
}

LIGHTGBM_GRID = {
    "n_estimators": [500, 1000],
    "learning_rate": [0.01, 0.05, 0.10],
    "max_depth": [4, 6],
    "num_leaves": [15, 31],
}

LGB_MIN_DATA_IN_LEAF = 5

N_SPLITS_CV = 5

# Forecast scenarios: {days: {"n_lags": ..., "rolling": [...]}}
SCENARIOS = {
    7: {"n_lags": 7, "rolling": [7]},
    10: {"n_lags": 10, "rolling": [7, 10]},
    14: {"n_lags": 14, "rolling": [7, 14]},
}

# Combined forecast window (days 8-14 following 17 March 2026).
FORECAST_START = "2026-03-18"
FORECAST_END = "2026-03-31"

# ISPU category thresholds (Ministerial Regulation No. 14/2020) — 4 categories.
FORECAST_ISPU_LABELS = ["Good", "Moderate", "Unhealthy", "Very Unhealthy"]
FORECAST_ISPU_THRESHOLDS = {
    "pm25": [0, 15.5, 55.4, 150.4, float("inf")],
    "pm10": [0, 50, 150, 350, float("inf")],
    "so2": [0, 52, 180, 400, float("inf")],
    "no2": [0, 80, 200, 565, float("inf")],
    "o3": [0, 120, 235, 400, float("inf")],
    "co": [0, 4000, 8000, 15000, float("inf")],
}
CATEGORY_COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]


# ---------------------------------------------------------------------------
# 1. Hyperparameter tuning
# ---------------------------------------------------------------------------


def grid_combinations(grid: dict):
    """Yield every value combination from a grid dict."""
    keys = list(grid.keys())
    for combo in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def count_combinations(*grids: dict) -> int:
    """Total number of combinations across all grids."""
    return sum(int(np.prod([len(v) for v in g.values()])) for g in grids)


def build_catboost(params: dict, **overrides) -> CatBoostRegressor:
    return CatBoostRegressor(
        **params, loss_function="RMSE", random_seed=RANDOM_STATE, verbose=0, **overrides
    )


def build_lightgbm(params: dict, **overrides) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        **params,
        min_data_in_leaf=LGB_MIN_DATA_IN_LEAF,
        random_state=RANDOM_STATE,
        verbose=-1,
        **overrides,
    )


def tune_catboost(X_tr, y_tr, X_val, y_val, grid: dict | None = None):
    """Find the CatBoost hyperparameters with the lowest validation RMSE."""
    grid = grid or CATBOOST_GRID
    best_rmse, best_params, best_model = np.inf, None, None
    for params in grid_combinations(grid):
        model = build_catboost(params)
        model.fit(X_tr, y_tr)
        rmse = float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))
        if rmse < best_rmse:
            best_rmse, best_params, best_model = rmse, params, model
    return best_model, best_params, best_rmse


def tune_lightgbm(X_tr, y_tr, X_val, y_val, grid: dict | None = None):
    """Find the LightGBM hyperparameters with the lowest validation RMSE."""
    grid = grid or LIGHTGBM_GRID
    best_rmse, best_params, best_model = np.inf, None, None
    for params in grid_combinations(grid):
        model = build_lightgbm(params)
        model.fit(X_tr, y_tr)
        rmse = float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))
        if rmse < best_rmse:
            best_rmse, best_params, best_model = rmse, params, model
    return best_model, best_params, best_rmse


def run_tuning(train_df: pd.DataFrame, val_df: pd.DataFrame, verbose: bool = True):
    """Tune separately per (model, target) on the train/val split."""
    best_params_cb: dict[str, dict] = {}
    best_params_lgb: dict[str, dict] = {}
    val_rmse_cb: dict[str, float] = {}
    val_rmse_lgb: dict[str, float] = {}

    if verbose:
        n_cb = count_combinations(CATBOOST_GRID)
        n_lgb = count_combinations(LIGHTGBM_GRID)
        print("=" * 78)
        print("1. HYPERPARAMETER TUNING")
        print("=" * 78)
        print(f"CatBoost combinations : {n_cb} | LightGBM : {n_lgb}")
        print(f"Fits per target : {n_cb + n_lgb} | for 6 targets : {(n_cb + n_lgb) * 6}\n")

    for target in POLLUTANT_COLS:
        feats = get_features(target, train_df.columns)
        X_tr, y_tr = train_df[feats], train_df[target]
        X_val, y_val = val_df[feats], val_df[target]

        _, p_cb, rmse_cb = tune_catboost(X_tr, y_tr, X_val, y_val)
        best_params_cb[target] = p_cb
        val_rmse_cb[target] = rmse_cb

        _, p_lgb, rmse_lgb = tune_lightgbm(X_tr, y_tr, X_val, y_val)
        best_params_lgb[target] = p_lgb
        val_rmse_lgb[target] = rmse_lgb

        if verbose:
            print(f"  {target.upper():<6}")
            print(f"    CatBoost best  | RMSE_val={rmse_cb:7.4f} | {p_cb}")
            print(f"    LightGBM best  | RMSE_val={rmse_lgb:7.4f} | {p_lgb}")

    return {
        "best_params_cb": best_params_cb,
        "best_params_lgb": best_params_lgb,
        "val_rmse_cb": val_rmse_cb,
        "val_rmse_lgb": val_rmse_lgb,
    }


# ---------------------------------------------------------------------------
# 2. Cross-validation TimeSeriesSplit
# ---------------------------------------------------------------------------


def cross_validate(
    trainval_df: pd.DataFrame,
    best_params_cb: dict[str, dict],
    best_params_lgb: dict[str, dict],
    n_splits: int = N_SPLITS_CV,
    verbose: bool = True,
) -> pd.DataFrame:
    """TimeSeriesSplit evaluation using the best hyperparameters per target.

    Folds whose target variance is ~0 are skipped because the metrics are meaningless.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    if verbose:
        print("\n" + "=" * 90)
        print(f"2. TIME SERIES SPLIT CROSS-VALIDATION ({n_splits}-fold, best hyperparams)")
        print("=" * 90)

    for target in POLLUTANT_COLS:
        feats = get_features(target, trainval_df.columns)
        X = trainval_df[feats]
        y = trainval_df[target]

        rmse_cb, mae_cb, r2_cb = [], [], []
        rmse_lg, mae_lg, r2_lg = [], [], []

        for tr_idx, te_idx in tscv.split(X):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            if y_te.std() < 1e-3:
                continue

            m_cb = build_catboost(best_params_cb[target])
            m_cb.fit(X_tr, y_tr)
            p_cb = m_cb.predict(X_te)

            m_lg = build_lightgbm(best_params_lgb[target])
            m_lg.fit(X_tr, y_tr)
            p_lg = m_lg.predict(X_te)

            rmse_cb.append(np.sqrt(mean_squared_error(y_te, p_cb)))
            mae_cb.append(mean_absolute_error(y_te, p_cb))
            r2_cb.append(r2_score(y_te, p_cb))
            rmse_lg.append(np.sqrt(mean_squared_error(y_te, p_lg)))
            mae_lg.append(mean_absolute_error(y_te, p_lg))
            r2_lg.append(r2_score(y_te, p_lg))

        rows.append(
            {
                "Parameter": target.upper(),
                "RMSE_CB": round(np.mean(rmse_cb), 3),
                "MAE_CB": round(np.mean(mae_cb), 3),
                "R²_CB": round(np.mean(r2_cb), 3),
                "RMSE_LGB": round(np.mean(rmse_lg), 3),
                "MAE_LGB": round(np.mean(mae_lg), 3),
                "R²_LGB": round(np.mean(r2_lg), 3),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Final models & test-set evaluation
# ---------------------------------------------------------------------------


def train_final_models(
    trainval_df: pd.DataFrame, best_params_cb: dict, best_params_lgb: dict, verbose: bool = True
):
    """Train the final models on train+val with the best hyperparameters."""
    models_cb: dict = {}
    models_lgb: dict = {}

    if verbose:
        print("\nTraining final models on train+val...")

    for target in POLLUTANT_COLS:
        feats = get_features(target, trainval_df.columns)
        X_tr, y_tr = trainval_df[feats], trainval_df[target]

        m_cb = build_catboost(best_params_cb[target])
        m_cb.fit(X_tr, y_tr)
        models_cb[target] = m_cb

        m_lg = build_lightgbm(best_params_lgb[target])
        m_lg.fit(X_tr, y_tr)
        models_lgb[target] = m_lg

        if verbose:
            print(f"  ✓ {target.upper():<6} model final trained")

    return models_cb, models_lgb


def _regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
    }


def evaluate_on_test(
    test_df: pd.DataFrame, models_cb: dict, models_lgb: dict, verbose: bool = True
) -> pd.DataFrame:
    """Evaluate every target on the test set, plus pick the best model."""
    rows = []
    predictions: dict[str, dict[str, np.ndarray]] = {}

    for target in POLLUTANT_COLS:
        feats = get_features(target, test_df.columns)
        X_te, y_te = test_df[feats], test_df[target]

        p_cb = np.clip(models_cb[target].predict(X_te), 0, None)
        p_lgb = np.clip(models_lgb[target].predict(X_te), 0, None)

        m_cb = _regression_metrics(y_te, p_cb)
        m_lgb = _regression_metrics(y_te, p_lgb)

        rows.append(
            {
                "Parameter": target.upper(),
                "MAE_CB": round(m_cb["MAE"], 3),
                "RMSE_CB": round(m_cb["RMSE"], 3),
                "R²_CB": round(m_cb["R²"], 3),
                "MAE_LGB": round(m_lgb["MAE"], 3),
                "RMSE_LGB": round(m_lgb["RMSE"], 3),
                "R²_LGB": round(m_lgb["R²"], 3),
                "Best Model": "CatBoost" if m_cb["RMSE"] <= m_lgb["RMSE"] else "LightGBM",
            }
        )
        predictions[target] = {"cb": p_cb, "lgb": p_lgb, "actual": y_te.to_numpy()}

    df_results = pd.DataFrame(rows)
    if verbose:
        print("\n" + "=" * 100)
        print("3. EVALUATION RESULTS — TEST SET")
        print("=" * 100)
        print(df_results.to_string(index=False))
        print("\nKeterangan:")
        print("  • MAE  : Mean Absolute Error")
        print("  • RMSE : Root Mean Squared Error (lower is better)")
        print("  • R²   : Coefficient of Determination (closer to 1 is better)")

    df_results.attrs["predictions"] = predictions
    return df_results


def rank_models_by_wins(df_results: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Ranking-based score: how many metrics (MAE/RMSE/R²) are won."""
    rows = []
    for _, row in df_results.iterrows():
        wins_lgb = (
            (0 if row["MAE_CB"] < row["MAE_LGB"] else (1 if row["MAE_CB"] > row["MAE_LGB"] else 0.5))
            + (0 if row["RMSE_CB"] < row["RMSE_LGB"] else (1 if row["RMSE_CB"] > row["RMSE_LGB"] else 0.5))
            + (0 if row["R²_CB"] > row["R²_LGB"] else (1 if row["R²_CB"] < row["R²_LGB"] else 0.5))
        )
        wins_cb = 3 - wins_lgb
        rows.append(
            {
                "Parameter": row["Parameter"],
                "CB Wins": wins_cb,
                "LGB Wins": wins_lgb,
                "Best": row["Best Model"],
            }
        )

    df_rank = pd.DataFrame(rows)
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY — NUMBER OF METRICS WON (MAE, RMSE, R²)")
        print("=" * 70)
        print(df_rank.to_string(index=False))

        n_cb = int((df_results["Best Model"] == "CatBoost").sum())
        n_lgb = int((df_results["Best Model"] == "LightGBM").sum())
        n_total = len(POLLUTANT_COLS)
        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print(f"  • CatBoost is best for {n_cb} of {n_total} parameters")
        print(f"  • LightGBM is best for {n_lgb} of {n_total} parameters")
        if n_cb > n_lgb:
            overall = "CatBoost"
        elif n_lgb > n_cb:
            overall = "LightGBM"
        else:
            overall = "Tie — choice depends on the target parameter"
        print(f"  • Best overall predictive model: {overall}")

    return df_rank


def plot_model_comparison(df_results: pd.DataFrame, name: str = "model_comparison") -> None:
    """Bar chart of RMSE & R² per parameter for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(POLLUTANT_COLS))
    width = 0.35

    for ax, (col_cb, col_lgb, ylabel, title) in zip(
        axes,
        [
            ("RMSE_CB", "RMSE_LGB", "RMSE", "RMSE Comparison on the Test Set"),
            ("R²_CB", "R²_LGB", "R²", "R² Comparison on the Test Set"),
        ],
    ):
        ax.bar(x - width / 2, df_results[col_cb], width, label="CatBoost", color="teal", edgecolor="black")
        ax.bar(x + width / 2, df_results[col_lgb], width, label="LightGBM", color="mediumpurple", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([p.upper() for p in POLLUTANT_COLS])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        if ylabel == "R²":
            ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for i, (cb, lg) in enumerate(zip(df_results[col_cb], df_results[col_lgb])):
            va = "bottom" if cb >= 0 else "top"
            ax.text(i - width / 2, cb, f"{cb:.2f}", ha="center", va=va, fontsize=8)
            va = "bottom" if lg >= 0 else "top"
            ax.text(i + width / 2, lg, f"{lg:.2f}", ha="center", va=va, fontsize=8)

    fig.suptitle("PERFORMANCE COMPARISON — CatBoost vs LightGBM — Test Set", fontweight="bold", fontsize=13)
    fig.tight_layout()
    _savefig(fig, name)


def plot_predictions_vs_actual(
    test_df: pd.DataFrame,
    models_cb: dict,
    models_lgb: dict,
    name: str = "predictions_vs_actual",
) -> None:
    """3x2 grid: actual vs both models' predictions on the test set."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, target in enumerate(POLLUTANT_COLS):
        feats = get_features(target, test_df.columns)
        X_te, y_te = test_df[feats], test_df[target]
        p_cb = np.clip(models_cb[target].predict(X_te), 0, None)
        p_lgb = np.clip(models_lgb[target].predict(X_te), 0, None)

        ax = axes[i]
        ax.plot(test_df.index, y_te.values, "o-", color="indianred", label="Actual", linewidth=2, markersize=4)
        ax.plot(test_df.index, p_cb, "s--", color="teal", label="CatBoost", linewidth=1.8, markersize=4, alpha=0.8)
        ax.plot(test_df.index, p_lgb, "^--", color="mediumpurple", label="LightGBM", linewidth=1.8, markersize=4, alpha=0.8)
        ax.set_title(f"{target.upper()} ({UNIT_MAP[target]})", fontweight="bold")
        ax.set_ylabel(UNIT_MAP[target])
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Prediction vs Actual on the Test Set", fontweight="bold", fontsize=14)
    fig.tight_layout()
    _savefig(fig, name)


# ---------------------------------------------------------------------------
# 4. Forecast scenarios 7 / 10 / 14 days
# ---------------------------------------------------------------------------


def build_scenario_features(
    df_input: pd.DataFrame,
    n_lags: int,
    rolling_windows,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Feature matrix using the lag/rolling scheme of a specific scenario."""
    df_res = df_input[POLLUTANT_COLS].copy().sort_index()
    df_res = df_res[~df_res.index.duplicated(keep="first")]

    for param in POLLUTANT_COLS:
        for i in range(1, n_lags + 1):
            df_res[f"{param}_lag_{i}"] = df_res[param].shift(i)

    for param in POLLUTANT_COLS:
        for w in rolling_windows:
            df_res[f"{param}_rmean_{w}d"] = df_res[param].shift(1).rolling(w).mean()

    add_temporal(df_res)

    return df_res.dropna() if drop_na else df_res


def add_temporal(df_res: pd.DataFrame) -> None:
    """Calendar features + cyclical encoding (identical to transform.py)."""
    index = df_res.index
    df_res["day_of_week"] = index.dayofweek
    df_res["day_of_month"] = index.day
    df_res["day_of_year"] = index.dayofyear
    df_res["week_of_year"] = index.isocalendar().week.astype(int)
    df_res["month"] = index.month
    df_res["quarter"] = index.quarter
    df_res["is_weekend"] = (index.dayofweek >= 5).astype(int)

    df_res["dow_sin"] = np.sin(2 * np.pi * df_res["day_of_week"] / 7)
    df_res["dow_cos"] = np.cos(2 * np.pi * df_res["day_of_week"] / 7)
    df_res["month_sin"] = np.sin(2 * np.pi * df_res["month"] / 12)
    df_res["month_cos"] = np.cos(2 * np.pi * df_res["month"] / 12)
    df_res["doy_sin"] = np.sin(2 * np.pi * df_res["day_of_year"] / 365)
    df_res["doy_cos"] = np.cos(2 * np.pi * df_res["day_of_year"] / 365)


def get_scenario_features(df_scn: pd.DataFrame, target: str, n_lags: int, rolling_windows) -> list[str]:
    """Model input feature list for a given scenario."""
    feats = []
    for param in POLLUTANT_COLS:
        for i in range(1, n_lags + 1):
            feats.append(f"{param}_lag_{i}")
        for w in rolling_windows:
            feats.append(f"{param}_rmean_{w}d")

    feats.extend(
        [
            "day_of_week", "day_of_month", "day_of_year", "week_of_year",
            "month", "quarter", "is_weekend",
            "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
        ]
    )
    return [f for f in feats if f in set(df_scn.columns) and f != target]


def run_scenarios(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    scenarios: dict | None = None,
    verbose: bool = True,
):
    """Train & evaluate models per 7/10/14-day scenario.

    Returns:
        df_scenario, scenario_models, scenario_predictions
    """
    scenarios = scenarios or SCENARIOS
    rows: list[dict] = []
    scenario_models: dict[int, dict] = {}
    scenario_predictions: dict[int, dict] = {}

    for scenario_days, cfg in scenarios.items():
        n_lags, rolling_w = cfg["n_lags"], cfg["rolling"]

        if verbose:
            print("\n" + "=" * 78)
            print(f"4. {scenario_days}-DAY SCENARIO  (n_lags={n_lags}, rolling={rolling_w})")
            print("=" * 78)

        df_tr_scn = build_scenario_features(df_train, n_lags, rolling_w)
        df_te_scn = build_scenario_features(df_test, n_lags, rolling_w)

        val_size = len(df_te_scn) // 2
        train_s = df_tr_scn.copy()
        val_s = df_te_scn.iloc[:val_size]
        test_s = df_te_scn.iloc[val_size:]
        trainval_s = pd.concat([train_s, val_s])

        if verbose:
            print(f"  Train={len(train_s)} | Val={len(val_s)} | Test={len(test_s)}")
            print(f"  Test period: {test_s.index.min().date()} -> {test_s.index.max().date()}")

        scenario_models[scenario_days] = {}
        scenario_predictions[scenario_days] = {"test_index": test_s.index, "targets": {}}

        for target in POLLUTANT_COLS:
            feats = get_scenario_features(df_tr_scn, target, n_lags, rolling_w)
            X_tr, y_tr = trainval_s[feats], trainval_s[target]
            X_te, y_te = test_s[feats], test_s[target]

            m_cb = build_catboost({})
            m_cb.fit(X_tr, y_tr)
            p_cb = np.clip(m_cb.predict(X_te), 0, None)

            m_lg = build_lightgbm({})
            m_lg.fit(X_tr, y_tr)
            p_lg = np.clip(m_lg.predict(X_te), 0, None)

            m_cb_m = _regression_metrics(y_te, p_cb)
            m_lg_m = _regression_metrics(y_te, p_lg)

            rows.append(
                {
                    "Scenario": f"{scenario_days}-Day",
                    "Parameter": target.upper(),
                    "RMSE_CB": m_cb_m["RMSE"], "MAE_CB": m_cb_m["MAE"], "R2_CB": m_cb_m["R²"],
                    "RMSE_LGB": m_lg_m["RMSE"], "MAE_LGB": m_lg_m["MAE"], "R2_LGB": m_lg_m["R²"],
                    "Best": "CatBoost" if m_cb_m["RMSE"] <= m_lg_m["RMSE"] else "LightGBM",
                }
            )

            scenario_predictions[scenario_days]["targets"][target] = {
                "cb": p_cb, "lgb": p_lg, "actual": y_te
            }
            scenario_models[scenario_days][target] = {
                "cb": m_cb,
                "lgb": m_lg,
                "best": "CatBoost" if m_cb_m["RMSE"] <= m_lg_m["RMSE"] else "LightGBM",
            }

    df_scenario = pd.DataFrame(rows)
    if verbose:
        print("\n" + "=" * 110)
        print("MULTI-SCENARIO EVALUATION ON THE TEST SET")
        print("=" * 110)
        print(df_scenario.to_string(index=False))

    return df_scenario, scenario_models, scenario_predictions


def plot_scenario_pivots(df_scenario: pd.DataFrame, name: str = "scenario_comparison") -> None:
    """Pivot RMSE/R² per scenario + comparison bar charts."""
    pivots = {
        "RMSE_CB": "RMSE CatBoost per Scenario",
        "RMSE_LGB": "RMSE LightGBM per Scenario",
        "R2_CB": "R² CatBoost per Scenario",
        "R2_LGB": "R² LightGBM per Scenario",
    }
    for col, title in pivots.items():
        print("\n" + "=" * 70)
        print(f"PIVOT -- {title}")
        print("=" * 70)
        print(df_scenario.pivot(index="Parameter", columns="Scenario", values=col).round(3).to_string())

    scenario_order = sorted(df_scenario["Scenario"].unique())
    x = np.arange(len(POLLUTANT_COLS))
    width = 0.8 / max(len(scenario_order), 1)
    palette = ["#1f77b4", "#2ca02c", "#d62728"]

    for metric, ylabel, suptitle in [
        ("RMSE", "RMSE", "RMSE COMPARISON -- Scenario 7 vs 10 vs 14 Days"),
        ("R2", "R²", "R² COMPARISON -- Scenario 7 vs 10 vs 14 Days"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        offset0 = -(len(scenario_order) - 1) / 2 * width
        for ax, model_name in zip(axes, ["CB", "LGB"]):
            col = f"{metric}_{model_name}"
            for i, scn in enumerate(scenario_order):
                sub = df_scenario[df_scenario["Scenario"] == scn].set_index("Parameter")
                vals = sub.loc[[p.upper() for p in POLLUTANT_COLS], col].to_numpy()
                bars = ax.bar(
                    x + offset0 + i * width, vals, width,
                    label=f"Scenario {scn}", color=palette[i % len(palette)], edgecolor="black",
                )
                for b, v in zip(bars, vals):
                    ax.text(
                        b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=7,
                    )
            if metric == "R2":
                ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([p.upper() for p in POLLUTANT_COLS])
            ax.set_ylabel(ylabel)
            model_label = "CatBoost" if model_name == "CB" else "LightGBM"
            ax.set_title(f"{metric if metric != 'R2' else 'R²'} per Scenario -- {model_label}", fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(suptitle, fontweight="bold", fontsize=13)
        fig.tight_layout()
        _savefig(fig, f"{name}_{metric.lower()}")


# ---------------------------------------------------------------------------
# 5. Combined forecast 18-31 March 2026
# ---------------------------------------------------------------------------


def combined_march_forecast(
    df_test: pd.DataFrame,
    df_test_capped: pd.DataFrame,
    scenario_models: dict,
    scenarios: dict | None = None,
    start: str = FORECAST_START,
    end: str = FORECAST_END,
    verbose: bool = True,
):
    """Combined forecast: days 1-7 from scenario 7, 8-10 from scenario 10, 11-14 from scenario 14.

    The original notebook stacks slices of the three scenario models' predictions
    into a single line. That behaviour is preserved here.

    Returns:
        results_march (list of dict), df_forecast, forecast_index
    """
    scenarios = scenarios or SCENARIOS
    pred_dates = pd.date_range(start, end)
    results_march: list[dict] = []
    df_forecast = pd.DataFrame(index=pred_dates, columns=POLLUTANT_COLS, dtype=float)

    for target in POLLUTANT_COLS:
        preds_by_scenario: dict[int, np.ndarray] = {}
        for scenario_days, cfg in scenarios.items():
            n_lags, rolling_w = cfg["n_lags"], cfg["rolling"]
            df_test_scn = build_scenario_features(df_test, n_lags, rolling_w)
            pred_df = df_test_scn.loc[pred_dates]
            feats = get_scenario_features(df_test_scn, target, n_lags, rolling_w)

            model_info = scenario_models[scenario_days][target]
            model = model_info["cb"] if model_info["best"] == "CatBoost" else model_info["lgb"]
            y_pred = np.clip(model.predict(pred_df[feats]), 0, None)
            preds_by_scenario[scenario_days] = y_pred

            for d, p in zip(pred_dates, y_pred):
                results_march.append(
                    {"Date": d.date(), "Scenario": f"{scenario_days}-Day",
                     "Parameter": target.upper(), "Prediction": p}
                )

        y_combined = np.concatenate(
            [preds_by_scenario[7][:7], preds_by_scenario[10][7:10], preds_by_scenario[14][10:14]]
        )
        df_forecast[target] = y_combined

    df_march = pd.DataFrame(results_march)
    df_forecast.index.name = "date"

    if verbose:
        print("\n" + "=" * 90)
        print("5. COMBINED FORECAST (7/10/14-DAY SCENARIOS)")
        print("=" * 90)
        pivot25 = (
            df_march[df_march["Date"] == pd.Timestamp("2026-03-25").date()]
            .pivot(index="Parameter", columns="Scenario", values="Prediction")
        )
        actual_25 = df_test_capped.loc["2026-03-25", POLLUTANT_COLS]
        actual_25.index = [c.upper() for c in actual_25.index]
        pivot25["Actual"] = actual_25
        print(pivot25.round(3).to_string())

    return results_march, df_forecast, df_forecast.index


def plot_combined_forecast(
    df_test_capped: pd.DataFrame,
    df_forecast: pd.DataFrame,
    name: str = "combined_march_forecast",
) -> None:
    """Chart of March actuals + combined forecast 18-31 March 2026."""
    pred_dates = df_forecast.index
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    axes = axes.flatten()

    for i, target in enumerate(POLLUTANT_COLS):
        ax = axes[i]
        actual_full = df_test_capped.loc["2026-03-01":"2026-03-31", target]
        ax.plot(actual_full.index, actual_full.to_numpy(), "o-", color="dimgray",
                label="Actual", alpha=0.7, markersize=4)
        ax.plot(pred_dates, df_forecast[target].to_numpy(), "o-", color="royalblue",
                label="Combined Forecast (7/10/14 days)", linewidth=2.5, markersize=5)
        ax.set_title(f"{target.upper()} ({UNIT_MAP[target]})", fontweight="bold")
        ax.set_ylabel(f"Concentration ({UNIT_MAP[target]})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.suptitle(
        "Direct Forecast 18-31 March 2026 — Single Combined Scenario Line\n"
        "Days 1-7 = 7-day scenario | Days 8-10 = 10-day scenario | Days 11-14 = 14-day scenario",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _savefig(fig, name)


def forecast_ispu_counts(
    forecast_values: pd.Series, col: str, thresholds: dict | None = None
) -> list[int]:
    """Count the days per ISPU category for a single forecast series."""
    thresholds = thresholds or FORECAST_ISPU_THRESHOLDS
    counts = [0, 0, 0, 0]
    thresh = thresholds.get(col.lower())
    if thresh is None:
        return counts
    for val in forecast_values.to_numpy():
        for idx in range(len(counts)):
            if thresh[idx] <= val < thresh[idx + 1]:
                counts[idx] += 1
                break
    return counts


def plot_ispu_forecast_distribution(
    df_forecast: pd.DataFrame,
    forecast_index,
    name: str = "ispu_forecast_distribution",
) -> pd.DataFrame:
    """ISPU category distribution over the forecast (14 days)."""
    forecast_only = df_forecast.loc[forecast_index]
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    table = {}

    for i, col in enumerate(POLLUTANT_COLS):
        ax = axes[i]
        counts = forecast_ispu_counts(forecast_only[col], col)
        table[col.upper()] = counts

        x = np.arange(len(FORECAST_ISPU_LABELS))
        bars = ax.bar(x, counts, width=0.6, color=CATEGORY_COLORS, edgecolor="black", zorder=3)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.15, f"{int(height)}",
                        ha="center", fontweight="bold", fontsize=10)

        ax.set_title(f"ISPU Category Distribution of Forecast: {col.upper()}",
                     fontweight="bold", fontsize=12)
        ax.set_xlabel("Air Quality Category", fontsize=9)
        ax.set_ylabel(f"Number of Days (Total = {len(forecast_only)} days)", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(FORECAST_ISPU_LABELS, rotation=10, fontsize=9)
        ax.set_ylim(0, max(counts) + 2 if max(counts) > 0 else 5)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)

    df_counts = pd.DataFrame(table, index=FORECAST_ISPU_LABELS)
    fig.suptitle(
        "Air Quality Category Distribution Analysis\n"
        "14-Day Self-Generated Forecast (Recursive Forecasting)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    _savefig(fig, name)
    return df_counts


# ---------------------------------------------------------------------------
# 6. Load stage orchestration
# ---------------------------------------------------------------------------


def _load_transformed_from_disk() -> dict:
    """Read every artifact produced by transform.py from disk."""
    required = {
        "train_features": TRANSFORM_DIR / "train_features.csv",
        "test_features": TRANSFORM_DIR / "test_features.csv",
        "train_ml_input": TRANSFORM_DIR / "train_ml_input.csv",
        "test_ml_input": TRANSFORM_DIR / "test_ml_input.csv",
        "test_capped": TRANSFORM_DIR / "test_capped.csv",
    }
    missing = [str(p) for p in required.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Transform artifacts are incomplete:\n  "
            + "\n  ".join(missing)
            + "\nRun `python transform.py` first."
        )

    frames = {
        name: pd.read_csv(path, index_col=0, parse_dates=True)
        for name, path in required.items()
    }
    train_df, val_df, test_df = chronological_split(
        frames["train_features"], frames["test_features"]
    )
    return {
        "df_train_features": frames["train_features"],
        "df_test_features": frames["test_features"],
        "df_train_ml": frames["train_ml_input"],
        "df_test_ml": frames["test_ml_input"],
        "df_test_capped": frames["test_capped"],
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def _save_models(models: dict, framework: str, params: dict) -> None:
    """Save the trained models + the best-hyperparameter table."""
    _save_table(
        pd.DataFrame(
            {target: [str(p)] for target, p in params.items()},
            index=["params"],
        ).T,
        f"best_params_{framework}",
    )
    if not SAVE_MODELS:
        return

    _ensure_dir(MODEL_DIR)
    for target, model in models.items():
        path = MODEL_DIR / f"{framework}_{target}.txt"
        try:
            # CatBoost exposes save_model(); LightGBM exposes it only on the
            # underlying Booster, so fall back to that.
            if hasattr(model, "save_model"):
                model.save_model(str(path))
            else:
                model.booster_.save_model(str(path))
        except Exception as exc:  # pragma: no cover - format model beda antar library
            print(f"  ! failed to save model {framework}/{target}: {exc}")


def run_load(artifacts: dict | None = None, verbose: bool = True) -> dict:
    """Run the whole load stage (tuning -> training -> evaluation -> forecast).

    Args:
        artifacts: output of run_transform(). If None, artifacts are read from
            disk; if they are missing, run_transform() is invoked for you.
    """
    if artifacts is None:
        try:
            artifacts = _load_transformed_from_disk()
        except FileNotFoundError as exc:
            if verbose:
                print(exc)
                print("Running transform.py first...")
            artifacts = run_transform(verbose=verbose)

    train_df = artifacts["train_df"]
    val_df = artifacts["val_df"]
    test_df = artifacts["test_df"]
    df_train_ml = artifacts["df_train_ml"]
    df_test_ml = artifacts["df_test_ml"]
    df_test_capped = artifacts["df_test_capped"]

    # --- 1. Hyperparameter tuning ---------------------------------------
    tuning = run_tuning(train_df, val_df, verbose=verbose)
    best_params_cb = tuning["best_params_cb"]
    best_params_lgb = tuning["best_params_lgb"]

    # --- 2. Cross-validation TimeSeriesSplit ----------------------------
    trainval_df = pd.concat([train_df, val_df])
    df_cv = cross_validate(trainval_df, best_params_cb, best_params_lgb, verbose=verbose)
    if verbose:
        print(df_cv.to_string(index=False))
    _save_table(df_cv, "cross_validation")

    # --- 3. Final models & test-set evaluation ------------------------------
    models_cb, models_lgb = train_final_models(
        trainval_df, best_params_cb, best_params_lgb, verbose=verbose
    )
    df_test_results = evaluate_on_test(test_df, models_cb, models_lgb, verbose=verbose)
    _save_table(df_test_results, "test_evaluation", index=False)
    df_rank = rank_models_by_wins(df_test_results, verbose=verbose)
    _save_table(df_rank, "ranking_model", index=False)
    plot_model_comparison(df_test_results)
    plot_predictions_vs_actual(test_df, models_cb, models_lgb)

    # --- 4. Forecast scenarios 7 / 10 / 14 days --------------------------
    df_scenario, scenario_models, _ = run_scenarios(
        df_train_ml, df_test_ml, verbose=verbose
    )
    _save_table(df_scenario, "scenario_results", index=False)
    plot_scenario_pivots(df_scenario)

    # --- 5. Combined 18-31 March forecast + ISPU forecast distribution ---
    results_march, df_forecast, forecast_index = combined_march_forecast(
        df_test_ml, df_test_capped, scenario_models, verbose=verbose
    )
    _ensure_dir(PREDICTION_DIR)
    pd.DataFrame(results_march).to_csv(
        PREDICTION_DIR / "march_predictions_per_scenario.csv", index=False
    )
    df_forecast.to_csv(
        PREDICTION_DIR / "combined_forecast_18_31_march.csv", index_label="date"
    )
    plot_combined_forecast(df_test_capped, df_forecast)
    df_counts = plot_ispu_forecast_distribution(df_forecast, forecast_index)
    _save_table(df_counts.reset_index(), "forecast_ispu_distribution")

    # --- 6. Save models & hyperparameter summary ----------------------
    _save_models(models_cb, "catboost", best_params_cb)
    _save_models(models_lgb, "lightgbm", best_params_lgb)
    if verbose:
        print("\n" + "=" * 78)
        print("BEST HYPERPARAMETERS")
        print("=" * 78)
        for label, params in (("CatBoost", best_params_cb), ("LightGBM", best_params_lgb)):
            print(f"\n--- {label} ---")
            for target, p in params.items():
                print(f"  {target.upper():<6} : {p}")
        print(f"\nLoad artifacts saved to: {ARTIFACT_DIR}")

    return {
        "df_cv": df_cv,
        "df_test_results": df_test_results,
        "df_rank": df_rank,
        "models_cb": models_cb,
        "models_lgb": models_lgb,
        "best_params_cb": best_params_cb,
        "best_params_lgb": best_params_lgb,
        "df_scenario": df_scenario,
        "scenario_models": scenario_models,
        "df_forecast": df_forecast,
        "df_forecast_ispu": df_counts,
    }


def main() -> dict:
    return run_load(verbose=True)


if __name__ == "__main__":
    main()