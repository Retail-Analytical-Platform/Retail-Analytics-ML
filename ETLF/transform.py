"""
TRANSFORM — Stage 2 of the ISPU Pollution Forecast pipeline.

Responsibilities (following the cell order of the ml_src_source.ipynb notebook):
    1. Cleaning & type standardisation (decimal comma, '-', negative values).
    2. Parse Indonesian dates -> datetime, set index, sort.
    3. Exploration & descriptive statistics.
    4. Sensitivity analysis for choosing the KNN imputation k.
    5. KNN imputation (fit on train, transform on test — no data leakage).
    6. Validate the KNN imputation with a synthetic missing-value test.
    7. Outlier policy: keep them, clip with IQR (bounds computed on training only).
    8. ISPU PM2.5 category classification (Permen LHK No. 14/2020).
    9. Feature engineering: lag, rolling mean, temporal + cyclical encoding.
   10. Chronological train / val / test split.

This script can be run directly:  python transform.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; plt.show() stays safe

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.impute import KNNImputer  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from extract import (  # noqa: E402
    ARTIFACT_DIR,
    RANDOM_STATE,
    load_extracted,
)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

TRANSFORM_DIR = ARTIFACT_DIR / "transformed"
FIGURE_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"

SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"
SAVE_PLOTS = os.environ.get("SAVE_PLOTS", "1") == "1"

POLLUTANT_COLS = ["co", "no2", "o3", "so2", "pm10", "pm25"]

UNIT_MAP = {
    "co": "ppm",
    "no2": "ppm",
    "o3": "ppm",
    "so2": "ppm",
    "pm10": "µg/m³",
    "pm25": "µg/m³",
}

COLORS = {
    "pm25": "crimson",
    "pm10": "darkorange",
    "o3": "forestgreen",
    "no2": "mediumblue",
    "so2": "purple",
    "co": "saddlebrown",
}

MONTH_MAP = {
    "Januari": "01",
    "Februari": "02",
    "Maret": "03",
    "April": "04",
    "Mei": "05",
    "Juni": "06",
    "Juli": "07",
    "Agustus": "08",
    "September": "09",
    "Oktober": "10",
    "November": "11",
    "Desember": "12",
}

# KNN imputation k. The notebook uses 7 (result of the k=3/5/7 sensitivity test).
KNN_K = 7
KNN_CANDIDATES = [3, 5, 7]
VALIDATION_MISSING_RATE = 0.10
VALIDATION_N_ITER = 30

# Maximum number of lags built during feature engineering (1..13).
MAX_LAGS = 13
# Number of lags used as model input.
MODEL_LAGS = 7
# Rolling-mean windows used by the model.
MODEL_ROLLING = (7, 14)

# ISPU PM2.5 categories (5 classes, Permen LHK No. 14/2020).
ISPU_BINS = [0, 50, 100, 200, 300, np.inf]
ISPU_LABELS = [
    "Good",
    "Moderate",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _savefig(fig, name: str) -> None:
    if SAVE_PLOTS:
        _ensure_dir(FIGURE_DIR)
        fig.savefig(FIGURE_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    _ensure_dir(TABLE_DIR)
    df.to_csv(TABLE_DIR / f"{name}.csv", index=index)


# ---------------------------------------------------------------------------
# 1. Cleaning & standardisation
# ---------------------------------------------------------------------------


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise dtypes, handle negative values, parse Indonesian dates."""
    df = df.copy()
    df = df.replace("-", np.nan)

    for col in POLLUTANT_COLS:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    df["date"] = df["date"].replace(MONTH_MAP, regex=True)
    df["date"] = pd.to_datetime(df["date"], format="%d %m %Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date").sort_index()

    return df


def load_and_preprocess(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw zone, then clean the train & test sets."""
    raw = load_extracted(verbose=verbose)
    df_train = preprocess_data(raw["train"])
    df_test = preprocess_data(raw["test"])

    if verbose:
        print("=" * 78)
        print("1. LOADING DATA & DATA CLEANING")
        print("=" * 78)
        print("=== TRAINING DATA (2022-2024) ===")
        print(
            f"Period    : {df_train.index.min().date()} to "
            f"{df_train.index.max().date()} ({len(df_train)} days)"
        )
        print("=== TESTING & VALIDATION DATA (2026) ===")
        print(
            f"Period    : {df_test.index.min().date()} to "
            f"{df_test.index.max().date()} ({len(df_test)} days)"
        )
        print("\nMissing data in training set:")
        print(
            missing_summary(df_train).to_string()
        )
        print("\nMissing data in test/validation set:")
        print(
            missing_summary(df_test).to_string()
        )

    return df_train, df_test


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of missing count and percentage per pollutant."""
    return (
        df[POLLUTANT_COLS]
        .isna()
        .sum()
        .to_frame("NaN")
        .assign(Percent=lambda x: (x["NaN"] / len(x) * 100).round(2))
    )


# ---------------------------------------------------------------------------
# 2. Exploration & descriptive statistics
# ---------------------------------------------------------------------------


def descriptive_stats(df_input: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Full formal descriptive statistics for every pollutant column."""
    rows = []
    for col in cols:
        s = df_input[col].dropna()
        rows.append(
            {
                "Parameter": col.upper(),
                "N Valid": int(s.count()),
                "N Missing": int(df_input[col].isna().sum()),
                "Mean": round(s.mean(), 3),
                "Median": round(s.median(), 3),
                "Std Dev": round(s.std(), 3),
                "Min": round(s.min(), 3),
                "Max": round(s.max(), 3),
                "Q1 (25%)": round(s.quantile(0.25), 3),
                "Q3 (75%)": round(s.quantile(0.75), 3),
                "Skewness": round(s.skew(), 3),
                "Kurtosis": round(s.kurtosis(), 3),
            }
        )
    return pd.DataFrame(rows).set_index("Parameter")


def plot_raw_profiles(df_raw: pd.DataFrame, name: str = "eda_raw_profiles") -> None:
    """Boxplot + histogram of every pollutant on the raw data."""
    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    for i, col in enumerate(POLLUTANT_COLS):
        sns.boxplot(y=df_raw[col], ax=axes[0, i], color="lightcoral")
        axes[0, i].set_title(f"{col.upper()}", fontweight="bold")
        axes[0, i].set_ylabel(UNIT_MAP[col] if i == 0 else "")

        sns.histplot(
            df_raw[col].dropna(),
            kde=True,
            ax=axes[1, i],
            color="indianred",
            edgecolor="black",
        )
        axes[1, i].set_title(f"Skew: {df_raw[col].skew():.2f}")
        axes[1, i].set_xlabel(UNIT_MAP[col])

    fig.suptitle("Raw Data Profile: Spread & Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, name)


def plot_post_preprocessing_distributions(
    df_capped: pd.DataFrame, name: str = "eda_post_imputation"
) -> None:
    """Distribution boxplots after preprocessing (outliers retained)."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 6, figsize=(20, 6))
    for i, col in enumerate(POLLUTANT_COLS):
        sns.boxplot(data=df_capped, y=col, ax=axes[i], color="skyblue", width=0.5)
        axes[i].set_title(col.upper(), fontweight="bold", fontsize=12)
        axes[i].set_ylabel(
            f"Concentration ({UNIT_MAP[col]})" if i == 0 else ""
        )
    fig.suptitle(
        "ISPU Parameter Distribution After Preprocessing (Outliers Kept)",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )
    fig.tight_layout()
    _savefig(fig, name)


def plot_trends_and_correlation(
    df_final: pd.DataFrame, name: str = "eda_trends_correlation"
) -> None:
    """Daily trend of every pollutant + correlation heatmap."""
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    for i, col in enumerate(POLLUTANT_COLS):
        axes[i].plot(df_final.index, df_final[col], color=COLORS[col], linewidth=1.5)
        axes[i].set_title(f"{col.upper()} ({UNIT_MAP[col]})", fontweight="bold")
        axes[i].grid(alpha=0.3)
    fig.suptitle("Daily Pollutant Trend", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, f"{name}_trends")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df_final[POLLUTANT_COLS].corr(),
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation Between Parameters (Final Data)", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, f"{name}_correlation")


# ---------------------------------------------------------------------------
# 3. KNN imputation
# ---------------------------------------------------------------------------


def knn_impute_with_k(df_input: pd.DataFrame, cols: list[str], k: int) -> pd.DataFrame:
    """KNN imputation for a given k (standalone, no shared scaler)."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_input[cols])
    imputer = KNNImputer(n_neighbors=k, weights="distance")
    imputed = imputer.fit_transform(scaled)
    return pd.DataFrame(
        scaler.inverse_transform(imputed), columns=cols, index=df_input.index
    )


def impute_knn(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    k: int = KNN_K,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, KNNImputer]:
    """KNN imputation: fit only on training, transform on testing.

    Also returns the scaler & imputer so load.py can reuse them.
    """
    df_impute = df_train.copy()
    df_test_impute = df_test.copy()

    scaler = MinMaxScaler()
    knn = KNNImputer(n_neighbors=k, weights="distance")

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_impute[POLLUTANT_COLS]),
        columns=POLLUTANT_COLS,
        index=df_impute.index,
    )
    df_imputed = pd.DataFrame(
        knn.fit_transform(df_scaled), columns=POLLUTANT_COLS, index=df_impute.index
    )
    df_impute[POLLUTANT_COLS] = scaler.inverse_transform(df_imputed)

    df_test_scaled = pd.DataFrame(
        scaler.transform(df_test_impute[POLLUTANT_COLS]),
        columns=POLLUTANT_COLS,
        index=df_test_impute.index,
    )
    df_test_imputed = pd.DataFrame(
        knn.transform(df_test_scaled),
        columns=POLLUTANT_COLS,
        index=df_test_impute.index,
    )
    df_test_impute[POLLUTANT_COLS] = scaler.inverse_transform(df_test_imputed)

    if verbose:
        print("=" * 78)
        print(f"2. MISSING-VALUE IMPUTATION WITH KNN (k={k})")
        print("=" * 78)
        print("Missing values after imputation (Training):")
        print(df_impute[POLLUTANT_COLS].isnull().sum().to_string())
        print("\nMissing values after imputation (Testing):")
        print(df_test_impute[POLLUTANT_COLS].isnull().sum().to_string())

    return df_impute, df_test_impute, scaler, knn


def knn_sensitivity_analysis(
    df_raw: pd.DataFrame, candidates: list[int] | None = None, k_reference: int = KNN_K
) -> pd.DataFrame:
    """Test how the imputed mean/std/skew respond to the choice of k."""
    candidates = candidates or KNN_CANDIDATES
    rows = []
    for k in candidates:
        df_k = knn_impute_with_k(df_raw, POLLUTANT_COLS, k)
        for col in POLLUTANT_COLS:
            rows.append(
                {
                    "k": k,
                    "Param": col.upper(),
                    "Mean": round(df_k[col].mean(), 3),
                    "Std": round(df_k[col].std(), 3),
                    "Skewness": round(df_k[col].skew(), 3),
                }
            )
    return pd.DataFrame(rows)


def print_sensitivity_report(df_sensitivity: pd.DataFrame, k_reference: int = KNN_K) -> pd.DataFrame:
    """Print the pivot table + relative deviation between k values."""
    pivot_mean = df_sensitivity.pivot(index="Param", columns="k", values="Mean")
    ks = list(pivot_mean.columns)

    print("=" * 70)
    print("3. SENSITIVITY ANALYSIS — CHOICE OF k (AVERAGE IMPUTED VALUES)")
    print("=" * 70)
    print(pivot_mean.to_string())

    print(f"\nRelative deviation (%) vs k={k_reference}:")
    for col in POLLUTANT_COLS:
        row = pivot_mean.loc[col.upper()]
        base = row[k_reference]
        parts = []
        for k in ks:
            if k == k_reference:
                continue
            dev = abs(row[k] - base) / base * 100 if base else 0.0
            parts.append(f"k={k} vs k={k_reference}: {dev:5.2f}%")
        print(f"  {col.upper():<6} | " + " | ".join(parts))

    return pivot_mean


def validate_knn_imputation(
    df_complete: pd.DataFrame,
    cols: list[str],
    k: int = KNN_K,
    missing_rate: float = VALIDATION_MISSING_RATE,
    n_iter: int = VALIDATION_N_ITER,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Synthetic missing test: drop X% at random -> impute -> compare.

    Metrics: RMSE, MAE and MAPE (MAPE skipped where the true value is 0).
    """
    rng = np.random.default_rng(seed)
    metrics = {col: {"rmse": [], "mae": [], "mape": []} for col in cols}

    for _ in range(n_iter):
        df_test = df_complete.copy()
        mask = rng.random(df_test.shape) < missing_rate
        truth = df_test.to_numpy(copy=True)
        df_masked = df_test.mask(mask)

        sc = MinMaxScaler()
        scaled = sc.fit_transform(df_masked)
        imputer = KNNImputer(n_neighbors=k, weights="distance")
        imputed = sc.inverse_transform(imputer.fit_transform(scaled))

        for j, col in enumerate(cols):
            col_mask = mask[:, j]
            if col_mask.sum() == 0:
                continue
            y_true = truth[col_mask, j]
            y_pred = imputed[col_mask, j]
            metrics[col]["rmse"].append(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics[col]["mae"].append(mean_absolute_error(y_true, y_pred))
            nonzero = y_true != 0
            if nonzero.sum() > 0:
                metrics[col]["mape"].append(
                    np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
                )

    summary = []
    for col in cols:
        summary.append(
            {
                "Parameter": col.upper(),
                "Original Mean": round(df_complete[col].mean(), 3),
                "RMSE": round(np.mean(metrics[col]["rmse"]), 3),
                "MAE": round(np.mean(metrics[col]["mae"]), 3),
                "MAPE (%)": round(np.mean(metrics[col]["mape"]), 2),
            }
        )
    return pd.DataFrame(summary).set_index("Parameter")


def compare_imputation_k(df_complete_cases: pd.DataFrame, candidates: list[int] | None = None) -> pd.DataFrame:
    """Compare average imputation RMSE/MAE for each value of k."""
    candidates = candidates or KNN_CANDIDATES
    rows = []
    for k in candidates:
        hv = validate_knn_imputation(df_complete_cases, POLLUTANT_COLS, k=k)
        rows.append(
            {
                "k": k,
                "Avg RMSE": round(hv["RMSE"].mean(), 3),
                "Avg MAE": round(hv["MAE"].mean(), 3),
            }
        )
    return pd.DataFrame(rows).set_index("k")


# ---------------------------------------------------------------------------
# 4. Outlier policy (IQR capping)
# ---------------------------------------------------------------------------


def apply_iqr_capping(
    df_impute: pd.DataFrame,
    df_test_impute: pd.DataFrame,
    multiplier: float = 1.5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clip extreme outliers with the IQR rule.

    Bounds are computed ONLY from the training data, then applied to testing,
    so test-set statistics never leak into the training process.
    """
    df_capped = df_impute.copy()
    df_test_capped = df_test_impute.copy()

    bounds: dict[str, tuple[float, float]] = {}
    for col in POLLUTANT_COLS:
        q1 = df_capped[col].quantile(0.25)
        q3 = df_capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        bounds[col] = (lower, upper)

        n_clipped = int(((df_capped[col] < lower) | (df_capped[col] > upper)).sum())
        df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
        df_test_capped[col] = df_test_capped[col].clip(lower=lower, upper=upper)

        if verbose:
            print(f"  {col.upper():<6}: [{lower:.2f}, {upper:.2f}] | "
                  f"{n_clipped} training values clipped")

    if verbose:
        print("\nRanges after capping:")
        for col in POLLUTANT_COLS:
            print(
                f"  {col.upper():<6} train: {df_capped[col].min():8.2f} – "
                f"{df_capped[col].max():8.2f} | test: {df_test_capped[col].min():8.2f} – "
                f"{df_test_capped[col].max():8.2f}"
            )

    return df_capped, df_test_capped


def print_variance_check(df_capped: pd.DataFrame, test_fraction: float = 0.2, verbose: bool = True) -> pd.DataFrame:
    """Check variance over the test period (last 20%) to detect stagnant data."""
    df_test_check = df_capped.sort_index()
    n_test = int(len(df_test_check) * test_fraction)
    test_window = df_test_check.tail(n_test)

    if verbose:
        print("\nVariance check over the test period (last 20%):")
        print(
            f"  Test period: {test_window.index.min().date()} -> "
            f"{test_window.index.max().date()}\n"
        )

    rows = []
    for col in POLLUTANT_COLS:
        rows.append(
            {
                "Parameter": col.upper(),
                "std": round(test_window[col].std(), 3),
                "variance": round(test_window[col].var(), 3),
                "range": round(test_window[col].max() - test_window[col].min(), 2),
            }
        )
    df_var = pd.DataFrame(rows).set_index("Parameter")
    if verbose:
        print(df_var.to_string())
    return df_var


# ---------------------------------------------------------------------------
# 5. ISPU PM2.5 category classification
# ---------------------------------------------------------------------------


def assign_ispu_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add the pm25_category column using the ISPU thresholds."""
    out = df.copy()
    out["pm25_category"] = pd.cut(
        out["pm25"], bins=ISPU_BINS, labels=ISPU_LABELS, right=True
    )
    return out


def category_distribution(df_raw: pd.DataFrame, df_final: pd.DataFrame) -> pd.DataFrame:
    """ISPU PM2.5 category distribution: raw data vs final data."""
    count_raw = df_raw["pm25_category"].value_counts().reindex(ISPU_LABELS).fillna(0).astype(int)
    count_final = (
        df_final["pm25_category"].value_counts().reindex(ISPU_LABELS).fillna(0).astype(int)
    )
    return pd.DataFrame(
        {
            "Raw Data": count_raw,
            "Final Data": count_final,
            "Difference (+/-)": count_final - count_raw,
        }
    )


def plot_category_distribution(
    df_ispu_categories: pd.DataFrame,
    total_raw: int,
    total_final: int,
    name: str = "ispu_category_distribution",
) -> None:
    """Bar chart comparing ISPU categories: raw data vs final data."""
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(ISPU_LABELS))
    width = 0.38

    bars1 = ax.bar(
        x - width / 2,
        df_ispu_categories["Raw Data"],
        width,
        label=f"Raw Data (n={total_raw} valid days)",
        color="indianred",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        df_ispu_categories["Final Data"],
        width,
        label=f"Final Data (n={total_final} days, after imputation+capping)",
        color="teal",
        edgecolor="black",
    )

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.5,
                    f"{int(height)}",
                    ha="center",
                    fontweight="bold",
                    fontsize=10,
                )

    ax.set_xlabel("ISPU PM2.5 Category", fontweight="bold")
    ax.set_ylabel("Number of Days", fontweight="bold")
    ax.set_title(
        "Distribution of Air-Quality Categories (PM2.5)\n"
        "Raw Data vs Data After Preprocessing",
        fontweight="bold",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ISPU_LABELS, rotation=15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _savefig(fig, name)


# ---------------------------------------------------------------------------
# 6. Feature engineering
# ---------------------------------------------------------------------------


def add_lag_features(df: pd.DataFrame, max_lags: int, target: pd.DataFrame | None = None):
    """Lags 1..max_lags for every pollutant."""
    for param in POLLUTANT_COLS:
        for i in range(1, max_lags + 1):
            target[f"{param}_lag_{i}"] = df[param].shift(i)


def add_rolling_features(
    df: pd.DataFrame, windows, target: pd.DataFrame | None = None
):
    """Rolling mean shifted by 1 day (avoids in-sample data leakage)."""
    for param in POLLUTANT_COLS:
        for w in windows:
            target[f"{param}_rmean_{w}d"] = (
                df[param].shift(1).rolling(window=w, min_periods=1).mean()
            )


def add_temporal_features(index: pd.DatetimeIndex, target: pd.DataFrame) -> None:
    """Calendar features + cyclical encoding (sin/cos) per day."""
    target["day_of_week"] = index.dayofweek
    target["day_of_month"] = index.day
    target["day_of_year"] = index.dayofyear
    target["week_of_year"] = index.isocalendar().week.astype(int)
    target["month"] = index.month
    target["quarter"] = index.quarter
    target["is_weekend"] = (index.dayofweek >= 5).astype(int)

    target["dow_sin"] = np.sin(2 * np.pi * target["day_of_week"] / 7)
    target["dow_cos"] = np.cos(2 * np.pi * target["day_of_week"] / 7)
    target["month_sin"] = np.sin(2 * np.pi * target["month"] / 12)
    target["month_cos"] = np.cos(2 * np.pi * target["month"] / 12)
    target["doy_sin"] = np.sin(2 * np.pi * target["day_of_year"] / 365)
    target["doy_cos"] = np.cos(2 * np.pi * target["day_of_year"] / 365)


def build_features(
    df_input: pd.DataFrame,
    max_lags: int = MAX_LAGS,
    rolling_windows=MODEL_ROLLING,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Build the feature matrix: lag + rolling + temporal + cyclical.

    Args:
        df_input: frame containing the pollutant columns with a datetime index.
        max_lags: number of lags to build (1..max_lags).
        rolling_windows: iterable of rolling-mean window sizes.
        drop_na: drop rows whose features are NaN (default True).
    """
    df_res = df_input[POLLUTANT_COLS].copy().sort_index()
    df_res = df_res[~df_res.index.duplicated(keep="first")]

    add_lag_features(df_res, max_lags, df_res)
    add_rolling_features(df_res, rolling_windows, df_res)
    add_temporal_features(df_res.index, df_res)

    return df_res.dropna() if drop_na else df_res


def get_features(
    target: str,
    available_cols,
    n_lags: int = MODEL_LAGS,
    rolling_windows=MODEL_ROLLING,
) -> list[str]:
    """Feature column list for a target (excluding the target column itself)."""
    feats: list[str] = []
    for param in POLLUTANT_COLS:
        for i in range(1, n_lags + 1):
            feats.append(f"{param}_lag_{i}")
        for w in rolling_windows:
            feats.append(f"{param}_rmean_{w}d")

    feats.extend(
        [
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "week_of_year",
            "month",
            "quarter",
            "is_weekend",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "doy_sin",
            "doy_cos",
        ]
    )
    return [f for f in feats if f in set(available_cols) and f != target]


def chronological_split(
    df_train_features: pd.DataFrame, df_test_features: pd.DataFrame, val_fraction: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split: train = 2022-2024, val/test = 2026 (halved 50:50)."""
    n_val = len(df_test_features) // 2
    return (
        df_train_features.copy(),
        df_test_features.iloc[:n_val].copy(),
        df_test_features.iloc[n_val:].copy(),
    )


# ---------------------------------------------------------------------------
# 7. Transform stage orchestration
# ---------------------------------------------------------------------------


def run_transform(verbose: bool = True) -> dict:
    """Run the whole transform stage and save its results to artifacts.

    Returns:
        dict holding the transformed frames, scaler/imputer, data splits and
        analytics tables (descriptive, sensitivity, validation, categories).
    """
    np.random.seed(RANDOM_STATE)
    verbose = verbose

    df_train, df_test = load_and_preprocess(verbose=verbose)
    df_raw = df_train.copy()

    # Back up the data before imputation (used by the synthetic validation)
    df_train_backup = df_train.copy()
    df_test_backup = df_test.copy()

    # --- Raw-data exploration ---------------------------------------------
    stats_raw = descriptive_stats(df_raw, POLLUTANT_COLS)
    plot_raw_profiles(df_raw)
    if verbose:
        print("\n" + "=" * 95)
        print("TABLE — DESCRIPTIVE STATISTICS (RAW DATA)")
        print("=" * 95)
        print(stats_raw.to_string())
    _save_table(stats_raw.reset_index(), "raw_descriptive_stats")

    # --- k sensitivity analysis ----------------------------------------
    df_sensitivity = knn_sensitivity_analysis(df_raw)
    pivot_mean = print_sensitivity_report(df_sensitivity)

    # --- Imputation validation ------------------------------------------------
    df_complete_cases = df_raw[POLLUTANT_COLS].dropna().copy()
    imputation_validation = validate_knn_imputation(df_complete_cases, POLLUTANT_COLS, k=KNN_K)
    k_comparison = compare_imputation_k(df_complete_cases)
    if verbose:
        print("\n" + "=" * 80)
        print(f"TABLE — KNN IMPUTATION VALIDATION (k={KNN_K}, missing={VALIDATION_MISSING_RATE:.0%})")
        print("=" * 80)
        print(imputation_validation.to_string())
        print("\nACCURACY COMPARISON ACROSS k VALUES:")
        print(k_comparison.to_string())
    _save_table(imputation_validation.reset_index(), "imputation_validation")
    _save_table(k_comparison.reset_index(), "k_comparison")

    # --- Imputation + capping ----------------------------------------------
    df_impute, df_test_impute, scaler, knn = impute_knn(df_train, df_test, k=KNN_K, verbose=verbose)
    df_capped, df_test_capped = apply_iqr_capping(
        df_impute, df_test_impute, verbose=verbose
    )
    df_variance = print_variance_check(df_capped, verbose=verbose)
    plot_post_preprocessing_distributions(df_capped)
    plot_trends_and_correlation(df_capped)

    # --- ISPU categories ----------------------------------------------------
    df_raw_cat = assign_ispu_category(df_raw)
    df_capped_cat = assign_ispu_category(df_capped)
    df_ispu_categories = category_distribution(df_raw_cat, df_capped_cat)
    total_raw = int(df_ispu_categories["Raw Data"].sum())
    total_final = int(df_ispu_categories["Final Data"].sum())
    if verbose:
        print("\n" + "=" * 65)
        print("DISTRIBUTION OF ISPU PM2.5 CATEGORIES (Permen LHK No. 14/2020)")
        print("=" * 65)
        print(df_ispu_categories.to_string())
        print("-" * 65)
        print(f"Classified days             : {total_raw} (raw) -> {total_final} (final)")
        print(
            f"Days with NaN data        : {len(df_raw) - total_raw} (raw)     -> "
            f"{len(df_raw) - total_final} (final)"
        )
    plot_category_distribution(df_ispu_categories, total_raw, total_final)
    _save_table(df_ispu_categories.reset_index(), "ispu_category_distribution")

    # --- Feature engineering ---------------------------------------------
    df_capped_clean = df_capped_cat.drop("pm25_category", errors="ignore")
    df_test_capped_clean = df_test_capped.copy()

    common_cols = sorted(set(df_capped_clean.columns) & set(df_test_capped_clean.columns))
    df_train_ml = df_capped_clean[common_cols]
    df_test_ml = df_test_capped_clean[common_cols]

    df_train_features = build_features(df_train_ml)
    df_test_features = build_features(df_test_ml)

    train_df, val_df, test_df = chronological_split(df_train_features, df_test_features)
    if verbose:
        print("\n" + "=" * 78)
        print("FEATURE ENGINEERING & CHRONOLOGICAL SPLIT")
        print("=" * 78)
        print(f"Features per target      : {len(get_features('pm25', df_train_features.columns))}")
        print(f"Shape data ML (Training) : {df_train_features.shape}")
        print(f"Shape data ML (Testing)  : {df_test_features.shape}")
        print(f"Train : {train_df.index.min().date()} → {train_df.index.max().date()} "
              f"({len(train_df)} days)")
        print(f"Val   : {val_df.index.min().date()} → {val_df.index.max().date()} "
              f"({len(val_df)} days)")
        print(f"Test  : {test_df.index.min().date()} → {test_df.index.max().date()} "
              f"({len(test_df)} days)")

    # --- Save artifacts ------------------------------------------------------
    _ensure_dir(TRANSFORM_DIR)
    df_train_features.to_csv(TRANSFORM_DIR / "train_features.csv", index_label="date")
    df_test_features.to_csv(TRANSFORM_DIR / "test_features.csv", index_label="date")
    df_capped.to_csv(TRANSFORM_DIR / "train_capped.csv", index_label="date")
    df_test_capped.to_csv(TRANSFORM_DIR / "test_capped.csv", index_label="date")
    # Plain input (no features) for the load stage / scenario forecasting.
    df_train_ml.to_csv(TRANSFORM_DIR / "train_ml_input.csv", index_label="date")
    df_test_ml.to_csv(TRANSFORM_DIR / "test_ml_input.csv", index_label="date")

    if verbose:
        print(f"\nTransform artifacts saved to: {TRANSFORM_DIR}")

    return {
        "df_train": df_train,
        "df_test": df_test,
        "df_raw": df_raw,
        "df_train_backup": df_train_backup,
        "df_test_backup": df_test_backup,
        "df_capped": df_capped,
        "df_test_capped": df_test_capped,
        "scaler": scaler,
        "knn": knn,
        "stats_raw": stats_raw,
        "df_sensitivity": df_sensitivity,
        "pivot_mean": pivot_mean,
        "imputation_validation": imputation_validation,
        "k_comparison": k_comparison,
        "df_ispu_categories": df_ispu_categories,
        "df_variance": df_variance,
        "df_train_features": df_train_features,
        "df_test_features": df_test_features,
        "df_train_ml": df_train_ml,
        "df_test_ml": df_test_ml,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def main() -> dict:
    return run_transform(verbose=True)


if __name__ == "__main__":
    main()