import pandas as pd

from src.features.calendar import CALENDAR_COLUMNS, add_calendar_features
from src.features.lags import add_lag_rolling_features, lag_roll_names
from src.pipeline_paths import DATA_DIR

DATASET_PATH = DATA_DIR / "forecast_dataset.parquet"
FEATURES_PATH = DATA_DIR / "features.parquet"


def save_frame(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_frame(path):
    return pd.read_parquet(path)


def feature_names(fcfg, xcfg):
    names = CALENDAR_COLUMNS + lag_roll_names(fcfg["target_column"], xcfg)
    return names + [c for c in xcfg["categorical_columns"] if c not in names]


def make_features(dataset, fcfg, xcfg, drop_incomplete=True):
    missing = [c for c in xcfg["feature_columns"] if c not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns required by features_config: {missing}")
    df = add_calendar_features(dataset, xcfg)
    df = add_lag_rolling_features(df, xcfg, fcfg["horizon"], fcfg["target_column"], fcfg["grain"])
    if drop_incomplete:
        df = df.dropna(subset=lag_roll_names(fcfg["target_column"], xcfg)).reset_index(drop=True)
    return df


def model_view(df, feature_columns, categorical, model):
    """Same feature set for every model; only the categorical encoding differs."""
    view = df[list(feature_columns)].copy()
    for col in categorical:
        if model == "catboost":
            view[col] = view[col].astype(str)
        elif model in ("lightgbm", "xgboost"):
            view[col] = view[col].astype("category")
        else:
            raise ValueError(f"No categorical handling for model '{model}'")
    return view