from pathlib import Path

import pandas as pd

from src.features.categorical import fit_categories, prepare_model_frame
from src.features.lags import drop_insufficient_history
from src.models.registry import build_model

ML_MODELS = ("catboost", "lightgbm", "xgboost")
MODELS_DIR = Path("artifacts/models")


def split_for_early_stopping(df, date_col, days):
    """Fit on everything before the last `days` days; the last `days` are the early-stopping set."""
    cut = df[date_col].max() - pd.Timedelta(days=days - 1)
    return df[df[date_col] < cut], df[df[date_col] >= cut]


def _plain(values):
    return [v.item() if hasattr(v, "item") else v for v in values]


def train_model(name, feats, fcfg, feat_cfg, model_cfg, run_id, out_dir=MODELS_DIR):
    date_col, target = fcfg["date_column"], fcfg["target_column"]
    cats = feat_cfg["categorical_columns"]

    data = drop_insufficient_history(feats, target)
    window = fcfg.get("training_window_days")
    if window:
        data = data[data[date_col] > data[date_col].max() - pd.Timedelta(days=window)]
    fit_part, es_part = split_for_early_stopping(data, date_col, fcfg["horizon"])
    if fit_part.empty or es_part.empty:
        raise ValueError("Not enough history for the training window plus the early-stopping set")

    categories = fit_categories(feats, cats)     # same category codes for train, validation, forecast
    model = build_model(name, model_cfg, categorical_columns=cats)
    Xf, _ = prepare_model_frame(fit_part, name, cats, date_col, target, categories)
    Xe, _ = prepare_model_frame(es_part, name, cats, date_col, target, categories)
    model.fit(Xf, fit_part[target], Xe, es_part[target])

    path = model.save(Path(out_dir) / name, run_id=run_id, feature_names=model.feature_names)
    return model, path