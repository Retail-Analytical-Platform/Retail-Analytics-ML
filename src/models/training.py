from pathlib import Path

import pandas as pd
import yaml

from src.config import CONFIG_DIR, load_config
from src.features.pipeline import feature_names, model_view
from src.models.base import get_model
from src.pipeline_paths import MODELS_DIR


def _tuned():
    path = Path(CONFIG_DIR) / "tuned_params.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}


def model_params(name, sample=False):
    spec = load_config("models_config")["models"][name]
    params = dict(spec["params"])
    params.update(_tuned().get(name, {}))
    if sample:
        params.update(spec.get("sample_params", {}))
    return params


def build_model(name, categorical, sample=False, params=None):
    cfg = load_config("models_config")
    return get_model(
        name, params=params or model_params(name, sample), seed=cfg["seed"],
        early_stopping_rounds=cfg["early_stopping_rounds"], categorical=categorical,
    )


def fit_with_holdout(model, X, y, dates, holdout_days):
    """Early stopping on the last holdout_days of the training rows, never on validation data."""
    cutoff = dates.max() - pd.Timedelta(days=holdout_days)
    inner = (dates <= cutoff).to_numpy()
    if inner.sum() == 0 or (~inner).sum() == 0:
        return model.fit(X, y)
    return model.fit(X[inner], y[inner], X[~inner], y[~inner])


def train_final(name, table, fcfg, xcfg, run_id, sample=False):
    feats, cat, target = feature_names(fcfg, xcfg), xcfg["categorical_columns"], fcfg["target_column"]
    start = table["date"].max() - pd.Timedelta(days=fcfg["training_window_days"] - 1)
    mask = table["date"] >= start
    view = model_view(table, feats, cat, name)
    model = build_model(name, cat, sample)
    fit_with_holdout(model, view[mask], table.loc[mask, target], table.loc[mask, "date"],
                     fcfg["early_stopping_holdout_days"])
    return model.save(MODELS_DIR / run_id / name, run_id)