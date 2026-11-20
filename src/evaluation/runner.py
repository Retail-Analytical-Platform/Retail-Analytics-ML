import logging

import numpy as np
import pandas as pd

from src.evaluation.backtest import fold_frames
from src.evaluation.metrics import METRICS, grouped_metrics

logger = logging.getLogger(__name__)


def run_backtest(df, date_col, grain, target, folds, predict_fn):
    """predict_fn(train, valid) must return forecasts aligned with valid's rows.
    Returns a long table: keys, date, fold, horizon_step, actual, forecast."""
    parts = []
    for fold in folds:
        train, valid = fold_frames(df, date_col, fold)
        forecast = np.asarray(predict_fn(train, valid), dtype=float)
        if forecast.shape[0] != len(valid):
            raise ValueError(f"Fold {fold['fold']}: got {forecast.shape[0]} forecasts for {len(valid)} rows")
        out = valid[[*grain, date_col]].copy()
        out["fold"] = fold["fold"]
        out["horizon_step"] = (pd.to_datetime(out[date_col]) - pd.Timestamp(fold["train_end"])).dt.days
        out["actual"] = valid[target].to_numpy(dtype=float)
        out["forecast"] = forecast
        missing = int(np.isnan(forecast).sum())
        if missing:
            logger.warning("Fold %s: %d rows have no forecast and are not scored", fold["fold"], missing)
        parts.append(out[np.isfinite(out["forecast"])])

    pred = pd.concat(parts, ignore_index=True)
    absent = {f["fold"] for f in folds} - set(pred["fold"])
    if absent:
        raise ValueError(f"No scored rows for fold(s): {sorted(absent)}")
    return pred


def summarize(pred):
    """Per-fold metrics, plus their mean and spread (std) across folds."""
    per_fold = grouped_metrics(pred, ["fold"])
    cols = list(METRICS)
    return {
        "per_fold": per_fold,
        "mean": per_fold[cols].mean(),
        "std": per_fold[cols].std(ddof=0),
    }