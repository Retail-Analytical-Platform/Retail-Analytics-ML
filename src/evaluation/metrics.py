import numpy as np
import pandas as pd


def _arrays(y_true, y_pred):
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return a, b


def mae(y_true, y_pred):
    a, b = _arrays(y_true, y_pred)
    return float(np.abs(b - a).mean())


def rmse(y_true, y_pred):
    a, b = _arrays(y_true, y_pred)
    return float(np.sqrt(((b - a) ** 2).mean()))


def smape(y_true, y_pred):
    """Percent. A term with zero actual and zero prediction counts as 0."""
    a, b = _arrays(y_true, y_pred)
    denom = np.abs(a) + np.abs(b)
    ratio = np.divide(2 * np.abs(b - a), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(100 * ratio.mean())


def wape(y_true, y_pred):
    a, b = _arrays(y_true, y_pred)
    total = np.abs(a).sum()
    return float("nan") if total == 0 else float(np.abs(b - a).sum() / total)


def bias(y_true, y_pred):
    """Mean of prediction minus actual. Positive means over-forecasting."""
    a, b = _arrays(y_true, y_pred)
    return float((b - a).mean())


METRICS = {"mae": mae, "rmse": rmse, "smape": smape, "wape": wape, "bias": bias}


def score_table(df, metric_names, by=None, true_col="y_true", pred_col="y_pred"):
    unknown = [m for m in metric_names if m not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}")
    groups = [((), df)] if not by else df.groupby(list(by))
    rows = []
    for key, part in groups:
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by or [], key))
        row.update({m: METRICS[m](part[true_col], part[pred_col]) for m in metric_names})
        rows.append(row)
    return pd.DataFrame(rows)