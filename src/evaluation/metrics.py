import numpy as np
import pandas as pd


def _clean(actual, forecast):
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError("actual and forecast must have the same length")
    if a.size == 0:
        raise ValueError("no values to score")
    if np.isnan(a).any() or np.isnan(f).any():
        raise ValueError("NaN in actual or forecast")
    return a, f


def mae(actual, forecast):
    a, f = _clean(actual, forecast)
    return float(np.mean(np.abs(f - a)))


def rmse(actual, forecast):
    a, f = _clean(actual, forecast)
    return float(np.sqrt(np.mean((f - a) ** 2)))


def bias(actual, forecast):
    """Mean error. Positive means the forecast is too high on average."""
    a, f = _clean(actual, forecast)
    return float(np.mean(f - a))


def wape(actual, forecast):
    """Sum of absolute errors over the sum of actuals. If actuals are all zero:
    0.0 when the forecast is also exact, otherwise NaN (never infinity)."""
    a, f = _clean(actual, forecast)
    num, den = float(np.sum(np.abs(f - a))), float(np.sum(np.abs(a)))
    if den == 0:
        return 0.0 if num == 0 else float("nan")
    return num / den


def smape(actual, forecast):
    """In percent, 0 to 200. A day where actual and forecast are both zero counts as 0."""
    a, f = _clean(actual, forecast)
    denom = np.abs(a) + np.abs(f)
    safe = np.where(denom == 0, 1.0, denom)
    terms = np.where(denom == 0, 0.0, 2 * np.abs(f - a) / safe)
    return float(100 * np.mean(terms))


METRICS = {"mae": mae, "rmse": rmse, "smape": smape, "wape": wape, "bias": bias}
RANKABLE = ("mae", "rmse", "smape", "wape")      # lower is better; bias cannot be ranked


def get_metric(name):
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(f"Unknown metric '{name}'; expected one of {sorted(METRICS)}") from None


def check_primary_metric(name):
    if name not in RANKABLE:
        raise ValueError(f"primary_metric must be one of {RANKABLE}, got {name!r}")


def compute_metrics(actual, forecast):
    return {name: fn(actual, forecast) for name, fn in METRICS.items()}


def grouped_metrics(df, group_cols, actual_col="actual", forecast_col="forecast"):
    """Metrics per group. Use group_cols like ['fold'], ['horizon_step'] or your series keys."""
    group_cols = list(group_cols)
    rows = []
    for key, g in df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row.update(compute_metrics(g[actual_col], g[forecast_col]))
        rows.append(row)
    return pd.DataFrame(rows)