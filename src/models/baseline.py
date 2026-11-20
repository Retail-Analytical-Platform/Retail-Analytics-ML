import numpy as np
import pandas as pd


def _last_values(train, date_col, grain, target):
    ordered = train.sort_values([*grain, date_col])
    return ordered.groupby(grain, sort=False)[target].last().rename("_naive").reset_index()


def naive_predict(train, valid, date_col, grain, target):
    """Repeat each series' last observed value. Returns an array aligned with valid's rows."""
    last = _last_values(train, date_col, grain, target)
    merged = valid[grain].merge(last, on=grain, how="left")
    return merged["_naive"].to_numpy(dtype=float)


def seasonal_naive_predict(train, valid, date_col, grain, target, season=7):
    """Repeat the value from the same weekday in the latest whole season on or before the
    origin. For horizons up to `season` days this is exactly the value `season` days earlier."""
    origin = pd.to_datetime(train[date_col]).max()
    dates = pd.to_datetime(valid[date_col])
    days_ahead = (dates - origin).dt.days
    if (days_ahead < 1).any():
        raise ValueError("Validation dates must come after the end of the training data")

    back = np.ceil(days_ahead / season).astype(int) * season
    source = valid[grain].copy()
    source["_src_date"] = dates - pd.to_timedelta(back, unit="D")

    hist = train[[*grain, date_col, target]].copy()
    hist[date_col] = pd.to_datetime(hist[date_col])
    hist = hist.rename(columns={date_col: "_src_date", target: "_seasonal"})

    seasonal = source.merge(hist, on=[*grain, "_src_date"], how="left")["_seasonal"].to_numpy(dtype=float)
    fallback = naive_predict(train, valid, date_col, grain, target)
    return np.where(np.isnan(seasonal), fallback, seasonal)


BASELINES = {"naive": naive_predict, "seasonal_naive": seasonal_naive_predict}