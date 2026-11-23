import numpy as np
import pandas as pd


def naive_forecast(history, future_dates, grain, target):
    grain = list(grain)
    last = history.sort_values("date").groupby(grain)[target].last().rename("y_pred").reset_index()
    future = pd.DataFrame({"date": pd.DatetimeIndex(future_dates)})
    return last.merge(future, how="cross")


def seasonal_naive_forecast(history, future_dates, grain, target, season=7):
    grain = list(grain)
    origin = history["date"].max()
    future = pd.DataFrame({"date": pd.DatetimeIndex(future_dates)})
    steps = (future["date"] - origin).dt.days
    shift = np.ceil(steps / season).astype(int) * season
    future["src_date"] = future["date"] - pd.to_timedelta(shift, unit="D")

    frame = history[grain].drop_duplicates().merge(future, how="cross")
    source = history[[*grain, "date", target]].rename(columns={"date": "src_date", target: "y_pred"})
    out = frame.merge(source, on=[*grain, "src_date"], how="left").drop(columns="src_date")
    if out["y_pred"].isna().any():
        raise ValueError("Not enough history for the seasonal naive forecast")
    return out


BASELINE_FUNCTIONS = {"naive": naive_forecast, "seasonal_naive": seasonal_naive_forecast}