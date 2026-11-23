import pandas as pd

STATS = ("mean", "std", "min", "max")


def lag_roll_names(target, features_config):
    names = [f"{target}_lag_{lag}" for lag in features_config["lags"]]
    names += [f"{target}_roll_{s}_{w}" for w in features_config["rolling_windows"] for s in STATS]
    return names


def add_lag_rolling_features(df, features_config, horizon, target, grain):
    lags = list(features_config["lags"])
    windows = list(features_config["rolling_windows"])
    grain = list(grain)
    if min(lags) < horizon:
        raise ValueError(f"Smallest lag {min(lags)} is below the horizon {horizon}: it would leak the future")

    out = df.sort_values([*grain, "date"]).reset_index(drop=True)
    gaps = out.groupby(grain)["date"].diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        raise ValueError("Each series needs one row per calendar day; build the dataset first")

    by_series = out.groupby(grain)[target]
    for lag in lags:
        out[f"{target}_lag_{lag}"] = by_series.shift(lag)

    shifted = by_series.shift(horizon)
    grouped = shifted.groupby([out[c] for c in grain])
    for w in windows:
        for stat in STATS:
            out[f"{target}_roll_{stat}_{w}"] = grouped.transform(
                lambda s, w=w, stat=stat: s.rolling(w, min_periods=w).agg(stat)
            )
    return out