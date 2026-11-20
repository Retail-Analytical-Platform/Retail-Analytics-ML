import pandas as pd


def _assert_daily(df, date_col, grain):
    steps = df.groupby(grain, sort=False)[date_col].diff().dropna()
    if (steps != pd.Timedelta(days=1)).any():
        raise ValueError(
            "Each series needs exactly one row per day; build it with the dataset builder first."
        )


def add_lag_rolling_features(df, date_col, grain, target, cfg, horizon):
    """cfg is the "lags_rolling" section of feature_config.yaml."""
    lags = sorted(cfg["lags"])
    windows = sorted(cfg["windows"])
    if min(lags) < horizon:
        raise ValueError(
            f"Smallest lag {min(lags)} is below the forecast horizon {horizon}; this would leak the future."
        )

    out = df.sort_values([*grain, date_col]).reset_index(drop=True)
    _assert_daily(out, date_col, grain)
    by_series = out.groupby(grain, sort=False)[target]

    for k in lags:
        out[f"{target}_lag_{k}"] = by_series.shift(k)

    # Rolling stats run on the target shifted by the horizon, so a window
    # never includes a value the forecast could not have known.
    out["_base"] = by_series.shift(horizon)
    base = out.groupby(grain, sort=False)["_base"]
    for w in windows:
        for stat in ("mean", "std", "min", "max"):
            out[f"{target}_roll_{stat}_{w}"] = base.transform(
                lambda s, w=w, stat=stat: getattr(s.rolling(w, min_periods=w), stat)()
            )
    return out.drop(columns="_base")


def drop_insufficient_history(df, target):
    """Drop the first rows of each series that lack enough history.
    Apply to training rows only; future rows keep their features."""
    cols = [c for c in df.columns if c.startswith((f"{target}_lag_", f"{target}_roll_"))]
    return df.dropna(subset=cols).reset_index(drop=True)