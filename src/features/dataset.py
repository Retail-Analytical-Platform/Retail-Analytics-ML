from pathlib import Path

import pandas as pd

DATASET_PATH = Path("artifacts/data/forecast_dataset.parquet")


def build_forecast_dataset(df, date_col, grain, target):
    """One row per series per day, no date gaps, sorted by series then date."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    # 1. aggregate to the chosen grain
    daily = df.groupby([*grain, date_col], as_index=False)[target].sum()

    # A day with no rows in any series is a data gap, not a zero-sales day.
    all_days = pd.date_range(daily[date_col].min(), daily[date_col].max(), freq="D")
    seen = pd.DatetimeIndex(daily[date_col].unique())
    missing = all_days.difference(seen)
    if len(missing):
        raise ValueError(
            f"No data on {len(missing)} day(s), first: {missing[0].date()}. "
            "Fix the source; do not zero-fill."
        )
    # 2-3. full date index per series, from its first sale to the last day in the data
    parts = []
    for key, g in daily.groupby(grain):
        key = key if isinstance(key, tuple) else (key,)
        idx = pd.date_range(g[date_col].min(), all_days[-1], freq="D", name=date_col)
        g = g.set_index(date_col).reindex(idx)
        g[target] = g[target].fillna(0)
        for col, val in zip(grain, key):
            g[col] = val
        parts.append(g.reset_index())

    # 4. sort by series, then date
    out = pd.concat(parts, ignore_index=True)
    return out[[*grain, date_col, target]].sort_values([*grain, date_col]).reset_index(drop=True)


def save_forecast_dataset(df, path=DATASET_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_forecast_dataset(path=DATASET_PATH):
    return pd.read_parquet(path)

def build_dataset(tables, config):
    target, grain = config["target_column"], list(config["grain"])
    fact = tables["fact_sales"].copy()
    calendar = pd.DatetimeIndex(
        pd.to_datetime(tables["dim_date"]["full_date"]).dt.normalize()
    ).unique().sort_values()

    fact["date"] = pd.to_datetime(fact["sold_at"]).dt.normalize()
    unknown = fact.loc[~fact["date"].isin(calendar), "date"].unique()
    if len(unknown):
        raise ValueError(
            f"{len(unknown)} sale date(s) are missing from dim_date, first {pd.Timestamp(unknown[0]).date()}"
        )

    daily = fact.groupby([*grain, "date"], as_index=False)[target].sum()
    series = fact[grain].drop_duplicates()
    full = series.merge(pd.DataFrame({"date": calendar}), how="cross")
    out = full.merge(daily, on=[*grain, "date"], how="left")
    out["is_zero_filled"] = out[target].isna()
    out[target] = out[target].fillna(0.0).astype(float)
    return out.sort_values([*grain, "date"]).reset_index(drop=True)