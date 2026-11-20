import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import build_features

GRAIN = ["store_key", "category"]
SERIES = [(1, "food"), (2, "toys")]
H = 7
ORIGIN = pd.Timestamp("2025-03-01")
CFG = {
    "calendar": {"holidays": ["2025-02-01"], "payday_days": [25, 1]},
    "lags_rolling": {"lags": [7, 14], "windows": [7, 14], "drop_incomplete_history": True},
    "categorical_columns": GRAIN,
}


def _data():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=90)
    parts = [
        pd.DataFrame(
            {"store_key": s, "category": c, "date": dates,
             "revenue": rng.normal(100, 10, len(dates)).round(2)}
        )
        for s, c in SERIES
    ]
    return pd.concat(parts, ignore_index=True)


def _build(df):
    return build_features(df, "date", GRAIN, "revenue", CFG, H)


def _leaky_future(df):
    out = _build(df)
    out["leak_next"] = out.groupby(GRAIN)["revenue"].shift(-1)       # uses tomorrow
    return out


def _leaky_now(df):
    out = _build(df)
    out["leak_now"] = out["revenue"]                                  # copies today
    return out


def _leaky_rolling_now(df):
    out = _build(df)
    out["leak_roll"] = out.groupby(GRAIN)["revenue"].transform(
        lambda s: s.rolling(7, min_periods=7).mean()                  # window includes today
    )
    return out


def _feature_cols(out):
    return [c for c in out.columns if c not in (*GRAIN, "date", "revenue")]


def _sorted(out):
    return out.sort_values([*GRAIN, "date"]).reset_index(drop=True)


def check_no_future_leak(build, df, origin):
    full = _sorted(build(df))
    cut = _sorted(build(df[df["date"] <= origin]))
    cols = _feature_cols(full)
    upto = full[full["date"] <= origin].reset_index(drop=True)
    pd.testing.assert_frame_equal(upto[cols], cut[cols])


def check_no_current_leak(build, df, origin):
    base = _sorted(build(df))
    changed = df.copy()
    mask = changed["date"] == origin
    changed.loc[mask, "revenue"] = changed.loc[mask, "revenue"] + 1000
    other = _sorted(build(changed))
    cols = _feature_cols(base)
    # rows earlier than origin + horizon cannot legitimately see the origin value
    blind = base["date"] < origin + pd.Timedelta(days=H)
    pd.testing.assert_frame_equal(base.loc[blind, cols], other.loc[blind, cols])


def test_real_features_have_no_future_leak():
    check_no_future_leak(_build, _data(), ORIGIN)


def test_real_features_have_no_current_leak():
    check_no_current_leak(_build, _data(), ORIGIN)


def test_future_rows_get_features_from_history_only():
    df = _data()
    last = df["date"].max()
    future = pd.concat(
        [
            pd.DataFrame(
                {"store_key": s, "category": c,
                 "date": pd.date_range(last + pd.Timedelta(days=1), periods=H),
                 "revenue": np.nan}
            )
            for s, c in SERIES
        ],
        ignore_index=True,
    )
    out = _build(pd.concat([df, future], ignore_index=True))
    cols = [c for c in out.columns if c.startswith(("revenue_lag_", "revenue_roll_"))]
    assert out.loc[out["date"] > last, cols].notna().all().all()


def test_check_catches_a_feature_from_the_future():
    with pytest.raises(AssertionError):
        check_no_future_leak(_leaky_future, _data(), ORIGIN)


def test_check_catches_a_feature_from_the_current_target():
    with pytest.raises(AssertionError):
        check_no_current_leak(_leaky_now, _data(), ORIGIN)
    with pytest.raises(AssertionError):
        check_no_current_leak(_leaky_rolling_now, _data(), ORIGIN)