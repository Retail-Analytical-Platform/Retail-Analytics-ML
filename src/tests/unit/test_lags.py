import pandas as pd
import pytest

from src.features.lags import add_lag_rolling_features, drop_insufficient_history

CFG = {"lags": [2, 3], "windows": [3]}
H = 2


def _frame():
    dates = pd.date_range("2025-01-01", periods=40)
    a = pd.DataFrame({"store": "A", "date": dates, "revenue": range(1, 41)})
    b = pd.DataFrame({"store": "B", "date": dates, "revenue": range(101, 141)})
    return pd.concat([a, b], ignore_index=True)


def _build(df=None):
    return add_lag_rolling_features(
        _frame() if df is None else df, "date", ["store"], "revenue", CFG, H
    )


def test_lag_and_rolling_values_and_series_isolation():
    out = _build()
    a = out[out["store"] == "A"].reset_index(drop=True)
    assert a.loc[10, "revenue_lag_2"] == 9          # value from 2 days earlier
    assert a.loc[10, "revenue_roll_mean_3"] == 8    # mean of days 6..8 (shifted by horizon)
    assert pd.isna(a.loc[1, "revenue_lag_2"])
    b = out[out["store"] == "B"].reset_index(drop=True)
    assert pd.isna(b.loc[0, "revenue_lag_2"])       # no bleed from the end of series A
    assert b.loc[10, "revenue_lag_2"] == 109


def test_rejects_lag_below_horizon():
    with pytest.raises(ValueError, match="leak"):
        add_lag_rolling_features(
            _frame(), "date", ["store"], "revenue", {"lags": [1], "windows": [3]}, H
        )


def test_changing_target_does_not_change_earlier_features():
    base = _build()
    df = _frame()
    df.loc[20, "revenue"] = 9999                    # series A, day index 20
    changed = _build(df)
    a0 = base[base["store"] == "A"].reset_index(drop=True)
    a1 = changed[changed["store"] == "A"].reset_index(drop=True)
    cols = [c for c in a0.columns if c.startswith(("revenue_lag_", "revenue_roll_"))]
    pd.testing.assert_frame_equal(a0.loc[:21, cols], a1.loc[:21, cols])  # days 20 and 21 unaffected
    assert a0.loc[22, "revenue_lag_2"] != a1.loc[22, "revenue_lag_2"]    # day 22 is the first that sees it


def test_drop_insufficient_history():
    out = drop_insufficient_history(_build(), "revenue")
    assert out.isna().sum().sum() == 0
    assert len(out) < 80