import numpy as np
import pandas as pd
import pytest

from src.models.baseline import naive_predict, seasonal_naive_predict

GRAIN = ["store"]

def _frame(days=31):
    dates = pd.date_range("2025-01-01", periods=days)
    a = pd.DataFrame({"store": "A", "date": dates, "y": np.arange(1, days + 1, dtype=float)})
    b = pd.DataFrame({"store": "B", "date": dates, "y": np.arange(101, 101 + days, dtype=float)})
    c_dates = pd.date_range("2025-01-18", periods=days - 17)           # starts late
    c = pd.DataFrame({"store": "C", "date": c_dates, "y": np.arange(1000, 1000 + len(c_dates), dtype=float)})
    return pd.concat([a, b, c], ignore_index=True)

def _split(valid_end):
    df = _frame()
    d = pd.to_datetime(df["date"])
    origin = pd.Timestamp("2025-01-21")
    return df[d <= origin], df[(d > origin) & (d <= pd.Timestamp(valid_end))]

def _at(valid, pred, store, date):
    s = pd.Series(pred, index=valid.index)
    on_date = pd.to_datetime(valid["date"]) == pd.Timestamp(date)
    return s[(valid["store"] == store) & on_date].iloc[0]

def test_naive_repeats_each_series_last_value():
    train, valid = _split("2025-01-28")
    pred = naive_predict(train, valid, "date", GRAIN, "y")
    assert _at(valid, pred, "A", "2025-01-25") == 21
    assert _at(valid, pred, "B", "2025-01-25") == 121
    assert _at(valid, pred, "C", "2025-01-25") == 1003

def test_seasonal_naive_uses_the_value_7_days_earlier_for_a_7_day_horizon():
    train, valid = _split("2025-01-28")
    pred = seasonal_naive_predict(train, valid, "date", GRAIN, "y")
    assert _at(valid, pred, "A", "2025-01-22") == 15      # Jan 15
    assert _at(valid, pred, "A", "2025-01-28") == 21      # Jan 21, the origin
    assert _at(valid, pred, "B", "2025-01-22") == 115

def test_seasonal_naive_never_uses_values_after_the_origin_for_a_longer_horizon():
    train, valid = _split("2025-01-31")
    pred = seasonal_naive_predict(train, valid, "date", GRAIN, "y")
    assert _at(valid, pred, "A", "2025-01-29") == 15      # two weeks back: Jan 15
    assert _at(valid, pred, "A", "2025-01-31") == 17      # Jan 17

def test_seasonal_naive_falls_back_to_naive_when_history_is_missing():
    train, valid = _split("2025-01-28")
    pred = seasonal_naive_predict(train, valid, "date", GRAIN, "y")
    assert _at(valid, pred, "C", "2025-01-22") == 1003    # no Jan 15 row for C

def test_validation_dates_must_follow_training():
    train, _ = _split("2025-01-28")
    with pytest.raises(ValueError, match="after the end"):
        seasonal_naive_predict(train, train.tail(3), "date", GRAIN, "y")