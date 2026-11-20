import pandas as pd

from src.features.calendar import add_calendar_features

CFG = {"holidays": ["2027-01-01"], "payday_days": [25, 1]}

def _future():
    return pd.DataFrame({"date": pd.date_range("2027-01-01", periods=31)})

def test_future_dates_have_no_nulls_without_a_target():
    out = add_calendar_features(_future(), "date", CFG)
    assert "revenue" not in out.columns
    assert out.isna().sum().sum() == 0

def test_flags_and_parts():
    out = add_calendar_features(_future(), "date", CFG).set_index("date")
    assert out.loc["2027-01-01", "is_holiday"] == 1
    assert out.loc["2027-01-02", "is_holiday"] == 0
    assert out.loc["2027-01-02", "is_weekend"] == 1     # Saturday
    assert out.loc["2027-01-04", "is_weekend"] == 0     # Monday
    assert out.loc["2027-01-01", "is_month_start"] == 1
    assert out.loc["2027-01-31", "is_month_end"] == 1
    assert out.loc["2027-01-25", "is_payday"] == 1
    assert out.loc["2027-01-25", "day_of_month"] == 25
    assert out.loc["2027-01-25", "quarter"] == 1

def test_input_frame_is_not_modified():
    df = _future()
    add_calendar_features(df, "date", CFG)
    assert list(df.columns) == ["date"]