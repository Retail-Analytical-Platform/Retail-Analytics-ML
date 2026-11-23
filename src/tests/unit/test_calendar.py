import pandas as pd

from src.features.calendar import CALENDAR_COLUMNS, add_calendar_features

XCFG = {"holidays": ["2026-01-01"], "payday_days": [25]}


def test_future_dates_without_target_are_fully_filled():
    future = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=40), "store_key": 1})
    out = add_calendar_features(future, XCFG)
    assert not out[CALENDAR_COLUMNS].isna().any().any()


def test_known_values():
    out = add_calendar_features(pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-25", "2026-01-31"])}), XCFG)
    assert out.loc[0, "day_of_week"] == 3          # Thursday
    assert out.loc[0, "is_holiday"] == 1 and out.loc[0, "is_month_start"] == 1
    assert out.loc[1, "is_payday"] == 1
    assert out.loc[2, "is_month_end"] == 1