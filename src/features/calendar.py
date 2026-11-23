import pandas as pd

CALENDAR_COLUMNS = [
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter",
    "is_weekend", "is_month_start", "is_month_end", "is_holiday", "is_payday",
]


def add_calendar_features(df, features_config):
    out = df.copy()
    d = out["date"]
    out["day_of_week"] = d.dt.day_of_week
    out["day_of_month"] = d.dt.day
    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter
    out["is_weekend"] = (d.dt.day_of_week >= 5).astype(int)
    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    out["is_holiday"] = d.isin(pd.to_datetime(features_config["holidays"])).astype(int)
    out["is_payday"] = d.dt.day.isin(features_config["payday_days"]).astype(int)
    return out