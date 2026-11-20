import pandas as pd


def add_calendar_features(df, date_col, cfg):
    out = df.copy()
    d = pd.to_datetime(out[date_col]).dt.normalize()

    out["day_of_week"] = d.dt.day_of_week          # Monday = 0
    out["day_of_month"] = d.dt.day
    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter

    out["is_weekend"] = (d.dt.day_of_week >= 5).astype(int)
    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)

    holidays = pd.to_datetime(cfg.get("holidays", [])).normalize()
    out["is_holiday"] = d.isin(holidays).astype(int)

    out["is_payday"] = d.dt.day.isin(cfg.get("payday_days", [])).astype(int)
    return out