import pandas as pd


def _fail(key, why):
    raise ValueError(f"feature_config.yaml: '{key}' {why}")


def _int_list(section_cfg, key, section, lo=1, hi=None):
    path = f"{section}.{key}"
    values = section_cfg.get(key)
    if not isinstance(values, list) or not values:
        _fail(path, "must be a non-empty list")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
            _fail(path, f"has an invalid value: {v!r}")


def validate_feature_config(cfg):
    for section in ("calendar", "lags_rolling", "categorical_columns"):
        if section not in cfg:
            _fail(section, "is missing")

    cal = cfg["calendar"]
    if not isinstance(cal, dict):
        _fail("calendar", "must be a mapping")
    holidays = cal.get("holidays", [])
    if not isinstance(holidays, list):
        _fail("calendar.holidays", "must be a list of dates")
    for h in holidays:
        try:
            pd.to_datetime(h)
        except (ValueError, TypeError):
            _fail("calendar.holidays", f"has an invalid date: {h!r}")
    if "payday_days" in cal:
        _int_list(cal, "payday_days", "calendar", 1, 31)

    lr = cfg["lags_rolling"]
    if not isinstance(lr, dict):
        _fail("lags_rolling", "must be a mapping")
    _int_list(lr, "lags", "lags_rolling")
    _int_list(lr, "windows", "lags_rolling")
    if not isinstance(lr.get("drop_incomplete_history", True), bool):
        _fail("lags_rolling.drop_incomplete_history", "must be true or false")

    cols = cfg["categorical_columns"]
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        _fail("categorical_columns", "must be a list of column names")