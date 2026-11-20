import copy

import pandas as pd
import pytest

from src.features.categorical import MODELS, fit_categories, prepare_model_frame
from src.features.feature_config import validate_feature_config
from src.features.pipeline import build_features

CATS = ["store_key", "category"]
GOOD = {
    "calendar": {"holidays": ["2025-01-01"], "payday_days": [25, 1]},
    "lags_rolling": {"lags": [7, 14], "windows": [7], "drop_incomplete_history": True},
    "categorical_columns": CATS,
}


def _df():
    dates = pd.date_range("2025-01-01", periods=10)
    parts = [
        pd.DataFrame(
            {"store_key": s, "category": c, "date": dates, "revenue": range(10), "revenue_lag_7": 1.0}
        )
        for s, c in [(1, "food"), (2, "toys")]
    ]
    return pd.concat(parts, ignore_index=True)


def _view(model, df=None, **kw):
    return prepare_model_frame(_df() if df is None else df, model, CATS, "date", "revenue", **kw)


def test_all_models_get_the_same_feature_columns():
    cols = [list(_view(m)[0].columns) for m in MODELS]
    assert cols[0] == cols[1] == cols[2]
    assert "date" not in cols[0] and "revenue" not in cols[0]


def test_catboost_view():
    X, params = _view("catboost")
    assert X["store_key"].map(lambda v: isinstance(v, str)).all()
    assert X["category"].map(lambda v: isinstance(v, str)).all()
    assert params == {"cat_features": CATS}
    assert X["revenue_lag_7"].dtype == float


def test_lightgbm_view():
    X, params = _view("lightgbm")
    assert str(X["store_key"].dtype) == "category"
    assert str(X["category"].dtype) == "category"
    assert params == {}


def test_xgboost_view():
    X, params = _view("xgboost")
    assert str(X["store_key"].dtype) == "category"
    assert params == {"enable_categorical": True}


def test_fixed_categories_keep_codes_consistent():
    df = _df()
    cats = fit_categories(df, CATS)
    X, _ = _view("lightgbm", df[df["store_key"] == 1], categories=cats)
    assert list(X["store_key"].cat.categories) == [1, 2]
    assert list(X["category"].cat.categories) == ["food", "toys"]


def test_unknown_model_and_missing_column():
    with pytest.raises(ValueError, match="Unknown model"):
        _view("randomforest")
    with pytest.raises(ValueError, match="not in the data"):
        prepare_model_frame(_df(), "lightgbm", ["nope"], "date", "revenue")


def test_valid_config_passes():
    validate_feature_config(GOOD)


def test_bad_config_names_the_key():
    bad = copy.deepcopy(GOOD)
    bad["lags_rolling"]["lags"] = [7, "x"]
    with pytest.raises(ValueError, match="lags_rolling.lags"):
        validate_feature_config(bad)

    bad = copy.deepcopy(GOOD)
    bad["calendar"]["payday_days"] = [40]
    with pytest.raises(ValueError, match="calendar.payday_days"):
        validate_feature_config(bad)

    bad = copy.deepcopy(GOOD)
    bad["calendar"]["holidays"] = ["not a date"]
    with pytest.raises(ValueError, match="calendar.holidays"):
        validate_feature_config(bad)

    missing = {k: v for k, v in GOOD.items() if k != "categorical_columns"}
    with pytest.raises(ValueError, match="categorical_columns"):
        validate_feature_config(missing)


def test_build_features_adds_calendar_and_lag_columns():
    df = _df()[["store_key", "category", "date", "revenue"]]
    out = build_features(df, "date", CATS, "revenue", GOOD, horizon=7)
    assert len(out) == len(df)
    assert {"day_of_week", "is_holiday", "revenue_lag_7", "revenue_roll_mean_7"} <= set(out.columns)