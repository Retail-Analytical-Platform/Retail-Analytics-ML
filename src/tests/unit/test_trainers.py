import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.ingestion import load_sample
from src.features.categorical import fit_categories, prepare_model_frame
from src.features.dataset import build_forecast_dataset
from src.features.lags import drop_insufficient_history
from src.features.pipeline import build_features
from src.models.registry import build_model, get_model, load_model
from src.models.training import ML_MODELS, split_for_early_stopping

ROOT = Path(__file__).resolve().parents[3]
GRAIN = ["store_key", "category"]
H = 7
CFG = {
    "calendar": {"holidays": ["2025-01-01"], "payday_days": [25, 1]},
    "lags_rolling": {"lags": [7, 14], "windows": [7, 14], "drop_incomplete_history": True},
    "categorical_columns": GRAIN,
}
SMALL = {
    "catboost": {"iterations": 30, "depth": 3, "learning_rate": 0.1},
    "lightgbm": {"n_estimators": 30, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 5},
    "xgboost": {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1},
}


@pytest.fixture(scope="module")
def data():
    ds = build_forecast_dataset(load_sample("fact_sales"), "sold_at", GRAIN, "quantity")
    feats = drop_insufficient_history(build_features(ds, "sold_at", GRAIN, "quantity", CFG, H), "quantity")
    cut = pd.Timestamp("2025-11-01")
    return feats[feats["sold_at"] < cut], feats[feats["sold_at"] >= cut], feats


@pytest.fixture(scope="module", params=ML_MODELS)
def fitted(request, data):
    name = request.param
    train, valid, feats = data
    cats = fit_categories(feats, GRAIN)
    Xt, _ = prepare_model_frame(train, name, GRAIN, "sold_at", "quantity", cats)
    Xv, _ = prepare_model_frame(valid, name, GRAIN, "sold_at", "quantity", cats)
    model = get_model(name, **SMALL[name], seed=1, early_stopping_rounds=5, categorical_columns=GRAIN)
    model.fit(Xt, train["quantity"], Xv, valid["quantity"])
    return name, model, Xv, valid


def test_trains_and_predicts(fitted):
    _, model, Xv, valid = fitted
    pred = np.asarray(model.predict(Xv))
    assert len(pred) == len(valid) and np.isfinite(pred).all() and (pred >= 0).all()


def test_save_and_load_give_identical_predictions(fitted, tmp_path):
    name, model, Xv, _ = fitted
    model.save(tmp_path, run_id="r1")
    loaded = load_model(tmp_path)
    assert loaded.name == name and loaded.feature_names == model.feature_names
    assert np.allclose(loaded.predict(Xv), model.predict(Xv))


def test_categorical_columns_are_handled_natively(fitted):
    name, model, _, _ = fitted
    if name == "catboost":
        assert len(model.model.get_cat_feature_indices()) == len(GRAIN)
    elif name == "lightgbm":
        assert len(model.booster.pandas_categorical) == len(GRAIN)
    else:
        types = model.model.get_booster().feature_types
        assert sum(t == "c" for t in types) == len(GRAIN)


def test_validation_set_was_used_for_early_stopping(fitted):
    name, model, _, _ = fitted
    if name == "catboost":
        assert model.model.get_best_iteration() is not None
    elif name == "lightgbm":
        assert model.booster.best_iteration > 0
    else:
        assert model.model.best_iteration >= 0


@pytest.mark.parametrize("name", ML_MODELS)
def test_params_come_from_models_config(name):
    cfg = load_config("models_config")
    params = build_model(name, cfg).params
    for key, value in {**cfg["shared"], **cfg["models"][name]}.items():
        assert params[key] == value


def test_registry_does_not_import_ml_libraries():
    code = (
        "import sys; import src.models.registry as r; r.available_models();"
        "bad = {'catboost','lightgbm','xgboost'} & set(sys.modules); assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_early_stopping_set_is_strictly_later_than_fit_set():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=60), "y": 1.0})
    fit, es = split_for_early_stopping(df, "date", 10)
    assert len(es) == 10 and fit["date"].max() < es["date"].min()
    assert len(fit) + len(es) == len(df)