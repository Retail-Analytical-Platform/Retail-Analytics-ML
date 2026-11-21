import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import load_sample
from src.features.categorical import fit_categories, prepare_model_frame
from src.features.dataset import build_forecast_dataset
from src.features.lags import drop_insufficient_history
from src.features.pipeline import build_features
from src.models.registry import get_model, load_model

GRAIN = ["store_key", "category"]
H = 7
CFG = {
    "calendar": {"holidays": [], "payday_days": [25, 1]},
    "lags_rolling": {"lags": [7, 14], "windows": [7, 14], "drop_incomplete_history": True},
    "categorical_columns": GRAIN,
}
SMALL = {
    "catboost": {"iterations": 30, "depth": 3, "learning_rate": 0.1},
    "lightgbm": {"n_estimators": 30, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 5},
    "xgboost": {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1},
}


@pytest.fixture(scope="module")
def frames():
    ds = build_forecast_dataset(load_sample("fact_sales"), "sold_at", GRAIN, "quantity")
    feats = drop_insufficient_history(build_features(ds, "sold_at", GRAIN, "quantity", CFG, H), "quantity")
    cut = pd.Timestamp("2025-11-01")
    return feats[feats["sold_at"] < cut], feats[feats["sold_at"] >= cut]


@pytest.mark.parametrize("name", list(SMALL))
def test_trains_predicts_and_round_trips(name, frames, tmp_path):
    train, valid = frames
    cats = fit_categories(pd.concat(frames), GRAIN)
    Xt, _ = prepare_model_frame(train, name, GRAIN, "sold_at", "quantity", cats)
    Xv, _ = prepare_model_frame(valid, name, GRAIN, "sold_at", "quantity", cats)
    model = get_model(name, **SMALL[name], seed=1, early_stopping_rounds=5, categorical_columns=GRAIN)
    model.fit(Xt, train["quantity"], Xv, valid["quantity"])
    pred = np.asarray(model.predict(Xv))
    assert len(pred) == len(valid) and np.isfinite(pred).all() and (pred >= 0).all()

    model.save(tmp_path, run_id="r1")
    assert np.allclose(load_model(tmp_path).predict(Xv), pred)