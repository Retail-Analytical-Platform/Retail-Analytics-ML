import pytest

from src.config import load_config
from src.data.ingestion import load_all
from src.features.dataset import build_dataset
from src.features.pipeline import feature_names, make_features, model_view
from src.forecasting.forecast import generate_forecast
from src.models.training import build_model, fit_with_holdout


def test_forecast_has_horizon_times_series_rows(tmp_path):
    pytest.importorskip("lightgbm")
    fcfg, xcfg = load_config("forecast_config"), load_config("features_config")
    tables = load_all(sample=True)
    data = make_features(build_dataset(tables, fcfg), fcfg, xcfg)
    cat, feats = xcfg["categorical_columns"], feature_names(fcfg, xcfg)
    view = model_view(data, feats, cat, "lightgbm")
    model = build_model("lightgbm", cat, sample=True)
    fit_with_holdout(model, view, data["quantity"], data["date"], fcfg["early_stopping_holdout_days"])
    model.save(tmp_path, "t")

    out = generate_forecast(tables, fcfg, xcfg, model_dir=tmp_path)
    series = data[fcfg["grain"]].drop_duplicates().shape[0]
    assert len(out) == fcfg["horizon"] * series
    assert out["forecast"].notna().all() and (out["forecast"] >= 0).all()