import json

import pytest

from src.evaluation.comparison import compare, load_records


def _rec(model, mean, std=0.01, secs=1.0, run_id="r1", **over):
    r = {
        "model": model, "run_id": run_id, "fit_seconds": secs,
        "target": "quantity", "grain": ["store_key"], "horizon": 28,
        "data_range": ["2024-01-01", "2025-12-31"],
        "training_window": {"window": "sliding"}, "n_scored_rows": 100,
        "validation": {"folds_hash": "abc"},
        "metrics": {"mean": {k: mean for k in ("wape", "mae", "rmse", "smape", "bias")},
                    "std": {k: std for k in ("wape", "mae", "rmse", "smape", "bias")}},
    }
    r.update(over)
    return r


RECS = [_rec("naive", 0.50), _rec("seasonal_naive", 0.40), _rec("catboost", 0.30),
        _rec("lightgbm", 0.35), _rec("xgboost", 0.33)]


def test_table_has_every_model_with_mean_std_and_time():
    t = compare(RECS, "wape")
    assert list(t["model"]) == ["catboost", "xgboost", "lightgbm", "seasonal_naive", "naive"]
    for col in ("wape_mean", "wape_std", "mae_mean", "mae_std", "fit_seconds"):
        assert col in t.columns and t[col].notna().all()


def test_percent_vs_baseline():
    t = compare(RECS, "wape").set_index("model")
    assert t.loc["catboost", "wape_vs_seasonal_naive_pct"] == pytest.approx(-25.0)


def test_different_folds_are_rejected():
    bad = [*RECS[:-1], _rec("xgboost", 0.33, validation={"folds_hash": "other"})]
    with pytest.raises(ValueError, match="same folds"):
        compare(bad, "wape")


@pytest.mark.parametrize("key,value", [("horizon", 7), ("target", "revenue"), ("n_scored_rows", 99)])
def test_different_conditions_are_rejected(key, value):
    bad = [*RECS[:-1], _rec("xgboost", 0.33, **{key: value})]
    with pytest.raises(ValueError, match=key):
        compare(bad, "wape")


def test_load_records_filters_by_run_and_keeps_latest(tmp_path):
    for i, (run, model) in enumerate([("r1", "naive"), ("r2", "naive"), ("r2", "catboost")]):
        (tmp_path / f"{i}.json").write_text(json.dumps(_rec(model, 0.4, run_id=run)))
    assert len(load_records(tmp_path)) == 2
    assert {r["run_id"] for r in load_records(tmp_path)} == {"r2"}
    assert len(load_records(tmp_path, run_id="r1")) == 1