import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import make_folds
from src.evaluation.runner import run_backtest, summarize
from src.evaluation.tracking import load_run_records, write_run_record
from src.models.baseline import naive_predict, seasonal_naive_predict

GRAIN = ["store"]
BT = {"n_folds": 3, "step": 7, "window": "expanding", "min_train_days": 30}
H = 7


def _frame(days=120):
    dates = pd.date_range("2025-01-01", periods=days)
    rng = np.random.default_rng(1)
    parts = [
        pd.DataFrame({"store": s, "date": dates, "y": rng.normal(100, 5, days).round(1)})
        for s in ("A", "B")
    ]
    return pd.concat(parts, ignore_index=True)


def _folds(df):
    d = pd.to_datetime(df["date"])
    return make_folds(d.min(), d.max(), H, BT, test_days=14)


def test_every_fold_gets_a_baseline_score():
    df = _frame()
    folds = _folds(df)
    for fn in (naive_predict, seasonal_naive_predict):
        pred = run_backtest(df, "date", GRAIN, "y", folds,
                            lambda tr, va, fn=fn: fn(tr, va, "date", GRAIN, "y"))
        summary = summarize(pred)
        assert len(summary["per_fold"]) == len(folds)
        assert pred["horizon_step"].between(1, H).all()


def test_perfect_forecast_scores_zero_error():
    df = _frame()
    pred = run_backtest(df, "date", GRAIN, "y", _folds(df), lambda tr, va: va["y"].to_numpy())
    assert summarize(pred)["mean"]["wape"] == pytest.approx(0.0)


def test_rows_without_a_forecast_are_not_scored():
    df = _frame()
    def only_a(tr, va):
        out = va["y"].to_numpy().copy()
        out[(va["store"] == "B").to_numpy()] = np.nan
        return out
    pred = run_backtest(df, "date", GRAIN, "y", _folds(df), only_a)
    assert set(pred["store"]) == {"A"}


def test_a_fold_with_no_scored_rows_is_an_error():
    df = _frame()
    with pytest.raises(ValueError, match="No scored rows"):
        run_backtest(df, "date", GRAIN, "y", _folds(df), lambda tr, va: np.full(len(va), np.nan))


def test_two_runs_write_two_records_and_load_as_two_rows(tmp_path):
    df = _frame()
    folds = _folds(df)
    pred = run_backtest(df, "date", GRAIN, "y", folds,
                        lambda tr, va: naive_predict(tr, va, "date", GRAIN, "y"))
    summary = summarize(pred)
    for run_id in ("run-1", "run-2"):
        write_run_record(run_id, "naive", {}, summary, ["2025-01-01", "2025-04-30"], "y", GRAIN, H,
                         {"method": "rolling_origin", "n_folds": 3}, out_dir=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2
    table = load_run_records(tmp_path)
    assert len(table) == 2
    assert {"run_id", "model", "wape_mean", "git_commit"} <= set(table.columns)
    record = json.loads((tmp_path / "run-1_naive.json").read_text())
    assert len(record["metrics"]["per_fold"]) == 3