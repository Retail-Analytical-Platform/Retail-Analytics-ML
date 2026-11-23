import itertools
import json

import numpy as np
import pandas as pd

from src.reporting.report import build_report

FCFG = {"output_dir": "forecasts"}


def _grid(dates):
    keys = pd.DataFrame(list(itertools.product([1, 2], ["A", "B"])), columns=["store_key", "category"])
    return keys.merge(pd.DataFrame({"date": dates}), how="cross")


def test_report_files_are_written(tmp_path):
    metrics, forecasts = tmp_path / "metrics", tmp_path / "forecasts"
    metrics.mkdir()
    forecasts.mkdir()
    rng = np.random.default_rng(0)

    past = _grid(pd.date_range("2025-12-01", periods=6))
    past["y_true"], past["y_pred"] = rng.random(len(past)), rng.random(len(past))
    past.to_parquet(metrics / "selected_backtest_predictions.parquet", index=False)

    future = _grid(pd.date_range("2026-01-01", periods=6))
    future["forecast"] = rng.random(len(future))
    future.to_parquet(forecasts / "forecast.parquet", index=False)

    row = {"primary_mean": 0.7, "primary_std": 0.01, "train_seconds": 1.0}
    (metrics / "comparison.json").write_text(json.dumps({
        "run_id": "r1", "primary_metric": "wape", "selected": "xgboost", "gate_passed": True,
        "candidates": [{"model": "xgboost", **row}],
        "baselines": [{"model": "seasonal_naive", **row}],
    }))

    path = build_report(FCFG, artifact_dir=tmp_path, metrics_dir=metrics)
    assert path.exists() and "xgboost" in path.read_text(encoding="utf-8")
    assert (path.parent / "metrics_table.csv").exists()
    assert len(pd.read_csv(path.parent / "forecast.csv")) == len(future)