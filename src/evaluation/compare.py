import time

import numpy as np

from src.evaluation.backtest import run_backtest
from src.evaluation.metrics import score_table
from src.evaluation.splits import load_folds, make_folds
from src.features.pipeline import (
    DATASET_PATH,
    FEATURES_PATH,
    feature_names,
    load_frame,
    model_view,
)
from src.models.baseline import BASELINE_FUNCTIONS
from src.models.names import BASELINES
from src.models.training import build_model, fit_with_holdout, model_params
from src.pipeline_paths import FOLDS_PATH, METRICS_DIR
from src.tracking import record_run


def get_folds(dataset, fcfg):
    if FOLDS_PATH.exists():
        return load_folds()
    return make_folds(dataset["date"].min(), dataset["date"].max(), fcfg)


def backtest_baseline(name, dataset, folds, fcfg):
    fn, grain, target = BASELINE_FUNCTIONS[name], fcfg["grain"], fcfg["target_column"]

    def predict(train, val):
        return fn(train, sorted(val["date"].unique()), grain, target)

    return run_backtest(dataset, folds, predict, grain, target)


def backtest_ml(name, table, folds, fcfg, xcfg, sample=False):
    feats, cat = feature_names(fcfg, xcfg), xcfg["categorical_columns"]
    grain, target = fcfg["grain"], fcfg["target_column"]
    view = model_view(table, feats, cat, name)
    seconds = []

    def predict(train, val):
        model = build_model(name, cat, sample)
        start = time.perf_counter()
        fit_with_holdout(model, view.loc[train.index], train[target], train["date"],
                         fcfg["early_stopping_holdout_days"])
        seconds.append(time.perf_counter() - start)
        out = val[[*grain, "date"]].copy()
        out["y_pred"] = np.clip(model.predict(view.loc[val.index]), 0, None)
        return out

    return run_backtest(table, folds, predict, grain, target), float(np.mean(seconds))


def summarize(preds, metric_names):
    per_fold = score_table(preds, metric_names, by=["fold"])
    return {
        "mean": per_fold[metric_names].mean().to_dict(),
        "std": per_fold[metric_names].std(ddof=0).to_dict(),
        "per_fold": per_fold.to_dict("records"),
    }


def backtest_and_record(name, run_id, fcfg, xcfg, ecfg, sample=False):
    dataset = load_frame(DATASET_PATH)
    folds = get_folds(dataset, fcfg)
    if name in BASELINES:
        preds, seconds, params = backtest_baseline(name, dataset, folds, fcfg), 0.0, {}
    else:
        preds, seconds = backtest_ml(name, load_frame(FEATURES_PATH), folds, fcfg, xcfg, sample)
        params = model_params(name, sample)
    summary = summarize(preds, ecfg["metrics"])
    summary["train_seconds"] = seconds
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(METRICS_DIR / f"{run_id}_{name}_predictions.parquet", index=False)
    span = [dataset["date"].min().date().isoformat(), dataset["date"].max().date().isoformat()]
    record_run(run_id, name, params, summary, fcfg, span)
    return summary