import argparse
import logging
import sys
from functools import partial

import pandas as pd

from src.config import load_config
from src.data.ingestion import SAMPLE_CONFIG, load_all
from src.data.validation import validate_tables
from src.evaluation.backtest import make_folds, save_folds
from src.evaluation.metrics import check_primary_metric
from src.evaluation.runner import run_backtest, summarize
from src.evaluation.tracking import write_run_record
from src.features.dataset import (
    build_forecast_dataset,
    load_forecast_dataset,
    save_forecast_dataset,
)
from src.logging_setup import setup_logging
from src.models.baseline import BASELINES
from src.run_id import new_run_id


def validate(args):
    print("Validate")
    config = load_config("forecast_config")
    if args.sample:
        config = {**config, **SAMPLE_CONFIG}
    tables = load_all(sample=args.sample)
    for name, df in tables.items():
        print(f"{name}: {len(df)} rows")
    failures = validate_tables(tables, config)
    if failures:
        raise ValueError("Data validation failed:\n- " + "\n- ".join(failures))
    print("All checks passed")

def features(args):
    cfg = load_config("forecast_config")
    df = load_all(sample=args.sample)["fact_sales"]
    ds = build_forecast_dataset(df, cfg["date_column"], cfg["grain"], cfg["target_column"])
    path = save_forecast_dataset(ds)
    print(f"Saved {len(ds)} rows to {path}")
    
def train(args):
    print("Train")
    
def backtest(args):
    cfg = load_config("forecast_config")
    model = args.model or "seasonal_naive"
    if model not in BASELINES:
        raise ValueError(f"Unknown model '{model}'; available now: {sorted(BASELINES)}")
    primary = cfg["primary_metric"]
    check_primary_metric(primary)

    date_col, grain, target, horizon = (
        cfg["date_column"], cfg["grain"], cfg["target_column"], cfg["horizon"]
    )
    df = load_forecast_dataset()          # written by the features stage
    dates = pd.to_datetime(df[date_col])
    bt = cfg["backtest"]
    folds = make_folds(dates.min(), dates.max(), horizon, bt, cfg["split"]["test_days"])
    save_folds(folds, bt["folds_path"])

    predict = partial(BASELINES[model], date_col=date_col, grain=grain, target=target)
    pred = run_backtest(df, date_col, grain, target, folds, predict)
    summary = summarize(pred)

    path = write_run_record(
        args.run_id, model, {"season": 7} if model == "seasonal_naive" else {}, summary,
        [dates.min().date().isoformat(), dates.max().date().isoformat()], target, grain, horizon,
        {"method": "rolling_origin", "window": bt.get("window", "expanding"),
         "n_folds": bt["n_folds"], "step": bt["step"], "test_days": cfg["split"]["test_days"]},
    )
    print(f"{model}: mean {primary} = {summary['mean'][primary]:.4f} over {len(folds)} folds; record {path}")

def select(args):
    print("Select")
def forecast(args):
    print("Forecast")
def report(args):
    print("Report")

STAGES = {
    "validate": validate,
    "features": features,
    "train": train,
    "backtest": backtest,
    "select": select,
    "forecast": forecast,
    "report": report,
}

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    parser.add_argument("--model")
    parser.add_argument("--target")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    run_id = new_run_id()
    args.run_id = run_id
    logging.getLogger(__name__).info("run_id=%s", run_id)

    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        try:
            STAGES[name](args)
        except Exception as e:
            print(f"Stage'{name}' failed: {e}", file=sys.stderr)
            return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())