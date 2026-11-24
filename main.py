import argparse
import logging
import os
import sys

from src.config import load_config
from src.data.ingestion import SAMPLE_CONFIG, load_all
from src.data.validation import validate_tables
from src.evaluation.compare import backtest_and_record
from src.evaluation.select import run_select
from src.evaluation.splits import make_folds, save_folds
from src.features.dataset import build_dataset
from src.features.pipeline import (
    DATASET_PATH,
    FEATURES_PATH,
    load_frame,
    make_features,
    save_frame,
)
from src.forecasting.forecast import generate_forecast, save_forecast
from src.logging_setup import setup_logging
from src.models.names import BASELINES, ML_MODELS
from src.models.training import train_final
from src.reporting.report import build_report
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
    print("Features")
    fcfg, xcfg = load_config("forecast_config"), load_config("features_config")
    dataset = build_dataset(load_all(sample=args.sample), fcfg)
    save_frame(dataset, DATASET_PATH)
    table = make_features(dataset, fcfg, xcfg)
    save_frame(table, FEATURES_PATH)
    save_folds(make_folds(dataset["date"].min(), dataset["date"].max(), fcfg))
    print(f"dataset: {len(dataset)} rows, features: {len(table)} rows")


def train(args):
    print("Train")
    if args.model and args.model not in ML_MODELS:
        raise ValueError(f"'{args.model}' is not a trainable model; choose from {ML_MODELS}")
    fcfg, xcfg = load_config("forecast_config"), load_config("features_config")
    table = load_frame(FEATURES_PATH)
    for name in [args.model] if args.model else ML_MODELS:
        print(f"{name}: saved to {train_final(name, table, fcfg, xcfg, args.run_id, args.sample)}")


def backtest(args):
    print("Backtest")
    allowed = (*ML_MODELS, *BASELINES)
    if args.model and args.model not in allowed:
        raise ValueError(f"Unknown model '{args.model}'; expected one of {allowed}")
    fcfg, xcfg = load_config("forecast_config"), load_config("features_config")
    ecfg = load_config("evaluation_config")
    primary = ecfg["primary_metric"]
    for name in [args.model] if args.model else allowed:
        summary = backtest_and_record(name, args.run_id, fcfg, xcfg, ecfg, args.sample)
        print(f"{name}: {primary}={summary['mean'][primary]:.4f}")

def select(args):
    print("Select")
    fcfg, xcfg = load_config("forecast_config"), load_config("features_config")
    ecfg = load_config("evaluation_config")
    if args.sample:
        fcfg = {**fcfg, "min_improvement": 0.0}
    summary = run_select(args.run_id, fcfg, xcfg, ecfg, args.sample)
    print(f"selected: {summary['selected']}")

def forecast(args):
    print("Forecast")
    fcfg, xcfg = load_config("forecast_config"), load_config("features_config")
    out = generate_forecast(load_all(sample=args.sample), fcfg, xcfg)
    print(f"{len(out)} forecast rows saved to {save_forecast(out, fcfg)}")

def report(args):
    print("Report")
    print(build_report(load_config("forecast_config")))

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
    parser.add_argument("--run-id", dest="run_id")
    args = parser.parse_args(argv)
    setup_logging()
    run_id = new_run_id()
    args.run_id = args.run_id or os.environ.get("RUN_ID") or new_run_id()
    logging.getLogger(__name__).info("run_id=%s", run_id)

    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        try:
            STAGES[name](args)
        except Exception as e:
            print(f"Stage '{name}' failed: {e}")
            return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())