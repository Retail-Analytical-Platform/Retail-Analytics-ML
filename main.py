import argparse
import logging
import sys

from src.config import load_config
from src.data.ingestion import SAMPLE_CONFIG, load_all
from src.data.validation import validate_tables
from src.logging_setup import setup_logging
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
def train(args):
    print("Train")
def backtest(args): 
    print("Backtest")
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