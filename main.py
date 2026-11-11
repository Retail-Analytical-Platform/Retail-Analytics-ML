import argparse
import logging
import sys

<<<<<<< HEAD
from src.logging_setup import setup_logging
from src.run_id import new_run_id

=======
>>>>>>> 0e6b1b4 (Add CI workflow, configuration files, and initial tests)

def validate(args):
    print("Validate")
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
<<<<<<< HEAD
    setup_logging()
    run_id = new_run_id()
    logging.getLogger(__name__).info("run_id=%s", run_id)
=======
>>>>>>> 0e6b1b4 (Add CI workflow, configuration files, and initial tests)

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