import argparse
import sys

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    parser.add_argument("--model")
    parser.add_argument("--target")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        try : 
            STAGES[name](args)
        except Exception as e:
            print(f"Stage'{name}' failed: {e}",file=sys.stderr)
            return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())