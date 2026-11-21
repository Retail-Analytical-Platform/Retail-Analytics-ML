import argparse
import logging
import sys

from src.config import load_config
from src.data.ingestion import SAMPLE_CONFIG, load_all
from src.data.validation import validate_tables
from src.features.dataset import (
    build_forecast_dataset,
    load_forecast_dataset,
    save_forecast_dataset,
)
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
    cfg = load_config("forecast_config")
    df = load_all(sample=args.sample)["fact_sales"]
    ds = build_forecast_dataset(df, cfg["date_column"], cfg["grain"], cfg["target_column"])
    path = save_forecast_dataset(ds)
    print(f"Saved {len(ds)} rows to {path}")
    
def train(args):
    from src.features.pipeline import build_features
    from src.models.training import ML_MODELS, train_model

    if args.model and args.model not in ML_MODELS:
        raise ValueError(f"'{args.model}' is not a trainable model; choose from {ML_MODELS}")
    fcfg = load_config("forecast_config")
    feat_cfg = load_config("features_config")
    model_cfg = load_config("models_config")
    try:
        ds = load_forecast_dataset()
    except FileNotFoundError:
        raise FileNotFoundError("Forecast dataset not found; run --stage features first") from None
    feats = build_features(ds, fcfg["date_column"], fcfg["grain"], fcfg["target_column"],
                           feat_cfg, fcfg["horizon"])
    for name in [args.model] if args.model else ML_MODELS:
        _, path = train_model(name, feats, fcfg, feat_cfg, model_cfg, args.run_id)
        print(f"{name}: saved to {path}")
    
def backtest(args):
    from src.evaluation.comparison import compare, load_records
    from src.evaluation.model_runner import (
        ALL_MODELS,
        get_folds,
        load_inputs,
        run_model,
    )
    from src.pipeline_paths import METRICS_DIR

    if args.model and args.model not in ALL_MODELS:
        raise ValueError(f"Unknown model '{args.model}'; expected one of {ALL_MODELS}")
    fcfg, feat_cfg, model_cfg, ds, feats = load_inputs(args)
    folds, fhash = get_folds(ds, fcfg)
    primary = fcfg["primary_metric"]
    for name in [args.model] if args.model else ALL_MODELS:
        rec = run_model(name, ds, feats, folds, fhash, fcfg, feat_cfg, model_cfg, args.run_id)
        m, s = rec["metrics"]["mean"], rec["metrics"]["std"]
        print(f"{name}: {primary}={m[primary]:.4f} (+/-{s[primary]:.4f}) fit={rec['fit_seconds']}s")
    if not args.model:
        table = compare(load_records(run_id=args.run_id), primary)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
        print(table.round(4).to_string(index=False))

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
            print(f"Stage '{name}' failed: {e}")
            return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())