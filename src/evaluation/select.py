import json
import math
import shutil

from src.evaluation.compare import backtest_and_record
from src.models.names import BASELINES
from src.pipeline_paths import METRICS_DIR, MODELS_DIR, SELECTED_DIR
from src.tracking import load_records

LOWER_IS_BETTER = ("mae", "rmse", "smape", "wape")


class QualityGateError(RuntimeError):
    pass


def _score(record, primary):
    m = record["metrics"]
    return (round(m["mean"][primary], 6), round(m["std"][primary], 6), round(m["train_seconds"], 3))


def choose(records, primary, margin):
    if primary not in LOWER_IS_BETTER:
        raise ValueError(f"primary_metric must be one of {LOWER_IS_BETTER}")
    candidates = [r for r in records if r["model"] not in BASELINES]
    baseline = next((r for r in records if r["model"] == "seasonal_naive"), None)
    if not candidates:
        raise ValueError("No trained model results found")
    if baseline is None:
        raise ValueError("Seasonal naive results are missing")
    for r in [*candidates, baseline]:
        if math.isnan(r["metrics"]["mean"][primary]):
            raise ValueError(f"{r['model']}: primary metric is NaN")
    ranked = sorted(candidates, key=lambda r: _score(r, primary))
    best = ranked[0]
    passed = best["metrics"]["mean"][primary] <= baseline["metrics"]["mean"][primary] * (1 - margin)
    return best, ranked, baseline, passed


def _row(record, primary):
    m = record["metrics"]
    return {"model": record["model"], "primary_mean": m["mean"][primary],
            "primary_std": m["std"][primary], "train_seconds": m["train_seconds"]}


def run_select(run_id, fcfg, xcfg, ecfg, sample=False):
    for name in BASELINES:
        backtest_and_record(name, run_id, fcfg, xcfg, ecfg, sample)
    records = load_records(run_id=run_id)
    primary, margin = ecfg["primary_metric"], fcfg["min_improvement"]
    best, ranked, baseline, passed = choose(records, primary, margin)

    summary = {
        "run_id": run_id, "primary_metric": primary, "min_improvement": margin,
        "selected": best["model"], "gate_passed": passed,
        "baselines": [_row(r, primary) for r in records if r["model"] in BASELINES],
        "candidates": [_row(r, primary) for r in ranked],
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise QualityGateError(
            f"{best['model']} {primary}={best['metrics']['mean'][primary]:.4f} does not beat "
            f"seasonal_naive {baseline['metrics']['mean'][primary]:.4f} by {margin:.0%}"
        )

    if SELECTED_DIR.exists():
        shutil.rmtree(SELECTED_DIR)
    shutil.copytree(MODELS_DIR / run_id / best["model"], SELECTED_DIR)
    shutil.copy(METRICS_DIR / f"{run_id}_{best['model']}_predictions.parquet",
                METRICS_DIR / "selected_backtest_predictions.parquet")
    return summary