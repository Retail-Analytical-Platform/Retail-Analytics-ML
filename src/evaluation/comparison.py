import json
from pathlib import Path

import pandas as pd

from src.pipeline_paths import RUNS_DIR

SAME_KEYS = ("target", "grain", "horizon", "data_range", "training_window", "n_scored_rows")
METRICS = ("wape", "mae", "rmse", "smape", "bias")


def load_records(runs_dir=RUNS_DIR, run_id=None):
    """With run_id: that run only. Without: the latest record per model."""
    records = [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*.json"))]
    if run_id:
        records = [r for r in records if r["run_id"] == run_id]
    latest = {}
    for r in sorted(records, key=lambda r: r["run_id"]):
        latest[r["model"]] = r
    return list(latest.values())


def check_comparable(records):
    if not records:
        raise ValueError("No run records found; run --stage backtest first")
    if len({r["validation"]["folds_hash"] for r in records}) != 1:
        raise ValueError("Models were not scored on the same folds; rerun the backtest for all of them")
    for key in SAME_KEYS:
        if len({json.dumps(r[key], sort_keys=True) for r in records}) != 1:
            raise ValueError(f"Models disagree on '{key}'; rerun the backtest for all of them")


def compare(records, primary, baseline="seasonal_naive"):
    check_comparable(records)
    others = [m for m in METRICS if m != primary]
    rows = []
    for r in records:
        mean, std = r["metrics"]["mean"], r["metrics"]["std"]
        row = {"model": r["model"], f"{primary}_mean": mean[primary], f"{primary}_std": std[primary]}
        for m in others:
            row[f"{m}_mean"], row[f"{m}_std"] = mean[m], std[m]
        row["fit_seconds"] = r["fit_seconds"]
        row["run_id"] = r["run_id"]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(f"{primary}_mean").reset_index(drop=True)
    base = table.loc[table["model"] == baseline, f"{primary}_mean"]
    if len(base) and base.iloc[0] != 0:
        table[f"{primary}_vs_{baseline}_pct"] = (table[f"{primary}_mean"] / base.iloc[0] - 1) * 100
    return table