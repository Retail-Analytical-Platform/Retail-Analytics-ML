import json
import os
import subprocess
from datetime import datetime, timezone

import pandas as pd

from src.pipeline_paths import METRICS_DIR


def git_commit():
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _plain(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value)}")


def record_run(run_id, model, params, metrics, fcfg, data_range, directory=METRICS_DIR):
    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_range": list(data_range),
        "target": fcfg["target_column"],
        "grain": list(fcfg["grain"]),
        "horizon": fcfg["horizon"],
        "validation": {
            "method": "rolling_origin", "window_type": fcfg["window_type"],
            "n_folds": fcfg["n_folds"], "step_days": fcfg["fold_step_days"],
        },
        "model": model,
        "params": params,
        "metrics": metrics,
        "git_commit": git_commit(),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}_{model}.json"
    path.write_text(json.dumps(record, indent=2, default=_plain) + "\n", encoding="utf-8")
    return path


def load_records(directory=METRICS_DIR, run_id=None):
    records = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "run_id" in record and "model" in record and (run_id is None or record["run_id"] == run_id):
            records.append(record)
    return records


def load_runs(directory=METRICS_DIR, run_id=None):
    return pd.json_normalize(load_records(directory, run_id))