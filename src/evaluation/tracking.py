import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

RUNS_DIR = "artifacts/metrics/runs"


def git_commit():
    sha = os.environ.get("GITHUB_SHA")           # set by GitHub Actions
    if sha:
        return sha
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if r.returncode == 0:
            return r.stdout.strip()
    except OSError:
        pass
    return "unknown"


def _clean(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None                               # keep the JSON valid
    return value


def write_run_record(run_id, model, params, summary, data_range, target, grain, horizon,
                     validation, out_dir=RUNS_DIR):
    record = {
        "run_id": run_id,
        "model": model,
        "parameters": params,
        "target": target,
        "grain": list(grain),
        "horizon": horizon,
        "data_range": data_range,
        "validation": validation,
        "metrics": {
            "mean": summary["mean"].to_dict(),
            "std": summary["std"].to_dict(),
            "per_fold": summary["per_fold"].to_dict(orient="records"),
        },
        "git_commit": git_commit(),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}_{model}.json"
    path.write_text(json.dumps(_clean(record), indent=2, sort_keys=True) + "\n")
    return path


def load_run_records(directory=RUNS_DIR):
    """All run records as one table, one row per record."""
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        r = json.loads(path.read_text())
        row = {k: r[k] for k in ("run_id", "model", "target", "horizon", "git_commit")}
        row["grain"] = ",".join(r["grain"])
        row["date_start"], row["date_end"] = r["data_range"]
        row.update({f"validation_{k}": v for k, v in r["validation"].items()})
        for name, value in r["metrics"]["mean"].items():
            row[f"{name}_mean"] = value
        for name, value in r["metrics"]["std"].items():
            row[f"{name}_std"] = value
        rows.append(row)
    return pd.DataFrame(rows)