import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]


def run_pipeline(artifact_dir):
    env = {**os.environ, "RETAIL_ARTIFACT_DIR": str(artifact_dir), "RUN_ID": "e2e"}
    env.pop("DATABASE_ROLE_URL", None)
    return subprocess.run(
        [sys.executable, "main.py", "--stage", "all", "--sample"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    a, b = tmp_path_factory.mktemp("a"), tmp_path_factory.mktemp("b")
    return a, b, run_pipeline(a), run_pipeline(b)


def test_all_stages_exit_zero_and_write_artifacts(runs):
    a, _, first, _ = runs
    assert first.returncode == 0, first.stderr
    for relative in (
        "data/forecast_dataset.parquet", "data/features.parquet", "folds/folds.json",
        "metrics/comparison.json", "models/selected/meta.json",
        "forecasts/forecast.csv", "forecasts/report/report.html",
    ):
        assert (a / relative).exists(), relative


def test_runs_are_reproducible(runs):
    a, b, first, second = runs
    assert first.returncode == 0 and second.returncode == 0
    pd.testing.assert_frame_equal(
        pd.read_parquet(a / "data/features.parquet"), pd.read_parquet(b / "data/features.parquet")
    )
    assert (a / "folds/folds.json").read_bytes() == (b / "folds/folds.json").read_bytes()
    pd.testing.assert_frame_equal(
        pd.read_parquet(a / "forecasts/forecast.parquet"), pd.read_parquet(b / "forecasts/forecast.parquet")
    )
    ca = json.loads((a / "metrics/comparison.json").read_text())
    cb = json.loads((b / "metrics/comparison.json").read_text())
    assert ca["selected"] == cb["selected"]
    for x, y in zip(ca["candidates"], cb["candidates"]):
        assert x["primary_mean"] == pytest.approx(y["primary_mean"], rel=1e-6)