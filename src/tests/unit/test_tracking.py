from src.tracking import load_runs, record_run

FCFG = {"target_column": "quantity", "grain": ["store_key"], "horizon": 28,
        "window_type": "expanding", "n_folds": 4, "fold_step_days": 28}


def test_two_runs_give_two_files_and_two_rows(tmp_path):
    for run_id in ("run-a", "run-b"):
        record_run(run_id, "catboost", {"depth": 6}, {"mean": {"wape": 0.2}}, FCFG, ["2024-01-01", "2025-12-31"], tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert len(load_runs(tmp_path)) == 2
    assert len(load_runs(tmp_path, run_id="run-a")) == 1