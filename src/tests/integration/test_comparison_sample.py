import pandas as pd

import main
from src.evaluation.comparison import compare, load_records
from src.evaluation.model_runner import ALL_MODELS


def test_all_five_models_are_scored_on_identical_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)               # artifacts/ is relative, so this run is isolated
    assert main.main(["--stage", "features", "--sample"]) == 0
    assert main.main(["--stage", "backtest", "--sample"]) == 0

    records = load_records()
    assert sorted(r["model"] for r in records) == sorted(ALL_MODELS)
    keys = [
        pd.read_parquet(r["predictions_path"])[["store_key", "category", "sold_at", "fold"]]
        .sort_values(["store_key", "category", "sold_at"]).reset_index(drop=True)
        for r in records
    ]
    for k in keys[1:]:
        pd.testing.assert_frame_equal(keys[0], k)

    table = compare(records, "wape")
    assert len(table) == 5 and table["fit_seconds"].notna().all()
    assert (table.set_index("model").loc[list(ALL_MODELS[2:]), "fit_seconds"] > 0).all()