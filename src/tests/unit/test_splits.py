import pandas as pd
import pytest

from src.evaluation.splits import (
    chronological_split,
    load_folds,
    make_folds,
    save_folds,
    split_frame,
    validate_folds,
)

CFG = {
    "horizon": 28, "fold_step_days": 28, "n_folds": 4, "val_days": 28, "test_days": 28,
    "window_type": "expanding", "training_window_days": 180,
}
MIN, MAX = "2024-01-01", "2025-12-31"


def test_expected_fold_dates():
    folds = make_folds(MIN, MAX, CFG)
    assert folds[-1] == {
        "fold": 3, "train_start": "2024-01-01", "train_end": "2025-11-05",
        "val_start": "2025-11-06", "val_end": "2025-12-03",
    }
    assert folds[0]["val_start"] == "2025-08-14"


def test_training_always_ends_before_validation_starts():
    for fold in make_folds(MIN, MAX, CFG):
        assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["val_start"])


def test_validation_windows_do_not_overlap_when_step_equals_horizon():
    folds = make_folds(MIN, MAX, CFG)
    for earlier, later in zip(folds, folds[1:]):
        assert pd.Timestamp(earlier["val_end"]) < pd.Timestamp(later["val_start"])


def test_folds_stay_out_of_the_test_period():
    test_start = pd.Timestamp(chronological_split(MIN, MAX, CFG)["test_start"])
    assert pd.Timestamp(make_folds(MIN, MAX, CFG)[-1]["val_end"]) < test_start


def test_chronological_split_is_ordered_and_complete():
    s = chronological_split(MIN, MAX, CFG)
    assert s["train_end"] < s["val_start"] <= s["val_end"] < s["test_start"] <= s["test_end"] == MAX


def test_sliding_window_limits_training_length():
    folds = make_folds(MIN, MAX, {**CFG, "window_type": "sliding"})
    last = folds[-1]
    assert (pd.Timestamp(last["train_end"]) - pd.Timestamp(last["train_start"])).days == 179


def test_too_many_folds_is_rejected():
    with pytest.raises(ValueError):
        make_folds(MIN, MAX, {**CFG, "n_folds": 50})


def test_validate_folds_catches_broken_date_order():
    folds = make_folds(MIN, MAX, CFG)
    folds[1]["train_end"] = folds[1]["val_start"]
    with pytest.raises(ValueError, match="training ends"):
        validate_folds(folds)


def test_split_frame_has_no_overlap():
    df = pd.DataFrame({"date": pd.date_range(MIN, MAX), "store_key": 1})
    for fold in make_folds(MIN, MAX, CFG):
        train, val = split_frame(df, fold)
        assert train["date"].max() < val["date"].min()


def test_fold_file_is_deterministic(tmp_path):
    save_folds(make_folds(MIN, MAX, CFG), tmp_path / "a.json")
    save_folds(make_folds(MIN, MAX, CFG), tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert load_folds(tmp_path / "a.json") == make_folds(MIN, MAX, CFG)
