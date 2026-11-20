import pandas as pd
import pytest

from src.evaluation.backtest import (
    assert_chronological,
    chronological_split,
    fold_frames,
    load_folds,
    make_folds,
    save_folds,
    split_frame,
)

FIRST, LAST = "2025-01-01", "2025-04-10"      # 100 days
H = 7
BT = {"n_folds": 3, "step": 7, "window": "expanding", "min_train_days": 30}


def test_chronological_split_is_ordered_and_sized():
    dates = pd.date_range(FIRST, LAST)
    s = chronological_split(dates, validation_days=10, test_days=20)
    assert (s["test"][1] - s["test"][0]).days + 1 == 20
    assert (s["validation"][1] - s["validation"][0]).days + 1 == 10
    assert s["train"][1] + pd.Timedelta(days=1) == s["validation"][0]
    assert s["validation"][1] + pd.Timedelta(days=1) == s["test"][0]
    assert s["test"][1] == dates[-1]


def test_split_frame_has_no_overlap():
    df = pd.DataFrame({"date": pd.date_range(FIRST, LAST), "y": 1})
    parts = split_frame(df, "date", chronological_split(df["date"], 10, 20))
    assert parts["train"]["date"].max() < parts["validation"]["date"].min()
    assert parts["validation"]["date"].max() < parts["test"]["date"].min()
    assert sum(len(p) for p in parts.values()) == len(df)


def test_folds_are_ordered_and_keep_the_test_period_untouched():
    folds = make_folds(FIRST, LAST, H, BT, test_days=14)
    assert len(folds) == 3
    assert folds[-1]["valid_end"] == "2025-03-27"          # 14 days before the last date
    assert pd.Timestamp(folds[-1]["valid_end"]) < pd.Timestamp(LAST) - pd.Timedelta(days=13)
    for f in folds:
        assert pd.Timestamp(f["train_end"]) < pd.Timestamp(f["valid_start"])
        assert (pd.Timestamp(f["valid_end"]) - pd.Timestamp(f["valid_start"])).days + 1 == H
    for a, b in zip(folds, folds[1:]):
        assert pd.Timestamp(a["valid_end"]) < pd.Timestamp(b["valid_start"])


def test_sliding_window_limits_training_length():
    folds = make_folds(FIRST, LAST, H, {**BT, "window": "sliding", "train_window_days": 30}, 14)
    for f in folds:
        days = (pd.Timestamp(f["train_end"]) - pd.Timestamp(f["train_start"])).days + 1
        assert days == 30


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="min_train_days"):
        make_folds(FIRST, LAST, H, {**BT, "min_train_days": 200}, 14)
    with pytest.raises(ValueError, match="at least the horizon"):
        make_folds(FIRST, LAST, H, {**BT, "step": 3}, 14)
    with pytest.raises(ValueError, match="window"):
        make_folds(FIRST, LAST, H, {**BT, "window": "weird"}, 14)


def test_a_broken_fold_is_caught():
    folds = make_folds(FIRST, LAST, H, BT, 14)
    folds[1]["train_end"] = folds[1]["valid_start"]          # training now reaches validation
    with pytest.raises(ValueError, match="before validation"):
        assert_chronological(folds)


def test_fold_frames_never_mix_dates():
    df = pd.DataFrame({"date": pd.date_range(FIRST, LAST), "y": 1})
    for f in make_folds(FIRST, LAST, H, BT, 14):
        train, valid = fold_frames(df, "date", f)
        assert train["date"].max() < valid["date"].min()


def test_saved_folds_are_identical_across_runs(tmp_path):
    folds = make_folds(FIRST, LAST, H, BT, 14)
    p1 = save_folds(folds, tmp_path / "a" / "folds.json")
    p2 = save_folds(make_folds(FIRST, LAST, H, BT, 14), tmp_path / "b" / "folds.json")
    assert p1.read_bytes() == p2.read_bytes()
    assert load_folds(p1) == folds