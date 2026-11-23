import json
from pathlib import Path

import pandas as pd

from src.evaluation.splits import split_frame


def chronological_split(dates, validation_days, test_days):
    """Train, validation and test periods in time order. Returns (start, end) pairs."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).unique().sort_values()
    first, last = dates[0], dates[-1]
    test_start = last - pd.Timedelta(days=test_days - 1)
    valid_end = test_start - pd.Timedelta(days=1)
    valid_start = valid_end - pd.Timedelta(days=validation_days - 1)
    train_end = valid_start - pd.Timedelta(days=1)
    if train_end < first:
        raise ValueError("Not enough history for the requested validation and test periods")
    return {
        "train": (first, train_end),
        "validation": (valid_start, valid_end),
        "test": (test_start, last),
    }

def assert_chronological(folds):
    prev_valid_end = None
    for f in folds:
        ts, te = pd.Timestamp(f["train_start"]), pd.Timestamp(f["train_end"])
        vs, ve = pd.Timestamp(f["valid_start"]), pd.Timestamp(f["valid_end"])
        if not (ts <= te < vs <= ve):
            raise ValueError(f"Fold {f['fold']}: training must end before validation starts")
        if prev_valid_end is not None and vs <= prev_valid_end:
            raise ValueError(f"Fold {f['fold']}: validation window overlaps the previous fold")
        prev_valid_end = ve


def make_folds(first_date, last_date, horizon, bt_cfg, test_days):
    """Rolling-origin folds. Each validates on the `horizon` days after its training end."""
    n, step = bt_cfg["n_folds"], bt_cfg["step"]
    window = bt_cfg.get("window", "expanding")
    if window not in ("expanding", "sliding"):
        raise ValueError(f"backtest.window must be 'expanding' or 'sliding', got {window!r}")
    if step < horizon:
        raise ValueError(f"backtest.step ({step}) must be at least the horizon ({horizon})")

    first, last = pd.Timestamp(first_date), pd.Timestamp(last_date)
    region_end = last - pd.Timedelta(days=test_days)   # the test period stays untouched
    day = pd.Timedelta(days=1)

    folds = []
    for k in range(n):
        valid_end = region_end - pd.Timedelta(days=(n - 1 - k) * step)
        valid_start = valid_end - pd.Timedelta(days=horizon - 1)
        train_end = valid_start - day
        if window == "sliding":
            train_start = max(first, train_end - pd.Timedelta(days=bt_cfg["train_window_days"] - 1))
        else:
            train_start = first
        if (train_end - train_start).days + 1 < bt_cfg["min_train_days"]:
            raise ValueError(f"Fold {k}: fewer than min_train_days of training data")
        folds.append({
            "fold": k,
            "train_start": train_start.date().isoformat(),
            "train_end": train_end.date().isoformat(),
            "valid_start": valid_start.date().isoformat(),
            "valid_end": valid_end.date().isoformat(),
        })
    assert_chronological(folds)
    return folds


def fold_frames(df, date_col, fold):
    d = pd.to_datetime(df[date_col])
    train = df[(d >= pd.Timestamp(fold["train_start"])) & (d <= pd.Timestamp(fold["train_end"]))]
    valid = df[(d >= pd.Timestamp(fold["valid_start"])) & (d <= pd.Timestamp(fold["valid_end"]))]
    return train, valid


def save_folds(folds, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(folds, indent=2, sort_keys=True) + "\n")
    return path


def load_folds(path):
    return json.loads(Path(path).read_text())



def run_backtest(dataset, folds, predict_fn, grain, target):
    """predict_fn(train, val) -> DataFrame with the grain columns, date and y_pred."""
    grain = list(grain)
    parts = []
    for fold in folds:
        train, val = split_frame(dataset, fold)
        pred = predict_fn(train, val)
        merged = (
            val[[*grain, "date", target]]
            .rename(columns={target: "y_true"})
            .merge(pred[[*grain, "date", "y_pred"]], on=[*grain, "date"], how="left", validate="one_to_one")
        )
        if merged["y_pred"].isna().any():
            raise ValueError(f"Fold {fold['fold']}: missing predictions")
        merged["fold"] = fold["fold"]
        merged["horizon_step"] = (merged["date"] - pd.Timestamp(fold["val_start"])).dt.days + 1
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)