import json

import pandas as pd

from src.pipeline_paths import FOLDS_PATH

DAY = pd.Timedelta(days=1)


def _iso(ts):
    return pd.Timestamp(ts).date().isoformat()


def chronological_split(min_date, max_date, config):
    min_date, max_date = pd.Timestamp(min_date), pd.Timestamp(max_date)
    test_start = max_date - pd.Timedelta(days=config["test_days"] - 1)
    val_start = test_start - pd.Timedelta(days=config["val_days"])
    split = {
        "train_start": min_date, "train_end": val_start - DAY,
        "val_start": val_start, "val_end": test_start - DAY,
        "test_start": test_start, "test_end": max_date,
    }
    if split["train_end"] < split["train_start"]:
        raise ValueError("Not enough history for train, validation and test periods")
    return {key: _iso(value) for key, value in split.items()}


def make_folds(min_date, max_date, config):
    min_date, max_date = pd.Timestamp(min_date), pd.Timestamp(max_date)
    if config["window_type"] not in ("expanding", "sliding"):
        raise ValueError("window_type must be 'expanding' or 'sliding'")
    n, step, horizon = config["n_folds"], config["fold_step_days"], config["horizon"]
    backtest_end = max_date - pd.Timedelta(days=config["test_days"])

    folds = []
    for i in range(n):
        val_end = backtest_end - pd.Timedelta(days=(n - 1 - i) * step)
        val_start = val_end - pd.Timedelta(days=horizon - 1)
        train_end = val_start - DAY
        if config["window_type"] == "expanding":
            train_start = min_date
        else:
            train_start = max(min_date, train_end - pd.Timedelta(days=config["training_window_days"] - 1))
        if train_end < train_start:
            raise ValueError(f"Fold {i} has no training data; lower n_folds or fold_step_days")
        folds.append({
            "fold": i, "train_start": _iso(train_start), "train_end": _iso(train_end),
            "val_start": _iso(val_start), "val_end": _iso(val_end),
        })
    validate_folds(folds, backtest_end)
    return folds


def validate_folds(folds, backtest_end=None):
    previous = None
    for fold in folds:
        train_end, val_start = pd.Timestamp(fold["train_end"]), pd.Timestamp(fold["val_start"])
        if not train_end < val_start:
            raise ValueError(f"Fold {fold['fold']}: training ends {train_end.date()}, validation starts {val_start.date()}")
        if backtest_end is not None and pd.Timestamp(fold["val_end"]) > pd.Timestamp(backtest_end):
            raise ValueError(f"Fold {fold['fold']}: validation reaches into the test period")
        if previous is not None and val_start <= previous:
            raise ValueError(f"Fold {fold['fold']}: folds are not in time order")
        previous = val_start


def save_folds(folds, path=FOLDS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(folds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_folds(path=FOLDS_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def split_frame(df, fold):
    d = df["date"]
    train = df[(d >= pd.Timestamp(fold["train_start"])) & (d <= pd.Timestamp(fold["train_end"]))]
    val = df[(d >= pd.Timestamp(fold["val_start"])) & (d <= pd.Timestamp(fold["val_end"]))]
    if len(train) and len(val) and train["date"].max() >= val["date"].min():
        raise ValueError(f"Fold {fold['fold']}: training and validation dates overlap")
    return train, val