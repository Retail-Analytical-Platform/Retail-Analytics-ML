import pandas as pd

from src.evaluation.splits import split_frame


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
