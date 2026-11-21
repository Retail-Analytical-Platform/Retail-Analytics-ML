import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.backtest import fold_frames, make_folds, save_folds
from src.evaluation.metrics import compute_metrics, grouped_metrics
from src.features.categorical import fit_categories, prepare_model_frame
from src.features.dataset import load_forecast_dataset
from src.features.lags import drop_insufficient_history
from src.features.pipeline import build_features
from src.models.registry import build_model
from src.models.training import ML_MODELS, split_for_early_stopping
from src.pipeline_paths import BACKTEST_DIR, RUNS_DIR

BASELINES = ("naive", "seasonal_naive")
ALL_MODELS = (*BASELINES, *ML_MODELS)


def load_inputs(args):
    fcfg = load_config("forecast_config")
    if getattr(args, "target", None):
        fcfg["target_column"] = args.target
    if getattr(args, "horizon", None):
        fcfg["horizon"] = args.horizon
    feat_cfg = load_config("features_config")
    model_cfg = load_config("models_config")
    try:
        ds = load_forecast_dataset()
    except FileNotFoundError:
        raise FileNotFoundError("Forecast dataset not found; run --stage features first") from None
    feats = build_features(ds, fcfg["date_column"], fcfg["grain"], fcfg["target_column"],
                           feat_cfg, fcfg["horizon"])
    return fcfg, feat_cfg, model_cfg, ds, feats


def get_folds(ds, fcfg):
    d = fcfg["date_column"]
    folds = make_folds(ds[d].min(), ds[d].max(), fcfg["horizon"], fcfg["backtest"],
                       fcfg["split"]["test_days"])
    save_folds(folds, fcfg["backtest"]["folds_path"])
    digest = hashlib.sha256(json.dumps(folds, sort_keys=True).encode()).hexdigest()[:12]
    return folds, digest


def _pred_frame(df, grain, date_col, target, forecast, fold):
    out = df[[*grain, date_col]].copy()
    out["actual"] = df[target].to_numpy()
    out["forecast"] = np.asarray(forecast, dtype=float)
    out["fold"] = fold["fold"]
    out["horizon_step"] = (df[date_col] - pd.Timestamp(fold["valid_start"])).dt.days.to_numpy() + 1
    return out.reset_index(drop=True)


def baseline_predictions(name, ds, folds, date_col, grain, target) -> tuple[pd.DataFrame, float, dict]:
    """naive: last training value. seasonal_naive: same weekday from the last 7 training days.
    A series with no history forecasts 0. If you want your own #26 code here, keep the return shape."""
    parts = []
    for fold in folds:
        train, valid = fold_frames(ds, date_col, fold)
        train_end = pd.Timestamp(fold["train_end"])
        if name == "naive":
            last = (train.sort_values(date_col).groupby(grain, sort=False).tail(1)[[*grain, target]]
                    .rename(columns={target: "forecast"}))
            merged = valid.merge(last, on=grain, how="left")
        elif name == "seasonal_naive":
            steps = (valid[date_col] - train_end).dt.days.to_numpy()
            source = valid[date_col] - pd.to_timedelta(7 * np.ceil(steps / 7).astype(int), unit="D")
            lookup = train[[*grain, date_col, target]].rename(
                columns={date_col: "_source", target: "forecast"})
            merged = valid.assign(_source=source.to_numpy()).merge(
                lookup, on=[*grain, "_source"], how="left")
        else:
            raise ValueError(f"Unknown baseline '{name}'")
        parts.append(_pred_frame(merged, grain, date_col, target,
                                 merged["forecast"].fillna(0.0), fold))
    return pd.concat(parts, ignore_index=True), 0.0, {}


def ml_predictions(name, feats, folds, fcfg, feat_cfg, model_cfg) -> tuple[pd.DataFrame, float, dict]:
    date_col, grain, target = fcfg["date_column"], fcfg["grain"], fcfg["target_column"]
    cats = feat_cfg["categorical_columns"]
    categories = fit_categories(feats, cats)          # same category codes in every fold
    parts, seconds, params = [], 0.0, {}
    for fold in folds:
        train, valid = fold_frames(feats, date_col, fold)
        train = drop_insufficient_history(train, target)
        fit_part, es_part = split_for_early_stopping(train, date_col, fcfg["horizon"])
        if fit_part.empty or es_part.empty:
            raise ValueError(f"Fold {fold['fold']}: not enough training rows")
        model = build_model(name, model_cfg, categorical_columns=cats)
        Xf, _ = prepare_model_frame(fit_part, name, cats, date_col, target, categories)
        Xe, _ = prepare_model_frame(es_part, name, cats, date_col, target, categories)
        Xv, _ = prepare_model_frame(valid, name, cats, date_col, target, categories)
        start = time.perf_counter()
        model.fit(Xf, fit_part[target], Xe, es_part[target])
        seconds += time.perf_counter() - start
        parts.append(_pred_frame(valid, grain, date_col, target, model.predict(Xv), fold))
        params = {k: v for k, v in model.params.items() if k != "categorical_columns"}
    return pd.concat(parts, ignore_index=True), seconds, params


def git_commit():
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _plain(o):
    return o.item() if hasattr(o, "item") else str(o)


def run_model(name, ds, feats, folds, fhash, fcfg, feat_cfg, model_cfg, run_id):
    """Backtest one model, save its fold predictions and its run record, return the record."""
    date_col, grain, target = fcfg["date_column"], fcfg["grain"], fcfg["target_column"]
    if name in BASELINES:
        preds, seconds, params = baseline_predictions(name, ds, folds, date_col, grain, target)
    elif name in ML_MODELS:
        preds, seconds, params = ml_predictions(name, feats, folds, fcfg, feat_cfg, model_cfg)
    else:
        raise ValueError(f"Unknown model '{name}'; expected one of {ALL_MODELS}")

    overall = compute_metrics(preds["actual"], preds["forecast"])
    per_fold = grouped_metrics(preds, ["fold"])
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = BACKTEST_DIR / f"{run_id}_{name}.parquet"
    preds.to_parquet(pred_path, index=False)

    bt = fcfg["backtest"]
    record = {
        "run_id": run_id,
        "model": name,
        "data_range": [ds[date_col].min().date().isoformat(), ds[date_col].max().date().isoformat()],
        "target": target,
        "grain": grain,
        "horizon": fcfg["horizon"],
        "training_window": {"window": bt.get("window", "expanding"),
                            "train_window_days": bt.get("train_window_days"),
                            "min_train_days": bt["min_train_days"]},
        "validation": {"method": "rolling_origin", "n_folds": len(folds), "folds_hash": fhash},
        "n_scored_rows": int(len(preds)),
        "params": params,
        "metrics": {
            "overall": overall,
            "per_fold": per_fold.to_dict("records"),
            "mean": {c: float(per_fold[c].mean()) for c in overall},
            "std": {c: float(per_fold[c].std()) for c in overall},   # sample std across folds
        },
        "fit_seconds": round(seconds, 3),
        "git_commit": git_commit(),
        "predictions_path": str(pred_path),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{run_id}_{name}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=_plain) + "\n")
    return record