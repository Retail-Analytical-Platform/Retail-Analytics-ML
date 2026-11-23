import numpy as np
import pandas as pd
import shap
from catboost import Pool

from src.pipeline_paths import METRICS_DIR


def native_importance(model):
    inner = model.model
    if hasattr(inner, "feature_importances_"):
        values = inner.feature_importances_
    else:                                              # a LightGBM Booster loaded from disk
        values = inner.feature_importance(importance_type="gain")
    return pd.Series(values, index=model.feature_columns, name="importance").sort_values(ascending=False)


def shap_importance(model, X):
    if model.name == "catboost":
        values = model.model.get_feature_importance(Pool(X, cat_features=list(model.categorical)), type="ShapValues")[:, :-1]
    else:
        values = shap.TreeExplainer(model.model).shap_values(X)
    return pd.Series(np.abs(values).mean(axis=0), index=model.feature_columns, name="mean_abs_shap").sort_values(ascending=False)


def error_breakdown(preds, by):
    d = preds.assign(error=preds["y_pred"] - preds["y_true"])
    d["abs_error"] = d["error"].abs()
    out = d.groupby(by).agg(
        rows=("error", "size"), mae=("abs_error", "mean"), bias=("error", "mean"),
        actual_sum=("y_true", "sum"), abs_error_sum=("abs_error", "sum"),
    )
    out["wape"] = out["abs_error_sum"] / out["actual_sum"].where(out["actual_sum"] != 0)
    return out.drop(columns=["actual_sum", "abs_error_sum"])


def worst_days(preds, n=10):
    d = preds.assign(abs_error=(preds["y_pred"] - preds["y_true"]).abs())
    return d.sort_values("abs_error", ascending=False).head(n)


def write_analysis(preds, model, X, out_dir=METRICS_DIR / "analysis"):
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = preds.assign(weekday=preds["date"].dt.day_of_week)
    native_importance(model).to_csv(out_dir / "importance.csv")
    shap_importance(model, X).to_csv(out_dir / "shap.csv")
    for key in ("store_key", "category", "weekday", "horizon_step", "fold"):
        error_breakdown(preds, [key]).to_csv(out_dir / f"errors_by_{key}.csv")
    worst_days(preds).to_csv(out_dir / "worst_days.csv", index=False)
    return out_dir