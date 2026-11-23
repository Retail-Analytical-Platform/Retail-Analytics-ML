import numpy as np
import pandas as pd

from src.features.dataset import build_dataset
from src.features.pipeline import make_features, model_view
from src.models.base import load_model
from src.pipeline_paths import ARTIFACT_DIR, SELECTED_DIR


def generate_forecast(tables, fcfg, xcfg, model_dir=SELECTED_DIR):
    model = load_model(model_dir)
    target, grain, horizon = fcfg["target_column"], list(fcfg["grain"]), fcfg["horizon"]

    dataset = build_dataset(tables, fcfg)
    last = dataset["date"].max()
    future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = dataset[grain].drop_duplicates().merge(pd.DataFrame({"date": future_dates}), how="cross")
    future[target] = np.nan
    future["is_zero_filled"] = False

    table = make_features(pd.concat([dataset, future], ignore_index=True), fcfg, xcfg, drop_incomplete=False)
    rows = table[table["date"] > last]
    view = model_view(table, model.feature_columns, xcfg["categorical_columns"], model.name)
    prediction = np.clip(model.predict(view.loc[rows.index, model.feature_columns]), 0, None)

    out = rows[[*grain, "date"]].copy()
    out["forecast"] = prediction
    expected = horizon * len(dataset[grain].drop_duplicates())
    if len(out) != expected:
        raise ValueError(f"Expected {expected} forecast rows, got {len(out)}")
    return out.reset_index(drop=True)


def save_forecast(df, fcfg):
    out_dir = ARTIFACT_DIR / fcfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "forecast.csv", index=False)
    df.to_parquet(out_dir / "forecast.parquet", index=False)
    return out_dir