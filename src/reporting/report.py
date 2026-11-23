import base64
import io
import json

import matplotlib.pyplot as plt
import pandas as pd

from src.pipeline_paths import ARTIFACT_DIR, METRICS_DIR

plt.switch_backend("Agg")


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _by_store(df, value_cols):
    """Sum categories so each store is one line."""
    return df.groupby(["store_key", "date"], as_index=False)[list(value_cols)].sum()


def _chart(df, value_cols, title):
    stores = sorted(df["store_key"].unique())
    fig, axes = plt.subplots(len(stores), 1, figsize=(10, 2.6 * len(stores)), sharex=True, squeeze=False)
    for ax, store in zip(axes[:, 0], stores):
        part = df[df["store_key"] == store].sort_values("date")
        for col, label in value_cols.items():
            ax.plot(part["date"], part[col], label=label)
        ax.set_title(f"Store {store} (all categories)", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return _png(fig)


def build_report(fcfg, artifact_dir=ARTIFACT_DIR, metrics_dir=METRICS_DIR):
    out_dir = artifact_dir / fcfg["output_dir"]
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    comparison = json.loads((metrics_dir / "comparison.json").read_text(encoding="utf-8"))
    table = pd.DataFrame(comparison["candidates"] + comparison["baselines"])
    forecast = pd.read_parquet(out_dir / "forecast.parquet")
    backtest = pd.read_parquet(metrics_dir / "selected_backtest_predictions.parquet")

    chart_backtest = _chart(
        _by_store(backtest, ["y_true", "y_pred"]), {"y_true": "Actual", "y_pred": "Forecast"},
        "Forecast against actual (validation folds)",
    )
    chart_future = _chart(
        _by_store(forecast, ["forecast"]), {"forecast": "Forecast"}, "Forecast for the next period",
    )

    table.to_csv(report_dir / "metrics_table.csv", index=False)
    forecast.to_csv(report_dir / "forecast.csv", index=False)
    gate = "passed" if comparison["gate_passed"] else "FAILED"
    html = f"""<!doctype html>
        <html><head><meta charset="utf-8"><title>Demand forecast report</title>
        <style>body{{font-family:sans-serif;max-width:960px;margin:2rem auto}}table{{border-collapse:collapse}}
        td,th{{border:1px solid #bbb;padding:4px 8px}}</style></head><body>
        <h1>Demand forecast report</h1>
        <p>Run {comparison['run_id']}. Selected model: <b>{comparison['selected']}</b>.
        Primary metric: {comparison['primary_metric']} (lower is better). Quality gate: {gate}.</p>
        <h2>Metrics</h2>{table.round(4).to_html(index=False)}
        <h2>Forecast against actual</h2><img src="data:image/png;base64,{chart_backtest}">
        <h2>Future forecast</h2><img src="data:image/png;base64,{chart_future}">
        <p>Charts sum all categories per store. Per store and category values are in forecast.csv.</p>
        </body></html>"""
    path = report_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
