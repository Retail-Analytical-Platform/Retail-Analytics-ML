import pandas as pd
import pytest

from src.features.dataset import (
    build_forecast_dataset,
    load_forecast_dataset,
    save_forecast_dataset,
)


def _rows(*items):
    return pd.DataFrame(items, columns=["date", "store_id", "revenue"])


def test_gaps_filled_with_zero_inside_each_series_range():
    df = _rows(("2025-01-01", 1, 10.0), ("2025-01-03", 1, 5.0), ("2025-01-02", 2, 7.0))
    out = build_forecast_dataset(df, "date", ["store_id"], "revenue")
    assert len(out) == 5
    s1 = out[out.store_id == 1].set_index("date")["revenue"]
    assert s1.loc[pd.Timestamp("2025-01-02")] == 0
    # store 2 first sold on Jan 2, so no invented history before that
    assert out[out.store_id == 2]["date"].min() == pd.Timestamp("2025-01-02")
    assert not out.duplicated(["store_id", "date"]).any()
    assert out.equals(out.sort_values(["store_id", "date"]).reset_index(drop=True))


def test_duplicate_rows_are_summed_to_the_grain():
    df = _rows(("2025-01-01", 1, 10.0), ("2025-01-01", 1, 5.0))
    out = build_forecast_dataset(df, "date", ["store_id"], "revenue")
    assert out["revenue"].tolist() == [15.0]


def test_day_missing_everywhere_raises():
    df = _rows(("2025-01-01", 1, 1.0), ("2025-01-03", 1, 1.0))
    with pytest.raises(ValueError, match="No data"):
        build_forecast_dataset(df, "date", ["store_id"], "revenue")


def test_two_column_grain_and_parquet_round_trip(tmp_path):
    df = pd.DataFrame(
        [("2025-01-01", 1, "a", 1.0), ("2025-01-02", 1, "b", 2.0), ("2025-01-02", 1, "a", 3.0)],
        columns=["date", "store_id", "category", "revenue"],
    )
    out = build_forecast_dataset(df, "date", ["store_id", "category"], "revenue")
    assert len(out) == 3  # (1,a): Jan1,Jan2 ; (1,b): Jan2
    path = save_forecast_dataset(out, tmp_path / "x" / "ds.parquet")
    assert load_forecast_dataset(path).equals(out)