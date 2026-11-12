import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import SAMPLE_CONFIG, load_all
from src.data.validation import validate_tables

CFG = {"min_fact_rows": 1, "expected_date_start": "2024-01-01", "expected_date_end": "2024-01-03"}


@pytest.fixture
def tables():
    """Valid data: the #15 fixture, freshly read for each test."""
    return load_all(sample=True)


def report(tables, config=SAMPLE_CONFIG):
    return "\n".join(validate_tables(tables, config))


def test_reports_null_duplicate_and_negative_together():
    fact = pd.DataFrame({
        "sale_id": [1, 1, 2], "date_key": [1, 2, None], "product_key": [1, 1, 1],
        "customer_key": [1, 1, 1], "store_key": [1, 1, 1],
        "sold_at": pd.to_datetime(["2024-01-01"] * 3), "quantity": [1, -2, 3],
        "unit_price": [1.0] * 3, "discount_pct": [0.0] * 3,
        "revenue": [1.0, 2.0, 3.0], "category": ["a"] * 3,
    })
    text = "\n".join(validate_tables({"fact_sales": fact}, CFG))
    assert "duplicate" in text
    assert "null" in text
    assert "negative" in text


def test_valid_data_has_no_failures(tables):
    assert validate_tables(tables, SAMPLE_CONFIG) == []


def test_missing_table(tables):
    del tables["dim_store"]
    assert "dim_store: table was not loaded" in report(tables)


def test_missing_column(tables):
    tables["dim_store"] = tables["dim_store"].drop(columns=["region"])
    assert "dim_store.region: column is missing" in report(tables)


def test_wrong_type(tables):
    tables["fact_sales"]["quantity"] = "many"
    assert "fact_sales.quantity: expected numeric" in report(tables)


def test_null_in_required_column(tables):
    fact = tables["fact_sales"]
    fact["date_key"] = fact["date_key"].astype(float)
    fact.loc[0, "date_key"] = np.nan
    assert "fact_sales.date_key: 1 null" in report(tables)


def test_null_customer_is_allowed(tables):
    # guest sales have no customer; the fixture contains some on purpose
    assert tables["fact_sales"]["customer_key"].isna().any()
    assert "customer_key" not in report(tables)


def test_duplicate_key(tables):
    fact = tables["fact_sales"]
    fact.loc[1, "sale_id"] = fact.loc[0, "sale_id"]
    assert "fact_sales.sale_id: 1 duplicate" in report(tables)


def test_negative_quantity(tables):
    tables["fact_sales"].loc[0, "quantity"] = -1
    assert "fact_sales.quantity: 1 negative" in report(tables)


def test_negative_revenue(tables):
    tables["fact_sales"].loc[0, "revenue"] = -1.0
    assert "fact_sales.revenue: 1 negative" in report(tables)


def test_too_few_fact_rows(tables):
    config = {**SAMPLE_CONFIG, "min_fact_rows": len(tables["fact_sales"]) + 1}
    assert "expected at least" in report(tables, config)


def test_missing_day_in_dim_date(tables):
    tables["dim_date"] = tables["dim_date"].drop(index=10)     # a gap inside the range
    assert "1 missing day" in report(tables)


def test_date_range_mismatch(tables):
    tables["dim_date"] = tables["dim_date"].iloc[:-1]          # last day gone, no gap
    assert "dim_date: range" in report(tables)
