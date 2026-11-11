import pandas as pd

from src.data.validation import validate_tables

CFG = {"min_fact_rows": 1, "expected_date_start": "2024-01-01", "expected_date_end": "2025-12-31"}


def test_reports_null_duplicate_and_negative_together():
    fact = pd.DataFrame({
        "sale_id": [1, 1, 2], "date_key": [1, 2, None], "product_key": [1, 1, 1],
        "customer_key": [1, 1, 1], "store_key": [1, 1, 1],
        "sold_at": pd.to_datetime(["2024-01-01"] * 3), "quantity": [1, -2, 3],
        "unit_price": [1.0] * 3, "discount_pct": [0.0] * 3,
        "revenue": [1.0, 2.0, 3.0], "category": ["a"] * 3,
    })
    report = "\n".join(validate_tables({"fact_sales": fact}, CFG))
    assert "duplicate" in report
    assert "null" in report
    assert "negative" in report