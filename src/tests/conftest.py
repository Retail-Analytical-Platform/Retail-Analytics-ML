import pandas as pd
import pytest


@pytest.fixture
def small_sales():
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "date": dates,
        "store_id": 1,
        "quantity": range(30),
        "revenue": [x * 10.0 for x in range(30)],
    })