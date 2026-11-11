import pandas as pd
from pandas.api import types as pt

KINDS = {
    "numeric": lambda s: pt.is_numeric_dtype(s) and not pt.is_bool_dtype(s),
    "text": pt.is_string_dtype,
    "datetime": pt.is_datetime64_any_dtype,
    "bool": pt.is_bool_dtype,
}

CONTRACT = {
    "dim_customer": {
        "key": "customer_key",
        "columns": {"customer_key": "numeric", "customer_name": "text", "email": "text",
                    "city": "text", "country": "text", "segment": "text"},
        "not_null": ["customer_key"],
    },
    "dim_date": {
        "key": "date_key",
        "columns": {"date_key": "numeric", "full_date": "datetime", "day_of_week": "text",
                    "month_name": "text", "quarter": "text", "is_weekend": "bool"},
        "not_null": ["date_key", "full_date"],
    },
    "dim_product": {
        "key": "product_key",
        "columns": {"product_key": "numeric", "product_name": "text", "category": "text",
                    "subcategory": "text", "unit_cost": "numeric", "list_price": "numeric"},
        "not_null": ["product_key", "category"],
    },
    "dim_store": {
        "key": "store_key",
        "columns": {"store_key": "numeric", "store_name": "text", "region": "text",
                    "store_type": "text"},
        "not_null": ["store_key"],
    },
    "fact_sales": {
        "key": "sale_id",
        "columns": {"sale_id": "numeric", "date_key": "numeric", "product_key": "numeric",
                    "customer_key": "numeric", "store_key": "numeric", "sold_at": "datetime",
                    "quantity": "numeric", "unit_price": "numeric", "discount_pct": "numeric",
                    "revenue": "numeric", "category": "text"},
        "not_null": ["sale_id", "date_key", "product_key", "store_key", "sold_at",
                     "quantity", "unit_price", "revenue"],
    },
}


def validate_tables(tables: dict[str, pd.DataFrame], config: dict) -> list[str]:
    """Return every failure found; an empty list means the data is valid."""
    failures = []
    for name, spec in CONTRACT.items():
        df = tables.get(name)
        if df is None:
            failures.append(f"{name}: table was not loaded")
            continue
        for col, kind in spec["columns"].items():
            if col not in df.columns:
                failures.append(f"{name}.{col}: column is missing")
            elif not KINDS[kind](df[col]):
                failures.append(f"{name}.{col}: expected {kind}, got {df[col].dtype}")
        for col in spec["not_null"]:
            if col in df.columns and (n := int(df[col].isna().sum())):
                failures.append(f"{name}.{col}: {n} null value(s)")
        key = spec["key"]
        if key in df.columns and (n := int(df[key].duplicated().sum())):
            failures.append(f"{name}.{key}: {n} duplicate key(s)")

    failures += _check_fact(tables.get("fact_sales"), config)
    failures += _check_dates(tables.get("dim_date"), config)
    return failures


def _check_fact(fact, config):
    if fact is None:
        return []
    failures = []
    for col in ("quantity", "revenue"):
        if col in fact.columns and pt.is_numeric_dtype(fact[col]):
            if n := int((fact[col] < 0).sum()):
                failures.append(f"fact_sales.{col}: {n} negative value(s)")
    if len(fact) < config["min_fact_rows"]:
        failures.append(f"fact_sales: {len(fact)} rows, expected at least {config['min_fact_rows']}")
    return failures


def _check_dates(dim_date, config):
    if dim_date is None or "full_date" not in dim_date.columns:
        return []
    if not pt.is_datetime64_any_dtype(dim_date["full_date"]):
        return []  # the type failure is already reported
    dates = pd.DatetimeIndex(dim_date["full_date"].dropna().dt.normalize())
    if dates.empty:
        return ["dim_date.full_date: no dates"]
    failures = []
    missing = pd.date_range(dates.min(), dates.max(), freq="D").difference(dates)
    if len(missing):
        failures.append(f"dim_date.full_date: {len(missing)} missing day(s), first {missing[0].date()}")
    start, end = pd.Timestamp(config["expected_date_start"]), pd.Timestamp(config["expected_date_end"])
    if dates.min() != start or dates.max() != end:
        failures.append(
            f"dim_date: range {dates.min().date()} to {dates.max().date()}, "
            f"expected {start.date()} to {end.date()}"
        )
    return failures
