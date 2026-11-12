import logging
import os
import time
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql

logger = logging.getLogger(__name__)

SCHEMA = os.environ.get("SCHEMA_NAME", "clean")
TABLES = ("dim_customer", "dim_date", "dim_product", "dim_store", "fact_sales")
RETRIES = 3
WAIT_SECONDS = 5

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
SAMPLE_DATE_COLUMNS = {"dim_date": ["full_date"], "fact_sales": ["sold_at"]}
SAMPLE_CONFIG = {
    "expected_date_start": "2024-01-01",
    "expected_date_end": "2025-12-31",
    "min_fact_rows": 1000,
}


def _connect() -> psycopg.Connection:
    url = os.environ.get("DATABASE_ROLE_URL")
    if not url:
        raise RuntimeError("database url is not set")

    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            return psycopg.connect(url, sslmode="require", connect_timeout=10)
        except psycopg.OperationalError as exc:
            last_error = exc
            logger.warning("Database connection failed (attempt %d/%d)", attempt, RETRIES)
            if attempt < RETRIES:
                time.sleep(WAIT_SECONDS)
    raise ConnectionError(
        f"Could not reach the database after {RETRIES} attempts.") from last_error


def _load_once(name: str) -> pd.DataFrame:
    if name not in TABLES:
        raise ValueError(f"Unknown table '{name}'")
    query = sql.SQL("select * from {}.{}").format(sql.Identifier(SCHEMA), sql.Identifier(name))
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query)
        if cur.description is None:
            raise RuntimeError(f"Query for table '{name}' returned no columns")
        columns = [col.name for col in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=columns)
    # numeric columns arrive as Decimal objects; models need floats
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, Decimal)).any():
            df[col] = df[col].astype(float)

    # timestamp columns can arrive as object dtype; make them real datetimes
    for col in df.columns:
        values = df[col].dropna()
        if df[col].dtype == object and not values.empty and values.map(
            lambda v: isinstance(v, date)
        ).all():
            df[col] = pd.to_datetime(df[col])
    return df

def load_table(name: str) -> pd.DataFrame:
    for attempt in range(1, RETRIES + 1):
        try:
            return _load_once(name)
        except psycopg.OperationalError as exc:
            logger.warning(
                "Reading %s failed (attempt %d/%d): %s",
                name, attempt, RETRIES, type(exc).__name__,
            )
            if attempt == RETRIES:
                raise
            time.sleep(WAIT_SECONDS)
    raise RuntimeError(f"Failed to load table '{name}' after {RETRIES} attempts")


def load_dim_customer() -> pd.DataFrame:
    return load_table("dim_customer")


def load_dim_date() -> pd.DataFrame:
    return load_table("dim_date")


def load_dim_product() -> pd.DataFrame:
    return load_table("dim_product")


def load_dim_store() -> pd.DataFrame:
    return load_table("dim_store")


def load_fact_sales() -> pd.DataFrame:
    return load_table("fact_sales")


def load_sample(name: str) -> pd.DataFrame:
    if name not in TABLES:
        raise ValueError(f"Unknown table '{name}'")
    path = FIXTURE_DIR / f"{name}.csv"
    return pd.read_csv(path, parse_dates=SAMPLE_DATE_COLUMNS.get(name, []))


def load_all(sample: bool = False) -> dict[str, pd.DataFrame]:
    if sample:
        return {name: load_sample(name) for name in TABLES}
    return {name: load_table(name) for name in TABLES}