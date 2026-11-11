import logging
import os
import time
from decimal import Decimal

import pandas as pd
import psycopg  # type: ignore[reportMissingImports]
from psycopg import sql  # type: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

SCHEMA = os.environ.get("SCHEMA_NAME", "clean")
TABLES = ("dim_customer", "dim_date", "dim_product", "dim_store", "fact_sales")
RETRIES = 3
WAIT_SECONDS = 5


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


def load_table(name: str) -> pd.DataFrame:
    if name not in TABLES:
        raise ValueError(f"Unknown table '{name}'")
    query = sql.SQL("select * from {}.{}").format(sql.Identifier(SCHEMA), sql.Identifier(name))
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query)
        columns = [col.name for col in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=columns)
    # numeric columns arrive as Decimal objects; models need floats
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, Decimal)).any():
            df[col] = df[col].astype(float)
    return df


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


def load_all() -> dict[str, pd.DataFrame]:
    return {name: load_table(name) for name in TABLES}