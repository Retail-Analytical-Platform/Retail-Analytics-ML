from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import psycopg
import pytest

from src.data import ingestion

URL_VAR = "DATABASE_ROLE_URL"
COLUMNS = ["id", "amount", "day"]
ROWS = [(1, Decimal("1.50"), date(2024, 1, 1)), (2, Decimal("2.50"), date(2024, 1, 2))]


class FakeCursor:
    description = [SimpleNamespace(name=c) for c in COLUMNS]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query):
        pass

    def fetchall(self):
        return ROWS


class FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor()


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    """A fake URL, and no real waiting between retries."""
    monkeypatch.setenv(URL_VAR, "postgresql://fake:fake@localhost:5432/fake")
    sleeps = []
    monkeypatch.setattr(ingestion.time, "sleep", sleeps.append)
    return sleeps


def patch_connect(monkeypatch, fail_times):
    calls = {"n": 0}

    def connect(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise psycopg.OperationalError("boom")
        return FakeConn()

    monkeypatch.setattr(ingestion.psycopg, "connect", connect)
    return calls


@pytest.mark.parametrize("name", ingestion.TABLES)
def test_normal_load_for_each_table(monkeypatch, name):
    patch_connect(monkeypatch, fail_times=0)
    df = ingestion.load_table(name)
    assert list(df.columns) == COLUMNS
    assert len(df) == 2
    assert pd.api.types.is_float_dtype(df["amount"])          # Decimal -> float
    assert pd.api.types.is_datetime64_any_dtype(df["day"])    # date -> datetime


def test_unknown_table_raises(monkeypatch):
    patch_connect(monkeypatch, fail_times=0)
    with pytest.raises(ValueError, match="Unknown table"):
        ingestion.load_table("not_a_table")


def test_missing_url_raises_and_never_connects(monkeypatch):
    monkeypatch.delenv(URL_VAR, raising=False)
    calls = patch_connect(monkeypatch, fail_times=0)
    with pytest.raises(RuntimeError, match="not set"):
        ingestion._connect()
    assert calls["n"] == 0


def test_unreachable_database_retries_then_raises(monkeypatch, fake_env):
    calls = patch_connect(monkeypatch, fail_times=99)
    with pytest.raises(ConnectionError, match="Could not reach the database"):
        ingestion._connect()
    assert calls["n"] == ingestion.RETRIES
    assert len(fake_env) == ingestion.RETRIES - 1        # no wait after the last try


def test_connect_succeeds_after_a_failed_attempt(monkeypatch):
    calls = patch_connect(monkeypatch, fail_times=1)
    assert ingestion._connect() is not None
    assert calls["n"] == 2


def test_read_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def always_fails(name):
        calls["n"] += 1
        raise psycopg.OperationalError("dropped")

    monkeypatch.setattr(ingestion, "_load_once", always_fails)
    with pytest.raises(psycopg.OperationalError):
        ingestion.load_table("fact_sales")
    assert calls["n"] == ingestion.RETRIES


def test_read_succeeds_after_a_dropped_read(monkeypatch):
    calls = {"n": 0}

    def flaky(name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise psycopg.OperationalError("dropped")
        return pd.DataFrame({"id": [1]})

    monkeypatch.setattr(ingestion, "_load_once", flaky)
    assert len(ingestion.load_table("fact_sales")) == 1
    assert calls["n"] == 2


def test_sample_load_needs_no_database(monkeypatch):
    monkeypatch.delenv(URL_VAR, raising=False)
    tables = ingestion.load_all(sample=True)
    assert set(tables) == set(ingestion.TABLES)
