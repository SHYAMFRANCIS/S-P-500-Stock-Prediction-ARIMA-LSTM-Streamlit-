"""
Test suite for MODULE 1: Data Collection & Ingestion Pipeline.
===============================================================
Covers fetching, validation, caching, retry logic, and S&P 500
ticker retrieval. Network-dependent tests are gated behind the
RUN_NETWORK_TESTS=1 environment variable so the suite stays
hermetic by default.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import sp500_stock_prediction.data_collector as dc
from sp500_stock_prediction.data_collector import (
    DataFetchError,
    StockDataCollector,
    ValidationError,
    retry_with_backoff,
)

NETWORK_TESTS = os.environ.get("RUN_NETWORK_TESTS") == "1"

requires_network = pytest.mark.skipif(
    not NETWORK_TESTS, reason="set RUN_NETWORK_TESTS=1 to enable network tests"
)


def make_valid_df(days: int = 30, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame that passes validate_data."""
    idx = pd.bdate_range(start=start, periods=max(days, 22))
    n = len(idx)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0.1, 1.0, size=n))
    close = np.maximum(close, 5.0)
    high = close * (1 + np.abs(rng.normal(0, 0.005, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, size=n)))
    open_ = low + (high - low) * rng.random(n)
    volume = rng.integers(1_000_000, 10_000_000, size=n).astype("int64")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.DatetimeIndex(idx, name="Date"),
    )


class FakeTicker:
    """Stand-in for yf.Ticker returning canned history data."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def history(self, *args, **kwargs) -> pd.DataFrame:
        return self._df.copy()


@pytest.fixture
def collector(tmp_path) -> StockDataCollector:
    return StockDataCollector(cache_db=str(tmp_path / "cache.db"), max_retries=3)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(dc.time, "sleep", lambda seconds: None)


@pytest.fixture
def patch_yf(monkeypatch):
    def _install(df: pd.DataFrame) -> None:
        monkeypatch.setattr(dc.yf, "Ticker", lambda symbol: FakeTicker(df))

    return _install


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return make_valid_df(days=30)


@requires_network
def test_fetch_single_valid_ticker(collector):
    df = collector.fetch_single("AAPL", period="1mo", interval="1d")
    assert not df.empty
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)


def test_fetch_single_invalid_ticker(collector, patch_yf, no_sleep):
    patch_yf(pd.DataFrame())
    with pytest.raises(DataFetchError):
        collector.fetch_single("INVALID_TICKER_XYZ")


def test_validate_data_pass(valid_df):
    assert StockDataCollector.validate_data(valid_df) is True


def test_validate_data_missing_columns(valid_df):
    broken = valid_df.drop(columns=["High"])
    assert StockDataCollector.validate_data(broken) is False


def test_validate_data_null_close(valid_df):
    broken = valid_df.copy()
    broken.iloc[0, broken.columns.get_loc("Close")] = np.nan
    assert StockDataCollector.validate_data(broken) is False


def test_validate_data_invalid_volume(valid_df):
    broken = valid_df.copy()
    broken.iloc[0, broken.columns.get_loc("Volume")] = 0
    assert StockDataCollector.validate_data(broken) is False


def test_validate_data_empty():
    assert StockDataCollector.validate_data(pd.DataFrame()) is False
    assert StockDataCollector.validate_data(None) is False


def test_cache_roundtrip(collector, valid_df):
    collector.cache_data("AAPL", valid_df)
    loaded = collector.load_from_cache("AAPL")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, valid_df)


def test_load_from_cache_expired(collector, valid_df):
    collector.cache_data("AAPL", valid_df)
    assert collector.load_from_cache("AAPL", max_age_hours=-1) is None
    assert collector.load_from_cache("MISSING_TICKER") is None


def test_fetch_single_uses_cache(collector, valid_df, patch_yf, monkeypatch):
    collector.cache_data("AAPL", valid_df)

    def boom(symbol):
        raise AssertionError("yfinance should not be called on cache hit")

    monkeypatch.setattr(dc.yf, "Ticker", boom)
    result = collector.fetch_single("AAPL")
    pd.testing.assert_frame_equal(result, valid_df)


def test_retry_decorator(monkeypatch):
    calls = {"count": 0}
    delays: list[float] = []
    monkeypatch.setattr(dc.time, "sleep", lambda s: delays.append(s))

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("transient failure")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3
    assert delays == [1.0, 2.0]


def test_retry_decorator_exhausted(no_sleep):
    calls = {"count": 0}

    @retry_with_backoff(max_retries=2, base_delay=0.01)
    def always_fails() -> None:
        calls["count"] += 1
        raise ConnectionError("down")

    with pytest.raises(DataFetchError):
        always_fails()
    assert calls["count"] == 2


def test_get_sp500_tickers(collector):
    tickers = collector.get_sp500_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) > 400
    assert len(set(tickers)) == len(tickers)
    assert all(isinstance(t, str) and t for t in tickers)


def test_fetch_batch_partial_success(collector, valid_df, patch_yf, monkeypatch):
    def fake_ticker(symbol: str):
        if symbol == "BAD":
            return FakeTicker(pd.DataFrame())
        return FakeTicker(valid_df)

    monkeypatch.setattr(dc.yf, "Ticker", fake_ticker)
    results = collector.fetch_batch(["GOOD", "BAD"], period="1mo")
    assert set(results.keys()) == {"GOOD"}
