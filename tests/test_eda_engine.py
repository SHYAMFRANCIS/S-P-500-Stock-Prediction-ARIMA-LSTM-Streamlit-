"""
Test suite for MODULE 2: Exploratory Data Analysis & Visualization Engine.
==========================================================================
All tests are hermetic (no network, no display backends). Matplotlib runs
in Agg mode via the eda_engine module import.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from sp500_stock_prediction.eda_engine import StockEDA


def make_ohlcv(days: int = 260, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV frame with enough rows for SMA(50) and BB(20)."""
    idx = pd.bdate_range(start="2023-01-02", periods=days)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, size=days)))
    high = close * (1 + np.abs(rng.normal(0, 0.006, size=days)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, size=days)))
    open_ = low + (high - low) * rng.random(days)
    volume = rng.integers(500_000, 20_000_000, size=days).astype("int64")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.DatetimeIndex(idx, name="Date"),
    )


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture(scope="module")
def eda(ohlcv) -> StockEDA:
    return StockEDA(df=ohlcv, ticker="AAPL")


def test_adf_test_structure(eda):
    result = eda.adf_test()
    assert set(result) >= {"statistic", "p_value", "is_stationary"}
    assert isinstance(result["statistic"], float)
    assert 0.0 <= result["p_value"] <= 1.0
    assert isinstance(result["is_stationary"], bool)


def test_kpss_test_structure(eda):
    result = eda.kpss_test()
    assert set(result) >= {"statistic", "p_value", "is_trend_stationary"}
    assert isinstance(result["is_trend_stationary"], bool)


def test_calculate_indicators_columns(eda):
    out = eda.calculate_indicators()
    for col in [
        "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Middle", "BB_Lower",
    ]:
        assert col in out.columns
        assert out[col].notna().any()
    assert len(out) == len(eda.df)
    original = eda.df
    assert list(original.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_rsi_range(eda):
    rsi = eda.calculate_indicators()["RSI_14"].dropna()
    assert not rsi.empty
    assert (rsi >= 0.0).all()
    assert (rsi <= 100.0).all()


def test_bollinger_order(eda):
    ind = eda.calculate_indicators().dropna(subset=["BB_Upper", "BB_Middle", "BB_Lower"])
    assert not ind.empty
    assert (ind["BB_Upper"] > ind["BB_Middle"]).all()
    assert (ind["BB_Middle"] > ind["BB_Lower"]).all()


def test_macd_consistency(eda):
    ind = eda.calculate_indicators()
    reconstructed = ind["MACD"] - ind["MACD_Signal"]
    pd.testing.assert_series_equal(ind["MACD_Hist"], reconstructed, check_names=False)


def test_sma_matches_pandas(eda):
    ind = eda.calculate_indicators()
    expected = eda.df["Close"].rolling(20, min_periods=20).mean()
    pd.testing.assert_series_equal(ind["SMA_20"], expected, check_names=False)


def test_plot_price_trend_returns_figure(eda):
    fig = eda.plot_price_trend()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


def test_plot_candlestick_returns_figure(eda):
    fig = eda.plot_candlestick(max_points=100)
    assert isinstance(fig, go.Figure)
    kinds = [type(trace).__name__ for trace in fig.data]
    assert "Candlestick" in kinds
    assert "Bar" in kinds


def test_plot_correlation_heatmap_returns_figure(eda):
    fig = eda.plot_correlation_heatmap()
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_plot_return_distribution_returns_figure(eda):
    fig = eda.plot_return_distribution()
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_generate_report_creates_file(eda, tmp_path):
    out_path = tmp_path / "report.html"
    written = eda.generate_report(str(out_path))
    assert written.exists()
    assert out_path.stat().st_size > 0


def test_report_contains_ticker(eda, tmp_path):
    out_path = tmp_path / "report.html"
    eda.generate_report(str(out_path))
    html = out_path.read_text(encoding="utf-8")
    assert "AAPL" in html
    assert "<html" in html.lower()


def test_invalid_dataframe_raises():
    with pytest.raises(ValueError):
        StockEDA(pd.DataFrame(), ticker="AAPL")


def test_missing_column_raises():
    df = make_ohlcv(days=30).drop(columns=["Volume"])
    with pytest.raises(ValueError, match="Volume"):
        StockEDA(df, ticker="AAPL")
