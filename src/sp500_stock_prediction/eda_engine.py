"""
MODULE 2: Exploratory Data Analysis & Visualization Engine
===========================================================
Purpose: Comprehensive EDA with statistical tests, interactive
visualizations, and automated insight generation for stock data.

Features:
- Statistical tests: ADF, KPSS, Jarque-Bera, Ljung-Box
- Distribution analysis: histogram, Q-Q plot, KDE
- Correlation analysis: heatmap, rolling correlation
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands
- Interactive plots: Plotly candlesticks, volume profiles
- Automated report generation: HTML export

Input:  OHLCV pandas DataFrame with a DatetimeIndex, plus a ticker symbol.
Output: Statistical test results, plotly/matplotlib figures, HTML report.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from jinja2 import Template
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)

COLORBLIND_PALETTE = "colorblind"
REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EDA Report - {{ ticker }}</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem auto; max-width: 1100px;
         color: #1f2933; background: #f9fafb; }
  h1 { border-bottom: 2px solid #2563eb; padding-bottom: 0.4rem; }
  h2 { color: #2563eb; margin-top: 2rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff; }
  th, td { border: 1px solid #d5dbe1; padding: 0.5rem 0.75rem; text-align: left; }
  th { background: #eef2f7; }
  .badge-pass { color: #047857; font-weight: bold; }
  .badge-fail { color: #b91c1c; font-weight: bold; }
  .chart-block { background: #fff; border: 1px solid #d5dbe1; border-radius: 8px;
                 padding: 0.75rem; margin: 1rem 0; }
  img { max-width: 100%; height: auto; }
</style>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
<h1>Exploratory Data Analysis: {{ ticker }}</h1>

<h2>Executive Summary</h2>
<ul>
  <li>Rows analysed: <strong>{{ n_rows }}</strong>
      ({{ date_start }} to {{ date_end }})</li>
  <li>Price stationarity (ADF): {{ adf_badge }} (p = {{ "%.4f"|format(adf_p) }})</li>
  <li>Trend stationarity (KPSS): {{ kpss_badge }} (p = {{ "%.4f"|format(kpss_p) }})</li>
  <li>Normality of daily returns (Jarque-Bera): {{ jb_badge }}
      (p = {{ "%.4f"|format(jb_p) }})</li>
  <li>Annualised volatility: <strong>{{ "%.2f%%"|format(volatility) }}</strong></li>
  <li>Total return over period: <strong>{{ "%.2f%%"|format(total_return) }}</strong></li>
</ul>

<h2>Key Statistics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {% for name, value in key_stats %}
  <tr><td>{{ name }}</td><td>{{ value }}</td></tr>
  {% endfor %}
</table>

<h2>Interactive Charts</h2>
<div class="chart-block">{{ price_chart_html }}</div>
<div class="chart-block">{{ candlestick_html }}</div>

<h2>Return Distribution &amp; Correlation</h2>
<div class="chart-block"><img src="{{ heatmap_png }}" alt="Correlation heatmap"></div>

<h2>Technical Indicator Summary (latest values)</h2>
<table>
  <tr><th>Indicator</th><th>Value</th></tr>
  {% for name, value in indicator_summary %}
  <tr><td>{{ name }}</td><td>{{ value }}</td></tr>
  {% endfor %}
</table>
</body>
</html>
"""


class StockEDA:
    """Exploratory analysis, indicator computation, and reporting for one ticker."""

    def __init__(self, df: pd.DataFrame, ticker: str) -> None:
        if df is None or df.empty:
            raise ValueError("df must be a non-empty OHLCV DataFrame")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col not in df.columns:
                raise ValueError(f"missing required column: {col}")
        self.df = df.copy()
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.sort_index()
        self.ticker = ticker.upper()
        logger.info("ticker=%s status=eda_initialised rows=%d", self.ticker, len(self.df))

    def _close(self) -> pd.Series:
        return self.df["Close"].astype(float)

    def _returns(self) -> pd.Series:
        return self._close().pct_change().dropna()

    def adf_test(self) -> Dict[str, Any]:
        """Augmented Dickey-Fuller test on closing prices.

        Returns:
            Dict with statistic, p_value, critical_values and is_stationary
            (True when p_value < 0.05).
        """
        close = self._close()
        stat, p_value, used_lag, n_obs, crit, _ = adfuller(close, autolag="AIC")
        result: Dict[str, Any] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "used_lag": int(used_lag),
            "n_obs": int(n_obs),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "is_stationary": bool(p_value < 0.05),
        }
        logger.info("ticker=%s test=adf p=%.4f stationary=%s", self.ticker, p_value,
                    result["is_stationary"])
        return result

    def kpss_test(self) -> Dict[str, Any]:
        """KPSS test on closing prices.

        Returns:
            Dict with statistic, p_value and is_trend_stationary
            (True when p_value > 0.05). Warnings from the statsmodels
            interpolation heuristic are suppressed.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            stat, p_value, lags, crit = kpss(self._close(), regression="ct", nlags="auto")
        result: Dict[str, Any] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "lags": int(lags),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "is_trend_stationary": bool(p_value > 0.05),
        }
        logger.info("ticker=%s test=kpss p=%.4f trend_stationary=%s", self.ticker, p_value,
                    result["is_trend_stationary"])
        return result

    def jarque_bera_test(self) -> Dict[str, float]:
        """Jarque-Bera normality test on daily returns."""
        jb_stat, jb_p, skew, kurtosis = jarque_bera(self._returns())
        return {
            "statistic": float(jb_stat),
            "p_value": float(jb_p),
            "skew": float(skew),
            "kurtosis": float(kurtosis),
            "is_normal": bool(jb_p > 0.05),
        }

    def ljung_box_test(self, lags: int = 20) -> pd.DataFrame:
        """Ljung-Box test on daily returns for autocorrelation."""
        return acorr_ljungbox(self._returns(), lags=[lags], return_df=True)

    def calculate_indicators(self) -> pd.DataFrame:
        """Compute SMA(20,50), EMA(12,26), RSI(14), MACD(12,26,9), BB(20,2).

        All computations are vectorised; boundary rows legitimately contain
        NaN where a window is not yet full.

        Returns:
            Copy of the source DataFrame with indicator columns appended.
        """
        df = self.df.copy()
        close = df["Close"].astype(float)

        df["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
        df["SMA_50"] = close.rolling(window=50, min_periods=50).mean()
        df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
        df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

        macd_line = df["EMA_12"] - df["EMA_26"]
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df["MACD"] = macd_line
        df["MACD_Signal"] = signal_line
        df["MACD_Hist"] = macd_line - signal_line

        bb_middle = close.rolling(window=20, min_periods=20).mean()
        bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
        df["BB_Middle"] = bb_middle
        df["BB_Upper"] = bb_middle + 2.0 * bb_std
        df["BB_Lower"] = bb_middle - 2.0 * bb_std

        return df

    def plot_price_trend(self) -> go.Figure:
        """Interactive price trend with SMA overlays."""
        fig = go.Figure()
        close = self._close()
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=close,
                mode="lines",
                name="Close",
                line=dict(color="#2563eb", width=2),
            )
        )
        sma20 = close.rolling(20, min_periods=1).mean()
        sma50 = close.rolling(50, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(x=self.df.index, y=sma20, mode="lines", name="SMA 20",
                       line=dict(color="#f59e0b", width=1.5))
        )
        fig.add_trace(
            go.Scatter(x=self.df.index, y=sma50, mode="lines", name="SMA 50",
                       line=dict(color="#059669", width=1.5))
        )
        fig.update_layout(
            title=f"{self.ticker} Price Trend",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    def plot_candlestick(self, max_points: int = 500) -> go.Figure:
        """OHLC candlestick chart with volume sub-panel."""
        view = self.df.tail(max_points)
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=view.index,
                open=view["Open"],
                high=view["High"],
                low=view["Low"],
                close=view["Close"],
                name=self.ticker,
            )
        )
        colors = ["#059669" if c >= o else "#dc2626" for o, c in zip(view["Open"], view["Close"])]
        fig.add_trace(go.Bar(x=view.index, y=view["Volume"], marker_color=colors,
                             name="Volume", yaxis="y2", opacity=0.4))
        fig.update_layout(
            title=f"{self.ticker} Candlestick + Volume",
            xaxis_rangeslider_visible=False,
            yaxis=dict(title="Price (USD)"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", type="log"),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    def plot_correlation_heatmap(self) -> matplotlib.figure.Figure:
        """Seaborn heatmap of correlations between OHLCV columns."""
        corr = self.df[["Open", "High", "Low", "Close", "Volume"]].corr()
        fig, ax = matplotlib.pyplot.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                    square=True, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(f"{self.ticker} OHLCV Correlation")
        fig.tight_layout()
        return fig

    def plot_return_distribution(self) -> matplotlib.figure.Figure:
        """Histogram with KDE and Q-Q inset for daily returns."""
        returns = self._returns()
        palette = sns.color_palette(COLORBLIND_PALETTE, 1)[0]
        fig, axes = matplotlib.pyplot.subplots(1, 2, figsize=(12, 4.5))
        sns.histplot(returns, kde=True, bins=60, color=palette, ax=axes[0])
        axes[0].set_title(f"{self.ticker} Daily Returns Distribution")
        axes[0].set_xlabel("Daily return")
        axes[0].set_ylabel("Frequency")
        stats.probplot(returns, dist="norm", plot=axes[1])
        axes[1].set_title(f"{self.ticker} Q-Q Plot (normal)")
        fig.tight_layout()
        return fig

    def generate_report(self, output_path: str) -> Path:
        """Render the automated EDA report to a standalone HTML file.

        Args:
            output_path: Destination path ending in ``.html``.

        Returns:
            The path of the written report.
        """
        indicators = self.calculate_indicators()
        adf_result = self.adf_test()
        kpss_result = self.kpss_test()
        jb_result = self.jarque_bera_test()

        returns = self._returns()
        volatility = float(returns.std() * np.sqrt(252) * 100)
        total_return = float(
            (self._close().iloc[-1] / self._close().iloc[0] - 1.0) * 100
        )

        key_stats = [
            ("Mean daily return", f"{returns.mean():.5f}"),
            ("Daily return std", f"{returns.std():.5f}"),
            ("Latest close", f"{self._close().iloc[-1]:.2f}"),
            ("Period high", f"{self.df['High'].max():.2f}"),
            ("Period low", f"{self.df['Low'].min():.2f}"),
            ("Average volume", f"{self.df['Volume'].mean():,.0f}"),
            (
                "ADF statistic vs 5% critical",
                f"{adf_result['statistic']:.3f} vs "
                f"{adf_result['critical_values'].get('5%', float('nan')):.3f}",
            ),
        ]

        latest = indicators.iloc[-1]
        indicator_summary = []
        for col in [
            "SMA_20", "SMA_50", "EMA_12", "EMA_26",
            "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Upper", "BB_Middle", "BB_Lower",
        ]:
            value = latest.get(col)
            indicator_summary.append((col, "n/a" if pd.isna(value) else f"{value:.4f}"))

        heatmap_buf = io.BytesIO()
        heatmap_fig = self.plot_correlation_heatmap()
        heatmap_fig.savefig(heatmap_buf, format="png", dpi=120)
        matplotlib.pyplot.close(heatmap_fig)
        heatmap_b64 = base64.b64encode(heatmap_buf.getvalue()).decode("ascii")

        html = Template(REPORT_TEMPLATE).render(
            ticker=self.ticker,
            n_rows=len(self.df),
            date_start=str(self.df.index.min().date()),
            date_end=str(self.df.index.max().date()),
            adf_badge=_badge(adf_result["is_stationary"], "stationary"),
            kpss_badge=_badge(kpss_result["is_trend_stationary"], "trend stationary"),
            jb_badge=_badge(jb_result["is_normal"], "normal"),
            adf_p=adf_result["p_value"],
            kpss_p=kpss_result["p_value"],
            jb_p=jb_result["p_value"],
            volatility=volatility,
            total_return=total_return,
            key_stats=key_stats,
            price_chart_html=self.plot_price_trend().to_html(
                full_html=False, include_plotlyjs=False
            ),
            candlestick_html=self.plot_candlestick().to_html(
                full_html=False, include_plotlyjs=False
            ),
            heatmap_png=f"data:image/png;base64,{heatmap_b64}",
            indicator_summary=indicator_summary,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logger.info("ticker=%s status=report_written path=%s", self.ticker, out)
        return out


def _badge(ok: bool, label: str) -> str:
    cls = "badge-pass" if ok else "badge-fail"
    verdict = "yes" if ok else "no"
    return f'<span class="{cls}">{verdict}</span> ({label})'
