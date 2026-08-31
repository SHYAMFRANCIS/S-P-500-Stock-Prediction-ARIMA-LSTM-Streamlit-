"""
MODULE 7: Streamlit Interactive Dashboard
==========================================
Purpose: User-friendly web interface for stock analysis, model comparison,
and real-time monitoring.

Run with:
    uv run streamlit run src/sp500_stock_prediction/dashboard.py

Tabs: Overview, EDA, ARIMA, LSTM, Compare, Live.
Data loading is cached with @st.cache_data; heavy objects (collector,
models) with @st.cache_resource / st.session_state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- pyarrow DLL blocked fallback (Windows App Control) ---
# Both st.dataframe and st.table require pyarrow on this env (blocked DLL).
# Fall back to pure-HTML via st.markdown which does NOT import pyarrow.
def _render_without_pyarrow(data):  # type: ignore[no-untyped-def]
    """Render DataFrame/Series without touching pyarrow."""
    try:
        if hasattr(data, "to_html"):
            html = data.to_html(classes="dataframe", border=0)  # type: ignore[attr-defined]
            return st.markdown(html, unsafe_allow_html=True)
        return st.text(str(data))
    except Exception:
        return st.text(str(data))


try:
    _orig_dataframe = st.dataframe
    _orig_table = st.table

    def _safe_dataframe(*args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return _orig_dataframe(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "pyarrow" in msg or "dll" in msg or "application control" in msg or "lib" in msg:
                kwargs.pop("use_container_width", None)
                kwargs.pop("hide_index", None)
                data = args[0] if args else kwargs.pop("data", None)
                if data is not None:
                    return _render_without_pyarrow(data)
                return _render_without_pyarrow(args[0] if args else kwargs)
            raise

    def _safe_table(*args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return _orig_table(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "pyarrow" in msg or "dll" in msg or "application control" in msg or "lib" in msg:
                data = args[0] if args else kwargs.get("data")
                if data is not None:
                    return _render_without_pyarrow(data)
                return _render_without_pyarrow(args[0] if args else kwargs)
            raise

    st.dataframe = _safe_dataframe  # type: ignore[assignment]
    st.table = _safe_table  # type: ignore[assignment]
except Exception:
    pass

st.set_page_config(
    page_title="S&P 500 Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]


# --------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_collector():
    from sp500_stock_prediction.data_collector import StockDataCollector

    return StockDataCollector(cache_db="dashboard_cache.db")


@st.cache_data(ttl=900, show_spinner="Fetching market data...")
def load_data(ticker: str, period: str) -> Optional[pd.DataFrame]:
    try:
        return get_collector().fetch_single(ticker, period=period)
    except Exception as exc:
        st.error(f"Could not load data for {ticker}: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def get_eda(df: pd.DataFrame, ticker: str):
    from sp500_stock_prediction.eda_engine import StockEDA

    return StockEDA(df=df, ticker=ticker)


# --------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------- #
def _metric_row(row) -> None:
    cols = st.columns(3)
    values = list(row.items())
    for i, (label, value) in enumerate(values[:3]):
        cols[i].metric(label, value)


def _show_price_with_sma(df: pd.DataFrame) -> go.Figure:
    close = df["Close"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=close, name="Close", mode="lines"))
    for window, color in [(20, "#f59e0b"), (50, "#059669")]:
        sma = close.rolling(window, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma, name=f"SMA {window}",
                                 mode="lines", line=dict(color=color)))
    fig.update_layout(title="Price with SMA overlays", template="plotly_white",
                      height=420, legend=dict(orientation="h", y=1.06))
    return fig


def _indicators_figure(indicators: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indicators.index, y=indicators["Close"],
                             name="Close", mode="lines"))
    for col, dash in [("BB_Upper", "dot"), ("BB_Lower", "dot"),
                      ("SMA_20", "dash"), ("EMA_12", "dashdot")]:
        series = indicators[col]
        fig.add_trace(go.Scatter(x=indicators.index, y=series, name=col,
                                 mode="lines", line=dict(dash=dash, width=1)))
    fig.update_layout(title="Technical Indicators", template="plotly_white",
                      height=440, legend=dict(orientation="h", y=1.08))
    return fig


def _forecast_figure(history: pd.Series, mean, ci) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index[-120:], y=history.iloc[-120:],
                             name="History", mode="lines"))
    fig.add_trace(go.Scatter(x=mean.index, y=mean, name="Forecast", mode="lines",
                             line=dict(dash="dash")))
    fig.add_trace(go.Scatter(
        x=list(mean.index) + list(mean.index[::-1]),
        y=list(ci["upper"]) + list(ci["lower"])[::-1],
        fill="toself", fillcolor="rgba(37,99,235,0.15)", name="95% CI",
        line=dict(width=0), showlegend=True,
    ))
    fig.update_layout(title="Forecast with Confidence Intervals",
                      template="plotly_white", height=430)
    return fig


def _prediction_vs_actual_figure(y_true, y_pred) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(y=y_pred, name="Predicted", mode="lines",
                             line=dict(dash="dash")))
    fig.update_layout(title="LSTM: Predicted vs Actual (test segment)",
                      template="plotly_white", height=430,
                      xaxis_title="Test sample", yaxis_title="Price")
    return fig


def train_lstm(df: pd.DataFrame, lookback: int, epochs: int, batch_size: int):
    """Train the LSTM with live progress; returns (predictor, history, preds)."""
    from sp500_stock_prediction.lstm_model import LSTMPredictor
    import tensorflow as tf

    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, bar: "st.delta_generator.DeltaGenerator"):
            super().__init__()
            self._bar = bar

        def on_epoch_end(self, epoch, logs=None):
            frac = min(1.0, (epoch + 1) / max(1, epochs))
            self._bar.progress(int(frac * 100),
                               text=f"Epoch {epoch + 1}/{epochs} - "
                                    f"val_loss={logs.get('val_loss', float('nan')):.5f}")

    predictor = LSTMPredictor(lookback_window=int(lookback), units=[50],
                              dropout_rate=0.2, learning_rate=0.001)
    X, y = predictor.prepare_sequences(df["Close"])
    split = int(len(X) * 0.9)
    bar = st.progress(0, text="Training...")
    history = predictor.fit(X[:split], y[:split], X[split:], y[split:],
                            epochs=epochs, batch_size=batch_size, patience=max(2, epochs // 3))
    bar.progress(100, text="Training complete")
    y_true = predictor.scaler.inverse_transform(y[split:]).ravel()
    y_pred = predictor.predict(X[split:])
    return predictor, history, (y_true, y_pred)


# --------------------------------------------------------------------- #
# Tab renderers
# --------------------------------------------------------------------- #
def render_overview(df: pd.DataFrame, ticker: str) -> None:
    closes = df["Close"].astype(float)
    last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
    change_pct = (last - prev) / prev * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Last Close", f"{last:,.2f}", f"{change_pct:+.2f}%")
    c2.metric("Volume", f"{float(df['Volume'].iloc[-1]):,.0f}")
    c3.metric("Rows Loaded", len(df))

    st.subheader("Key Statistics")
    stats = pd.DataFrame({
        "Statistic": ["Mean", "Std Dev", "Min", "Max"],
        "Close": [closes.mean(), closes.std(), closes.min(), closes.max()],
        "Daily Return": [
            closes.pct_change().mean(), closes.pct_change().std(),
            closes.pct_change().min(), closes.pct_change().max(),
        ],
    }).set_index("Statistic").round(4)
    st.dataframe(stats, use_container_width=True)

    st.plotly_chart(_show_price_with_sma(df), use_container_width=True)


def render_eda(df: pd.DataFrame, ticker: str) -> None:
    eda = get_eda(df, ticker)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("ADF Test (price)")
        adf = eda.adf_test()
        st.json(json.dumps({k: v for k, v in adf.items() if k != "critical_values"},
                           default=str))
        st.caption(f"Critical values: {adf['critical_values']}")
    with col_b:
        st.subheader("KPSS Test (price)")
        kpss_result = eda.kpss_test()
        st.json(json.dumps({k: v for k, v in kpss_result.items()
                            if k != "critical_values"}, default=str))
        jb = eda.jarque_bera_test()
        st.caption(f"Jarque-Bera on returns: stat={jb['statistic']:.2f}, "
                   f"p={jb['p_value']:.4g}")

    left, right = st.columns(2)
    with left:
        st.subheader("Return Distribution")
        st.pyplot(eda.plot_return_distribution(), clear_figure=True)
    with right:
        st.subheader("Correlation Heatmap")
        st.pyplot(eda.plot_correlation_heatmap(), clear_figure=True)

    st.subheader("Technical Indicators")
    indicators = eda.calculate_indicators()
    st.plotly_chart(_indicators_figure(indicators), use_container_width=True)

    html_path = Path("reports") / f"eda_report_{ticker.replace('^', '').replace('.', '_')}.html"
    if st.button("Generate HTML report", key="eda_report_btn"):
        try:
            out = eda.generate_report(str(html_path))
            st.success(f"Report written to {out}")
            st.download_button("Download report", data=out.read_bytes(),
                               file_name=out.name, mime="text/html")
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")


def render_arima(df: pd.DataFrame, ticker: str) -> None:
    st.subheader(f"ARIMA — {ticker}")
    mode = st.radio("Order selection", ["Auto (AIC grid search)", "Manual"],
                    horizontal=True, key="arima_mode")
    order = None
    if mode == "Manual":
        c1, c2, c3 = st.columns(3)
        p = c1.number_input("p", 0, 5, 1)
        d = c2.number_input("d", 0, 2, 1)
        q = c3.number_input("q", 0, 5, 1)
        order = (int(p), int(d), int(q))
    else:
        st.caption("Search space capped at max_p=3, max_d=1, max_q=3 for speed.")

    if st.button("Fit ARIMA model", type="primary"):
        from sp500_stock_prediction.arima_model import ARIMAPredictor

        try:
            with st.spinner("Fitting ARIMA..."):
                predictor = ARIMAPredictor(order=order)
                if order is None:
                    predictor.find_optimal_order(df["Close"], max_p=3, max_d=1, max_q=3)
                predictor.fit(df["Close"])
            st.session_state["arima_predictor"] = predictor
            st.session_state["arima_train_end"] = df.index[-1]
            st.success(f"Fitted ARIMA{predictor.order} (AIC={predictor._results.aic:.1f})")
        except Exception as exc:
            st.error(f"ARIMA fitting failed: {exc}")

    predictor: Optional[Any] = st.session_state.get("arima_predictor")
    if predictor is None:
        st.info("Fit a model to see forecasts and diagnostics.")
        return

    steps = st.slider("Forecast horizon (trading days)", 1, 30, 10, key="arima_steps")
    mean, ci = predictor.predict(steps=steps)
    st.plotly_chart(_forecast_figure(df["Close"], mean, ci), use_container_width=True)

    st.subheader("Residual Diagnostics")
    diag = predictor.residual_diagnostics()
    st.dataframe(pd.Series(diag).to_frame("value"), use_container_width=True)
    verdict = "white noise ✓" if diag["is_white_noise"] else "autocorrelated ✗"
    st.caption(f"Ljung-Box suggests residuals are {verdict}")


def render_lstm(df: pd.DataFrame, ticker: str) -> None:
    st.subheader(f"LSTM — {ticker}")
    c1, c2, c3 = st.columns(3)
    lookback = c1.slider("Lookback window", 10, 120, 60, step=5)
    epochs = c2.slider("Max epochs", 3, 100, 25)
    batch_size = c3.select_slider("Batch size", [8, 16, 32, 64], value=16)

    n_params = 0
    st.caption(f"Architecture: LSTM(50) → Dropout(0.2) → Dense(25) → Dense(1) "
               f"| lookback={lookback}")

    if st.button("Train LSTM", type="primary"):
        try:
            predictor, history, (y_true, y_pred) = train_lstm(
                df, lookback, epochs, batch_size
            )
            st.session_state["lstm_predictor"] = predictor
            st.session_state["lstm_preds"] = (y_true, y_pred)
            st.session_state["lstm_lookback"] = lookback
            st.line_chart({"train_loss": history.history["loss"],
                           "val_loss": history.history["val_loss"]})
            st.success("Training complete")
        except Exception as exc:
            st.error(f"LSTM training failed: {exc}")

    preds = st.session_state.get("lstm_preds")
    predictor = st.session_state.get("lstm_predictor")
    if preds and predictor is not None:
        y_true, y_pred = preds
        metrics = predictor.evaluate(y_true, y_pred)
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"{metrics['RMSE']:.4f}")
        m2.metric("MAE", f"{metrics['MAE']:.4f}")
        m3.metric("R²", f"{metrics['R2']:.4f}")
        st.plotly_chart(_prediction_vs_actual_figure(y_true, y_pred),
                        use_container_width=True)


def _get_predictions(df: pd.DataFrame, steps: int):
    arima = st.session_state.get("arima_predictor")
    if arima is not None:
        idx = pd.date_range(start=df.index[-1] + pd.offsets.BDay(1), periods=steps,
                            freq="B")
        mean, ci = arima.predict(steps=steps)
        mean.index = idx
        ci.index = idx
        arima_mean = mean
    else:
        arima_mean = None

    lstm = st.session_state.get("lstm_predictor")
    if lstm is not None:
        scaled_X, _ = lstm.prepare_sequences(df["Close"], fit_scaler=False)
        tail_X = scaled_X[-steps:]
        lstm_vals = lstm.predict(tail_X)
        hist_idx = df.index[-len(scaled_X):]
        lstm_series = pd.Series(lstm_vals, index=hist_idx[-steps:])
    else:
        lstm_series = None
    return arima_mean, lstm_series


def render_compare(df: pd.DataFrame, ticker: str) -> None:
    st.subheader(f"Model Comparison — {ticker}")
    steps = st.slider("Comparison horizon", 5, 30, 10, key="cmp_steps")

    arima_mean, lstm_series = _get_predictions(df, steps)
    if arima_mean is None or lstm_series is None:
        st.warning("Train **both** models (ARIMA and LSTM tabs) to compare them.")
        return

    # Evaluate each model against the actual tail it overlaps with.
    common_idx = arima_mean.index.intersection(lstm_series.index)
    if len(common_idx) < 3:
        st.warning("Forecasts do not overlap enough to compare; "
                   "retrain both models on the same period.")
        return
    actual_tail = df["Close"].reindex(common_idx).ffill()

    from sp500_stock_prediction.model_comparison import ModelComparator

    comparator = ModelComparator(actual_tail,
                                 arima_mean.reindex(common_idx).ffill(),
                                 lstm_series.reindex(common_idx).ffill())
    table = comparator.compare_metrics()
    st.dataframe(table, use_container_width=True)
    st.plotly_chart(comparator.plot_comparison(), use_container_width=True)

    winner = comparator.best_model()
    dm = comparator.diebold_mariano_test()
    st.markdown(f"## 🏆 Winner: {winner}")
    if dm["is_significant"]:
        st.caption(f"Diebold-Mariano confirms significance (DM="
                   f"{dm['dm_statistic']}, p={dm['p_value']}).")
    else:
        st.caption(f"Difference not statistically significant "
                   f"(DM={dm['dm_statistic']}, p={dm['p_value']}).")

    report = comparator.generate_winner_report()
    st.download_button("Download Markdown report", report, file_name=f"{ticker}_comparison.md",
                       mime="text/markdown")


def render_live(collector, ticker: str) -> None:
    st.subheader(f"Live Monitor — {ticker}")
    autorefresh = st.checkbox("Auto-refresh every 30s")
    if st.button("Refresh now", type="primary") or autorefresh:
        try:
            fresh = collector.fetch_single(ticker, period="1mo")
            st.session_state["live_df"] = fresh
        except Exception as exc:
            st.error(f"Fetch failed: {exc}")

    live_df: Optional[pd.DataFrame] = st.session_state.get("live_df")
    if live_df is None or live_df.empty:
        st.info("Press **Refresh now** to pull the latest prices.")
        return

    closes = live_df["Close"].astype(float)
    last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
    change_pct = (last - prev) / prev * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Last Price", f"{last:,.2f}", f"{change_pct:+.2f}%")
    c2.metric("Session High", f"{float(live_df['High'].iloc[-1]):,.2f}")
    c3.metric("Session Volume", f"{float(live_df['Volume'].iloc[-1]):,.0f}")

    baseline = st.session_state.get("live_baseline_close")
    threshold = st.sidebar.slider("Retrain/alert threshold (%)", 1, 20, 5,
                                  key="alert_threshold") / 100.0
    if baseline is not None:
        move = abs(last - baseline) / baseline
        alert = move > threshold
        if alert:
            st.error(f"🚨 ALERT: price moved {move * 100:.2f}% from baseline "
                     f"{baseline:,.2f} (threshold {threshold * 100:.0f}%)")
        else:
            st.success(f"No alert: {move * 100:.2f}% from baseline "
                       f"(threshold {threshold * 100:.0f}%)")
    st.session_state["live_baseline_close"] = last

    arima = st.session_state.get("arima_predictor")
    if arima is not None:
        mean, ci = arima.predict(steps=1)
        st.metric("Next-day forecast (ARIMA)", f"{float(mean.iloc[0]):,.2f}",
                  f"[{float(ci['lower'].iloc[0]):,.2f}, {float(ci['upper'].iloc[0]):,.2f}]")
    else:
        st.caption("Train the ARIMA model to see next-day forecasts here.")

    if autorefresh:
        import time

        time.sleep(30)
        st.rerun()


# --------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------- #
def main() -> None:
    st.title("📈 S&P 500 Stock Prediction Dashboard")
    st.caption(f"Session started {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Ticker symbol", "^GSPC").strip().upper()
        period = st.selectbox("Historical period", PERIOD_OPTIONS, index=3)
        st.divider()
        st.caption("Models are stored in this browser session. "
                   "Refresh resets trained state.")

    df = load_data(ticker, period)
    if df is None or df.empty:
        st.stop()

    overview_tab, eda_tab, arima_tab, lstm_tab, compare_tab, live_tab = st.tabs(
        ["Overview", "EDA", "ARIMA", "LSTM", "Compare", "Live"]
    )
    with overview_tab:
        render_overview(df, ticker)
    with eda_tab:
        render_eda(df, ticker)
    with arima_tab:
        render_arima(df, ticker)
    with lstm_tab:
        render_lstm(df, ticker)
    with compare_tab:
        render_compare(df, ticker)
    with live_tab:
        render_live(get_collector(), ticker)


if __name__ == "__main__":
    main()
