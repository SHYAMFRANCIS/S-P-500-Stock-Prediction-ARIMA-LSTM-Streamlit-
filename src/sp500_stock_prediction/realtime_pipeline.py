"""
MODULE 6: Real-Time Data Pipeline & Streaming
==============================================
Purpose: Continuous data ingestion, model retraining triggers, and
real-time prediction serving for live trading hours.

Features:
- Scheduled data fetching via the `schedule` library in a background thread
- Change detection: retrain when the latest close moves beyond a threshold
- Prediction serving: FastAPI endpoints (/health, /predict, /metrics)
- WebSocket streaming for real-time price snapshots
- Alert/retrain event log with graceful shutdown
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import schedule
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from sp500_stock_prediction.data_collector import (
    CacheError,
    DataFetchError,
    StockDataCollector,
    ValidationError,
)

logger = logging.getLogger(__name__)

RETRAIN_CALLBACK = Callable[[str], None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RealtimePipeline:
    """Orchestrates scheduled ingestion, change detection and serving."""

    def __init__(
        self,
        tickers: List[str],
        retrain_threshold: float = 0.05,
        cache_db: str = "realtime_cache.db",
        retrain_callback: Optional[RETRAIN_CALLBACK] = None,
        prediction_fn: Optional[Callable[[str, pd.DataFrame], Dict[str, Any]]] = None,
    ) -> None:
        if not tickers:
            raise ValueError("tickers must be a non-empty list")
        self.tickers = [t.strip().upper() for t in tickers]
        self.retrain_threshold = float(retrain_threshold)
        self.collector = StockDataCollector(cache_db=cache_db)
        self.latest_data: Dict[str, pd.DataFrame] = {}
        self.retrain_events: List[Dict[str, Any]] = []
        self.retrain_callback = retrain_callback
        self.prediction_fn = prediction_fn or self._default_prediction
        self._stop_event = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None
        logger.info("status=pipeline_init tickers=%s threshold=%.3f",
                    self.tickers, self.retrain_threshold)

    # ------------------------------------------------------------------ #
    # Scheduling
    # ------------------------------------------------------------------ #
    def start_scheduler(self, interval_minutes: int = 15) -> None:
        """Start the cron-like background scheduler (non-blocking)."""
        if self.is_scheduler_running():
            logger.warning("status=scheduler_already_running")
            return
        interval = max(1, int(interval_minutes))
        schedule.clear()
        schedule.every(interval).minutes.do(self.run_scheduled_job)
        self._stop_event.clear()

        def loop() -> None:
            while not self._stop_event.is_set():
                schedule.run_pending()
                self._stop_event.wait(0.05)

        self._scheduler_thread = threading.Thread(
            target=loop, name="rt-pipeline-scheduler", daemon=True
        )
        self._scheduler_thread.start()
        logger.info("status=scheduler_started interval_min=%d", interval)

    def stop_scheduler(self) -> None:
        """Signal the scheduler thread to stop and clear pending jobs."""
        self._stop_event.set()
        thread = self._scheduler_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._scheduler_thread = None
        schedule.clear()
        logger.info("status=scheduler_stopped")

    def is_scheduler_running(self) -> bool:
        thread = self._scheduler_thread
        return bool(thread is not None and thread.is_alive())

    def run_scheduled_job(self) -> Dict[str, Any]:
        """One scheduler iteration: fetch, detect changes, alert/retrain."""
        summary: Dict[str, Any] = {"timestamp": _utc_now_iso(), "tickers": {}}
        previous = dict(self.latest_data)
        try:
            fetched = self.fetch_latest_data()
        except (DataFetchError, ValidationError) as exc:
            logger.error("status=scheduled_fetch_failed error=%s", exc)
            summary["error"] = str(exc)
            return summary

        for ticker, df_new in fetched.items():
            changed = self.detect_significant_change(df_new, previous.get(ticker))
            summary["tickers"][ticker] = {"rows": len(df_new), "significant_change": changed}
            if changed:
                self.trigger_retrain(ticker)
        self.latest_data = fetched
        logger.info("status=scheduled_job_complete %s", {k: v["rows"] for k, v
                                                         in summary["tickers"].items()})
        return summary

    # ------------------------------------------------------------------ #
    # Ingestion / change detection / retraining
    # ------------------------------------------------------------------ #
    def fetch_latest_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch the most recent ~5 days of data for every ticker."""
        results = self.collector.fetch_batch(self.tickers, period="5d")
        failures = [t for t in self.tickers if t not in results]
        if failures:
            logger.warning("status=fetch_partial failed=%s", failures)
        self.latest_data.update(results)
        return results

    def detect_significant_change(
        self,
        new_data: Optional[pd.DataFrame],
        old_data: Optional[pd.DataFrame],
        threshold: Optional[float] = None,
    ) -> bool:
        """True when the last close moved more than the retrain threshold."""
        if new_data is None or new_data.empty or old_data is None or old_data.empty:
            return False
        limit = float(threshold) if threshold is not None else self.retrain_threshold

        def last_close(df: pd.DataFrame) -> float:
            closes = df["Close"].dropna()
            return float(closes.iloc[-1])

        old_close = last_close(old_data)
        new_close = last_close(new_data)
        if np.isnan(old_close) or np.isnan(new_close):
            return True
        if old_close == 0.0:
            return True
        rel_change = abs(new_close - old_close) / abs(old_close)
        return bool(rel_change > limit)

    def trigger_retrain(self, ticker: str) -> Dict[str, Any]:
        """Log and dispatch a retrain event for one ticker."""
        event = {
            "ticker": ticker.upper(),
            "timestamp": _utc_now_iso(),
            "reason": "significant_price_change",
            "status": "queued",
        }
        self.retrain_events.append(event)
        logger.warning("status=retrain_triggered ticker=%s", ticker.upper())
        if self.retrain_callback is not None:
            try:
                self.retrain_callback(ticker.upper())
                event["status"] = "dispatched"
            except Exception as exc:
                event["status"] = f"callback_failed: {exc}"
                logger.error("status=retrain_callback_failed ticker=%s error=%s",
                             ticker, exc)
        return event

    # ------------------------------------------------------------------ #
    # Prediction serving
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_prediction(ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Persistence forecast with a volatility-based confidence band."""
        closes = df["Close"].dropna().astype(float)
        returns = closes.pct_change().dropna()
        sigma = float(returns.std()) if len(returns) > 1 else 0.01
        predicted = float(closes.iloc[-1])
        ci_low = predicted * (1 - 1.96 * sigma)
        ci_high = predicted * (1 + 1.96 * sigma)
        return {
            "predicted_price": round(predicted, 4),
            "confidence_interval": [round(ci_low, 4), round(ci_high, 4)],
            "model_used": "persistence_baseline",
        }

    def get_latest_prediction(self, ticker: str) -> Dict[str, Any]:
        """Serve a prediction for one ticker from the freshest cached data."""
        ticker = ticker.strip().upper()
        df = self.latest_data.get(ticker)
        if df is None or df.empty:
            try:
                df = self.collector.load_from_cache(ticker, max_age_hours=24 * 7)
            except CacheError as exc:
                logger.warning("status=cache_read_failed ticker=%s error=%s", ticker, exc)
                df = None
        if df is None or df.empty:
            raise LookupError(f"no data available for {ticker}; run fetch_latest_data()")

        payload = self.prediction_fn(ticker, df.copy())
        response = {
            "ticker": ticker,
            **payload,
            "as_of_close": float(df["Close"].dropna().iloc[-1]),
            "timestamp": _utc_now_iso(),
        }
        logger.info("status=prediction_served ticker=%s model=%s",
                    ticker, response.get("model_used"))
        return response

    def stream_snapshot(self) -> List[Dict[str, Any]]:
        """Latest price snapshot for all tickers (WebSocket payloads)."""
        snapshot = []
        for ticker in self.tickers:
            df = self.latest_data.get(ticker)
            if df is None or df.empty:
                continue
            closes = df["Close"].dropna().astype(float)
            prev = float(closes.iloc[-2]) if len(closes) >= 2 else float(closes.iloc[-1])
            last = float(closes.iloc[-1])
            snapshot.append({
                "ticker": ticker,
                "price": round(last, 4),
                "change_pct": round((last - prev) / prev * 100.0, 4) if prev else 0.0,
                "timestamp": _utc_now_iso(),
            })
        return snapshot


def create_app(pipeline: Optional[RealtimePipeline] = None) -> FastAPI:
    """FastAPI app factory bound to a pipeline instance."""
    pipeline = pipeline or RealtimePipeline(tickers=["^GSPC"])
    app = FastAPI(title="S&P 500 Realtime Pipeline", version="0.1.0")
    app.state.pipeline = pipeline

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "healthy", "timestamp": _utc_now_iso()}

    @app.get("/predict/{ticker}")
    def predict(ticker: str) -> Dict[str, Any]:
        try:
            return pipeline.get_latest_prediction(ticker)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/metrics")
    def metrics() -> Dict[str, Any]:
        events = pipeline.retrain_events
        return {
            "tickers": pipeline.tickers,
            "cached_tickers": list(pipeline.latest_data.keys()),
            "retrain_events_total": len(events),
            "retrain_events": events[-20:],
            "scheduler_running": pipeline.is_scheduler_running(),
            "timestamp": _utc_now_iso(),
        }

    @app.websocket("/ws/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            await websocket.send_json({"type": "snapshot", "data": pipeline.stream_snapshot()})
            while True:
                message = await websocket.receive_text()
                if message == "close":
                    break
                if message == "snapshot":
                    await websocket.send_json(
                        {"type": "snapshot", "data": pipeline.stream_snapshot()}
                    )
        except WebSocketDisconnect:
            logger.info("status=ws_client_disconnected")
        finally:
            await websocket.close()
            logger.info("status=ws_closed")

    return app
