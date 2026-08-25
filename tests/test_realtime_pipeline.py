"""
Test suite for MODULE 6: Real-Time Data Pipeline & Streaming.
==============================================================
Fully hermetic: yfinance is mocked, the scheduler is exercised via
direct job invocation plus a short-lived thread lifecycle check, and
the FastAPI app is tested with TestClient (in-process).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import sp500_stock_prediction.realtime_pipeline as rtp
from sp500_stock_prediction.realtime_pipeline import (
    RealtimePipeline,
    create_app,
)


def make_ohlcv(last_close: float = 100.0, days: int = 6) -> pd.DataFrame:
    idx = pd.bdate_range(end="2024-06-14", periods=days)
    closes = np.linspace(last_close * 0.98, last_close, days)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(days, 1_000_000),
        },
        index=pd.DatetimeIndex(idx, name="Date"),
    )


@pytest.fixture
def mock_fetch(monkeypatch):
    """Replace collector.fetch_batch with canned data; record calls."""
    calls: dict[str, object] = {"count": 0, "period": None}

    def fake_batch(self, tickers, period="2y", interval="1d"):
        calls["count"] += 1
        calls["period"] = period
        return {t: make_ohlcv(100.0 + 10 * i) for i, t in enumerate(tickers)}

    monkeypatch.setattr(rtp.StockDataCollector, "fetch_batch", fake_batch)
    return calls


@pytest.fixture
def pipeline(mock_fetch) -> RealtimePipeline:
    pipe = RealtimePipeline(tickers=["AAPL", "MSFT"], retrain_threshold=0.05,
                            cache_db=":memory:")
    # Seed with baseline data so change detection has something to compare to.
    pipe.latest_data = {t: make_ohlcv(100.0 + 10 * i) for i, t in
                        enumerate(pipe.tickers)}
    return pipe


def test_detect_significant_change_true(pipeline):
    new = make_ohlcv(last_close=120.0)   # +20% vs baseline 106.0 -> >5%
    assert pipeline.detect_significant_change(new, pipeline.latest_data["AAPL"]) is True


def test_detect_significant_change_false(pipeline):
    new = make_ohlcv(last_close=104.0)   # ~4% move vs baseline 100 -> below threshold
    assert pipeline.detect_significant_change(new, pipeline.latest_data["AAPL"]) is False


def test_detect_change_empty_data_returns_false(pipeline):
    assert pipeline.detect_significant_change(make_ohlcv(), None) is False
    assert pipeline.detect_significant_change(pd.DataFrame(), make_ohlcv()) is False


def test_fetch_latest_data(pipeline, mock_fetch):
    result = pipeline.fetch_latest_data()
    assert set(result.keys()) == {"AAPL", "MSFT"}
    assert mock_fetch["period"] == "5d"
    for df in result.values():
        assert not df.empty and "Close" in df.columns


def test_trigger_retrain_logs_event(pipeline):
    events_before = len(pipeline.retrain_events)
    event = pipeline.trigger_retrain("aapl")
    assert event["ticker"] == "AAPL"
    assert len(pipeline.retrain_events) == events_before + 1
    assert datetime.fromisoformat(event["timestamp"])


def test_trigger_retrain_callback_invoked():
    seen: list[str] = []
    pipe = RealtimePipeline(tickers=["T"], cache_db=":memory:",
                            retrain_callback=seen.append)
    pipe.trigger_retrain("T")
    assert seen == ["T"]
    assert pipe.retrain_events[-1]["status"] == "dispatched"


def test_get_latest_prediction_structure(pipeline):
    prediction = pipeline.get_latest_prediction("AAPL")
    assert {"predicted_price", "confidence_interval", "model_used",
            "timestamp"} <= set(prediction)
    lo, hi = prediction["confidence_interval"]
    assert lo <= hi
    assert datetime.fromisoformat(prediction["timestamp"])


def test_get_latest_prediction_unknown_ticker(pipeline):
    with pytest.raises(LookupError):
        pipeline.get_latest_prediction("NOPE")


def test_custom_prediction_fn_used(pipeline):
    pipeline.prediction_fn = lambda ticker, df: {
        "predicted_price": 42.0, "confidence_interval": [40.0, 44.0],
        "model_used": "custom",
    }
    result = pipeline.get_latest_prediction("MSFT")
    assert result["predicted_price"] == 42.0
    assert result["model_used"] == "custom"


def test_scheduler_lifecycle(pipeline):
    assert not pipeline.is_scheduler_running()
    pipeline.start_scheduler(interval_minutes=15)
    assert pipeline.is_scheduler_running()
    # Directly invoke the registered job instead of waiting a minute.
    schedule_jobs = list(rtp.schedule.jobs)
    assert schedule_jobs, "scheduler should have registered at least one job"
    summary = pipeline.run_scheduled_job()
    assert set(summary["tickers"].keys()) == {"AAPL", "MSFT"}
    pipeline.stop_scheduler()
    assert not pipeline.is_scheduler_running()
    assert not rtp.schedule.jobs


def test_run_scheduled_job_triggers_retrain_on_big_move(pipeline, monkeypatch):
    def spike_batch(self, tickers, period="2y", interval="1d"):
        return {t: make_ohlcv(200.0 + 10 * i) for i, t in enumerate(tickers)}

    monkeypatch.setattr(rtp.StockDataCollector, "fetch_batch", spike_batch)
    events_before = len(pipeline.retrain_events)
    pipeline.run_scheduled_job()
    assert len(pipeline.retrain_events) > events_before
    assert all(e["ticker"] in pipeline.tickers for e in pipeline.retrain_events)


# ---------------------------------------------------------------------- #
# FastAPI surface
# ---------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client() -> TestClient:
    module_pipe = RealtimePipeline(
        tickers=["AAPL"], retrain_threshold=0.05, cache_db=":memory:"
    )
    module_pipe.latest_data = {"AAPL": make_ohlcv()}
    app = create_app(module_pipe)
    with TestClient(app) as tc:
        yield tc


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    datetime.fromisoformat(body["timestamp"])


def test_predict_endpoint(client):
    response = client.get("/predict/AAPL")
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert "predicted_price" in body


def test_predict_endpoint_unknown_returns_404(client):
    response = client.get("/predict/UNKNOWN_XYZ")
    assert response.status_code == 404


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["tickers"] == ["AAPL"]
    assert "retrain_events_total" in body
    assert "cached_tickers" in body


def test_websocket_stream(client):
    with client.websocket_connect("/ws/stream") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"
        rows = snapshot["data"]
        assert rows and rows[0]["ticker"] == "AAPL"
        assert "price" in rows[0]
        ws.send_text("close")
