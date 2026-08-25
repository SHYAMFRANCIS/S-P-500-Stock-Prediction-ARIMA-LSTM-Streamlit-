"""
Test suite for MODULE 4: LSTM Deep Learning Forecasting Model.
===============================================================
Uses a small lookback, tiny network and 2-3 epochs so the suite
runs in seconds while still exercising every code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500_stock_prediction.lstm_model import LSTMPredictor


def make_series(n: int = 160, seed: int = 5) -> pd.Series:
    """Smooth synthetic price series (trend + sine + noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    prices = 100 + 0.1 * t + 8 * np.sin(t / 7.0) + rng.normal(0, 0.5, size=n)
    idx = pd.bdate_range(start="2023-01-02", periods=n)
    return pd.Series(prices, index=pd.DatetimeIndex(idx, name="Date"), name="Close")


@pytest.fixture(scope="module")
def series() -> pd.Series:
    return make_series()


@pytest.fixture(scope="module")
def trained(series) -> LSTMPredictor:
    predictor = LSTMPredictor(lookback_window=10, units=[8, 8], dropout_rate=0.1,
                              learning_rate=0.01)
    X, y = predictor.prepare_sequences(series.iloc[:120])
    split = len(X) - 20
    predictor.fit(X[:split], y[:split], X[split:], y[split:], epochs=3, batch_size=16,
                  patience=2)
    return predictor


def test_prepare_sequences_shape(series):
    predictor = LSTMPredictor(lookback_window=60)
    short = make_series(n=100)
    X, y = predictor.prepare_sequences(short)
    assert X.shape == (40, 60, 1)
    assert y.shape == (40, 1)
    assert 0.0 <= float(X.min()) and float(X.max()) <= 1.0


def test_prepare_sequences_too_short_raises():
    predictor = LSTMPredictor(lookback_window=30)
    with pytest.raises(ValueError):
        predictor.prepare_sequences(make_series(n=25))


def test_build_model_output():
    predictor = LSTMPredictor(lookback_window=10, units=[8, 8])
    model = predictor.build_model()
    assert model.input_shape == (None, 10, 1)
    assert model.output_shape == (None, 1)
    loss_name = model.loss if isinstance(model.loss, str) else model.loss.__name__
    assert loss_name == "mse"


def test_fit_returns_history(trained):
    history = getattr(trained, "_last_history", None)
    assert history is not None or trained.model is not None
    # Re-run fit briefly to capture the returned History object explicitly.
    predictor = LSTMPredictor(lookback_window=10, units=[4], dropout_rate=0.1,
                              learning_rate=0.01)
    X, y = predictor.prepare_sequences(make_series(n=80))
    hist = predictor.fit(X[:-10], y[:-10], X[-10:], y[-10:], epochs=2, batch_size=16)
    assert hasattr(hist, "history")
    assert "loss" in hist.history
    assert "val_loss" in hist.history


def test_predict_shape(trained, series):
    predictor = LSTMPredictor(lookback_window=10, units=[8, 8])
    predictor.model = trained.model
    predictor.scaler = trained.scaler
    X, y = predictor.prepare_sequences(series.iloc[-50:], fit_scaler=False)
    preds = predictor.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()
    # Inverse-transformed predictions should be near the raw price range.
    assert preds.min() > 50.0 and preds.max() < 200.0


def test_evaluate_metrics(trained, series):
    predictor = LSTMPredictor(lookback_window=10, units=[8, 8])
    predictor.model = trained.model
    predictor.scaler = trained.scaler
    X, y = predictor.prepare_sequences(series.iloc[-50:], fit_scaler=False)
    preds = predictor.predict(X)
    truth = predictor.scaler.inverse_transform(y).ravel()
    metrics = LSTMPredictor.evaluate(truth, preds)
    assert set(metrics) == {"MAE", "MSE", "RMSE", "R2", "MAPE"}
    for value in metrics.values():
        assert np.isfinite(value)


def test_save_load_model(trained, tmp_path):
    path = tmp_path / "lstm_model"
    trained.save_model(str(path))

    restored = LSTMPredictor(lookback_window=99)
    restored.load_model(str(path))

    assert restored.lookback_window == 10
    X, _ = restored.prepare_sequences(make_series(n=120), fit_scaler=False)
    original_preds = trained.predict(X)
    loaded_preds = restored.predict(X)
    np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)


def test_scaler_persistence(trained, tmp_path):
    path = tmp_path / "scaler_check"
    trained.save_model(str(path))

    restored = LSTMPredictor()
    restored.load_model(str(path))

    probe = np.array([[trained.scaler.data_min_[0]], [trained.scaler.data_max_[0]]])
    expected = trained.scaler.transform(probe)
    actual = restored.scaler.transform(probe)
    np.testing.assert_allclose(expected, actual)

    json_path = path.with_name("scaler_check_scaler.json")
    assert json_path.exists()


def test_mape_zero_handling():
    metrics = LSTMPredictor.evaluate(np.array([0.0, 10.0, 20.0]),
                                     np.array([1.0, 11.0, 21.0]))
    assert np.isfinite(metrics["MAPE"])
