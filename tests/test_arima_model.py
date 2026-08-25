"""
Test suite for MODULE 3: ARIMA Statistical Forecasting Model.
==============================================================
Hermetic tests on synthetic series with small grids to keep runtime low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500_stock_prediction.arima_model import ARIMAPredictor


def make_stationary_series(n: int = 200, seed: int = 11) -> pd.Series:
    """AR(1) process around a constant mean — stationary, ARIMA-friendly."""
    rng = np.random.default_rng(seed)
    phi, shocks = 0.6, rng.normal(0, 1.0, size=n)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = 5.0 + phi * (values[t - 1] - 5.0) + shocks[t]
    idx = pd.bdate_range(start="2023-01-02", periods=n)
    return pd.Series(values, index=pd.DatetimeIndex(idx, name="Date"), name="Close")


def make_random_walk(n: int = 220, seed: int = 3) -> pd.Series:
    """Random walk with drift — I(1), needs d=1."""
    rng = np.random.default_rng(seed)
    walk = 100.0 + np.cumsum(rng.normal(0.05, 1.0, size=n))
    idx = pd.bdate_range(start="2023-01-02", periods=n)
    return pd.Series(walk, index=pd.DatetimeIndex(idx, name="Date"), name="Close")


@pytest.fixture(scope="module")
def stationary() -> pd.Series:
    return make_stationary_series()


@pytest.fixture(scope="module")
def random_walk() -> pd.Series:
    return make_random_walk()


def test_find_optimal_order_returns_tuple(stationary):
    predictor = ARIMAPredictor()
    order = predictor.find_optimal_order(stationary, max_p=2, max_d=1, max_q=2)
    assert isinstance(order, tuple)
    assert len(order) == 3
    assert all(isinstance(v, int) and v >= 0 for v in order)


def test_find_optimal_order_prefers_d1_on_random_walk(random_walk):
    predictor = ARIMAPredictor()
    order = predictor.find_optimal_order(random_walk, max_p=2, max_d=1, max_q=2)
    assert order[1] == 1


def test_fit_predict_consistency(stationary):
    predictor = ARIMAPredictor(order=(1, 0, 1))
    train = stationary.iloc[:-10]
    predictor.fit(train)
    mean, _ = predictor.predict(steps=10)
    assert isinstance(mean, pd.Series)
    assert len(mean) == 10
    assert np.isfinite(mean).all()
    assert not mean.isnull().any()


def test_predict_before_fit_raises():
    predictor = ARIMAPredictor(order=(1, 0, 0))
    with pytest.raises(RuntimeError):
        predictor.predict(steps=5)


def test_forecast_has_confidence_intervals(stationary):
    predictor = ARIMAPredictor(order=(1, 0, 1))
    predictor.fit(stationary.iloc[:-12])
    mean, ci = predictor.predict(steps=12, alpha=0.05)
    assert {"lower", "upper"}.issubset(ci.columns)
    assert list(ci.index) == list(mean.index)
    assert (ci["lower"] <= ci["upper"]).all()
    assert ((mean >= ci["lower"]) & (mean <= ci["upper"])).all()
    # Wider alpha -> wider interval
    _, ci_wide = predictor.predict(steps=12, alpha=0.20)
    assert (ci_wide["upper"] - ci_wide["lower"] <= ci["upper"] - ci["lower"]).all()


def test_evaluate_metrics_structure(stationary):
    predictor = ARIMAPredictor(order=(1, 0, 1))
    split = int(len(stationary) * 0.9)
    predictor.fit(stationary.iloc[:split])
    metrics = predictor.evaluate(stationary.iloc[split:])
    assert set(metrics) == {"MAE", "MSE", "RMSE", "R2", "MAPE"}
    for value in metrics.values():
        assert isinstance(value, float)
        assert np.isfinite(value)
    assert metrics["MAE"] >= 0
    assert metrics["MSE"] >= 0
    assert metrics["RMSE"] >= 0


def test_residual_diagnostics(stationary):
    predictor = ARIMAPredictor(order=(1, 0, 1))
    predictor.fit(stationary)
    diag = predictor.residual_diagnostics()
    assert "ljung_box_p_value" in diag
    assert 0.0 <= diag["ljung_box_p_value"] <= 1.0
    for key in ["ljung_box_stat", "residual_mean", "residual_std",
                "jarque_bera_stat", "jarque_bera_p_value"]:
        assert key in diag
        assert np.isfinite(diag[key])


def test_save_load_model(stationary, tmp_path):
    predictor = ARIMAPredictor(order=(1, 0, 1))
    predictor.fit(stationary.iloc[:-8])
    original_mean, original_ci = predictor.predict(steps=8)

    path = tmp_path / "model.pkl"
    predictor.save_model(str(path))

    restored = ARIMAPredictor(order=(9, 9, 9))
    restored.load_model(str(path))
    loaded_mean, loaded_ci = restored.predict(steps=8)

    assert restored.order == (1, 0, 1)
    pd.testing.assert_series_equal(original_mean, loaded_mean)
    pd.testing.assert_frame_equal(original_ci, loaded_ci)


def test_mape_zero_handling(stationary):
    predictor = ARIMAPredictor(order=(1, 0, 1))
    predictor.fit(stationary.iloc[:-6])
    test_with_zeros = stationary.iloc[-6:].copy().astype(float)
    test_with_zeros.iloc[0] = 0.0
    test_with_zeros.iloc[3] = -0.0
    metrics = predictor.evaluate(test_with_zeros)
    assert np.isfinite(metrics["MAPE"])
    assert metrics["MAPE"] >= 0


def test_short_series_raises():
    short = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2024-01-01", periods=3))
    predictor = ARIMAPredictor(order=(1, 0, 0))
    with pytest.raises(ValueError, match="30"):
        predictor.fit(short)


def test_dataframe_input_uses_close_column():
    df = make_stationary_series(120).to_frame()
    df["Volume"] = 1000
    predictor = ARIMAPredictor(order=(1, 0, 0)).fit(df.iloc[:110])
    assert predictor._results is not None
    mean, _ = predictor.predict(steps=5)
    assert len(mean) == 5


def test_adf_helper(random_walk, stationary):
    assert ARIMAPredictor.adf_is_stationary(random_walk) is False
    assert ARIMAPredictor.adf_is_stationary(stationary.diff().dropna()) is True
