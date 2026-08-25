"""
Test suite for MODULE 5: Model Comparison & Ensemble Engine.
=============================================================
Synthetic forecasts with a known quality gap so winners, DM
significance and ensemble arithmetic are deterministic.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sp500_stock_prediction.model_comparison import ModelComparator


def make_scenario(n: int = 120, seed: int = 13):
    """Actual series plus two forecast series: LSTM close, ARIMA noisy."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    actual = pd.Series(
        100 + 0.15 * t + 6 * np.sin(t / 9.0),
        index=pd.bdate_range("2023-01-02", periods=n),
        name="Close",
    )
    arima_pred = actual + rng.normal(0, 3.0, size=n)
    lstm_pred = actual + rng.normal(0, 0.8, size=n)
    return actual, arima_pred.rename("ARIMA"), lstm_pred.rename("LSTM")


@pytest.fixture(scope="module")
def scenario():
    return make_scenario()


@pytest.fixture(scope="module")
def comparator(scenario) -> ModelComparator:
    actual, arima_pred, lstm_pred = scenario
    return ModelComparator(actual, arima_pred, lstm_pred)


def test_compare_metrics_structure(comparator):
    table = comparator.compare_metrics()
    assert list(table.index) == ["MAE", "MSE", "RMSE", "R2", "MAPE"]
    assert list(table.columns) == ["ARIMA", "LSTM", "Winner"]
    assert set(table["Winner"].unique()) <= {"ARIMA", "LSTM"}


def test_winner_by_rmse(comparator):
    table = comparator.compare_metrics()
    assert table.loc["RMSE", "Winner"] == "LSTM"
    assert comparator.best_model() == "LSTM"


def test_diebold_mariano_keys(comparator):
    dm = comparator.diebold_mariano_test()
    assert {"dm_statistic", "p_value", "is_significant"} <= set(dm)
    assert 0.0 <= dm["p_value"] <= 1.0
    assert isinstance(dm["is_significant"], bool)


def test_dm_detects_better_model(comparator):
    # LSTM errors (sigma=0.8) are far smaller than ARIMA's (sigma=3.0)
    dm = comparator.diebold_mariano_test()
    assert dm["dm_statistic"] > 0
    assert dm["is_significant"] is True


def test_dm_not_significant_when_equal():
    rng = np.random.default_rng(99)
    n = 200
    actual = pd.Series(np.full(n, 50.0))
    pred_a = actual + rng.normal(0, 2.0, size=n)
    pred_b = actual + rng.normal(0, 2.0, size=n)
    dm = ModelComparator(actual, pred_a, pred_b).diebold_mariano_test()
    assert dm["is_significant"] is False


def test_ensemble_values(comparator):
    ensemble = comparator.ensemble_prediction(weights=(0.3, 0.7))
    expected = 0.3 * comparator.arima_pred + 0.7 * comparator.lstm_pred
    pd.testing.assert_series_equal(ensemble, expected, check_names=False)


def test_ensemble_weights_validation(comparator):
    for bad in [(0.5, 0.6), (1.5,), (-0.2, 1.2), (0.7, 0.7)]:
        with pytest.raises(ValueError):
            comparator.ensemble_prediction(weights=bad)


def test_ensemble_beats_worst_model(comparator):
    ensemble_rmse = float(np.sqrt(np.mean((comparator.actual
                                           - comparator.ensemble_prediction()) ** 2)))
    metrics = comparator.compare_metrics()
    assert ensemble_rmse < max(metrics.loc["RMSE", "ARIMA"],
                               metrics.loc["RMSE", "LSTM"])


def test_plot_comparison_returns_figure(comparator):
    fig = comparator.plot_comparison()
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_metric_bars_returns_figure(comparator):
    fig = comparator.plot_metric_bars()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_generate_winner_report(comparator):
    report = comparator.generate_winner_report()
    assert isinstance(report, str)
    assert "# Model Comparison Report" in report
    assert "RMSE" in report
    assert "**LSTM**" in report


def test_misaligned_inputs_are_aligned(scenario):
    actual, arima_pred, lstm_pred = scenario
    shifted_lstm = lstm_pred.copy()
    shifted_lstm.iloc[:5] = np.nan
    comparator = ModelComparator(actual, arima_pred, shifted_lstm)
    assert len(comparator.actual) == len(actual) - 5


def test_empty_actual_raises():
    with pytest.raises(ValueError):
        ModelComparator(pd.Series(dtype=float), pd.Series(dtype=float),
                        pd.Series(dtype=float))


def test_bad_weights_sum_raises(scenario):
    actual, arima_pred, lstm_pred = scenario
    comparator = ModelComparator(actual, arima_pred, lstm_pred)
    with pytest.raises(ValueError, match="sum"):
        comparator.ensemble_prediction(weights=(0.9, 0.9))
