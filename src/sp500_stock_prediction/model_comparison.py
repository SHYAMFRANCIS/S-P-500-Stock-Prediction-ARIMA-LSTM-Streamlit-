"""
MODULE 5: Model Comparison & Ensemble Engine
=============================================
Purpose: Compare ARIMA vs LSTM performance, generate comparison reports,
and optionally create ensemble predictions.

Features:
- Side-by-side metric comparison (MAE/MSE/RMSE/R2/MAPE)
- Statistical significance testing (Diebold-Mariano, one-sided)
- Visualization: Actual vs Predicted overlays, residual distributions,
  metric bar charts
- Ensemble: Weighted average of ARIMA + LSTM predictions
- Automated winner selection based on RMSE and Markdown reporting
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

METRICS = ("MAE", "MSE", "RMSE", "R2", "MAPE")


def _metric_values(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100.0)
        if nonzero.any()
        else float("nan")
    )
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4),
    }


class ModelComparator:
    """Side-by-side evaluation of two forecast series against ground truth."""

    def __init__(
        self,
        actual: pd.Series,
        arima_pred: pd.Series,
        lstm_pred: pd.Series,
        model_names: Tuple[str, str] = ("ARIMA", "LSTM"),
    ) -> None:
        if actual is None or len(actual) == 0:
            raise ValueError("actual must be a non-empty series")
        self.model_a, self.model_b = model_names
        self.actual, self.arima_pred, self.lstm_pred = self._align(
            actual, arima_pred, lstm_pred
        )
        if len(self.actual) < 3:
            raise ValueError("need at least 3 aligned observations to compare")
        logger.info("status=comparator_init n=%d models=%s vs %s",
                    len(self.actual), self.model_a, self.model_b)

    @staticmethod
    def _align(
        actual: pd.Series, arima_pred: pd.Series, lstm_pred: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        def as_series(obj: Any, name: str) -> pd.Series:
            s = obj if isinstance(obj, pd.Series) else pd.Series(np.asarray(obj, dtype=float))
            return pd.to_numeric(s, errors="coerce").astype(float)

        actual_s = as_series(actual, "actual")
        arima_s = as_series(arima_pred, "arima").reindex(actual_s.index)
        lstm_s = as_series(lstm_pred, "lstm").reindex(actual_s.index)

        frame = pd.concat(
            {"actual": actual_s, "arima": arima_s, "lstm": lstm_s}, axis=1
        ).replace([np.inf, -np.inf], np.nan).dropna()
        return frame["actual"], frame["arima"], frame["lstm"]

    def compare_metrics(self) -> pd.DataFrame:
        """Metric table: one row per metric, columns per model plus Winner."""
        m_arima = _metric_values(self.actual.to_numpy(), self.arima_pred.to_numpy())
        m_lstm = _metric_values(self.actual.to_numpy(), self.lstm_pred.to_numpy())

        table = pd.DataFrame({self.model_a: m_arima, self.model_b: m_lstm})
        winners = []
        for metric in METRICS:
            va, vb = table.loc[metric, self.model_a], table.loc[metric, self.model_b]
            if metric in ("R2",):
                winners.append(self.model_a if va >= vb else self.model_b)
            else:
                winners.append(self.model_a if va <= vb else self.model_b)
        table["Winner"] = winners
        logger.info("status=metrics_compared winner_rmse=%s", winners[METRICS.index("RMSE")])
        return table

    def diebold_mariano_test(self, power: int = 2, horizon: int = 1) -> Dict[str, float]:
        """One-sided Diebold-Mariano test.

        Null hypothesis: both models have equal forecast accuracy.
        Alternative: the second model (LSTM) is more accurate, i.e. its
        losses are smaller. Uses a Newey-West/HAC variance estimate with
        ``horizon`` lags to absorb autocorrelation in the loss differential.

        Returns:
            Dict with dm_statistic, p_value and is_significant (alpha=0.05).
        """
        e_a = (self.actual - self.arima_pred).to_numpy()
        e_b = (self.actual - self.lstm_pred).to_numpy()
        d = np.abs(e_a) ** power - np.abs(e_b) ** power
        n = len(d)
        d_bar = float(d.mean())
        demeaned = d - d_bar
        gamma0 = float(np.mean(demeaned**2))
        var_sum = gamma0
        for lag in range(1, min(horizon, n - 1) + 1):
            weight = 1.0 - lag / (horizon + 1.0)
            cov = float(np.mean(demeaned[lag:] * demeaned[:-lag]))
            var_sum += 2.0 * weight * cov
        variance = max(var_sum, 1e-12) / n
        dm_stat = d_bar / np.sqrt(variance)

        # One-sided: H1 = second model better => d > 0 => large DM.
        p_one_sided = float(1.0 - stats.norm.cdf(dm_stat))
        result = {
            "dm_statistic": round(float(dm_stat), 4),
            "p_value": round(p_one_sided, 6),
            "mean_loss_diff": round(d_bar, 6),
            "n_observations": n,
            "is_significant": bool(p_one_sided < 0.05),
        }
        logger.info("status=dm_test dm=%.4f p=%.4f significant=%s",
                    dm_stat, p_one_sided, result["is_significant"])
        return result

    def ensemble_prediction(
        self, weights: Tuple[float, float] = (0.5, 0.5)
    ) -> pd.Series:
        """Weighted average of both forecasts.

        Raises:
            ValueError: If weights do not have two components, are negative,
                or do not sum to 1.0 within tolerance.
        """
        w = np.asarray(weights, dtype=float)
        if w.shape != (2,) or (w < 0).any() or abs(w.sum() - 1.0) > 1e-9:
            raise ValueError(f"weights must be two non-negative values summing "
                             f"to 1.0, got {weights!r}")
        ensemble = w[0] * self.arima_pred + w[1] * self.lstm_pred
        ensemble.name = "ensemble"
        logger.info("status=ensemble weights=%s rmse=%.4f",
                    tuple(w), float(np.sqrt(np.mean((self.actual - ensemble) ** 2))))
        return ensemble

    def plot_comparison(self) -> matplotlib.figure.Figure:
        """Two panels: Actual-vs-models overlay, residual distributions."""
        fig, (ax_forecast, ax_residual) = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
        )
        palette = sns.color_palette("colorblind", 3)
        idx = self.actual.index

        ax_forecast.plot(idx, self.actual, label="Actual", color=palette[0],
                         linewidth=2, marker="o", markersize=3)
        ax_forecast.plot(idx, self.arima_pred, label=self.model_a, color=palette[1],
                         linewidth=1.5, linestyle="--")
        ax_forecast.fill_between(
            idx,
            self.arima_pred - self.actual.std(),
            self.arima_pred + self.actual.std(),
            color=palette[1], alpha=0.10,
        )
        ax_forecast.plot(idx, self.lstm_pred, label=self.model_b, color=palette[2],
                         linewidth=1.5, linestyle="-.")
        ax_forecast.set_title("Actual vs Model Forecasts")
        ax_forecast.set_xlabel("Date")
        ax_forecast.set_ylabel("Price")
        ax_forecast.legend()
        ax_forecast.grid(alpha=0.3)

        resid_a = (self.actual - self.arima_pred).to_numpy()
        resid_b = (self.actual - self.lstm_pred).to_numpy()
        sns.histplot(resid_a, kde=True, color=palette[1], label=f"{self.model_a} residuals",
                     bins=25, ax=ax_residual, alpha=0.6)
        sns.histplot(resid_b, kde=True, color=palette[2], label=f"{self.model_b} residuals",
                     bins=25, ax=ax_residual, alpha=0.6)
        ax_residual.axvline(0.0, color="black", linewidth=1, linestyle=":")
        ax_residual.set_title("Residual Distributions")
        ax_residual.set_xlabel("Error")
        ax_residual.legend()

        fig.tight_layout()
        return fig

    def plot_metric_bars(self) -> matplotlib.figure.Figure:
        """Grouped bar chart of every metric except MSE-dominated scales."""
        table = self.compare_metrics().drop(columns=["Winner"])
        plot_table = table.drop(index="MSE")
        fig, ax = plt.subplots(figsize=(9, 5))
        palette = sns.color_palette("colorblind", 2)
        x = np.arange(len(plot_table.index))
        width = 0.35
        ax.bar(x - width / 2, plot_table[self.model_a], width,
               label=self.model_a, color=palette[0])
        ax.bar(x + width / 2, plot_table[self.model_b], width,
               label=self.model_b, color=palette[1])
        ax.set_xticks(x, plot_table.index)
        ax.set_title("Forecast Error Metrics by Model")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def best_model(self) -> str:
        """Name of the model with the lower RMSE."""
        table = self.compare_metrics()
        return str(table.loc["RMSE", "Winner"])

    def generate_winner_report(self) -> str:
        """Markdown summary: metrics, DM significance, recommendation."""
        table = self.compare_metrics()
        dm = self.diebold_mariano_test()
        winner = self.best_model()
        loser = self.model_b if winner == self.model_a else self.model_a

        verdict = (
            "statistically significant" if dm["is_significant"]
            else "not statistically significant"
        )
        lines = [
            f"# Model Comparison Report: {self.model_a} vs {self.model_b}",
            "",
            "## Metrics",
            "",
            table.to_string(),
            "",
            "## Diebold-Mariano Test",
            "",
            f"- DM statistic: **{dm['dm_statistic']}**",
            f"- One-sided p-value: **{dm['p_value']}**",
            f"- Verdict: {verdict} at alpha=0.05 in favour of {self.model_b}",
            "",
            "## Recommendation",
            "",
            f"Use **{winner}**: it achieves the lowest RMSE "
            f"({table.loc['RMSE', winner]:.4f} vs {table.loc['RMSE', loser]:.4f}).",
        ]
        if not dm["is_significant"]:
            lines.append(
                "The difference is not statistically significant; an equally "
                "weighted ensemble may offer robustness."
            )
        report = "\n".join(lines)
        logger.info("status=report_generated winner=%s", winner)
        return report
