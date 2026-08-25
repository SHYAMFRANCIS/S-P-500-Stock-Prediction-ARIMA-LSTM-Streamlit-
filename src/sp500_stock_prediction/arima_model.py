"""
MODULE 3: ARIMA Statistical Forecasting Model
==============================================
Purpose: Traditional time-series forecasting using ARIMA with automatic
parameter selection, diagnostics, and residual analysis.

Features:
- Auto ARIMA: Grid search p,d,q using AIC/BIC
- Manual ARIMA: User-specified (p,d,q)
- Residual diagnostics: Ljung-Box, Jarque-Bera, residual moments
- Forecast with confidence intervals
- Model persistence: Save/load with pickle

Input:  Univariate price series (pandas Series, or DataFrame with a
        "Close" column).
Output: Point forecasts with confidence intervals, evaluation metrics,
        residual diagnostics, persisted model files.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_ORDER: Tuple[int, int, int] = (1, 1, 1)


class ARIMAPredictor:
    """ARIMA forecaster with automatic order selection and diagnostics."""

    def __init__(self, order: Optional[Tuple[int, int, int]] = None) -> None:
        self.order = tuple(order) if order is not None else None
        self._results: Any = None
        self._train_index: Optional[pd.Index] = None
        logger.info("status=arima_init order=%s", self.order)

    @staticmethod
    def _to_series(data: "pd.DataFrame | pd.Series") -> pd.Series:
        if isinstance(data, pd.DataFrame):
            if "Close" not in data.columns:
                raise ValueError("DataFrame input must contain a 'Close' column")
            series = data["Close"].astype(float)
        else:
            series = pd.Series(data, dtype=float)
        series = series.dropna()
        return series.sort_index() if isinstance(series.index, pd.DatetimeIndex) else series

    def find_optimal_order(
        self,
        df: "pd.DataFrame | pd.Series",
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
    ) -> Tuple[int, int, int]:
        """Grid-search (p,d,q) minimising AIC.

        Non-converging or degenerate fits (LinAlgError, ValueError,
        HessianInversionWarning) are skipped. Falls back to (1,1,1)
        when nothing converges.
        """
        series = self._to_series(df)

        # Cap d at what the data supports: difference until variance explodes
        best_order: Optional[Tuple[int, int, int]] = None
        best_aic = np.inf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.simplefilter("error", HessianInversionWarning)
            warnings.filterwarnings("error", message=".*Hessian.*invert.*")
            for p, d, q in product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
                if p == 0 and q == 0:
                    continue
                try:
                    fit = ARIMA(series, order=(p, d, q), enforce_stationarity=False,
                                enforce_invertibility=False).fit()
                    aic = float(fit.aic)
                except (Warning, np.linalg.LinAlgError, ValueError, IndexError) as exc:
                    logger.debug("order=(%d,%d,%d) skipped: %s", p, d, q, exc)
                    continue
                if np.isfinite(aic) and aic < best_aic:
                    best_aic, best_order = aic, (p, d, q)
                logger.debug("order=(%d,%d,%d) aic=%.2f", p, d, q, aic)

        if best_order is None:
            best_order = DEFAULT_FALLBACK_ORDER
            logger.warning("status=no_convergence fallback_order=%s", best_order)

        logger.info("status=order_selected order=%s aic=%.2f", best_order, best_aic)
        self.order = best_order
        return best_order

    def fit(self, train_data: "pd.DataFrame | pd.Series") -> "ARIMAPredictor":
        """Fit ARIMA. Runs the AIC grid search first when no order was given."""
        series = self._to_series(train_data)
        if len(series) < 30:
            raise ValueError(f"need at least 30 observations to fit, got {len(series)}")
        if self.order is None:
            self.order = self.find_optimal_order(series)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._results = ARIMA(
                series, order=self.order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit()
        self._train_index = series.index
        logger.info("status=fitted order=%s aic=%.2f n=%d",
                    self.order, float(self._results.aic), len(series))
        return self

    def predict(
        self, steps: int, alpha: float = 0.05
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Forecast ``steps`` ahead with (1-alpha) confidence intervals.

        Returns:
            Tuple of (forecast_mean, ci) where ci has columns
            ``lower`` and ``upper`` aligned to the forecast index.
        """
        if self._results is None:
            raise RuntimeError("model is not fitted yet; call fit() first")
        forecast = self._results.get_forecast(steps=steps)
        mean = pd.Series(
            np.asarray(forecast.predicted_mean, dtype=float),
            name="forecast",
        )
        ci_raw = forecast.conf_int(alpha=alpha)
        lower_col = [c for c in ci_raw.columns if str(c).startswith("lower")]
        upper_col = [c for c in ci_raw.columns if str(c).startswith("upper")]
        ci = pd.DataFrame(
            {
                "lower": np.asarray(ci_raw[lower_col[0]] if lower_col else ci_raw.iloc[:, 0],
                                    dtype=float),
                "upper": np.asarray(ci_raw[upper_col[0]] if upper_col else ci_raw.iloc[:, -1],
                                    dtype=float),
            },
            index=mean.index,
        )
        if isinstance(self._train_index, pd.DatetimeIndex):
            freq = self._train_index.freq or pd.infer_freq(self._train_index)
            if freq is not None:
                start = self._train_index[-1] + pd.tseries.frequencies.to_offset(freq)
                new_idx = pd.date_range(start=start, periods=steps, freq=freq)
                mean.index = new_idx
                ci.index = new_idx
        logger.info("status=predicted steps=%d alpha=%.2f", steps, alpha)
        return mean, ci

    def evaluate(self, test_data: "pd.DataFrame | pd.Series") -> Dict[str, float]:
        """Score the model against held-out data.

        Produces a one-step-ahead rolling-free comparison: the model
        forecasts ``len(test)`` steps from the end of training and the
        result is scored against ``test``.

        Returns:
            Dict with MAE, MSE, RMSE, R2, MAPE rounded to 4 decimals.
        """
        if self._results is None:
            raise RuntimeError("model is not fitted yet; call fit() first")
        actual = self._to_series(test_data).astype(float)
        pred_mean, _ = self.predict(steps=len(actual))
        predicted = np.asarray(pred_mean, dtype=float)[: len(actual)]
        actual_arr = np.asarray(actual, dtype=float)

        mae = float(mean_absolute_error(actual_arr, predicted))
        mse = float(mean_squared_error(actual_arr, predicted))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(actual_arr, predicted)) if len(actual_arr) > 1 else float("nan")

        nonzero = actual_arr != 0
        if nonzero.any():
            mape = float(
                np.mean(np.abs((actual_arr[nonzero] - predicted[nonzero])
                               / actual_arr[nonzero])) * 100.0
            )
        else:
            mape = float("nan")

        metrics = {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "MAPE": round(mape, 4),
        }
        logger.info("status=evaluated metrics=%s", metrics)
        return metrics

    def residual_diagnostics(self) -> Dict[str, Any]:
        """Analyse in-sample residuals for whiteness and normality."""
        if self._results is None:
            raise RuntimeError("model is not fitted yet; call fit() first")
        resid = pd.Series(np.asarray(self._results.resid, dtype=float))
        resid = resid.replace([np.inf, -np.inf], np.nan).dropna()
        lb_df = acorr_ljungbox(resid, lags=[min(10, max(1, len(resid) // 5))],
                               return_df=True)
        jb_stat, jb_p, _, _ = jarque_bera(resid)
        diagnostics: Dict[str, Any] = {
            "ljung_box_stat": round(float(lb_df["lb_stat"].iloc[0]), 4),
            "ljung_box_p_value": round(float(lb_df["lb_pvalue"].iloc[0]), 4),
            "residual_mean": round(float(resid.mean()), 4),
            "residual_std": round(float(resid.std()), 4),
            "jarque_bera_stat": round(float(jb_stat), 4),
            "jarque_bera_p_value": round(float(jb_p), 4),
            "is_white_noise": bool(lb_df["lb_pvalue"].iloc[0] > 0.05),
        }
        logger.info("status=diagnostics %s", diagnostics)
        return diagnostics

    def save_model(self, path: str) -> None:
        """Pickle the fitted model state to disk."""
        if self._results is None:
            raise RuntimeError("nothing to save; model is not fitted")
        payload = {
            "order": self.order,
            "results": self._results,
            "train_index": self._train_index,
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("status=model_saved path=%s", out)

    def load_model(self, path: str) -> "ARIMAPredictor":
        """Restore a previously saved model into this instance."""
        with open(Path(path), "rb") as fh:
            payload = pickle.load(fh)
        self.order = tuple(payload["order"])
        self._results = payload["results"]
        self._train_index = payload.get("train_index")
        logger.info("status=model_loaded path=%s order=%s", path, self.order)
        return self

    @staticmethod
    def adf_is_stationary(series: "pd.DataFrame | pd.Series",
                          alpha: float = 0.05) -> bool:
        """Convenience ADF check used to pick differencing order upstream."""
        close = ARIMAPredictor._to_series(series)
        return bool(adfuller(close)[1] < alpha)
