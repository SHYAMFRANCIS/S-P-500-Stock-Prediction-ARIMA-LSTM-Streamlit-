"""
MODULE 4: LSTM Deep Learning Forecasting Model
===============================================
Purpose: Deep learning-based time-series prediction using LSTM networks
with sequence preparation, regularization, and early stopping.

Features:
- Sequence preparation: Sliding window with configurable lookback
- LSTM architecture: Configurable layers, units, dropout
- Training: Adam optimizer, MSE loss, early stopping, LR reduction
- Evaluation: MAE/MSE/RMSE/R2/MAPE (same protocol as ARIMA)
- Model persistence: Save/load Keras models + scaler parameters
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """LSTM forecaster with sliding-window sequences and scaling."""

    def __init__(
        self,
        lookback_window: int = 60,
        units: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.lookback_window = int(lookback_window)
        self.units = list(units) if units is not None else [50, 50]
        self.dropout_rate = float(dropout_rate)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.scaler: Optional[MinMaxScaler] = None
        self.model: Optional[tf.keras.Model] = None
        self.set_seeds(self.seed)

    @staticmethod
    def set_seeds(seed: int = 42, deterministic: bool = False) -> None:
        """Seed python, numpy and tensorflow RNGs for reproducibility."""
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
        logger.info("status=seeds_set seed=%d deterministic=%s", seed, deterministic)

    @staticmethod
    def _to_array(data: "pd.Series | np.ndarray") -> np.ndarray:
        values = data.to_numpy(dtype=float) if isinstance(data, pd.Series) else np.asarray(
            data, dtype=float
        )
        values = values[~np.isnan(values)]
        if not np.isfinite(values).all():
            raise ValueError("input data contains NaN or Inf")
        return values.reshape(-1, 1)

    def prepare_sequences(
        self, data: "pd.Series | np.ndarray", fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale data and build sliding-window supervised pairs.

        Args:
            data: Raw series (pass TRAINING data only when ``fit_scaler``).
            fit_scaler: Fit a fresh MinMaxScaler before transforming.

        Returns:
            X of shape ``(samples, lookback_window, 1)`` and y of shape
            ``(samples, 1)``, both in scaled space.
        """
        values = self._to_array(data)
        if len(values) <= self.lookback_window:
            raise ValueError(
                f"need more than {self.lookback_window} observations "
                f"for lookback_window={self.lookback_window}, got {len(values)}"
            )
        if fit_scaler or self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(values)
            logger.info("status=scaler_fitted min=%.4f max=%.4f",
                        self.scaler.data_min_[0], self.scaler.data_max_[0])
        scaled = self.scaler.transform(values).astype("float32")

        X = np.lib.stride_tricks.sliding_window_view(scaled.ravel(), self.lookback_window)
        X = np.expand_dims(X[:-1], axis=-1)
        y = scaled[self.lookback_window:]
        logger.info("status=sequences_ready samples=%d lookback=%d", len(X),
                    self.lookback_window)
        return X.astype("float32"), y.astype("float32")

    def build_model(self) -> tf.keras.Model:
        """Stacked LSTM -> Dropout -> Dense(25) -> Dense(1), Adam/mse."""
        inputs = tf.keras.Input(shape=(self.lookback_window, 1))
        x = inputs
        for i, layer_units in enumerate(self.units):
            returns_seq = i < len(self.units) - 1
            x = tf.keras.layers.LSTM(layer_units, return_sequences=returns_seq)(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Dense(25, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        self.model = model
        logger.info("status=model_built units=%s params=%d",
                    self.units, model.count_params())
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        verbose: int = 0,
    ) -> tf.keras.callbacks.History:
        """Train with early stopping and plateau LR reduction."""
        if self.model is None:
            self.build_model()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(1, patience // 2), min_lr=1e-6
            ),
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )
        logger.info("status=trained epochs_run=%d final_val_loss=%.6f",
                    len(history.history["loss"]), history.history["val_loss"][-1])
        return history

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict and inverse-transform back to the original price scale."""
        if self.model is None:
            raise RuntimeError("model is not built/trained yet; call build_model()/fit()")
        if self.scaler is None:
            raise RuntimeError("scaler missing; call prepare_sequences() first")
        scaled_pred = self.model.predict(X_test, verbose=0)
        original = self.scaler.inverse_transform(scaled_pred.reshape(-1, 1))
        logger.info("status=predicted samples=%d", len(original))
        return original.ravel()

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """MAE/MSE/RMSE/R2/MAPE rounded to 4dp; MAPE ignores zero actuals."""
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        metrics = {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(float(np.sqrt(mse)), 4),
            "R2": round(float(r2_score(y_true, y_pred)), 4),
            "MAPE": round(float(_safe_mape(y_true, y_pred)), 4),
        }
        logger.info("status=evaluated metrics=%s", metrics)
        return metrics

    def save_model(self, path: str) -> None:
        """Persist the Keras model (.keras) plus scaler parameters (.json)."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("nothing to save; build/train the model first")
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        model_path = base.with_suffix(".keras")
        self.model.save(model_path)

        scaler_meta = {
            "feature_range": list(self.scaler.feature_range),
            "data_min": float(self.scaler.data_min_[0]),
            "data_max": float(self.scaler.data_max_[0]),
            "config": {
                "lookback_window": self.lookback_window,
                "units": self.units,
                "dropout_rate": self.dropout_rate,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
            },
        }
        scaler_path = base.with_name(base.stem + "_scaler.json")
        scaler_path.write_text(json.dumps(scaler_meta, indent=2), encoding="utf-8")
        logger.info("status=model_saved path=%s scaler=%s", model_path, scaler_path)

    def load_model(self, path: str) -> "LSTMPredictor":
        """Restore Keras model and scaler parameters saved by save_model."""
        base = Path(path)
        model_path = base.with_suffix(".keras")
        scaler_path = base.with_name(base.stem + "_scaler.json")

        self.model = tf.keras.models.load_model(model_path)
        meta = json.loads(scaler_path.read_text(encoding="utf-8"))

        self.scaler = MinMaxScaler(feature_range=tuple(meta["feature_range"]))
        ref = np.array([[meta["data_min"]], [meta["data_max"]]], dtype=float)
        self.scaler.fit(ref)

        cfg = meta.get("config", {})
        self.lookback_window = cfg.get("lookback_window", self.lookback_window)
        self.units = cfg.get("units", self.units)
        self.dropout_rate = cfg.get("dropout_rate", self.dropout_rate)
        self.learning_rate = cfg.get("learning_rate", self.learning_rate)
        logger.info("status=model_loaded path=%s", model_path)
        return self


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    nonzero = y_true != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0)
