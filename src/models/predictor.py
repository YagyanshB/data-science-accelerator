"""Prediction utilities for the housing price models.

All models produce predictions in log-space (``log1p(sale_price)``).
This module provides helpers to back-transform those predictions into
pounds sterling (``expm1``) and to run end-to-end inference from raw
feature data.
"""

import logging
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol – any fitted trainer is compatible
# ---------------------------------------------------------------------------


@runtime_checkable
class _FittedModel(Protocol):
    """Structural type for any fitted trainer with a ``predict`` method."""

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class HousingPredictor:
    """Wraps a fitted model to produce sale-price predictions in pounds.

    The predictor handles the log → price back-transformation
    (``np.expm1``) so that callers always work in interpretable units.

    Args:
        model: Any fitted object that exposes a ``predict(X)`` method
            returning log-space predictions (e.g. :class:`LGBMTrainer`,
            :class:`XGBTrainer`, :class:`RidgeTrainer`).
    """

    def __init__(self, model: _FittedModel) -> None:
        if not isinstance(model, _FittedModel):
            raise TypeError(
                "model must have a predict(X) method. "
                f"Got {type(model).__name__}."
            )
        self.model = model

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw model predictions in log-space.

        Args:
            X: Feature matrix with the same columns used during training.

        Returns:
            1-D array of log-transformed price predictions.
        """
        log_preds = self.model.predict(X)
        logger.debug("predict_log: min=%.3f, max=%.3f", log_preds.min(), log_preds.max())
        return log_preds

    def predict_price(self, X: pd.DataFrame) -> np.ndarray:
        """Return sale-price predictions in pounds sterling.

        Applies ``np.expm1`` to reverse the ``np.log1p`` target
        transformation applied during training.

        Args:
            X: Feature matrix with the same columns used during training.

        Returns:
            1-D array of predicted sale prices (£).
        """
        log_preds = self.predict_log(X)
        prices = np.expm1(log_preds)
        logger.info(
            "predict_price: median=£%.0f, min=£%.0f, max=£%.0f",
            np.median(prices),
            prices.min(),
            prices.max(),
        )
        return prices

    def predict_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with both log and price predictions.

        Args:
            X: Feature matrix.

        Returns:
            DataFrame with columns:
                - ``predicted_log_price``: Raw model output.
                - ``predicted_sale_price``: Back-transformed price in £.
        """
        log_preds = self.predict_log(X)
        return pd.DataFrame(
            {
                "predicted_log_price": log_preds,
                "predicted_sale_price": np.expm1(log_preds),
            },
            index=X.index,
        )
