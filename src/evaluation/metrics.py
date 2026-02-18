"""Evaluation metrics for out-of-time housing price prediction.

All metrics operate on **original-scale** (pounds sterling) predictions
so that reported numbers are directly interpretable.  The caller is
responsible for back-transforming log-space predictions before passing
them here (use :meth:`HousingPredictor.predict_price`).

Primary metric: **Median Absolute Percentage Error (MdAPE)** – robust to
outliers and gives % accuracy consistency across price ranges, which
aligns with the log-transform rationale.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> Dict[str, float]:
    """Compute a standard suite of regression metrics on price-scale values.

    Args:
        y_true: Ground-truth sale prices in pounds sterling.
        y_pred: Predicted sale prices in pounds sterling.
        label: Optional label for log output (e.g. ``"val"`` or ``"test"``).

    Returns:
        Dictionary with the following keys:

        - ``rmse``      – Root Mean Squared Error (£).
        - ``mae``       – Mean Absolute Error (£).
        - ``mape``      – Mean Absolute Percentage Error (%).
        - ``mdape``     – Median Absolute Percentage Error (%) [primary].
        - ``r2``        – Coefficient of Determination.
        - ``within_10pct`` – Fraction of predictions within 10 % of truth.
        - ``within_20pct`` – Fraction of predictions within 20 % of truth.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different shapes.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Guard against true-zero prices (post outlier filter this is very rare).
    nonzero = y_true != 0
    ape = np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]) * 100
    mape = float(np.mean(ape))
    mdape = float(np.median(ape))

    within_10 = float(np.mean(ape <= 10.0))
    within_20 = float(np.mean(ape <= 20.0))

    results: Dict[str, float] = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "mdape": mdape,
        "r2": r2,
        "within_10pct": within_10,
        "within_20pct": within_20,
    }

    prefix = f"[{label}] " if label else ""
    logger.info(
        "%sRMSE=£%.0f | MAE=£%.0f | MdAPE=%.2f%% | R²=%.4f | "
        "within10%%=%.1f%% | within20%%=%.1f%%",
        prefix,
        rmse,
        mae,
        mdape,
        r2,
        within_10 * 100,
        within_20 * 100,
    )

    return results


def metrics_to_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert a dict of {split_name: metrics_dict} into a tidy DataFrame.

    Args:
        results: Mapping from split label (e.g. ``"val"``) to the dict
            returned by :func:`compute_metrics`.

    Returns:
        DataFrame with splits as rows and metric names as columns.
    """
    return pd.DataFrame(results).T.rename_axis("split")
