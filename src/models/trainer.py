"""Model training classes for housing price prediction.

Each trainer wraps a specific algorithm with a consistent
``fit`` / ``predict`` interface.  All models are trained on the
log-transformed target (``log_sale_price``); back-transformation to
pounds sterling is handled by :mod:`src.models.predictor`.

Available trainers:
    - :class:`LGBMTrainer` – LightGBM gradient boosting.
    - :class:`XGBTrainer`  – XGBoost gradient boosting.
    - :class:`RidgeTrainer` – Ridge linear regression (baseline).
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


class LGBMTrainer(BaseEstimator, RegressorMixin):
    """LightGBM gradient-boosting regressor with early stopping.

    Args:
        n_estimators: Maximum number of boosting rounds.
        learning_rate: Step size shrinkage.
        num_leaves: Maximum number of leaves per tree.
        min_child_samples: Minimum data points in a leaf.
        subsample: Row sub-sampling ratio per tree.
        colsample_bytree: Feature sub-sampling ratio per tree.
        early_stopping_rounds: Stop if validation loss does not improve
            for this many consecutive rounds.
        random_state: Random seed for reproducibility.
        extra_params: Additional keyword arguments forwarded to the
            LightGBM ``LGBMRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.extra_params = extra_params or {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LGBMTrainer":
        """Train the LightGBM model with optional early stopping.

        Args:
            X_train: Training feature matrix.
            y_train: Log-transformed training target.
            X_val: Validation feature matrix for early stopping.
                Required when early stopping is used.
            y_val: Validation target for early stopping.

        Returns:
            Fitted trainer (self).
        """
        import lightgbm as lgb  # lazy import to avoid hard dependency at module load

        params: Dict[str, Any] = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=-1,
            **self.extra_params,
        )

        self.model_ = lgb.LGBMRegressor(**params)

        callbacks = [lgb.log_evaluation(period=100)]
        fit_kwargs: Dict[str, Any] = {"callbacks": callbacks}

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=False
                )
            )

        self.model_.fit(X_train, y_train, **fit_kwargs)
        logger.info(
            "LightGBM trained. Best iteration: %s",
            getattr(self.model_, "best_iteration_", "N/A"),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions in log-space.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted log-prices.
        """
        return self.model_.predict(X)

    @property
    def feature_importances_(self) -> pd.Series:
        """Return feature importances (gain) as a sorted Series.

        Returns:
            Series mapping feature name to importance score.
        """
        return pd.Series(
            self.model_.feature_importances_,
            index=self.model_.feature_name_,
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


class XGBTrainer(BaseEstimator, RegressorMixin):
    """XGBoost gradient-boosting regressor with early stopping.

    Args:
        n_estimators: Maximum number of boosting rounds.
        learning_rate: Step size shrinkage (``eta``).
        max_depth: Maximum tree depth.
        subsample: Row sub-sampling ratio per tree.
        colsample_bytree: Feature sub-sampling ratio per tree.
        early_stopping_rounds: Stop if validation loss does not improve
            for this many consecutive rounds.
        random_state: Random seed for reproducibility.
        extra_params: Additional keyword arguments forwarded to
            ``XGBRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.extra_params = extra_params or {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBTrainer":
        """Train the XGBoost model with optional early stopping.

        Args:
            X_train: Training feature matrix.
            y_train: Log-transformed training target.
            X_val: Validation feature matrix for early stopping.
            y_val: Validation target for early stopping.

        Returns:
            Fitted trainer (self).
        """
        from xgboost import XGBRegressor  # lazy import

        params: Dict[str, Any] = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            tree_method="hist",
            verbosity=0,
            **self.extra_params,
        )

        self.model_ = XGBRegressor(**params)

        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
            self.model_.set_params(early_stopping_rounds=self.early_stopping_rounds)

        self.model_.fit(X_train, y_train, **fit_kwargs)
        logger.info(
            "XGBoost trained. Best iteration: %s",
            getattr(self.model_, "best_iteration", "N/A"),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions in log-space.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted log-prices.
        """
        return self.model_.predict(X)

    @property
    def feature_importances_(self) -> pd.Series:
        """Return feature importances (gain) as a sorted Series.

        Returns:
            Series mapping feature name to importance score.
        """
        names = (
            self.model_.get_booster().feature_names
            or [f"f{i}" for i in range(len(self.model_.feature_importances_))]
        )
        return pd.Series(
            self.model_.feature_importances_, index=names
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Ridge (baseline)
# ---------------------------------------------------------------------------


class RidgeTrainer(BaseEstimator, RegressorMixin):
    """Ridge regression baseline with standard scaling.

    Features are z-score scaled before fitting because Ridge is
    sensitive to feature magnitude.

    Args:
        alpha: L2 regularisation strength.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "RidgeTrainer":
        """Fit the Ridge regressor.

        Args:
            X_train: Training feature matrix.
            y_train: Log-transformed training target.
            X_val: Ignored (included for API consistency).
            y_val: Ignored (included for API consistency).

        Returns:
            Fitted trainer (self).
        """
        self.imputer_ = SimpleImputer(strategy="median")
        X_imputed = self.imputer_.fit_transform(X_train)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X_scaled, y_train)
        logger.info("Ridge trained with alpha=%.4f.", self.alpha)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions in log-space.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted log-prices.
        """
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        return self.model_.predict(X_scaled)

    def get_feature_weights(self, feature_names: pd.Index) -> pd.Series:
        """Return Ridge coefficients as a sorted Series.

        Args:
            feature_names: Index of feature names matching training columns.

        Returns:
            Series mapping feature name to coefficient magnitude (sorted).
        """
        return pd.Series(
            np.abs(self.model_.coef_), index=feature_names
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

_TRAINER_REGISTRY: Dict[str, type] = {
    "lgbm": LGBMTrainer,
    "xgb": XGBTrainer,
    "ridge": RidgeTrainer,
}


def get_trainer(name: str, **kwargs: Any) -> Any:
    """Instantiate a trainer by name.

    Args:
        name: One of ``"lgbm"``, ``"xgb"``, or ``"ridge"``.
        **kwargs: Passed to the trainer's constructor.

    Returns:
        Instantiated (unfitted) trainer.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    if name not in _TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown trainer '{name}'. Choose from: {list(_TRAINER_REGISTRY)}"
        )
    return _TRAINER_REGISTRY[name](**kwargs)
