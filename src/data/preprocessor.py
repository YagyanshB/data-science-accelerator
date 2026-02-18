"""Raw data preprocessing pipeline for the housing price dataset.

This module transforms the raw Excel data into a clean, analysis-ready
DataFrame.  The :class:`HousingPreprocessor` is sklearn-compatible
(``fit`` / ``transform``) so it can be embedded in a Pipeline and fitted
*only* on training data, preventing data leakage.

Stateful parameters learned during ``fit``:
    - ``price_lower_bound_``: 0.5th-percentile sale price used to filter
      anomalous transactions from the training fold.
    - ``floor_height_median_``: Median floor height used to impute missing
      values in both train and inference data.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column groups – centralised so they're easy to update
# ---------------------------------------------------------------------------

# Columns dropped because they have > 90 % missing values.
_HIGH_NULL_COLS = [
    "sheating_energy_eff",
    "sheating_env_eff",
    "floor_env_eff",
    "flat_storey_count",
]

# Columns dropped because they are zero-variance or redundant given others.
_ZERO_VARIANCE_COLS = ["record_status", "town_or_city", "county"]

# Columns dropped because of extreme class imbalance / no predictive signal.
_IMBALANCED_COLS = [
    "report_type",
    "mechanical_ventilation",
    "wind_turbine_count",
    "ppd_category_type",
    "heat_loss_corridor",
    "number_open_fireplaces",
    "unheated_corridor_length",
]

# Raw datetime columns consumed during date-feature extraction.
_DATE_COLS = [
    "date_of_transfer",
    "inspection_date",
    "lodgement_date",
    "lodgement_datetime",
]


class HousingPreprocessor(BaseEstimator, TransformerMixin):
    """Clean and standardise the raw housing dataset.

    The transformer performs column renaming, null/imbalance dropping,
    datetime feature extraction, and log-transforms the sale price target.

    Args:
        price_lower_quantile: Quantile used to determine the minimum
            acceptable sale price.  Rows below this threshold in the
            *training* fold are considered anomalous and excluded.
            Defaults to 0.005 (0.5th percentile).
        current_year: Year used to compute ``property_age``.
            Defaults to 2026.
    """

    def __init__(
        self,
        price_lower_quantile: float = 0.005,
        current_year: int = 2026,
    ) -> None:
        self.price_lower_quantile = price_lower_quantile
        self.current_year = current_year

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "HousingPreprocessor":
        """Learn parameters from training data only.

        Args:
            df: Raw training DataFrame (output of :class:`DataIngestor`).
            y: Ignored; present for sklearn compatibility.

        Returns:
            Fitted transformer (self).
        """
        df_prep = self._rename_and_lower(df)
        df_prep = self._drop_cols(df_prep, _HIGH_NULL_COLS + _ZERO_VARIANCE_COLS)

        self.price_lower_bound_: float = df_prep["sale_price"].quantile(
            self.price_lower_quantile
        )
        logger.info(
            "Learned price lower bound: £%.0f (%.3f-quantile)",
            self.price_lower_bound_,
            self.price_lower_quantile,
        )

        # Learn floor-height median on the *non-anomalous* training rows.
        df_filt = df_prep[df_prep["sale_price"] >= self.price_lower_bound_]
        self.floor_height_median_: float = float(df_filt["floor_height"].median())
        logger.info("Learned floor-height median: %.3f", self.floor_height_median_)

        return self

    def transform(
        self,
        df: pd.DataFrame,
        filter_outliers: bool = False,
    ) -> pd.DataFrame:
        """Apply all preprocessing steps to a DataFrame.

        Args:
            df: Raw DataFrame (output of :class:`DataIngestor`).
            filter_outliers: When ``True`` (training fold only), rows with
                ``sale_price`` below the fitted lower bound are removed.
                Set to ``False`` for validation / test folds.

        Returns:
            Preprocessed DataFrame containing ``log_sale_price`` and
            ``sale_year``, ready for :class:`FeatureEngineer`.
        """
        df = df.copy()

        # 1. Rename & lower-case columns
        df = self._rename_and_lower(df)

        # 2. Drop structurally useless columns
        df = self._drop_cols(df, _HIGH_NULL_COLS + _ZERO_VARIANCE_COLS, silent=True)

        # 3. Optionally remove anomalous price rows (training only)
        if filter_outliers:
            before = len(df)
            df = df[df["sale_price"] >= self.price_lower_bound_].copy()
            logger.info(
                "Outlier filter removed %d rows (sale_price < £%.0f).",
                before - len(df),
                self.price_lower_bound_,
            )

        # 4. Log-transform the target; drop raw price
        df["log_sale_price"] = np.log1p(df["sale_price"])
        df = df.drop(columns=["sale_price"])

        # 5. Drop imbalanced / low-signal columns
        df = self._drop_cols(df, _IMBALANCED_COLS, silent=True)

        # 6. Parse datetimes and extract temporal features
        df = self._extract_date_features(df)

        # 7. Impute floor height using training median
        df["floor_height"] = df["floor_height"].fillna(self.floor_height_median_)

        logger.info(
            "Preprocessing complete: %d rows × %d columns.", *df.shape
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rename_and_lower(df: pd.DataFrame) -> pd.DataFrame:
        """Rename ambiguous columns and lower-case all column names.

        The raw data contains both a ``Property Type`` column (property
        structure code) and a ``PROPERTY_TYPE`` column (full description).
        They are renamed to avoid collision after lower-casing.

        Args:
            df: Raw DataFrame.

        Returns:
            DataFrame with disambiguated, lower-cased column names.
        """
        df = df.rename(
            columns={
                "Property Type": "property_structure",
                "PROPERTY_TYPE": "property_category",
            }
        )
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        return df

    @staticmethod
    def _drop_cols(
        df: pd.DataFrame,
        cols: list,
        silent: bool = False,
    ) -> pd.DataFrame:
        """Drop columns that exist in ``cols``, ignoring missing ones.

        Args:
            df: Input DataFrame.
            cols: Column names to drop.
            silent: Suppress log output when ``True``.

        Returns:
            DataFrame with specified columns removed.
        """
        to_drop = [c for c in cols if c in df.columns]
        if to_drop and not silent:
            logger.info("Dropping columns: %s", to_drop)
        return df.drop(columns=to_drop)

    @staticmethod
    def _extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetime columns and derive temporal features.

        Derived features:
            - ``sale_year``: Year of the sale (used for OOT splitting).
            - ``days_inspection_to_sale``: Days between EPC inspection and
              transfer, capturing compliance recency.

        All raw datetime columns are dropped after extraction.

        Args:
            df: DataFrame with raw date columns.

        Returns:
            DataFrame with datetime columns replaced by numeric features.
        """
        for col in _DATE_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "date_of_transfer" in df.columns:
            df["sale_year"] = df["date_of_transfer"].dt.year

        if "date_of_transfer" in df.columns and "inspection_date" in df.columns:
            df["days_inspection_to_sale"] = (
                df["date_of_transfer"] - df["inspection_date"]
            ).dt.days

        df = df.drop(
            columns=[c for c in _DATE_COLS if c in df.columns]
        )
        return df
