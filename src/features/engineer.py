"""Feature engineering pipeline for the housing price dataset.

This module derives domain-specific features from the preprocessed
DataFrame and encodes categorical variables.  The
:class:`FeatureEngineer` is sklearn-compatible (``fit`` / ``transform``)
and must be fitted *only* on training data.

Stateful parameters learned during ``fit``:
    - ``floor_level_median_``: Median cleaned floor-level used to fill
      missing values.
    - ``cat_encoder_``: :class:`sklearn.preprocessing.OrdinalEncoder`
      fitted on training categorical columns, mapping unseen values to
      ``-1`` at inference time.
    - ``cat_columns_``: List of categorical column names seen at fit time.
"""

import logging
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EFFICIENCY_MAPPING = {
    "Very Poor": 1,
    "Poor": 2,
    "Average": 3,
    "Good": 4,
    "Very Good": 5,
}

_FABRIC_EFF_COLS = [
    "roof_energy_eff",
    "walls_energy_eff",
    "windows_energy_eff",
    "floor_energy_eff",
]

_SYSTEM_EFF_COLS = [
    "mainheat_energy_eff",
    "mainheatc_energy_eff",
    "hot_water_energy_eff",
    "lighting_energy_eff",
]

_ENV_EFF_COLS = [
    "hot_water_env_eff",
    "windows_env_eff",
    "walls_env_eff",
    "roof_env_eff",
    "mainheat_env_eff",
    "mainheatc_env_eff",
    "lighting_env_eff",
]

_ENERGY_COST_INPUT_COLS = [
    "lighting_cost_current",
    "heating_cost_current",
    "hot_water_cost_current",
    "lighting_cost_potential",
    "heating_cost_potential",
    "hot_water_cost_potential",
    "potential_energy_efficiency",
    "current_energy_efficiency",
    "environment_impact_potential",
    "environment_impact_current",
    "co2_emissions_potential",
    "co2_emissions_current",
]

_PROPERTY_METRIC_INPUT_COLS = [
    "co2_emiss_curr_per_floor_area",
    "total_floor_area",
    "floor_height",
]

_DESCRIPTION_DROP_COLS = [
    "walls_description",
    "windows_description",
    "roof_description",
    "lodgement_year",
    "days_lodgement_to_sale",
    "mains_gas_flag",
    "secondheat_description",
    "main_heating_controls",
]

_REDUNDANT_RATING_COLS = [
    "current_energy_rating",
    "potential_energy_rating",
    "energy_consumption_current",
    "energy_consumption_potential",
    "energy_tariff",
    "number_heated_rooms",
]

_FLOOR_LEVEL_MAP = {
    "ground": 0,
    "ground floor": 0,
    "0": 0,
    "nodata!": np.nan,
    "no data!": np.nan,
    "1st": 1,
    "1": 1,
    "2nd": 2,
    "2": 2,
    "3rd": 3,
    "3": 3,
    "4th": 4,
    "4": 4,
    "5th": 5,
    "5": 5,
    "6th": 6,
    "6": 6,
    "7th": 7,
    "7": 7,
    "8th": 8,
    "8": 8,
    "9th": 9,
    "9": 9,
    "10th": 10,
    "10": 10,
    "11": 11,
    "mid floor": np.nan,
    "top floor": np.nan,
    "basement": -1,
    "-1": -1,
    "20+": 20,
    "21st or above": 21,
}

_KNOWN_TRANSACTION_TYPES = {
    "marketed sale",
    "rental",
    "new dwelling",
    "non marketed sale",
}


# ---------------------------------------------------------------------------
# Helper functions (pure, no state)
# ---------------------------------------------------------------------------


def _extract_build_year(value: object) -> Optional[float]:
    """Parse a raw ``construction_age_band`` value into a numeric build year.

    Handles formats including:
    - Exact years (``"2021"``)
    - Ranges (``"1900-1929"`` → midpoint)
    - ``"before 1900"`` → 1890
    - ``"2007 onwards"`` → 2007
    - ``"NO DATA"`` / ``"INVALID"`` → NaN

    Args:
        value: Raw cell value.

    Returns:
        Numeric build year or ``np.nan``.
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    if s.isdigit():
        return int(s)

    if "before 1900" in s.lower():
        return 1890.0

    m = re.search(r"(\d{4})-(\d{4})", s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2

    m = re.search(r"(\d{4})\s+onwards", s, re.IGNORECASE)
    if m:
        return float(m.group(1))

    if "no data" in s.lower() or "invalid" in s.lower():
        return np.nan

    return np.nan


def _clean_windows(desc: object) -> str:
    """Classify a windows description into a coarse category.

    Args:
        desc: Raw windows description string.

    Returns:
        One of ``"Double/high perf"``, ``"Triple"``, ``"Single"``,
        or ``"Other"``.
    """
    d = str(desc).lower()
    if "double" in d or "high performance" in d:
        return "Double/high perf"
    if "triple" in d:
        return "Triple"
    if "single" in d:
        return "Single"
    return "Other"


def _clean_roof(desc: object) -> str:
    """Classify a roof description into a coarse category.

    Args:
        desc: Raw roof description string.

    Returns:
        One of ``"No_External_Roof"``, ``"Pitched"``, ``"Flat"``,
        ``"Roof_Room"``, or ``"Other"``.
    """
    d = str(desc).lower()
    if "another dwelling above" in d:
        return "No_External_Roof"
    if "pitched" in d:
        return "Pitched"
    if "flat" in d:
        return "Flat"
    if "roof room" in d:
        return "Roof_Room"
    return "Other"


def _clean_walls(desc: object) -> str:
    """Classify a walls description into a coarse category.

    Args:
        desc: Raw walls description string.

    Returns:
        One of ``"Solid brick"``, ``"Cavity wall insulated"``,
        ``"Cavity wall uninsulated"``, ``"System built"``, or ``"Other"``.
    """
    d = str(desc).lower()
    if "solid brick" in d:
        return "Solid brick"
    if "cavity wall" in d:
        return (
            "Cavity wall insulated"
            if ("insulated" in d or "filled" in d)
            else "Cavity wall uninsulated"
        )
    if "system built" in d:
        return "System built"
    return "Other"


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive domain features and encode categoricals.

    The transformer is stateful: it learns median floor-level and the
    categorical encoder vocabulary from training data.

    Args:
        current_year: Reference year for computing ``property_age``.
            Defaults to 2026.
    """

    def __init__(self, current_year: int = 2026) -> None:
        self.current_year = current_year

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Learn floor-level median, district wealth index, and categorical vocabularies.

        Args:
            df: Preprocessed training DataFrame (output of
                :class:`HousingPreprocessor`).
            y: Training target (``log_sale_price``) aligned with ``df``.
                When provided, a district-level median target encoding is
                computed and stored as ``district_wealth_``.  Must be a
                pandas Series or array with the same index/length as ``df``.

        Returns:
            Fitted transformer (self).
        """
        df_tmp = self._apply_stateless_transforms(df.copy())

        # Learn floor-level median from training data.
        self.floor_level_median_: float = float(
            df_tmp["floor_level_clean"].median()
        )
        logger.info("Learned floor-level median: %.1f", self.floor_level_median_)

        # Identify remaining categorical columns and fit encoder.
        df_tmp["floor_level_clean"] = df_tmp["floor_level_clean"].fillna(
            self.floor_level_median_
        )

        # District wealth index: median log_sale_price per district (train only).
        # Uses positional alignment so index mismatches are not a risk.
        if y is not None and "district" in df_tmp.columns:
            _y = pd.Series(
                np.asarray(y),
                index=df_tmp.index,
                name="target",
            )
            self.district_wealth_: dict = (
                _y.groupby(df_tmp["district"]).median().to_dict()
            )
            self.district_wealth_global_median_: float = float(_y.median())
            df_tmp = df_tmp.drop(columns=["district"])
            logger.info(
                "Learned district wealth index for %d districts.",
                len(self.district_wealth_),
            )
        else:
            self.district_wealth_ = {}
            self.district_wealth_global_median_ = float("nan")

        self.cat_columns_: List[str] = self._get_cat_columns(df_tmp)
        self.cat_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        if self.cat_columns_:
            self.cat_encoder_.fit(df_tmp[self.cat_columns_].astype(str))
            logger.info(
                "OrdinalEncoder fitted on %d categorical columns.",
                len(self.cat_columns_),
            )

        return self

    def transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Apply all feature engineering steps.

        Args:
            df: Preprocessed DataFrame (output of
                :class:`HousingPreprocessor`).
            y: Ignored; present for sklearn compatibility.

        Returns:
            Feature-engineered DataFrame with only numeric columns (plus
            ``log_sale_price`` if present).
        """
        df = df.copy()

        # Stateless transformations (identical for train, val, test).
        df = self._apply_stateless_transforms(df)

        # Stateful: impute floor level using training median.
        df["floor_level_clean"] = df["floor_level_clean"].fillna(
            self.floor_level_median_
        )

        # District wealth index: replace district string with numeric median price.
        # Unseen districts fall back to the global training median.
        if self.district_wealth_ and "district" in df.columns:
            df["district_wealth_index"] = (
                df["district"]
                .map(self.district_wealth_)
                .fillna(self.district_wealth_global_median_)
            )
            df = df.drop(columns=["district"])

        # Encode remaining categoricals.
        if self.cat_columns_:
            present = [c for c in self.cat_columns_ if c in df.columns]
            df[present] = self.cat_encoder_.transform(
                df[present].astype(str)
            )

        logger.info(
            "Feature engineering complete: %d rows × %d columns.", *df.shape
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_stateless_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all deterministic (non-stateful) feature transforms.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        df = self._compute_energy_cost_features(df)
        df = self._compute_property_metrics(df)
        df = self._encode_efficiency_ratings(df)
        df = self._drop_redundant_rating_cols(df)
        df = self._clean_description_cols(df)
        df = self._extract_construction_age(df)
        df = self._clean_floor_level(df)
        df = self._clean_transaction_type(df)
        return df

    @staticmethod
    def _compute_energy_cost_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive energy-cost saving and efficiency features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with derived energy features; raw input columns dropped.
        """
        df["current_energy_cost"] = (
            df["lighting_cost_current"]
            + df["heating_cost_current"]
            + df["hot_water_cost_current"]
        )
        df["potential_energy_cost"] = (
            df["lighting_cost_potential"]
            + df["heating_cost_potential"]
            + df["hot_water_cost_potential"]
        )
        df["energy_cost_savings_pct"] = (
            df["current_energy_cost"] - df["potential_energy_cost"]
        ) / df["current_energy_cost"].replace(0, np.nan)

        df["energy_efficiency_gain_pct"] = (
            (df["potential_energy_efficiency"] - df["current_energy_efficiency"])
            / df["current_energy_efficiency"].replace(0, np.nan)
        ).fillna(0)

        df["environmental_improvement"] = (
            df["environment_impact_current"] - df["environment_impact_potential"]
        )
        df["co2_reduction"] = (
            df["co2_emissions_current"] - df["co2_emissions_potential"]
        )

        cols_to_drop = _ENERGY_COST_INPUT_COLS + [
            "current_energy_cost",
            "potential_energy_cost",
        ]
        return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    @staticmethod
    def _compute_property_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Derive property volume and total CO2 emission features.

        Args:
            df: Input DataFrame (must contain ``total_floor_area``,
                ``floor_height``, and ``co2_emiss_curr_per_floor_area``).

        Returns:
            DataFrame with derived metrics; raw input columns dropped.
        """
        df["property_volume"] = df["total_floor_area"] * df["floor_height"]
        df["total_co2_emissions"] = (
            df["co2_emiss_curr_per_floor_area"] * df["total_floor_area"]
        )
        return df.drop(
            columns=[c for c in _PROPERTY_METRIC_INPUT_COLS if c in df.columns]
        )

    @staticmethod
    def _encode_efficiency_ratings(df: pd.DataFrame) -> pd.DataFrame:
        """Map EPC efficiency labels to ordinal integers and build indices.

        Encoding:
            ``Very Poor=1, Poor=2, Average=3, Good=4, Very Good=5``

        Derived indices:
            - ``fabric_index``: Weighted combination of roof (×1.5),
              walls (×1.5), floor, and windows ratings.
            - ``systems_index``: Sum of heating, controls, hot-water, and
              lighting (×0.5) ratings.

        Missing values are imputed via the row-wise mean of the relevant
        group before computing the index.

        Args:
            df: Input DataFrame containing raw EPC efficiency columns.

        Returns:
            DataFrame with efficiency indices; raw EPC columns dropped.
        """
        all_eff_cols = _FABRIC_EFF_COLS + _SYSTEM_EFF_COLS

        for col in all_eff_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().map(_EFFICIENCY_MAPPING)

        # Row-wise imputation within each group.
        fabric_present = [c for c in _FABRIC_EFF_COLS if c in df.columns]
        system_present = [c for c in _SYSTEM_EFF_COLS if c in df.columns]

        if fabric_present:
            fabric_avg = df[fabric_present].mean(axis=1).round().fillna(3)
            for col in fabric_present:
                df[col] = df[col].fillna(fabric_avg).astype(int)

            df["fabric_index"] = (
                df["roof_energy_eff"] * 1.5
                + df["walls_energy_eff"] * 1.5
                + df["floor_energy_eff"]
                + df["windows_energy_eff"]
            )

        if system_present:
            system_avg = df[system_present].mean(axis=1).round().fillna(3)
            for col in system_present:
                df[col] = df[col].fillna(system_avg).astype(int)

            df["systems_index"] = (
                df["mainheat_energy_eff"]
                + df["mainheatc_energy_eff"]
                + df["hot_water_energy_eff"]
                + df["lighting_energy_eff"] * 0.5
            )

        # Drop env-eff columns (regulatory signal, not market signal).
        all_drop = all_eff_cols + _ENV_EFF_COLS
        return df.drop(columns=[c for c in all_drop if c in df.columns])

    @staticmethod
    def _drop_redundant_rating_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Remove energy-rating columns superseded by derived features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with redundant rating columns removed.
        """
        return df.drop(
            columns=[c for c in _REDUNDANT_RATING_COLS if c in df.columns]
        )

    @staticmethod
    def _clean_description_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Convert free-text description columns into coarse categories.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with ``walls_clean``, ``windows_clean``, and
            ``roof_clean`` categorical columns; raw descriptions dropped.
        """
        if "walls_description" in df.columns:
            df["walls_clean"] = df["walls_description"].fillna("Other").apply(
                _clean_walls
            )
        if "windows_description" in df.columns:
            df["windows_clean"] = df["windows_description"].fillna("Other").apply(
                _clean_windows
            )
        if "roof_description" in df.columns:
            df["roof_clean"] = df["roof_description"].fillna("Other").apply(
                _clean_roof
            )
        return df.drop(
            columns=[c for c in _DESCRIPTION_DROP_COLS if c in df.columns]
        )

    def _extract_construction_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse ``construction_age_band`` and compute ``property_age``.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with ``property_age`` added; raw age-band columns
            dropped.
        """
        if "construction_age_band" in df.columns:
            df["build_year"] = df["construction_age_band"].apply(
                _extract_build_year
            )
            df["property_age"] = self.current_year - df["build_year"]
            df = df.drop(columns=["construction_age_band", "build_year"])
        return df

    @staticmethod
    def _clean_floor_level(df: pd.DataFrame) -> pd.DataFrame:
        """Map raw ``floor_level`` strings to numeric values.

        Missing / unrecognised values are set to NaN and imputed later
        with the training median (stateful step).

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with ``floor_level_clean``; raw column dropped.
        """
        if "floor_level" in df.columns:
            df["floor_level_clean"] = (
                df["floor_level"]
                .astype(str)
                .str.lower()
                .map(lambda x: _FLOOR_LEVEL_MAP.get(x, np.nan))
            )
            df = df.drop(columns=["floor_level"])
        return df

    @staticmethod
    def _clean_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
        """Collapse rare transaction types into an 'Other' category.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with ``transaction_type_clean``; raw column dropped.
        """
        if "transaction_type" in df.columns:
            df["transaction_type_clean"] = df["transaction_type"].apply(
                lambda x: x if x in _KNOWN_TRANSACTION_TYPES else "Other"
            )
            df = df.drop(columns=["transaction_type"])
        return df

    @staticmethod
    def _get_cat_columns(df: pd.DataFrame) -> List[str]:
        """Return names of remaining object/category columns, excluding the target.

        Args:
            df: Transformed DataFrame.

        Returns:
            List of categorical column names.
        """
        exclude = {"log_sale_price"}
        return [
            c
            for c in df.select_dtypes(include=["object", "category", "str"]).columns
            if c not in exclude
        ]
