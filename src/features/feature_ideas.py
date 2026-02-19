"""Interview cheat-sheet: additional engineered features NOT in the live pipeline.

Each function is self-contained and copy-paste ready.

WHERE TO INSERT (engineer.py):
    Stateless → add a call inside ``_apply_stateless_transforms()``
    Stateful  → add fit logic to ``FeatureEngineer.fit()``, call in ``transform()``

COLUMN STATE at the natural insertion point
(after ``_apply_stateless_transforms``, before categorical encoding):

    ✓ Available:   property_volume, property_age, fabric_index, systems_index,
                   energy_cost_savings_pct, energy_efficiency_gain_pct,
                   co2_reduction, total_co2_emissions, environmental_improvement,
                   floor_level_clean, walls_clean, windows_clean, roof_clean,
                   transaction_type_clean, sale_year, days_inspection_to_sale,
                   district, property_type, property_structure, property_category,
                   duration, number_habitable_rooms, multi_glaze_proportion,
                   fixed_lighting_outlets_count, low_energy_fixed_light_count,
                   log_sale_price  (target column, present on all splits)

    ✗ Dropped:     total_floor_area, floor_height, number_heated_rooms,
                   all raw energy-cost columns, all raw EPC rating columns

PREREQUISITE tags below call out when a pipeline change is needed first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Keep in sync with FeatureEngineer.current_year
CURRENT_YEAR = 2026


# ===========================================================================
# 1. PRICE-BASED FEATURES
# ===========================================================================


def add_price_per_sqm(df: pd.DataFrame) -> pd.DataFrame:
    """Price per square metre of total floor area.

    PREREQUISITE: ``total_floor_area`` must still be present.
        Remove it from ``_PROPERTY_METRIC_INPUT_COLS`` in engineer.py so
        ``_compute_property_metrics`` does not drop it before this runs.
        Call this function *before* ``_compute_property_metrics``.

    Args:
        df: DataFrame with ``log_sale_price`` and ``total_floor_area``.

    Returns:
        DataFrame with ``price_per_sqm`` added.

    Example:
        # In _apply_stateless_transforms, BEFORE _compute_property_metrics:
        # df = add_price_per_sqm(df)
    """
    sale_price = np.expm1(df["log_sale_price"])
    df["price_per_sqm"] = sale_price / df["total_floor_area"].replace(0, np.nan)
    return df


def add_price_per_room(df: pd.DataFrame) -> pd.DataFrame:
    """Price divided by number of habitable rooms.

    Args:
        df: DataFrame with ``log_sale_price`` and ``number_habitable_rooms``.

    Returns:
        DataFrame with ``price_per_room`` added.

    Example:
        # df = add_price_per_room(df)
    """
    sale_price = np.expm1(df["log_sale_price"])
    df["price_per_room"] = sale_price / df["number_habitable_rooms"].replace(0, np.nan)
    return df


def add_price_relative_to_district_median(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of this property's price to its district median.

    Values > 1 mean the property is pricier than the typical district sale.
    Uses ``district_wealth_index`` (median log_sale_price per district)
    computed statefully by ``FeatureEngineer``.

    Insert in ``FeatureEngineer.transform()`` *after* the
    ``district_wealth_index`` block so the column is already present.

    Args:
        df: DataFrame with ``log_sale_price`` and ``district_wealth_index``.

    Returns:
        DataFrame with ``price_relative_to_district`` added.

    Example:
        # In FeatureEngineer.transform(), after the district_wealth_index block:
        # df = add_price_relative_to_district_median(df)
    """
    df["price_relative_to_district"] = (
        np.expm1(df["log_sale_price"])
        / np.expm1(df["district_wealth_index"]).replace(0, np.nan)
    )
    return df


# ===========================================================================
# 2. PROPERTY AGE FEATURES
# ===========================================================================


def add_age_at_sale(df: pd.DataFrame) -> pd.DataFrame:
    """How old the property was at the point of sale (years).

    Derived as: ``sale_year - (CURRENT_YEAR - property_age)``
    i.e. ``sale_year - build_year``.

    Args:
        df: DataFrame with ``property_age`` and ``sale_year``.

    Returns:
        DataFrame with ``age_at_sale`` added.

    Example:
        # df = add_age_at_sale(df)
    """
    build_year = CURRENT_YEAR - df["property_age"]
    df["age_at_sale"] = df["sale_year"] - build_year
    return df


def add_age_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Coarse ordinal bucket of property age.

    Buckets (ordinal-encoded):
        0 → new (≤ 10 yrs)
        1 → 10–20 yrs
        2 → 20–50 yrs
        3 → 50+ yrs

    Args:
        df: DataFrame with ``property_age``.

    Returns:
        DataFrame with ``age_bucket`` (float) added.

    Example:
        # df = add_age_bucket(df)
    """
    df["age_bucket"] = pd.cut(
        df["property_age"],
        bins=[-np.inf, 10, 20, 50, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(float)
    return df


def add_is_period_property(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if the property was built before 1900.

    Args:
        df: DataFrame with ``property_age``.

    Returns:
        DataFrame with ``is_period_property`` (0/1 int) added.

    Example:
        # df = add_is_period_property(df)
    """
    build_year = CURRENT_YEAR - df["property_age"]
    df["is_period_property"] = (build_year < 1900).astype(int)
    return df


# ===========================================================================
# 3. ENERGY IMPROVEMENT FEATURES
# ===========================================================================


def add_energy_rating_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric gap between potential and current EPC efficiency score.

    PREREQUISITE: ``current_energy_efficiency`` and
        ``potential_energy_efficiency`` must still be present.
        They are currently dropped inside ``_compute_energy_cost_features``
        via ``_ENERGY_COST_INPUT_COLS``.  Apply this function *before* that
        step in ``_apply_stateless_transforms``.

    Args:
        df: DataFrame with ``potential_energy_efficiency`` and
            ``current_energy_efficiency``.

    Returns:
        DataFrame with ``energy_rating_gap`` added.

    Example:
        # In _apply_stateless_transforms, BEFORE _compute_energy_cost_features:
        # df = add_energy_rating_gap(df)
    """
    df["energy_rating_gap"] = (
        df["potential_energy_efficiency"] - df["current_energy_efficiency"]
    )
    return df


def add_cost_savings_potential(df: pd.DataFrame) -> pd.DataFrame:
    """Absolute annual running-cost saving if upgraded to potential rating (£/yr).

    PREREQUISITE: Raw cost columns must still be present (see
        ``_ENERGY_COST_INPUT_COLS``).  Apply *before* ``_compute_energy_cost_features``.

    Args:
        df: DataFrame with current and potential lighting/heating/hot-water cost columns.

    Returns:
        DataFrame with ``cost_savings_potential`` added.

    Example:
        # In _apply_stateless_transforms, BEFORE _compute_energy_cost_features:
        # df = add_cost_savings_potential(df)
    """
    current_total = (
        df["lighting_cost_current"]
        + df["heating_cost_current"]
        + df["hot_water_cost_current"]
    )
    potential_total = (
        df["lighting_cost_potential"]
        + df["heating_cost_potential"]
        + df["hot_water_cost_potential"]
    )
    df["cost_savings_potential"] = current_total - potential_total
    return df


def add_efficiency_score_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise mean of ``fabric_index`` and ``systems_index``.

    A single scalar summary of overall EPC efficiency across all building
    elements.  Call *after* ``_encode_efficiency_ratings`` which produces
    ``fabric_index`` and ``systems_index``.

    Args:
        df: DataFrame with ``fabric_index`` and/or ``systems_index``.

    Returns:
        DataFrame with ``efficiency_score_composite`` added.

    Example:
        # In _apply_stateless_transforms, AFTER _encode_efficiency_ratings:
        # df = add_efficiency_score_composite(df)
    """
    cols = [c for c in ["fabric_index", "systems_index"] if c in df.columns]
    if cols:
        df["efficiency_score_composite"] = df[cols].mean(axis=1)
    return df


# ===========================================================================
# 4. LOCATION FEATURES
# ===========================================================================


def add_district_avg_price(
    df: pd.DataFrame,
    district_map: dict,
    global_mean: float,
) -> pd.DataFrame:
    """Target-encode district with *mean* log_sale_price (stateful).

    NOTE: The pipeline already has ``district_wealth_index`` using *median*.
    Use this variant if mean encoding gives better CV score, or if you want
    to test both simultaneously.

    Fit snippet (in FeatureEngineer.fit()):
        self.dist_avg_map_    = y.groupby(df["district"]).mean().to_dict()
        self.dist_avg_global_ = float(y.mean())

    Args:
        df: DataFrame with ``district`` string column.
        district_map: Mapping district → mean log_sale_price (from fit).
        global_mean: Fallback value for districts unseen at fit time.

    Returns:
        DataFrame with ``district_avg_price`` added.

    Example:
        # Transform:
        # df = add_district_avg_price(df, self.dist_avg_map_, self.dist_avg_global_)
    """
    df["district_avg_price"] = df["district"].map(district_map).fillna(global_mean)
    return df


def add_district_transaction_volume(
    df: pd.DataFrame,
    volume_map: dict,
    global_volume: float,
) -> pd.DataFrame:
    """Number of training transactions per district (stateful).

    High volume → well-represented market; low volume → sparse/rural.
    Useful as a reliability/confidence weight for other district features.

    Fit snippet (in FeatureEngineer.fit()):
        self.dist_vol_map_    = df["district"].value_counts().to_dict()
        self.dist_vol_global_ = float(df["district"].value_counts().median())

    Args:
        df: DataFrame with ``district`` string column.
        volume_map: Mapping district → transaction count.
        global_volume: Fallback for unseen districts.

    Returns:
        DataFrame with ``district_transaction_volume`` added.

    Example:
        # Transform:
        # df = add_district_transaction_volume(df, self.dist_vol_map_, self.dist_vol_global_)
    """
    df["district_transaction_volume"] = (
        df["district"].map(volume_map).fillna(global_volume)
    )
    return df


def add_is_london(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if the district name contains 'london'.

    Must be applied *before* the ``district_wealth_index`` block in
    ``FeatureEngineer.transform()``, which drops the ``district`` column.

    Args:
        df: DataFrame with ``district`` string column.

    Returns:
        DataFrame with ``is_london`` (0/1 int) added.

    Example:
        # In FeatureEngineer.transform(), BEFORE the district_wealth_index block:
        # df = add_is_london(df)
    """
    df["is_london"] = (
        df["district"].str.lower().str.contains("london", na=False).astype(int)
    )
    return df


# ===========================================================================
# 5. TIME-BASED FEATURES
# ===========================================================================


def add_days_on_market(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy for days on market: EPC inspection-to-sale gap, clamped ≥ 0.

    ``days_inspection_to_sale`` is computed by ``HousingPreprocessor`` and is
    already in the DataFrame — this simply clips negatives (re-inspections
    close to sale date).

    Args:
        df: DataFrame with ``days_inspection_to_sale``.

    Returns:
        DataFrame with ``days_on_market`` added.

    Example:
        # df = add_days_on_market(df)
    """
    df["days_on_market"] = df["days_inspection_to_sale"].clip(lower=0)
    return df


def add_sale_month_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Sin/cos encoding of sale month to capture seasonality.

    PREREQUISITE: ``HousingPreprocessor._extract_date_features`` currently
        only extracts ``sale_year``.  Add the following line there:
            df["sale_month"] = df["date_of_transfer"].dt.month

    Args:
        df: DataFrame with ``sale_month`` (integers 1–12).

    Returns:
        DataFrame with ``sale_month_sin`` and ``sale_month_cos`` added;
        raw ``sale_month`` dropped.

    Example:
        # df = add_sale_month_cyclical(df)
    """
    df["sale_month_sin"] = np.sin(2 * np.pi * df["sale_month"] / 12)
    df["sale_month_cos"] = np.cos(2 * np.pi * df["sale_month"] / 12)
    df = df.drop(columns=["sale_month"], errors="ignore")
    return df


def add_is_q4_sale(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if the sale occurred in Q4 (October–December).

    PREREQUISITE: Same as ``add_sale_month_cyclical`` — requires ``sale_month``
        to be extracted in ``HousingPreprocessor``.

    Args:
        df: DataFrame with ``sale_month`` (integers 1–12).

    Returns:
        DataFrame with ``is_q4_sale`` (0/1 int) added.

    Example:
        # df = add_is_q4_sale(df)
    """
    df["is_q4_sale"] = (df["sale_month"] >= 10).astype(int)
    return df


def add_years_since_last_renovation(df: pd.DataFrame) -> pd.DataFrame:
    """Years elapsed since last recorded renovation.

    PREREQUISITE: Requires a ``last_renovation_year`` column not currently
        in the dataset.  Would need to be joined from an external source
        (e.g. planning-application records, VOA data).

    Args:
        df: DataFrame with ``last_renovation_year`` (numeric year).

    Returns:
        DataFrame with ``years_since_last_renovation`` added.

    Example:
        # df = add_years_since_last_renovation(df)
    """
    df["years_since_last_renovation"] = CURRENT_YEAR - df["last_renovation_year"]
    return df


# ===========================================================================
# 6. INTERACTION FEATURES
# ===========================================================================


def add_floor_area_x_energy_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction: property volume × energy cost savings percentage.

    Large properties with high savings potential may carry a green premium.
    Uses ``property_volume`` as the size proxy (``total_floor_area`` is
    dropped earlier in the pipeline by ``_compute_property_metrics``).

    Args:
        df: DataFrame with ``property_volume`` and ``energy_cost_savings_pct``.

    Returns:
        DataFrame with ``volume_x_energy_savings`` added.

    Example:
        # df = add_floor_area_x_energy_rating(df)
    """
    df["volume_x_energy_savings"] = (
        df["property_volume"] * df["energy_cost_savings_pct"].fillna(0)
    )
    return df


def add_district_x_property_type(df: pd.DataFrame) -> pd.DataFrame:
    """Combined categorical: district + property_type string concatenation.

    Captures sub-market effects (e.g. a flat in Kensington vs a flat in
    Manchester).  The resulting string is picked up by ``OrdinalEncoder``
    in the existing categorical encoding step.

    Must be applied *before* district and property_type are encoded/dropped
    in ``FeatureEngineer.transform()``.

    Args:
        df: DataFrame with ``district`` and ``property_type`` string columns.

    Returns:
        DataFrame with ``district_x_property_type`` (string) added.

    Example:
        # In FeatureEngineer.transform(), BEFORE the district/property_type blocks:
        # df = add_district_x_property_type(df)
    """
    df["district_x_property_type"] = (
        df["district"].astype(str) + "__" + df["property_type"].astype(str)
    )
    return df


def add_rooms_per_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Habitable rooms per floor-level unit (density proxy).

    Uses ``floor_level_clean`` (0 = ground floor) as a rough proxy for how
    many floors are below the unit.  Floors = floor_level_clean + 1 to
    avoid dividing by zero.

    Must be called after ``_clean_floor_level`` has produced
    ``floor_level_clean`` (NaNs imputed with training median first is ideal).

    Args:
        df: DataFrame with ``number_habitable_rooms`` and ``floor_level_clean``.

    Returns:
        DataFrame with ``rooms_per_floor`` added.

    Example:
        # df = add_rooms_per_floor(df)
    """
    floors = df["floor_level_clean"].fillna(0) + 1
    df["rooms_per_floor"] = df["number_habitable_rooms"] / floors
    return df


# ===========================================================================
# 7. RATIO FEATURES
# ===========================================================================


def add_heated_rooms_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of habitable rooms that are heated (clipped to [0, 1]).

    PREREQUISITE: ``number_heated_rooms`` is currently dropped inside
        ``_drop_redundant_rating_cols`` via ``_REDUNDANT_RATING_COLS``.
        Remove it from that list in engineer.py and call this function
        *before* ``_drop_redundant_rating_cols``.  The function drops
        ``number_heated_rooms`` itself afterwards.

    Args:
        df: DataFrame with ``number_heated_rooms`` and ``number_habitable_rooms``.

    Returns:
        DataFrame with ``heated_rooms_ratio`` added; ``number_heated_rooms`` dropped.

    Example:
        # In _apply_stateless_transforms, BEFORE _drop_redundant_rating_cols:
        # df = add_heated_rooms_ratio(df)
    """
    df["heated_rooms_ratio"] = (
        df["number_heated_rooms"]
        / df["number_habitable_rooms"].replace(0, np.nan)
    ).clip(0, 1)
    df = df.drop(columns=["number_heated_rooms"], errors="ignore")
    return df


def add_extension_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of floor area attributable to extensions (count / area proxy).

    NOTE: ``extension_count`` is a count of extensions, not their area, so
    this is an approximation — treat as a relative signal, not an exact ratio.

    PREREQUISITE: ``total_floor_area`` is dropped by ``_compute_property_metrics``.
        Apply *before* that step in ``_apply_stateless_transforms``.

    Args:
        df: DataFrame with ``extension_count`` and ``total_floor_area``.

    Returns:
        DataFrame with ``extension_ratio`` added.

    Example:
        # In _apply_stateless_transforms, BEFORE _compute_property_metrics:
        # df = add_extension_ratio(df)
    """
    df["extension_ratio"] = (
        df["extension_count"].fillna(0)
        / df["total_floor_area"].replace(0, np.nan)
    )
    return df


def add_glazing_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Proportion of glazing that is double/triple glazed (clipped to [0, 1]).

    ``multi_glaze_proportion`` is already a fraction in [0, 1]; this simply
    aliases it with a clearer name and ensures it is clipped.

    Args:
        df: DataFrame with ``multi_glaze_proportion``.

    Returns:
        DataFrame with ``glazing_ratio`` added.

    Example:
        # df = add_glazing_ratio(df)
    """
    df["glazing_ratio"] = df["multi_glaze_proportion"].clip(0, 1)
    return df
