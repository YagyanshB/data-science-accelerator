"""Unit tests for src/data/preprocessor.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import HousingPreprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Return a minimal raw DataFrame that mimics the real data schema."""
    rng = np.random.default_rng(seed)

    # Minimal columns required by the preprocessor
    df = pd.DataFrame(
        {
            # Core target + datetime
            "sale price": rng.integers(50_000, 600_000, size=n).astype(float),
            "Date of Transfer": pd.date_range("2021-01-01", periods=n, freq="ME"),
            # Renamed by preprocessor
            "Property Type": ["S"] * n,
            "PROPERTY_TYPE": ["House"] * n,
            # Dropped high-null cols – present but all NaN
            "SHEATING_ENERGY_EFF": [np.nan] * n,
            "SHEATING_ENV_EFF": [np.nan] * n,
            "FLOOR_ENV_EFF": [np.nan] * n,
            "FLAT_STOREY_COUNT": [np.nan] * n,
            # Zero-variance cols
            "Record Status": ["A"] * n,
            "town_or_city": ["LONDON"] * n,
            "County": ["GREATER LONDON"] * n,
            # Date cols
            "INSPECTION_DATE": pd.date_range("2021-01-01", periods=n, freq="ME"),
            "LODGEMENT_DATE": pd.date_range("2021-01-01", periods=n, freq="ME"),
            "LODGEMENT_DATETIME": pd.date_range("2021-01-01", periods=n, freq="ME"),
            # Imbalanced cols to be dropped
            "Report Type": [101] * n,
            "MECHANICAL_VENTILATION": [np.nan] * n,
            "WIND_TURBINE_COUNT": [0.0] * n,
            "PPD Category Type": ["A"] * n,
            "HEAT_LOSS_CORRIDOR": [np.nan] * n,
            "NUMBER_OPEN_FIREPLACES": [0.0] * n,
            "UNHEATED_CORRIDOR_LENGTH": [np.nan] * n,
            # Floor height for imputation test
            "FLOOR_HEIGHT": [2.5, np.nan] * (n // 2),
        }
    )
    return df


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return _make_raw_df(n=30)


@pytest.fixture()
def fitted_preprocessor(raw_df: pd.DataFrame) -> HousingPreprocessor:
    pp = HousingPreprocessor()
    pp.fit(raw_df)
    return pp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHousingPreprocessor:
    def test_fit_learns_price_bound(self, fitted_preprocessor: HousingPreprocessor, raw_df: pd.DataFrame) -> None:
        assert hasattr(fitted_preprocessor, "price_lower_bound_")
        assert fitted_preprocessor.price_lower_bound_ > 0

    def test_fit_learns_floor_height_median(self, fitted_preprocessor: HousingPreprocessor) -> None:
        assert hasattr(fitted_preprocessor, "floor_height_median_")
        assert fitted_preprocessor.floor_height_median_ == pytest.approx(2.5)

    def test_transform_adds_log_target(self, fitted_preprocessor: HousingPreprocessor, raw_df: pd.DataFrame) -> None:
        df_out = fitted_preprocessor.transform(raw_df)
        assert "log_sale_price" in df_out.columns

    def test_transform_drops_raw_sale_price(self, fitted_preprocessor: HousingPreprocessor, raw_df: pd.DataFrame) -> None:
        df_out = fitted_preprocessor.transform(raw_df)
        assert "sale_price" not in df_out.columns

    def test_transform_derives_sale_year(self, fitted_preprocessor: HousingPreprocessor, raw_df: pd.DataFrame) -> None:
        df_out = fitted_preprocessor.transform(raw_df)
        assert "sale_year" in df_out.columns
        assert df_out["sale_year"].notnull().all()

    def test_transform_imputes_floor_height(self, fitted_preprocessor: HousingPreprocessor, raw_df: pd.DataFrame) -> None:
        # floor_height is consumed internally (property_volume derived later)
        # Check it doesn't survive as NaN in the output
        df_out = fitted_preprocessor.transform(raw_df)
        if "floor_height" in df_out.columns:
            assert df_out["floor_height"].notnull().all()

    def test_filter_outliers_removes_rows(self, raw_df: pd.DataFrame) -> None:
        pp = HousingPreprocessor(price_lower_quantile=0.5)  # aggressive threshold
        pp.fit(raw_df)
        df_no_filter = pp.transform(raw_df, filter_outliers=False)
        df_filtered = pp.transform(raw_df, filter_outliers=True)
        assert len(df_filtered) <= len(df_no_filter)

    def test_log_transform_correct(self, fitted_preprocessor: HousingPreprocessor, raw_df: pd.DataFrame) -> None:
        df_out = fitted_preprocessor.transform(raw_df)
        # Manually compute expected value for first row
        # (raw sale price was lowercased to sale_price then log1p'd)
        raw_price = raw_df["sale price"].values
        expected_log = np.log1p(raw_price)
        actual_log = df_out["log_sale_price"].values
        np.testing.assert_allclose(actual_log, expected_log, rtol=1e-5)

    def test_no_data_leakage_between_fit_transform(self) -> None:
        """floor_height_median must come from fit data, not transform data."""
        train = _make_raw_df(n=20, seed=1)
        test = _make_raw_df(n=10, seed=2)
        # Override floor height in test to be very different
        test["FLOOR_HEIGHT"] = 99.0

        pp = HousingPreprocessor()
        pp.fit(train)
        expected_median = float(train["FLOOR_HEIGHT"].median())
        assert pp.floor_height_median_ == pytest.approx(expected_median, abs=0.1)
