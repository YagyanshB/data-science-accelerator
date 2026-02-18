"""Data quality checks for the housing price dataset."""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Runs schema validation and data quality reports on a DataFrame.

    These methods are stateless – they inspect the data and raise/log
    issues without fitting any parameters for later use.
    """

    # Columns that must be present in the raw dataset.
    REQUIRED_COLUMNS: List[str] = [
        "sale price",
        "Date of Transfer",
        "Property Type",
        "PROPERTY_TYPE",
        "TOTAL_FLOOR_AREA",
        "CONSTRUCTION_AGE_BAND",
    ]

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Assert that all required raw columns are present.

        Args:
            df: Raw DataFrame immediately after loading.

        Raises:
            ValueError: If any required column is missing.
        """
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Raw DataFrame is missing required columns: {missing}"
            )
        logger.info("Schema validation passed – all required columns present.")

    def report_nulls(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Return a summary of missing-value rates for the top-N columns.

        Args:
            df: DataFrame to inspect.
            top_n: Number of columns with the most nulls to return.

        Returns:
            DataFrame with columns ``missing_count`` and ``missing_pct``,
            sorted descending by ``missing_pct``.
        """
        summary = pd.DataFrame(
            {
                "missing_count": df.isnull().sum(),
                "missing_pct": df.isnull().mean() * 100,
            }
        ).sort_values("missing_pct", ascending=False)

        high_null = summary[summary["missing_pct"] > 0].head(top_n)
        if not high_null.empty:
            logger.info(
                "Top-%d columns by missing rate:\n%s", top_n, high_null.to_string()
            )
        return summary

    def report_cardinality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return unique-value counts for all object/category columns.

        Args:
            df: DataFrame to inspect.

        Returns:
            DataFrame with columns ``dtype`` and ``n_unique`` for categorical
            columns, sorted descending.
        """
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        summary = pd.DataFrame(
            {
                "dtype": df[cat_cols].dtypes,
                "n_unique": df[cat_cols].nunique(),
            }
        ).sort_values("n_unique", ascending=False)
        logger.info("Cardinality report:\n%s", summary.to_string())
        return summary

    def run_all(self, df: pd.DataFrame) -> None:
        """Run all quality checks and log results.

        Args:
            df: Raw DataFrame to check.
        """
        self.validate_schema(df)
        self.report_nulls(df)
        self.report_cardinality(df)
        logger.info("All quality checks complete.")
