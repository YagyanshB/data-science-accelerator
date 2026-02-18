"""Data loading module for housing price predictions."""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DataIngestor:
    """Handles secure and validated data loading from local Excel files.

    Args:
        file_path: Path to the Excel file. Falls back to DATA_PATH env var
            or 'data/raw_data.xlsx' if not provided.
    """

    def __init__(self, file_path: Optional[Union[str, Path]] = None) -> None:
        self.file_path = Path(
            file_path or os.getenv("DATA_PATH", "data/raw_data.xlsx")
        )

    def load_excel_data(self, sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        """Load an Excel file into a DataFrame with validation.

        Args:
            sheet_name: Sheet index or name to load. Defaults to first sheet.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the file path does not exist.
            ValueError: If the loaded DataFrame is empty.
        """
        logger.info("Attempting to load data from: %s", self.file_path)

        if not self.file_path.exists():
            msg = f"File not found: {self.file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            df = pd.read_excel(
                self.file_path, sheet_name=sheet_name, engine="openpyxl"
            )
        except Exception as exc:
            logger.error("Failed to read Excel file: %s", exc)
            raise

        if df.empty:
            raise ValueError(f"Loaded DataFrame from '{self.file_path}' is empty.")

        logger.info("Successfully loaded %d rows and %d columns.", *df.shape)
        return df
