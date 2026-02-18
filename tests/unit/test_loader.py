"""Unit tests for src/data/loader.py."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import DataIngestor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_excel(tmp_path: Path) -> Path:
    """Write a minimal valid Excel file and return its path."""
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    path = tmp_path / "sample.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDataIngestor:
    def test_loads_existing_file(self, sample_excel: Path) -> None:
        ingestor = DataIngestor(sample_excel)
        df = ingestor.load_excel_data()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert list(df.columns) == ["col_a", "col_b"]
        assert len(df) == 3

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        ingestor = DataIngestor(tmp_path / "nonexistent.xlsx")
        with pytest.raises(FileNotFoundError, match="File not found"):
            ingestor.load_excel_data()

    def test_raises_for_empty_sheet(self, tmp_path: Path) -> None:
        """An Excel with an empty sheet should raise ValueError."""
        path = tmp_path / "empty.xlsx"
        pd.DataFrame().to_excel(path, index=False, engine="openpyxl")
        ingestor = DataIngestor(path)
        with pytest.raises(ValueError, match="empty"):
            ingestor.load_excel_data()

    def test_default_path_uses_env(self, monkeypatch: pytest.MonkeyPatch, sample_excel: Path) -> None:
        monkeypatch.setenv("DATA_PATH", str(sample_excel))
        ingestor = DataIngestor()
        assert ingestor.file_path == sample_excel
