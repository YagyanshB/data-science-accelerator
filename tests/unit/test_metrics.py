"""Unit tests for src/evaluation/metrics.py."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics, metrics_to_dataframe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions(self) -> None:
        y = np.array([100_000.0, 200_000.0, 300_000.0])
        metrics = compute_metrics(y, y)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mdape"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["within_10pct"] == pytest.approx(1.0)
        assert metrics["within_20pct"] == pytest.approx(1.0)

    def test_known_mape(self) -> None:
        y_true = np.array([100_000.0, 200_000.0])
        y_pred = np.array([110_000.0, 200_000.0])  # 10 % error on first, 0 % on second
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mape"] == pytest.approx(5.0, abs=0.1)  # mean of [10, 0]
        assert metrics["mdape"] == pytest.approx(5.0, abs=0.1)

    def test_within_bands(self) -> None:
        y_true = np.array([100_000.0, 100_000.0, 100_000.0, 100_000.0])
        y_pred = np.array([105_000.0, 115_000.0, 125_000.0, 130_000.0])
        # APEs: 5%, 15%, 25%, 30%
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["within_10pct"] == pytest.approx(0.25, abs=0.01)
        assert metrics["within_20pct"] == pytest.approx(0.50, abs=0.01)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    def test_returns_all_keys(self) -> None:
        y = np.array([200_000.0, 350_000.0, 150_000.0])
        y_pred = y * 1.05
        metrics = compute_metrics(y, y_pred)
        expected_keys = {"rmse", "mae", "mape", "mdape", "r2", "within_10pct", "within_20pct"}
        assert set(metrics.keys()) == expected_keys

    def test_label_does_not_affect_values(self) -> None:
        y = np.array([100_000.0, 200_000.0])
        y_pred = np.array([110_000.0, 190_000.0])
        m1 = compute_metrics(y, y_pred, label="val")
        m2 = compute_metrics(y, y_pred, label="test")
        for k in m1:
            assert m1[k] == pytest.approx(m2[k])


class TestMetricsToDataFrame:
    def test_shape(self) -> None:
        results = {
            "lgbm": {"val": {"rmse": 10.0, "mae": 8.0}, "test": {"rmse": 12.0, "mae": 9.0}},
            "ridge": {"val": {"rmse": 20.0, "mae": 15.0}, "test": {"rmse": 22.0, "mae": 16.0}},
        }
        # Flatten to split-level dict for the function signature
        flat = {
            "lgbm/val": results["lgbm"]["val"],
            "lgbm/test": results["lgbm"]["test"],
            "ridge/val": results["ridge"]["val"],
            "ridge/test": results["ridge"]["test"],
        }
        df = metrics_to_dataframe(flat)
        assert df.shape == (4, 2)
        assert "rmse" in df.columns
        assert "mae" in df.columns
