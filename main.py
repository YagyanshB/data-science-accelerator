"""Entry point for the housing price prediction pipeline.

Usage
-----
    python main.py                       # uses configs/config.yaml
    python main.py --config path/to.yaml
    python main.py --models lgbm xgb    # select a subset of models
    python main.py --data path/to.xlsx  # override data path

Pipeline steps
--------------
1. Load raw Excel data.
2. Run data quality checks.
3. Preprocess (rename, clean, date extraction, log-transform target).
4. Out-of-time split: train ≤ 2023 | val = 2024 | test > 2024.
5. Fit feature engineering on training fold, transform all folds.
6. Train all selected models with early stopping on val.
7. Evaluate on val and test; print a comparison table.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Bootstrap logging before any local imports so module-level loggers work.
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO", fmt: Optional[str] = None) -> None:
    fmt = fmt or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)


_setup_logging()  # default until config is loaded
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

from src.data.loader import DataIngestor
from src.data.quality import DataQualityChecker
from src.data.preprocessor import HousingPreprocessor
from src.features.engineer import FeatureEngineer
from src.models.trainer import get_trainer
from src.models.predictor import HousingPredictor
from src.evaluation.metrics import compute_metrics, metrics_to_dataframe


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    config_path: str = "configs/config.yaml",
    data_path: Optional[str] = None,
    model_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Execute the full training and evaluation pipeline.

    Args:
        config_path: Path to the YAML configuration file.
        data_path: Override for the raw data path in the config.
        model_names: List of model names to train.  Defaults to all models
            defined in the config (``["lgbm", "xgb", "ridge"]``).

    Returns:
        Nested dict: ``{model_name: {"val": metrics, "test": metrics}}``.
    """
    # ------------------------------------------------------------------ #
    # 0. Config                                                           #
    # ------------------------------------------------------------------ #
    cfg = load_config(config_path)
    log_cfg = cfg.get("logging", {})
    _setup_logging(
        level=log_cfg.get("level", "INFO"),
        fmt=log_cfg.get("format"),
    )

    raw_path = data_path or cfg["data"]["raw_path"]
    sheet = cfg["data"].get("sheet_name", 0)
    train_cutoff: int = cfg["splits"]["train_cutoff"]
    val_cutoff: int = cfg["splits"]["val_cutoff"]
    model_cfg: dict = cfg.get("models", {})

    if model_names is None:
        model_names = list(model_cfg.keys())

    logger.info(
        "Pipeline config: data=%s | train≤%d | val=%d | test>%d | models=%s",
        raw_path,
        train_cutoff,
        val_cutoff,
        val_cutoff,
        model_names,
    )

    # ------------------------------------------------------------------ #
    # 1. Load                                                             #
    # ------------------------------------------------------------------ #
    ingestor = DataIngestor(raw_path)
    df_raw = ingestor.load_excel_data(sheet_name=sheet)

    # ------------------------------------------------------------------ #
    # 2. Quality checks                                                   #
    # ------------------------------------------------------------------ #
    checker = DataQualityChecker()
    checker.run_all(df_raw)

    # ------------------------------------------------------------------ #
    # 3. Preprocessing – fit only on future training rows                 #
    #    We need sale_year to split, so run a stateless preview first.    #
    # ------------------------------------------------------------------ #
    pre_cfg = cfg.get("preprocessing", {})
    preprocessor = HousingPreprocessor(
        price_lower_quantile=pre_cfg.get("price_lower_quantile", 0.005),
        current_year=pre_cfg.get("current_year", 2026),
    )

    # Fit on the full dataset to learn bounds, then split properly.
    # (sale_year is derived from date_of_transfer by the preprocessor.)
    df_pre_full = preprocessor.fit(df_raw).transform(df_raw, filter_outliers=False)

    # ------------------------------------------------------------------ #
    # 4. Out-of-time split                                                #
    # ------------------------------------------------------------------ #
    train_mask = df_pre_full["sale_year"] <= train_cutoff
    val_mask = (df_pre_full["sale_year"] > train_cutoff) & (
        df_pre_full["sale_year"] <= val_cutoff
    )
    test_mask = df_pre_full["sale_year"] > val_cutoff

    logger.info(
        "OOT split sizes → train: %d | val: %d | test: %d",
        train_mask.sum(),
        val_mask.sum(),
        test_mask.sum(),
    )

    if test_mask.sum() == 0:
        logger.warning(
            "No test rows found (sale_year > %d). "
            "Check the date range in your data.",
            val_cutoff,
        )

    # Re-run transform with outlier filter enabled on training fold only.
    preprocessor.fit(df_raw[train_mask.values])  # refit on train only
    df_train_pre = preprocessor.transform(
        df_raw[train_mask.values], filter_outliers=True
    )
    df_val_pre = preprocessor.transform(df_raw[val_mask.values], filter_outliers=False)
    df_test_pre = preprocessor.transform(
        df_raw[test_mask.values], filter_outliers=False
    )

    # ------------------------------------------------------------------ #
    # 5. Feature engineering – fit on train, transform all                #
    # ------------------------------------------------------------------ #
    target_col = cfg["target"]["log_column"]

    engineer = FeatureEngineer(current_year=pre_cfg.get("current_year", 2026))
    engineer.fit(df_train_pre.drop(columns=[target_col]))

    def _split_xy(df: pd.DataFrame):
        y = df[target_col].values
        X = engineer.transform(df.drop(columns=[target_col]))
        return X, y

    X_train, y_train = _split_xy(df_train_pre)
    X_val, y_val = _split_xy(df_val_pre)
    X_test, y_test = _split_xy(df_test_pre) if test_mask.sum() > 0 else (None, None)

    logger.info(
        "Feature matrix shapes → train: %s | val: %s | test: %s",
        X_train.shape,
        X_val.shape,
        X_test.shape if X_test is not None else "N/A",
    )

    # ------------------------------------------------------------------ #
    # 6. Train models                                                     #
    # ------------------------------------------------------------------ #
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for name in model_names:
        logger.info("=" * 60)
        logger.info("Training model: %s", name.upper())
        params = model_cfg.get(name, {})
        trainer = get_trainer(name, **params)
        trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        predictor = HousingPredictor(trainer)

        # ---- Validation ------------------------------------------------
        y_val_pred = predictor.predict_price(X_val)
        val_metrics = compute_metrics(
            np.expm1(y_val), y_val_pred, label=f"{name}/val"
        )

        # ---- Test -------------------------------------------------------
        test_metrics: Dict[str, float] = {}
        if X_test is not None and len(y_test) > 0:
            y_test_pred = predictor.predict_price(X_test)
            test_metrics = compute_metrics(
                np.expm1(y_test), y_test_pred, label=f"{name}/test"
            )

        all_results[name] = {"val": val_metrics, "test": test_metrics}

    # ------------------------------------------------------------------ #
    # 7. Summary table                                                    #
    # ------------------------------------------------------------------ #
    _print_summary(all_results)
    return all_results


def _print_summary(
    results: Dict[str, Dict[str, Dict[str, float]]]
) -> None:
    """Print a formatted comparison of all model results.

    Args:
        results: Nested dict returned by :func:`run_pipeline`.
    """
    rows = []
    for model, splits in results.items():
        for split, metrics in splits.items():
            if metrics:
                rows.append({"model": model, "split": split, **metrics})

    if not rows:
        logger.warning("No metrics to display.")
        return

    df = pd.DataFrame(rows).set_index(["model", "split"])
    logger.info("\n\n=== FINAL RESULTS ===\n%s\n", df.to_string())
    print("\n=== FINAL RESULTS ===")
    print(df.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Housing price OOT prediction pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Override raw data path from config.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["lgbm", "xgb", "ridge"],
        help="Models to train (default: all).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        config_path=args.config,
        data_path=args.data,
        model_names=args.models,
    )
