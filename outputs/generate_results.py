"""Generate presentation-ready results from the housing price pipeline.

Produces:
  outputs/reports/oot_mae_by_year.csv
  outputs/reports/feature_importance.csv
  outputs/reports/mae_by_price_band.csv
  outputs/reports/mae_by_property_type.csv

Walk-forward strategy
---------------------
For each test year Y (from 1998 to 2025):
  - Train LightGBM on all rows with sale_year < Y
  - Test on rows with sale_year == Y
Preprocessing (medians, encoder vocabulary) is fit once on the full
dataset so that walk-forward results are fast and consistent;
for the presentation the dominant source of variance is model skill,
not imputation values.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Path setup – allow running from repo root or outputs/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import DataIngestor
from src.data.quality import DataQualityChecker
from src.data.preprocessor import HousingPreprocessor
from src.features.engineer import FeatureEngineer
from src.evaluation.metrics import compute_metrics

logging.basicConfig(
    level=logging.WARNING,          # suppress noise during batch runs
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

OUT_DIR = ROOT / "outputs" / "reports"
DATA_PATH = ROOT / "Lead Data Scientist Pre Work 1.xlsx"

# LightGBM params – fixed iterations (no early stopping) for walk-forward
# consistency; fast and reproducible across all 28 folds.
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)

PRICE_BANDS = [
    (0,        100_000,   "<£100k"),
    (100_000,  200_000,   "£100–200k"),
    (200_000,  300_000,   "£200–300k"),
    (300_000,  500_000,   "£300–500k"),
    (500_000,  1_000_000, "£500k–1M"),
    (1_000_000, np.inf,   ">£1M"),
]

MIN_TRAIN_YEAR = 1997   # need ≥ 2 full years before testing
FIRST_TEST_YEAR = 1998  # first year we report OOT results


def assign_price_band(price: float) -> str:
    for lo, hi, label in PRICE_BANDS:
        if lo <= price < hi:
            return label
    return ">£1M"


# ===========================================================================
# 1. LOAD & PREPROCESS (single pass on full data)
# ===========================================================================

print("Loading data …")
ingestor = DataIngestor(DATA_PATH)
df_raw = ingestor.load_excel_data()

# Keep columns needed for slicing *before* they're dropped by the pipeline
# property_structure = D/S/T/F/O
aux_cols = ["sale price", "Property Type"]
df_aux = df_raw[aux_cols].copy()
df_aux.columns = ["sale_price_raw", "property_type"]

print("Preprocessing …")
preprocessor = HousingPreprocessor()
preprocessor.fit(df_raw)                             # fit on full data
df_pre = preprocessor.transform(df_raw, filter_outliers=False)

# Apply outlier filter to get the clean index (same rows we'll model)
price_lb = preprocessor.price_lower_bound_
keep_mask = df_aux["sale_price_raw"] >= price_lb
df_pre   = df_pre[keep_mask].reset_index(drop=True)
df_aux   = df_aux[keep_mask].reset_index(drop=True)

print("Feature engineering …")
engineer = FeatureEngineer()
TARGET = "log_sale_price"
engineer.fit(df_pre.drop(columns=[TARGET]))
X_all = engineer.transform(df_pre.drop(columns=[TARGET]))
y_all = df_pre[TARGET].values                        # log-space
sale_prices = df_aux["sale_price_raw"].values        # £ for metrics
prop_types  = df_aux["property_type"].values
years       = df_pre["sale_year"].values.astype(int)

print(f"Feature matrix shape: {X_all.shape}")
print(f"Year range: {years.min()} – {years.max()}")

# ===========================================================================
# 2. WALK-FORWARD OOT EVALUATION BY YEAR
# ===========================================================================

print("\nRunning walk-forward OOT by year …")
oot_rows = []

for test_year in range(FIRST_TEST_YEAR, int(years.max()) + 1):
    train_mask = years < test_year
    test_mask  = years == test_year

    n_train = train_mask.sum()
    n_test  = test_mask.sum()

    if n_train < 100 or n_test < 10:
        print(f"  {test_year}: skipped (train={n_train}, test={n_test})")
        continue

    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_te, y_te = X_all[test_mask],  y_all[test_mask]
    p_te = sale_prices[test_mask]   # £

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_tr, y_tr)

    log_preds = model.predict(X_te)
    pred_prices = np.expm1(log_preds)
    true_prices = np.expm1(y_te)

    mae = float(np.mean(np.abs(true_prices - pred_prices)))
    mape = float(np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100)
    r2 = float(1 - np.sum((true_prices - pred_prices)**2) /
                   np.sum((true_prices - true_prices.mean())**2))

    oot_rows.append({
        "test_year": test_year,
        "mae": round(mae, 0),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "n_samples": int(n_test),
        "n_train": int(n_train),
    })
    print(f"  {test_year}: n_test={n_test:4d}  MAE=£{mae:,.0f}  MAPE={mape:.1f}%  R²={r2:.3f}")

df_oot = pd.DataFrame(oot_rows)
df_oot.to_csv(OUT_DIR / "oot_mae_by_year.csv", index=False)
print(f"\nSaved oot_mae_by_year.csv  ({len(df_oot)} rows)")

# ===========================================================================
# 3. FEATURE IMPORTANCE  (model trained on 1995-2023, tested on 2024)
# ===========================================================================

print("\nFitting final model (train≤2023, val=2024) for feature importance …")
train_m = years <= 2023
val_m   = years == 2024

model_final = lgb.LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": 1000})
model_final.fit(
    X_all[train_m], y_all[train_m],
    eval_set=[(X_all[val_m], y_all[val_m])],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

importances = pd.Series(
    model_final.feature_importances_,
    index=X_all.columns,
    name="importance",
).sort_values(ascending=False)

top15 = importances.head(15).reset_index()
top15.columns = ["feature", "importance"]
total = top15["importance"].sum()
top15["importance_pct"] = (top15["importance"] / importances.sum() * 100).round(2)
top15 = top15[["feature", "importance_pct"]]

top15.to_csv(OUT_DIR / "feature_importance.csv", index=False)
print("Saved feature_importance.csv")
print(top15.to_string(index=False))

# ===========================================================================
# 4. MAE BY PRICE BAND  (use final model predictions on 2024 test set)
# ===========================================================================

print("\nComputing MAE by price band …")

# Use all data where we can compare pred vs true (hold-out 2024)
test_mask_2024 = years == 2024
X_te24 = X_all[test_mask_2024]
true24 = np.expm1(y_all[test_mask_2024])
pred24 = np.expm1(model_final.predict(X_te24))
pt24   = prop_types[test_mask_2024]

band_rows = []
for lo, hi, label in PRICE_BANDS:
    mask = (true24 >= lo) & (true24 < hi)
    n = mask.sum()
    if n < 5:
        continue
    t, p = true24[mask], pred24[mask]
    mae  = float(np.mean(np.abs(t - p)))
    mape = float(np.mean(np.abs((t - p) / t)) * 100)
    band_rows.append({
        "price_band": label,
        "mae": round(mae, 0),
        "mape": round(mape, 2),
        "n_samples": int(n),
    })

df_bands = pd.DataFrame(band_rows)
df_bands.to_csv(OUT_DIR / "mae_by_price_band.csv", index=False)
print("Saved mae_by_price_band.csv")
print(df_bands.to_string(index=False))

# ===========================================================================
# 5. MAE BY PROPERTY TYPE
# ===========================================================================

print("\nComputing MAE by property type …")

TYPE_LABELS = {
    "D": "Detached",
    "S": "Semi-Detached",
    "T": "Terraced",
    "F": "Flat/Maisonette",
    "O": "Other",
}

type_rows = []
for code, label in TYPE_LABELS.items():
    mask = pt24 == code
    n = mask.sum()
    if n < 5:
        continue
    t, p = true24[mask], pred24[mask]
    mae  = float(np.mean(np.abs(t - p)))
    mape = float(np.mean(np.abs((t - p) / t)) * 100)
    type_rows.append({
        "property_type": label,
        "code": code,
        "mae": round(mae, 0),
        "mape": round(mape, 2),
        "n_samples": int(n),
    })

df_types = pd.DataFrame(type_rows)
df_types.to_csv(OUT_DIR / "mae_by_property_type.csv", index=False)
print("Saved mae_by_property_type.csv")
print(df_types.to_string(index=False))

# ===========================================================================
# 6. OVERALL METRICS SUMMARY
# ===========================================================================

print("\n" + "=" * 65)
print("SUMMARY FOR PRESENTATION")
print("=" * 65)

# Weighted average MAE across all walk-forward years
weights = df_oot["n_samples"].values
w_avg_mae  = float(np.average(df_oot["mae"].values, weights=weights))
w_avg_mape = float(np.average(df_oot["mape"].values, weights=weights))

best_row   = df_oot.loc[df_oot["mae"].idxmin()]
worst_row  = df_oot.loc[df_oot["mae"].idxmax()]
recent     = df_oot[df_oot["test_year"] == 2024].iloc[0] if 2024 in df_oot["test_year"].values else None

# Val/test metrics from final model
val_true  = np.expm1(y_all[years == 2024])
val_pred  = np.expm1(model_final.predict(X_all[years == 2024]))
val_mae   = float(np.mean(np.abs(val_true - val_pred)))
val_r2    = float(1 - np.sum((val_true - val_pred)**2) /
                     np.sum((val_true - val_true.mean())**2))
val_mdape = float(np.median(np.abs((val_true - val_pred) / val_true)) * 100)

summary = {
    "weighted_avg_mae":    round(w_avg_mae, 0),
    "weighted_avg_mape":   round(w_avg_mape, 2),
    "best_year":           int(best_row["test_year"]),
    "best_year_mae":       round(float(best_row["mae"]), 0),
    "worst_year":          int(worst_row["test_year"]),
    "worst_year_mae":      round(float(worst_row["mae"]), 0),
    "mae_2024":            round(val_mae, 0),
    "r2_2024":             round(val_r2, 4),
    "mdape_2024":          round(val_mdape, 2),
    "n_test_years":        len(df_oot),
    "model":               "LightGBM",
    "train_strategy":      "Expanding OOT window",
}

df_summary = pd.DataFrame([summary]).T.reset_index()
df_summary.columns = ["metric", "value"]
df_summary.to_csv(OUT_DIR / "summary.csv", index=False)

print(f"  Model:                    LightGBM (expanding OOT window)")
print(f"  Test years evaluated:     {len(df_oot)} years ({int(df_oot['test_year'].min())}–{int(df_oot['test_year'].max())})")
print(f"  Weighted avg MAE:        £{w_avg_mae:,.0f}")
print(f"  Weighted avg MAPE:        {w_avg_mape:.1f}%")
print(f"  Best  year: {int(best_row['test_year'])}            MAE=£{float(best_row['mae']):,.0f}")
print(f"  Worst year: {int(worst_row['test_year'])}            MAE=£{float(worst_row['mae']):,.0f}")
print(f"  2024 OOT MAE:            £{val_mae:,.0f}")
print(f"  2024 Median APE:          {val_mdape:.1f}%")
print(f"  2024 R²:                  {val_r2:.4f}")
print("=" * 65)
print("\nAll files saved to outputs/reports/")
print("  oot_mae_by_year.csv")
print("  feature_importance.csv")
print("  mae_by_price_band.csv")
print("  mae_by_property_type.csv")
print("  summary.csv")
