# INTERVIEW CHEAT SHEET — Housing Price Prediction Pipeline
> Cmd+F to find anything. All code blocks are copy-paste ready.

---

## 1. FILE MAP

| File | What it does | Key classes / functions | Critical lines |
|------|-------------|------------------------|----------------|
| `main.py` | Full pipeline entry point | `run_pipeline()`, `load_config()`, `_parse_args()` | OOT split: L160–164 · Feature eng: L195–208 · Train loop: L222–245 |
| `configs/config.yaml` | All tunable parameters | — | Splits: L16–17 · LGBM: L24–31 · XGB: L34–41 · Ridge: L43–44 |
| `src/data/loader.py` | Load raw Excel file | `DataIngestor.load_excel_data()` | Class: L13 · load: L26 · env var fallback: L23 |
| `src/data/quality.py` | Schema + null + cardinality checks | `DataQualityChecker.run_all()` | Required cols: L19 · validate: L28 · nulls: L44 · run_all: L89 |
| `src/data/preprocessor.py` | Clean raw data, log-transform target | `HousingPreprocessor.fit()` / `.transform()` | Drop lists: L29–56 · fit: L86 · outlier filter: L141 · log-price: L151 · dates: L217 |
| `src/features/engineer.py` | Derive features, encode categoricals | `FeatureEngineer.fit()` / `.transform()` | Class: L265 · fit: L283 · district wealth: L312 · transform: L346 · stateless: L394 |
| `src/features/feature_ideas.py` | **Reference only** — 21 ready-to-add features | 21 standalone functions | Sections: price, age, energy, location, time, interaction, ratio |
| `src/models/trainer.py` | Model wrappers with consistent fit/predict | `LGBMTrainer`, `XGBTrainer`, `RidgeTrainer`, `get_trainer()` | LGBM: L32 · XGB: L153 · Ridge: L268 · registry: L352 |
| `src/models/predictor.py` | Log→£ back-transformation | `HousingPredictor.predict_price()` | Class: L35 · predict_price: L68 · predict_df: L90 |
| `src/evaluation/metrics.py` | RMSE, MAE, MdAPE, R², within-10/20% | `compute_metrics()`, `metrics_to_dataframe()` | compute_metrics: L23 · within-10%: L67 |
| `outputs/generate_results.py` | Walk-forward OOT + feature importance + segmented MAE | — | Walk-forward loop: L128 · feature importance: L173 · price bands: L215 |

### engineer.py Drop-List Constants (add column name here to remove a feature)

| Constant | Line | What gets dropped |
|----------|------|-------------------|
| `_ENERGY_COST_INPUT_COLS` | L64 | Raw cost cols after energy features computed |
| `_PROPERTY_METRIC_INPUT_COLS` | L79 | `total_floor_area`, `floor_height` after volume/CO2 computed |
| `_REDUNDANT_RATING_COLS` | L96 | `number_heated_rooms` + EPC rating cols |
| `_DESCRIPTION_DROP_COLS` | L85 | Free-text walls/windows/roof descriptions |
| `_IMBALANCED_COLS` | preprocessor.py L40 | Low-signal cols dropped in preprocessing |

---

## 2. QUICK MODIFICATIONS TABLE

| Task | File | Line | What to change |
|------|------|------|----------------|
| Change train cutoff year | `configs/config.yaml` | L16 | `train_cutoff: 2023` |
| Change val cutoff year | `configs/config.yaml` | L17 | `val_cutoff: 2024` |
| LGBM learning rate | `configs/config.yaml` | L25 | `learning_rate: 0.05` |
| LGBM num_leaves (complexity) | `configs/config.yaml` | L26 | `num_leaves: 63` → 31 (simpler) / 127 (complex) |
| LGBM early stopping patience | `configs/config.yaml` | L29 | `early_stopping_rounds: 50` |
| LGBM n_estimators cap | `configs/config.yaml` | L24 | `n_estimators: 1000` |
| XGB max_depth | `configs/config.yaml` | L37 | `max_depth: 6` |
| Ridge alpha | `configs/config.yaml` | L44 | `alpha: 1.0` (higher = more regularised) |
| Remove a feature | `src/features/engineer.py` | L64–103 | Add col name to relevant `_*_COLS` constant |
| Add stateless feature | `src/features/engineer.py` | L394–411 | Add `df = _new_fn(df)` inside `_apply_stateless_transforms()` |
| Add stateful feature | `src/features/engineer.py` | L312 / L368 | Add fit block after L329, transform block after L376 |
| Price outlier cutoff | `configs/config.yaml` | L9 | `price_lower_quantile: 0.005` |
| Reference year for property_age | `configs/config.yaml` | L10 | `current_year: 2026` |
| Re-enable a dropped column | `src/data/preprocessor.py` | L40–48 | Remove from `_IMBALANCED_COLS` |
| Keep total_floor_area | `src/features/engineer.py` | L79–83 | Remove from `_PROPERTY_METRIC_INPUT_COLS` |
| Keep number_heated_rooms | `src/features/engineer.py` | L96–103 | Remove from `_REDUNDANT_RATING_COLS` |
| Add sale_month | `src/data/preprocessor.py` | L238 | `df["sale_month"] = df["date_of_transfer"].dt.month` |
| Run only one model | CLI | — | `python main.py --models lgbm` |
| Override data path | CLI | — | `python main.py --data path/to/file.xlsx` |

---

## 3. COPY-PASTE CODE BLOCKS

### A. Add New Feature (stateless)

```python
# === PASTE INTO src/features/engineer.py ===

# Step 1 — add a new @staticmethod to FeatureEngineer:
@staticmethod
def _compute_rooms_per_floor(df: pd.DataFrame) -> pd.DataFrame:
    floors = df["floor_level_clean"].fillna(0) + 1
    df["rooms_per_floor"] = df["number_habitable_rooms"] / floors
    return df

# Step 2 — call it inside _apply_stateless_transforms() (line ~411),
# AFTER _clean_floor_level() which produces floor_level_clean:
df = self._compute_rooms_per_floor(df)
```

```python
# === PRICE-BASED FEATURE (log_sale_price is already in df) ===
@staticmethod
def _compute_price_per_room(df: pd.DataFrame) -> pd.DataFrame:
    sale_price = np.expm1(df["log_sale_price"])
    df["price_per_room"] = sale_price / df["number_habitable_rooms"].replace(0, np.nan)
    return df

# Call inside _apply_stateless_transforms():
df = self._compute_price_per_room(df)
```

---

### B. Change Hyperparameters

```yaml
# === configs/config.yaml — LGBM more regularised ===
lgbm:
  n_estimators: 2000
  learning_rate: 0.02       # slower = better generalisation
  num_leaves: 31            # simpler trees
  min_child_samples: 50     # larger leaves
  subsample: 0.7
  colsample_bytree: 0.7
  early_stopping_rounds: 100
  random_state: 42
```

```yaml
# === configs/config.yaml — XGB more regularised ===
xgb:
  n_estimators: 2000
  learning_rate: 0.02
  max_depth: 4
  subsample: 0.7
  colsample_bytree: 0.7
  early_stopping_rounds: 100
  random_state: 42
```

```python
# === RidgeCV — find best alpha automatically ===
from sklearn.linear_model import RidgeCV
import numpy as np

alphas = np.logspace(-2, 4, 20)   # 0.01 to 10000
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_absolute_error")
ridge_cv.fit(X_train_scaled, y_train)
print("Best alpha:", ridge_cv.alpha_)
```

---

### C. Create Ensemble (Simple & Weighted Average)

```python
# === PASTE after the training loop in main.py ===
import numpy as np
from src.evaluation.metrics import compute_metrics

# Collect log-space predictions from each trained model
log_preds_val  = {}
log_preds_test = {}

for name in model_names:
    tr = get_trainer(name, **model_cfg.get(name, {}))
    tr.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    log_preds_val[name]  = tr.predict(X_val)
    if X_test is not None:
        log_preds_test[name] = tr.predict(X_test)

# --- Simple average ensemble ---
ens_val   = np.column_stack(list(log_preds_val.values())).mean(axis=1)
compute_metrics(np.expm1(y_val), np.expm1(ens_val), label="simple_ensemble/val")

# --- Inverse-MAE weighted average ---
val_maes  = {n: np.mean(np.abs(np.expm1(p) - np.expm1(y_val)))
             for n, p in log_preds_val.items()}
inv_mae   = {n: 1/v for n, v in val_maes.items()}
total     = sum(inv_mae.values())
weights   = {n: v/total for n, v in inv_mae.items()}
print("Weights:", weights)

ens_weighted = sum(w * log_preds_val[n] for n, w in weights.items())
compute_metrics(np.expm1(y_val), np.expm1(ens_weighted), label="weighted_ensemble/val")
```

---

### D. Add Neural Network (MLPRegressor)

```python
# === PASTE INTO src/models/trainer.py ===
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class MLPTrainer(BaseEstimator, RegressorMixin):
    """MLP neural network with built-in imputation and scaling."""

    def __init__(
        self,
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        cols = X_train.columns
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(self.imputer_.fit_transform(X_train), columns=cols)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imp)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
            verbose=False,
        )
        self.model_.fit(X_scaled, y_train)
        self.feature_names_in_ = cols
        return self

    def predict(self, X):
        X_imp = pd.DataFrame(self.imputer_.transform(X), columns=self.feature_names_in_)
        return self.model_.predict(self.scaler_.transform(X_imp))

# Register it (add after the _TRAINER_REGISTRY dict at trainer.py:L352):
_TRAINER_REGISTRY["mlp"] = MLPTrainer

# Add to configs/config.yaml under models:
# mlp:
#   hidden_layer_sizes: [256, 128, 64]
#   alpha: 0.001
#   max_iter: 500
```

---

### E. Add SHAP Explanations

```python
# pip install shap
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- TreeExplainer (fast, works for LGBM and XGB) ---
# lgbm_trainer is the fitted LGBMTrainer instance
explainer   = shap.TreeExplainer(lgbm_trainer.model_)
shap_values = explainer.shap_values(X_val)   # (n_samples, n_features)

# Summary bar — mean |SHAP| per feature
shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
plt.tight_layout(); plt.savefig("outputs/shap_bar.png", dpi=150); plt.close()

# Beeswarm — direction + magnitude per sample
shap.summary_plot(shap_values, X_val, show=False)
plt.tight_layout(); plt.savefig("outputs/shap_beeswarm.png", dpi=150); plt.close()

# Force plot for a single prediction (idx = row index)
idx = 0
shap.force_plot(
    explainer.expected_value, shap_values[idx], X_val.iloc[idx],
    matplotlib=True, show=False,
)
plt.savefig("outputs/shap_force.png", dpi=150, bbox_inches="tight"); plt.close()

# Top 5 features by mean |SHAP|
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=X_val.columns
).sort_values(ascending=False)
print(mean_abs_shap.head())
```

---

### F. Change Train/Test Split

```yaml
# === OPTION 1: Edit configs/config.yaml ===
splits:
  train_cutoff: 2022   # train ≤2022, val=2023, test>2023
  val_cutoff: 2023
```

```python
# === OPTION 2: Edit split masks in main.py (around line 160) ===

# Current default:
train_mask = df_pre_full["sale_year"] <= train_cutoff           # ≤2023
val_mask   = (df_pre_full["sale_year"] > train_cutoff) & \
             (df_pre_full["sale_year"] <= val_cutoff)           # 2024
test_mask  = df_pre_full["sale_year"] > val_cutoff             # >2024

# Rolling 2-year val window:
train_mask = df_pre_full["sale_year"] <= 2021
val_mask   = df_pre_full["sale_year"].isin([2022, 2023])
test_mask  = df_pre_full["sale_year"] >= 2024

# Random 80/20 (NON-OOT — for comparison only):
from sklearn.model_selection import train_test_split
idx_tr, idx_vl = train_test_split(df_pre_full.index, test_size=0.2, random_state=42)
train_mask = df_pre_full.index.isin(idx_tr)
val_mask   = df_pre_full.index.isin(idx_vl)
test_mask  = pd.Series(False, index=df_pre_full.index)
```

---

### G. Ablation Test (Remove a Feature and Compare)

```python
# === PASTE INTO main.py or notebook after training ===
import numpy as np
from src.models.trainer import get_trainer
from src.evaluation.metrics import compute_metrics

def ablation_test(X_train, y_train, X_val, y_val, feature_to_drop, model_cfg):
    """Drop one feature, retrain LGBM, compare val MAE to baseline."""
    # Baseline
    base = get_trainer("lgbm", **model_cfg.get("lgbm", {}))
    base.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    base_mae = compute_metrics(
        np.expm1(y_val), np.expm1(base.predict(X_val)), label="baseline"
    )["mae"]

    # Ablated
    X_tr_abl  = X_train.drop(columns=[feature_to_drop], errors="ignore")
    X_val_abl = X_val.drop(columns=[feature_to_drop], errors="ignore")
    abl = get_trainer("lgbm", **model_cfg.get("lgbm", {}))
    abl.fit(X_tr_abl, y_train, X_val=X_val_abl, y_val=y_val)
    abl_mae = compute_metrics(
        np.expm1(y_val), np.expm1(abl.predict(X_val_abl)), label=f"no_{feature_to_drop}"
    )["mae"]

    delta = abl_mae - base_mae
    print(f"\nRemoving '{feature_to_drop}': MAE delta = £{delta:+,.0f}")
    print("  Positive delta = feature WAS useful; negative = removal helped")
    return {"baseline_mae": base_mae, "ablated_mae": abl_mae, "delta": delta}

# Usage:
ablation_test(X_train, y_train, X_val, y_val, "district_wealth_index", model_cfg)
ablation_test(X_train, y_train, X_val, y_val, "property_age", model_cfg)
```

---

### H. Add Cross-Validation (TimeSeriesSplit)

```python
# === PASTE INTO main.py or notebook ===
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import numpy as np

# Sort by time so splits respect temporal order
df_sorted = df_train_pre.sort_values("sale_year")
target_col = "log_sale_price"
X_cv = engineer.transform(df_sorted.drop(columns=[target_col]))
y_cv = df_sorted[target_col].values

tscv = TimeSeriesSplit(n_splits=5, gap=0)
fold_maes = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_cv)):
    X_tr, X_vl = X_cv.iloc[tr_idx], X_cv.iloc[val_idx]
    y_tr, y_vl = y_cv[tr_idx], y_cv[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    preds = np.expm1(model.predict(X_vl))
    mae   = np.mean(np.abs(np.expm1(y_vl) - preds))
    fold_maes.append(mae)
    print(f"  Fold {fold+1}: MAE = £{mae:,.0f}")

print(f"\nCV Mean MAE: £{np.mean(fold_maes):,.0f} ± £{np.std(fold_maes):,.0f}")
```

---

### I. Segmented Analysis (MAE by Price Band / Property Type)

```python
# === PASTE after predictions in main.py or notebook ===
import numpy as np
import pandas as pd

def segment_mae(y_true, y_pred, labels, name="segment"):
    """MAE and MAPE broken down by a categorical label array."""
    rows = []
    for seg in sorted(set(labels)):
        mask = labels == seg
        n = mask.sum()
        if n < 5:
            continue
        t, p = y_true[mask], y_pred[mask]
        mae  = float(np.mean(np.abs(t - p)))
        mape = float(np.mean(np.abs((t - p) / t)) * 100)
        rows.append({name: seg, "n": n, "MAE": f"£{mae:,.0f}", "MAPE": f"{mape:.1f}%"})
    df_out = pd.DataFrame(rows).set_index(name)
    print(f"\nMAE by {name}:\n{df_out.to_string()}")
    return df_out

def assign_band(price):
    if price < 100_000:    return "<£100k"
    if price < 200_000:    return "£100-200k"
    if price < 300_000:    return "£200-300k"
    if price < 500_000:    return "£300-500k"
    if price < 1_000_000:  return "£500k-1M"
    return ">£1M"

y_true  = np.expm1(y_val)
y_pred  = np.expm1(lgbm_trainer.predict(X_val))
bands   = np.array([assign_band(p) for p in y_true])
segment_mae(y_true, y_pred, bands, "price_band")

# Property type (carry property_structure through before engineer.transform):
# prop_types = df_val_pre["property_structure"].values  # D/S/T/F/O
# segment_mae(y_true, y_pred, prop_types, "property_type")
```

---

### J. Add Stacking Ensemble

```python
# === PASTE INTO main.py or notebook ===
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from src.evaluation.metrics import compute_metrics

lgbm_est = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
)
xgb_est = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    tree_method="hist", verbosity=0,
)
ridge_pipe = SkPipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scl", StandardScaler()),
    ("reg", Ridge(alpha=1.0)),
])

stacker = StackingRegressor(
    estimators=[("lgbm", lgbm_est), ("xgb", xgb_est), ("ridge", ridge_pipe)],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    passthrough=False,
    n_jobs=-1,
)

# Impute NaNs first (sklearn StackingRegressor doesn't handle NaN natively)
imp = SimpleImputer(strategy="median")
X_tr_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
X_vl_imp = pd.DataFrame(imp.transform(X_val),       columns=X_val.columns)

stacker.fit(X_tr_imp, y_train)
stack_preds = np.expm1(stacker.predict(X_vl_imp))
compute_metrics(np.expm1(y_val), stack_preds, label="stacking/val")
```

---

## 4. COMMON COMMANDS

```bash
# Run full pipeline (all 3 models: lgbm, xgb, ridge)
python main.py

# Run only LightGBM
python main.py --models lgbm

# Run LightGBM + XGBoost
python main.py --models lgbm xgb

# Override data path
python main.py --data "Lead Data Scientist Pre Work 1.xlsx"

# Use a different config
python main.py --config configs/config.yaml

# Regenerate all CSVs in outputs/reports/
python outputs/generate_results.py

# Run all tests
pytest tests/ -v

# Run single test file
pytest tests/unit/test_engineer.py -v

# Install all dependencies
pip install -r requirements.txt

# Quick feature audit — print all features after engineering
python -c "
import pandas as pd, numpy as np
from src.data.loader import DataIngestor
from src.data.preprocessor import HousingPreprocessor
from src.features.engineer import FeatureEngineer
df = DataIngestor().load_excel_data()
pre = HousingPreprocessor().fit(df).transform(df)
eng = FeatureEngineer()
eng.fit(pre.drop(columns=['log_sale_price']), y=pre['log_sale_price'])
X = eng.transform(pre.drop(columns=['log_sale_price']))
print(X.columns.tolist())
print('Shape:', X.shape)
"
```

---

## 5. KEY NUMBERS TO REMEMBER

| Metric | Value |
|--------|-------|
| **Primary metric** | MdAPE (Median Absolute % Error) |
| Model | **LightGBM**, expanding OOT window |
| Total OOT years evaluated | **28** (1998–2025) |
| Weighted avg MAE across all years | **£117,624** |
| Weighted avg MAPE across all years | **31.2%** |
| 2024 OOT MAE | **£177,758** |
| 2024 MdAPE | **19.97%** |
| 2024 R² | **0.277** |
| Best year | **1998** — MAE £42,057 |
| Worst year | **2020** — MAE £259,819 (COVID) |
| Train split | sale_year ≤ 2023 |
| Val split | sale_year = 2024 |
| Test split | sale_year > 2024 |

### Feature Importance — Top 10 (LightGBM gain, train≤2023 / val=2024)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `district` | **17.24%** |
| 2 | `property_volume` | 8.08% |
| 3 | `days_inspection_to_sale` | 5.83% |
| 4 | `sale_year` | 5.03% |
| 5 | `total_co2_emissions` | 4.41% |
| 6 | `fixed_lighting_outlets_count` | 4.38% |
| 7 | `property_structure` | 3.91% |
| 8 | `energy_cost_savings_pct` | 3.90% |
| 9 | `property_age` | 3.48% |
| 10 | `energy_efficiency_gain_pct` | 3.44% |

### MAE by Property Type (2024 holdout)

| Type | Code | MAE | MAPE | n |
|------|------|-----|------|---|
| Flat/Maisonette | F | £128,583 | 28.2% | 451 |
| Terraced | T | £149,120 | 25.8% | 251 |
| Semi-Detached | S | £164,528 | 26.3% | 148 |
| Detached | D | £244,577 | 36.3% | 37 |
| **Other** | **O** | **£1,172,662** | **194.7%** | **29** ← worst segment |

### MAE by Price Band (2024 holdout)

| Band | MAE | MAPE | n |
|------|-----|------|---|
| £200–300k | £74,193 | 29.4% | 148 |
| £300–500k | £83,965 | **21.7%** | 316 ← best accuracy |
| £100–200k | £93,612 | 62.0% | 90 |
| £500k–1M | £153,385 | 22.2% | 263 |
| <£100k | £154,783 | 325.5% | 13 ← worst % (sparse) |
| >£1M | £866,695 | 37.9% | 86 |

---

## 6. TROUBLESHOOTING

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `FileNotFoundError: Lead Data Scientist Pre Work 1.xlsx` | Data file not in repo root | `python main.py --data path/to/file.xlsx` or `export DATA_PATH=...` |
| `ValueError: Raw DataFrame is missing required columns` | Column name mismatch | Check `DataQualityChecker.REQUIRED_COLUMNS` at `quality.py:L19` |
| `ModuleNotFoundError: lightgbm` | Not installed | `pip install lightgbm` |
| `ModuleNotFoundError: xgboost` | Not installed | `pip install xgboost` |
| `ModuleNotFoundError: shap` | Not installed | `pip install shap` |
| `KeyError: 'district'` in transform | Already dropped before your code runs | Move code above the district_wealth_index block at `engineer.py:L368` |
| `KeyError: 'total_floor_area'` | Dropped by `_compute_property_metrics` | Remove from `_PROPERTY_METRIC_INPUT_COLS` at `engineer.py:L79` |
| `KeyError: 'number_heated_rooms'` | Dropped by `_drop_redundant_rating_cols` | Remove from `_REDUNDANT_RATING_COLS` at `engineer.py:L96` |
| `KeyError: 'sale_month'` | Preprocessor only extracts `sale_year` | Add `df["sale_month"] = df["date_of_transfer"].dt.month` at `preprocessor.py:L238` |
| NaN in predictions / LGBM | Unexpected NaN propagation in new feature | LGBM handles NaN natively — check your feature function for divide-by-zero |
| NaN in Ridge predictions | Ridge doesn't handle NaN — uses `SimpleImputer` | Handled internally by `RidgeTrainer`; ensure imputer is fitted first |
| OrdinalEncoder unknown value warning | New categorical level at inference | Already handled: `unknown_value=-1` at `engineer.py:L334` |
| `Shape mismatch` in `compute_metrics` | y_true / y_pred out of sync | Check X_test is not None before evaluating; verify mask alignment |
| `No test rows found` warning | All data ≤ val_cutoff | Raise `val_cutoff` in config or check year range in your data |
| Test MAE >> val MAE | Data drift or COVID spike in test set | Expected for 2020; inspect `oot_mae_by_year.csv` year-by-year |
| LGBM early stopping fires immediately | Val set too small | Verify `df_val_pre` has enough rows; check `val_cutoff` config value |

---

## 7. VERBAL ANSWERS

**Why OOT (out-of-time) validation instead of random split?**
> Property prices have strong temporal structure — a random split leaks future price levels into training, making the model look artificially better than it is in production. OOT mimics real deployment: the model never sees data from the years it is predicting. This matters especially around structural breaks like the 2008 crash, 2014–2016 London surge, and 2020 COVID.

**Why log-transform the target?**
> Sale prices are right-skewed with a long tail of multi-million-pound properties. Log-transforming makes the distribution approximately normal, reduces the influence of extreme values on gradient-based training, and makes MdAPE the natural evaluation metric. Back-transformation is `np.expm1()`.

**Why LightGBM as the primary model?**
> It handles mixed numeric/categorical features, missing values, and non-linear interactions natively with minimal preprocessing. Gradient boosting consistently outperforms linear models on tabular data. It's fast — early stopping finds the right number of trees automatically. XGBoost is our cross-check; Ridge is the interpretable baseline.

**What is the top feature and why?**
> `district` at 17.24% importance. Location is the dominant price driver in property — a flat in Kensington vs Manchester can differ by 10×. We encode it as `district_wealth_index`: the median log sale price per district, learned from training data only to prevent target leakage.

**What is your worst-performing segment?**
> The "Other" property category (code O) with MdAPE ~195% — but this is only 29 samples of non-standard commercial/unusual properties the model was never designed for. Among standard residential types, **Detached** (D) is hardest at MAE £244k / MAPE 36% — widest price range, fewest examples. Best segment is **£300–500k** at 21.7% MAPE.

**Why MdAPE as the primary metric?**
> Median is robust to the extreme errors from "Other" properties and £1M+ luxury homes. MAE in pound terms is hard to compare across price ranges — a £50k error on a £100k property is catastrophic; the same on a £1M property is trivial. MdAPE gives consistent interpretation across the full price spectrum.

**Why expanding OOT window (train < Y, test = Y) rather than a fixed rolling window?**
> We want to simulate production behaviour where we always use all available historical data. A fixed rolling window discards older data unnecessarily. The 2020 COVID spike is a useful stress test — the model trained on pre-pandemic data predicts pandemic prices badly, which is honest and expected rather than a bug.

**How would you improve the model?**
> Quick wins in `feature_ideas.py`: (1) `district_x_property_type` interaction to capture sub-market effects, (2) `price_per_sqm` if we preserve `total_floor_area`, (3) cyclical `sale_month` encoding. Bigger wins: (1) external data — school ratings, crime indices, transport links per postcode/district; (2) stacking LGBM + XGB + Ridge with a Ridge meta-learner; (3) a separate specialist model for the >£1M luxury segment where errors are largest in £ terms.
