
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
TRAIN_FILE = OUTPUT / "training_dataset.csv"
MODEL_FILE = OUTPUT / "paper_method1_model.joblib"

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"{TRAIN_FILE} not found. Run run_paper_comparison_updated.py first."
    )

train = pd.read_csv(TRAIN_FILE)

if "Y" not in train.columns:
    raise ValueError("Training dataset must contain the target column Y.")

features = [
    c for c in train.columns
    if c.startswith("CURRENT_")
    and "FUTURE" not in c
    and not c.endswith("_TARGET")
]

if not features:
    raise ValueError("No CURRENT_ feature columns found.")

y = pd.to_numeric(train["Y"], errors="coerce")
X = train[features].apply(pd.to_numeric, errors="coerce")

valid = y.notna()
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

usable = [
    c for c in X.columns
    if X[c].notna().any() and X[c].nunique(dropna=True) > 1
]

if not usable:
    raise ValueError("No usable training features remain.")

X = X[usable]

imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(
    imputer.fit_transform(X),
    columns=usable,
)

correlations = {}

for col in usable:
    if X_imp[col].nunique() <= 1:
        continue

    rho, _ = spearmanr(X_imp[col], y)

    if np.isfinite(rho):
        correlations[col] = float(rho)

if not correlations:
    raise ValueError("Could not calculate usable Spearman correlations.")

selected_corr = [
    col for col, rho in correlations.items()
    if abs(rho) >= 0.50
]

if not selected_corr:
    selected_corr = [
        max(correlations, key=lambda col: abs(correlations[col]))
    ]

X_corr = X_imp[selected_corr]
n_features = min(20, X_corr.shape[1])

if n_features < X_corr.shape[1]:
    selector = RFE(
        estimator=LinearRegression(),
        n_features_to_select=n_features,
        step=1,
    )
    selector.fit(X_corr, y)
    selected_features = X_corr.columns[selector.support_].tolist()
else:
    selected_features = X_corr.columns.tolist()

model = RandomForestRegressor(
    n_estimators=400,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_imp[selected_features], y)

bundle = {
    "model": model,
    "imputer": imputer,
    "usable_features": usable,
    "selected_features": selected_features,
    "target": "Y",
    "model_name": "Paper-inspired Random Forest",
}

joblib.dump(bundle, MODEL_FILE)

pd.DataFrame({
    "selected_feature": selected_features
}).to_csv(
    OUTPUT / "paper_method1_selected_features.csv",
    index=False,
)

print("Method 1 model saved successfully.")
print(f"Training examples: {len(X)}")
print(f"Usable features: {len(usable)}")
print(f"Selected features: {len(selected_features)}")
print(f"Model artifact: {MODEL_FILE}")
