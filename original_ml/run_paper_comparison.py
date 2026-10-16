from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "np1": DATA_DIR / "MDS-UPDRS_Part_I.csv",
    "np2": DATA_DIR / "MDS-UPDRS_Part_II.csv",
    "np3": DATA_DIR / "MDS-UPDRS_Part_III.csv",
    "demo": DATA_DIR / "Demographics.csv",
}

RANDOM_STATE = 42
TEST_SIZE = 0.25            # Paper-inspired 75/25 patient split
TARGET_WINDOW_DAYS = (335, 395)
MIN_HISTORY_VISITS = 2
CORRELATION_THRESHOLD = 0.50
MAX_RFE_FEATURES = 20

# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    return df


def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns include: {list(df.columns[:40])}"
        )


def parse_date(df):
    if "INFODT" not in df.columns:
        raise ValueError(
            "INFODT was not found. Check the visit-date column in your PPMI files."
        )

    df = df.copy()
    df["INFODT"] = pd.to_datetime(df["INFODT"], errors="coerce")
    return df.dropna(subset=["PATNO", "INFODT"])


def numeric_features(df, exclude):
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() > 0:
            df[c] = converted
            cols.append(c)
    return cols


def metrics(y_true, y_pred, n_features):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    if n > n_features + 1:
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adjusted_r2 = np.nan

    return {
        "N": n,
        "Features": n_features,
        "MAE": mae,
        "RMSE": rmse,
        "Test_R2": r2,
        "Adjusted_R2": adjusted_r2,
    }


def print_results(results):
    result_df = pd.DataFrame(results).set_index("Model")
    print("\n" + "=" * 100)
    print("FINAL TEST-SET COMPARISON")
    print("=" * 100)
    print(result_df.round(4).to_string())
    print("\nLower MAE/RMSE is better. Higher test R² is better.")
    print(
        "Adjusted R² is a supplementary calculation, not directly equivalent "
        "to the paper's reported adjusted R²."
    )


# ============================================================
# LOAD AND PREPARE PPMI DATA
# ============================================================

print("Loading PPMI files...")

np1 = parse_date(load_csv(FILES["np1"]))
np2 = parse_date(load_csv(FILES["np2"]))
np3 = parse_date(load_csv(FILES["np3"]))
demo = load_csv(FILES["demo"])

require_columns(np1, ["PATNO", "INFODT", "NP1RTOT"], "Part I")
require_columns(np2, ["PATNO", "INFODT", "NP2PTOT"], "Part II")
require_columns(np3, ["PATNO", "INFODT", "NP3TOT"], "Part III")
require_columns(demo, ["PATNO"], "Demographics")

# Exclude the NP3 "unable to rate" value.
np3["NP3TOT"] = pd.to_numeric(np3["NP3TOT"], errors="coerce")
np3 = np3[np3["NP3TOT"].between(0, 132)].copy()

# Prefer OFF-state examinations when PDSTATE is available.
if "PDSTATE" in np3.columns:
    state = np3["PDSTATE"].astype(str).str.strip().str.upper()
    off = np3[state.eq("OFF")].copy()

    # If a patient/date has no OFF record, retain an available record.
    # If OFF exists for that patient/date, use OFF only.
    off_keys = set(zip(off["PATNO"], off["INFODT"]))
    fallback = np3[
        ~np3.apply(
            lambda r: (r["PATNO"], r["INFODT"]) in off_keys,
            axis=1
        )
    ]
    np3 = pd.concat([off, fallback], ignore_index=True)

# Remove duplicate patient/date records deterministically.
# This does not guarantee identical handling to the paper's original data.
for name, df in [("NP1", np1), ("NP2", np2), ("NP3", np3)]:
    print(f"{name}: {len(df):,} rows, {df['PATNO'].nunique():,} patients")

np1 = np1.sort_values("INFODT").drop_duplicates(
    ["PATNO", "INFODT"], keep="last"
)
np2 = np2.sort_values("INFODT").drop_duplicates(
    ["PATNO", "INFODT"], keep="last"
)
np3 = np3.sort_values("INFODT").drop_duplicates(
    ["PATNO", "INFODT"], keep="last"
)

# Prefix clinical columns to prevent name collisions.
def prefix_clinical(df, prefix):
    keys = ["PATNO", "INFODT"]
    rename = {
        c: f"{prefix}_{c}"
        for c in df.columns
        if c not in keys
    }
    return df.rename(columns=rename)


np1p = prefix_clinical(np1, "NP1")
np2p = prefix_clinical(np2, "NP2")
np3p = prefix_clinical(np3, "NP3")

# Use actual assessment dates; only dates shared by all three domains
# become matched assessment occasions.
visits = (
    np1p.merge(np2p, on=["PATNO", "INFODT"], how="inner")
        .merge(np3p, on=["PATNO", "INFODT"], how="inner")
)

visits = visits.sort_values(["PATNO", "INFODT"]).reset_index(drop=True)

# Add demographic variables. Keep one row per patient.
demo = demo.drop_duplicates("PATNO", keep="last").copy()

# Do not treat patient identifiers as model features.
visits = visits.merge(demo, on="PATNO", how="left", suffixes=("", "_DEMO"))

print(f"\nMatched assessment occasions: {len(visits):,}")
print(f"Patients with matched assessments: {visits['PATNO'].nunique():,}")

# ============================================================
# BUILD FUTURE-NP3 PREDICTION EXAMPLES
# ============================================================

# Current-year clinical and demographic predictors:
# item-level and numeric clinical fields, including current NP3TOT.
exclude = {
    "PATNO", "INFODT", "EVENT_ID", "PAG_NAME", "PDSTATE",
    "NP3_NP3TOT", "NP3_NP3TOT_FUTURE",
    "NP1_NP1RTOT_FUTURE", "NP2_NP2PTOT_FUTURE",
}

# Avoid columns that are identifiers, visit labels, or free-text fields.
exclude_prefixes = (
    "EVENT_ID", "PAG_NAME", "PDSTATE", "INFODT",
)

candidate_cols = []
for c in visits.columns:
    if c in exclude or c.endswith("_ID"):
        continue
    if c.startswith(exclude_prefixes):
        continue
    if c in {"PATNO"}:
        continue
    if pd.api.types.is_datetime64_any_dtype(visits[c]):
        continue

    numeric = pd.to_numeric(visits[c], errors="coerce")
    if numeric.notna().sum() > 0:
        visits[c] = numeric
        candidate_cols.append(c)

# Remove constant fields and fields with no usable values.
candidate_cols = [
    c for c in candidate_cols
    if visits[c].notna().any() and visits[c].nunique(dropna=True) > 1
]

# Remove possible future or outcome columns from input features.
candidate_cols = [
    c for c in candidate_cols
    if "FUTURE" not in c and not c.endswith("_TARGET")
]

if "NP3_NP3TOT" not in visits.columns:
    raise ValueError(
        "Expected NP3_NP3TOT after merging Part III. "
        "Check your NP3 total column name."
    )

examples = []
histories = []

for patno, patient in visits.groupby("PATNO", sort=False):
    patient = patient.sort_values("INFODT").reset_index(drop=True)

    for i in range(len(patient)):
        current = patient.iloc[i]
        index_date = current["INFODT"]

        prior = patient.iloc[:i + 1].copy()
        if len(prior) < MIN_HISTORY_VISITS:
            continue

        # Require sufficient observations in the previous 12 months.
        history_start = index_date - pd.Timedelta(days=365)
        recent = prior.loc[prior["INFODT"].ge(history_start)].copy()
        if len(recent) < MIN_HISTORY_VISITS:
            continue

        # Select the closest future NP3 assessment in the target window.
        future = patient[
            patient["INFODT"] > index_date
        ].copy()

        if future.empty:
            continue

        future["GAP_DAYS"] = (
            future["INFODT"] - index_date
        ).dt.days

        future = future[
            future["GAP_DAYS"].between(
                TARGET_WINDOW_DAYS[0],
                TARGET_WINDOW_DAYS[1]
            )
        ]

        if future.empty:
            continue

        target_row = future.iloc[
            (future["GAP_DAYS"] - 365).abs().argmin()
        ]
        y_future = float(target_row["NP3_NP3TOT"])
        current_np3 = pd.to_numeric(
            pd.Series([current["NP3_NP3TOT"]]), errors="coerce"
        ).iloc[0]

        if pd.isna(y_future) or pd.isna(current_np3):
            continue

        # Paper-inspired rule: exclude samples where NP3 decreases.
        # This is applied as a sensitivity/reproduction choice and may
        # change the target population compared with your existing model.
        if y_future < current_np3:
            continue

        row = {
            "PATNO": patno,
            "INDEX_DATE": index_date,
            "TARGET_DATE": target_row["INFODT"],
            "TARGET_GAP_DAYS": int(target_row["GAP_DAYS"]),
            "Y": y_future,
            "LAST_NP3": float(current_np3),
        }

        # Current-visit clinical and demographic features.
        for c in candidate_cols:
            row[f"CURRENT_{c}"] = current[c]

        # Historical summary features for the temporal Random Forest.
        for domain_col in [
            "NP1_NP1RTOT",
            "NP2_NP2PTOT",
            "NP3_NP3TOT",
        ]:
            if domain_col not in recent.columns:
                continue

            vals = pd.to_numeric(
                recent[domain_col], errors="coerce"
            ).dropna()

            if len(vals) == 0:
                continue

            row[f"TEMP_{domain_col}_LAST"] = vals.iloc[-1]
            row[f"TEMP_{domain_col}_MEAN"] = vals.mean()
            row[f"TEMP_{domain_col}_MIN"] = vals.min()
            row[f"TEMP_{domain_col}_MAX"] = vals.max()

            if len(vals) >= 2:
                x = np.arange(len(vals), dtype=float)
                row[f"TEMP_{domain_col}_SLOPE"] = np.polyfit(
                    x, vals.to_numpy(dtype=float), 1
                )[0]
            else:
                row[f"TEMP_{domain_col}_SLOPE"] = np.nan

        row["HISTORY_VISITS"] = len(recent)
        examples.append(row)

if not examples:
    raise ValueError(
        "No eligible prediction examples were created. Check the date "
        "columns, matched assessment dates, target window, and visit coverage."
    )

data = pd.DataFrame(examples).sort_values(
    ["INDEX_DATE", "PATNO"]
).reset_index(drop=True)

print(f"\nPrediction examples after target-window and decrease filters: {len(data):,}")
print(f"Unique patients in prediction examples: {data['PATNO'].nunique():,}")
print(
    "Target gap (days): "
    f"median={data['TARGET_GAP_DAYS'].median():.0f}, "
    f"min={data['TARGET_GAP_DAYS'].min()}, "
    f"max={data['TARGET_GAP_DAYS'].max()}"
)

# ============================================================
# ONE PATIENT-LEVEL TRAIN/TEST SPLIT FOR EVERY MODEL
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(data, data["Y"], groups=data["PATNO"])
)

train = data.iloc[train_idx].copy()
test = data.iloc[test_idx].copy()
# Save constructed prediction dataset and train/test datasets
data.to_csv(OUTPUT_DIR / "all_prediction_examples.csv", index=False)
train.to_csv(OUTPUT_DIR / "training_dataset.csv", index=False)
test.to_csv(OUTPUT_DIR / "testing_dataset.csv", index=False)

print(f"\nDatasets saved in: {OUTPUT_DIR}")

train_patients = set(train["PATNO"])
test_patients = set(test["PATNO"])

assert train_patients.isdisjoint(test_patients), "Patient leakage detected!"

print("\n" + "=" * 100)
print("SPLIT DETAILS")
print("=" * 100)
print(f"Train examples: {len(train)}")
print(f"Test examples:  {len(test)}")
print(f"Train patients: {len(train_patients)}")
print(f"Test patients:  {len(test_patients)}")
print(f"Patient overlap: {len(train_patients & test_patients)}")

y_train = train["Y"].astype(float)
y_test = test["Y"].astype(float)

results = []

# ============================================================
# 1. LAST-SCORE BASELINE
# ============================================================

baseline_pred = test["LAST_NP3"].astype(float).to_numpy()

results.append({
    "Model": "Last NP3 score baseline",
    **metrics(
        y_test,
        baseline_pred,
        n_features=1
    )
})

# ============================================================
# 2. PAPER-INSPIRED FEATURE SELECTION
#    Spearman filtering and RFE use TRAINING DATA ONLY.
# ============================================================

paper_features = [
    c for c in candidate_cols
    if f"CURRENT_{c}" in data.columns
]

X_train_raw = train[[f"CURRENT_{c}" for c in candidate_cols]].copy()
X_test_raw = test[[f"CURRENT_{c}" for c in candidate_cols]].copy()

# Remove features that are completely missing in the training set.
usable = [
    c for c in X_train_raw.columns
    if X_train_raw[c].notna().any()
    and X_train_raw[c].nunique(dropna=True) > 1
]

X_train_raw = X_train_raw[usable]
X_test_raw = X_test_raw[usable]

# Imputation statistics are learned only from the training set.
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train_raw),
    columns=usable,
    index=train.index
)
X_test_imp = pd.DataFrame(
    imputer.transform(X_test_raw),
    columns=usable,
    index=test.index
)

# Spearman filtering against the outcome, calculated on training data only.
# This selects features associated with the target; it is not a causal test.
correlations = {}
for c in X_train_imp.columns:
    if X_train_imp[c].nunique() <= 1:
        continue
    rho, _ = spearmanr(X_train_imp[c], y_train)
    if np.isfinite(rho):
        correlations[c] = rho

selected_corr = [
    c for c, rho in correlations.items()
    if abs(rho) >= CORRELATION_THRESHOLD
]

# Fallback only if the threshold leaves no features.
if not selected_corr and correlations:
    selected_corr = [
        max(correlations, key=lambda c: abs(correlations[c]))
    ]

if not selected_corr:
    raise ValueError(
        "Spearman filtering retained no usable features. "
        "Inspect feature missingness and data types."
    )

print("\n" + "=" * 100)
print("PAPER-INSPIRED FEATURE SELECTION")
print("=" * 100)
print(f"Numeric candidate features: {len(candidate_cols)}")
print(f"Usable training features: {len(usable)}")
print(f"Features after Spearman filtering: {len(selected_corr)}")
print("Selected features:")
for c in selected_corr:
    print(f"  {c}: Spearman rho = {correlations[c]:.3f}")

Xtr_corr = X_train_imp[selected_corr].copy()
Xte_corr = X_test_imp[selected_corr].copy()

# RFE: fit only on training data.
# Use LinearRegression as the ranking estimator.
n_rfe = min(MAX_RFE_FEATURES, Xtr_corr.shape[1])

if n_rfe < Xtr_corr.shape[1]:
    rfe = RFE(
        estimator=LinearRegression(),
        n_features_to_select=n_rfe,
        step=1
    )
    rfe.fit(Xtr_corr, y_train)
    rfe_features = Xtr_corr.columns[rfe.support_].tolist()
else:
    rfe_features = Xtr_corr.columns.tolist()

print(f"\nFeatures retained by training-only RFE: {len(rfe_features)}")
for c in rfe_features:
    print(f"  {c}")
# Save selected features for the paper-inspired models
pd.DataFrame({
    "selected_feature": rfe_features
}).to_csv(OUTPUT_DIR / "paper_inspired_selected_features.csv", index=False)
Xtr_paper = Xtr_corr[rfe_features]
Xte_paper = Xte_corr[rfe_features]

# ============================================================
# 3. PAPER-INSPIRED REGRESSION MODELS
# ============================================================

paper_models = {
    "Paper-inspired Linear Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Paper-inspired Ridge": Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
    "Paper-inspired Bayesian Ridge": Pipeline([
        ("scale", StandardScaler()),
        ("model", BayesianRidge()),
    ]),
    "Paper-inspired Random Forest": RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Paper-inspired Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        loss="squared_error",
        random_state=RANDOM_STATE
    ),
}

for name, model in paper_models.items():
    model.fit(Xtr_paper, y_train)
    pred = model.predict(Xte_paper)

    results.append({
        "Model": name,
        **metrics(
            y_test,
            pred,
            n_features=Xtr_paper.shape[1]
        )
    })

# ============================================================
# 4. TEMPORAL RANDOM FOREST
#    Historical LAST / MEAN / MIN / MAX / SLOPE summaries.
# ============================================================

temporal_features = [
    c for c in data.columns
    if c.startswith("TEMP_")
]

if not temporal_features:
    raise ValueError(
        "No temporal features were generated. "
        "Check that NP1_NP1RTOT, NP2_NP2PTOT and NP3_NP3TOT exist."
    )

Xtr_temp = train[temporal_features].copy()
Xte_temp = test[temporal_features].copy()

temp_imputer = SimpleImputer(strategy="median")
Xtr_temp = pd.DataFrame(
    temp_imputer.fit_transform(Xtr_temp),
    columns=temporal_features,
    index=train.index
)
Xte_temp = pd.DataFrame(
    temp_imputer.transform(Xte_temp),
    columns=temporal_features,
    index=test.index
)

temporal_rf = RandomForestRegressor(
    n_estimators=400,
    min_samples_leaf=12,
    max_features=0.7,
    max_depth=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

temporal_rf.fit(Xtr_temp, y_train)
temporal_pred = temporal_rf.predict(Xte_temp)

results.append({
    "Model": "Temporal Random Forest",
    **metrics(
        y_test,
        temporal_pred,
        n_features=Xtr_temp.shape[1]
    )
})

# ============================================================
# PRINT RESULTS ONLY — NO OUTPUT FILES ARE CREATED
# ============================================================

print_results(results)

print("\n" + "=" * 100)
print("REPRODUCTION NOTES")
print("=" * 100)
print("1. All models use the same patient-level train/test partition.")
print("2. Spearman filtering and RFE are fitted using training data only.")
print("3. Imputation is fitted using training data only.")
print("4. Test R² is ordinary, unadjusted R².")
print("5. Adjusted R² is shown separately and depends on the feature count.")
print("6. Samples with a lower future NP3 score than current NP3 were excluded.")
print("7. The target is a future NP3 assessment score, not a direct measure")
print("   of disease progression.")
print("8. This is paper-inspired, not an exact reproduction of Ma et al.")
print("9. Only matched NP1/NP2/NP3 assessment dates are used in this script.")
print("10. Verify source-column definitions and the paper's exact preprocessing")
print("    rules before treating these results as a direct published comparison.")