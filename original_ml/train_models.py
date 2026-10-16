
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

df = pd.read_csv(OUT / "prediction_dataset.csv")

features = [
    "NP1RTOT", "NP2PTOT", "NP3TOT_CURRENT",
    "NP1RTOT_CHANGE", "NP2PTOT_CHANGE", "NP3TOT_CHANGE"
]
target = "NP3TOT_FUTURE"

df = df.dropna(subset=features + [target, "PATNO"]).copy()

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42
)
train_idx, test_idx = next(
    splitter.split(df[features], df[target], groups=df["PATNO"])
)

train = df.iloc[train_idx]
test = df.iloc[test_idx]

X_train, X_test = train[features], test[features]
y_train, y_test = train[target], test[target]

def evaluate(name, actual, predicted):
    return {
        "Model": name,
        "Test_examples": len(actual),
        "Test_patients": test["PATNO"].nunique(),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }

# 1. Baseline: assume no change in NP3
baseline_pred = X_test["NP3TOT_CURRENT"].to_numpy()
results = [
    evaluate("Persistence baseline", y_test, baseline_pred)
]

# 2. Original Random Forest
original = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
original.fit(X_train, y_train)
original_pred = original.predict(X_test)
results.append(evaluate("Original Random Forest", y_test, original_pred))

# 3. Tune Random Forest using training patients only
param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_leaf": [2, 3, 5, 8, 12],
    "max_features": [0.7, 1.0]
}

cv = GroupKFold(n_splits=4)
search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=1),
    param_distributions=param_grid,
    n_iter=15,
    scoring="neg_mean_absolute_error",
    cv=cv,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train, groups=train["PATNO"])
best_model = search.best_estimator_
tuned_pred = best_model.predict(X_test)

results.append(evaluate("Tuned Random Forest", y_test, tuned_pred))

joblib.dump(best_model, OUT / "random_forest_model.joblib")

test_results = test[
    ["PATNO", "INDEX_DATE", "TARGET_DATE", target]
].copy()
test_results["CURRENT_NP3"] = X_test["NP3TOT_CURRENT"]
test_results["PREDICTED_NP3"] = tuned_pred
test_results.to_csv(OUT / "random_forest_predictions.csv", index=False)

metrics = pd.DataFrame(results)
metrics.to_csv(OUT / "model_metrics.csv", index=False)

print("========== MODEL COMPARISON ==========")
print(metrics.round(3).to_string(index=False))
print("\nBest parameters:", search.best_params_)
print("Best cross-validation MAE:", round(-search.best_score_, 3))
print("\nSaved results in outputs/.")

