
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT = Path(__file__).resolve().parent / "outputs"

files = {
    "Retuned Temporal RF": (
        "shared_model_predictions.csv",
        "RandomForest",
    ),
    "GRU": (
        "shared_model_predictions.csv",
        "GRU",
    ),
    "Previously Tuned RF": (
        "shared_tuned_rf_predictions.csv",
        "Tuned_RF",
    ),
    "Chronos-T5 Small": (
        "chronos_shared_predictions.csv",
        "Chronos",
    ),
}

merged = None

for model_name, (filename, prediction_col) in files.items():
    path = OUT / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding model script first."
        )

    df = pd.read_csv(path)

    if "EXAMPLE_ID" not in df.columns:
        raise ValueError(f"{filename} has no EXAMPLE_ID column.")

    actual_col = next(
        (c for c in ["Actual", "Actual_RF", "Actual_GRU"]
         if c in df.columns),
        None,
    )

    if actual_col is None or prediction_col not in df.columns:
        raise ValueError(
            f"Expected actual score and '{prediction_col}' "
            f"in {filename}. Found: {list(df.columns)}"
        )

    df = df[["EXAMPLE_ID", actual_col, prediction_col]].copy()
    df = df.rename(columns={
        actual_col: f"Actual_{model_name}",
        prediction_col: model_name,
    })

    if merged is None:
        merged = df
    else:
        merged = merged.merge(
            df,
            on="EXAMPLE_ID",
            how="inner",
            validate="one_to_one",
        )

actual_cols = [f"Actual_{name}" for name in files]

reference = merged[actual_cols[0]]

for col in actual_cols[1:]:
    if not np.allclose(reference, merged[col], atol=1e-6):
        raise ValueError(
            f"Actual target scores disagree between models: {col}"
        )

merged["Actual"] = reference
prediction_cols = list(files.keys())

merged = merged.dropna(subset=["Actual"] + prediction_cols)

if len(merged) == 0:
    raise ValueError("No common test examples across all four models.")

y_true = merged["Actual"].to_numpy()

results = []

for model_name in prediction_cols:
    pred = merged[model_name].to_numpy()

    results.append({
        "Model": model_name,
        "Test_examples": len(merged),
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, pred)),
        "R2": r2_score(y_true, pred),
    })

comparison = pd.DataFrame(results).sort_values("MAE")

comparison.to_csv(OUT / "all_four_model_comparison.csv", index=False)
merged.to_csv(OUT / "all_four_model_predictions.csv", index=False)

print("\n========== ALL FOUR MODELS: SHARED TEST SET ==========")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nCommon test examples: {len(merged)}")
print("Saved comparison:", OUT / "all_four_model_comparison.csv")
print("Saved predictions:", OUT / "all_four_model_predictions.csv")
