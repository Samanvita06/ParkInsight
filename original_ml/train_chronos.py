
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronos import ChronosPipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"

DATASET_FILE = OUTPUT / "shared_examples.csv"
NP3_FILE = DATA / "MDS-UPDRS_Part_III.csv"

MODEL_NAME = "amazon/chronos-t5-small"
RANDOM_STATE = 42


def load_np3_history():
    df = pd.read_csv(NP3_FILE, low_memory=False)

    df["INFODT"] = pd.to_datetime(df["INFODT"], errors="coerce")
    df["NP3TOT"] = pd.to_numeric(df["NP3TOT"], errors="coerce")

    state = (
        df["PDSTATE"].astype("string").str.strip().str.upper()
        if "PDSTATE" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )

    form = (
        df["PAG_NAME"].astype("string").str.strip().str.upper()
        if "PAG_NAME" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )

    off_mask = state.eq("OFF") | (
        state.isna() & form.eq("NUPDR3OF")
    )

    df = df[
        off_mask
        & df["NP3TOT"].between(0, 132)
        & df["NP3TOT"].ne(101)
    ].dropna(subset=["PATNO", "INFODT", "NP3TOT"])

    # Use one score per patient per date.
    df = (
        df.groupby(["PATNO", "INFODT"], as_index=False)["NP3TOT"]
        .mean()
        .sort_values(["PATNO", "INFODT"])
    )

    return df


def main():
    dataset = pd.read_csv(DATASET_FILE)
    dataset["INDEX_DATE"] = pd.to_datetime(dataset["INDEX_DATE"])
    dataset["TARGET_DATE"] = pd.to_datetime(dataset["TARGET_DATE"])

    train = dataset[dataset["SPLIT"] == "train"].copy()
    test = dataset[dataset["SPLIT"] == "test"].copy()

    assert set(train["PATNO"]).isdisjoint(set(test["PATNO"]))

    np3 = load_np3_history()

    print(f"Train examples: {len(train)}")
    print(f"Test examples: {len(test)}")
    print(f"Train patients: {train['PATNO'].nunique()}")
    print(f"Test patients: {test['PATNO'].nunique()}")
    print(f"Loading pretrained model: {MODEL_NAME}")

    pipeline = ChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    predictions = []

    for i, row in enumerate(test.itertuples(index=False), start=1):
        history = np3[
            (np3["PATNO"] == row.PATNO)
            & (np3["INFODT"] <= row.INDEX_DATE)
        ]["NP3TOT"].to_numpy(dtype=np.float32)

        if len(history) < 2:
            predictions.append(np.nan)
            continue

        context = torch.tensor(history, dtype=torch.float32)

        with torch.no_grad():
            forecast = pipeline.predict(
                context,
                prediction_length=1,
                num_samples=20,
            )

        prediction = torch.quantile(
            forecast[0, :, 0].float(), 0.5
        ).item()

        predictions.append(float(np.clip(prediction, 0, 132)))

        if i % 10 == 0:
            print(f"Processed {i}/{len(test)} test examples")

    test["Chronos"] = predictions

    result = test[
        ["EXAMPLE_ID", "PATNO", "NP3TOT_FUTURE", "Chronos"]
    ].rename(columns={"NP3TOT_FUTURE": "Actual"})

    result.to_csv(OUTPUT / "chronos_shared_predictions.csv", index=False)

    evaluated = result.dropna(subset=["Chronos", "Actual"])

    metrics = pd.DataFrame([{
        "Model": "Chronos-T5 Small",
        "Test_examples": len(evaluated),
        "Test_patients": evaluated["PATNO"].nunique(),
        "MAE": mean_absolute_error(
            evaluated["Actual"], evaluated["Chronos"]
        ),
        "RMSE": np.sqrt(mean_squared_error(
            evaluated["Actual"], evaluated["Chronos"]
        )),
        "R2": r2_score(evaluated["Actual"], evaluated["Chronos"]),
    }])

    metrics.to_csv(OUTPUT / "chronos_shared_metrics.csv", index=False)

    print("\n========== CHRONOS SHARED-SPLIT RESULTS ==========")
    print(metrics.round(3).to_string(index=False))
    print(f"\nSaved results in: {OUTPUT}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
