
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MIN_HISTORY_VISITS = 2
HISTORY_MONTHS = 12
TARGET_MONTHS = 12
TARGET_TOLERANCE_DAYS = 30
TEST_SIZE = 0.20
SEED = 42


def load_domain(filename, total, maximum):
    df = pd.read_csv(DATA / filename, low_memory=False)

    df["INFODT"] = pd.to_datetime(df["INFODT"], errors="coerce")
    df[total] = pd.to_numeric(df[total], errors="coerce")

    df = df.dropna(subset=["PATNO", "INFODT", total])
    df = df[df[total].between(0, maximum)]

    return (
        df.groupby(["PATNO", "INFODT"], as_index=False)[total]
        .mean()
    )


def load_np3():
    df = pd.read_csv(
        DATA / "MDS-UPDRS_Part_III.csv",
        low_memory=False,
    )

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

    # Prefer explicit OFF assessments. Use OFF-form fallback only
    # when PDSTATE is missing.
    off = state.eq("OFF") | (
        state.isna() & form.eq("NUPDR3OF")
    )

    df = df[
        off
        & df["NP3TOT"].between(0, 132)
        & df["NP3TOT"].ne(101)
    ].dropna(subset=["PATNO", "INFODT", "NP3TOT"])

    # If multiple eligible OFF assessments exist on a date,
    # use their mean for that date.
    return (
        df.groupby(["PATNO", "INFODT"], as_index=False)["NP3TOT"]
        .mean()
    )


def slope(history, column, index_date):
    # Time measured in months relative to the index date.
    x = (
        history["INFODT"] - index_date
    ).dt.total_seconds().to_numpy() / (30.4375 * 86400)

    y = history[column].to_numpy(dtype=float)

    if len(np.unique(x)) < 2:
        return np.nan

    return float(np.polyfit(x, y, 1)[0])


def main():
    print("Loading PPMI assessment files...")

    np1 = load_domain("MDS-UPDRS_Part_I.csv", "NP1RTOT", 24)
    np2 = load_domain("MDS-UPDRS_Part_II.csv", "NP2PTOT", 52)
    np3 = load_np3()

    # A visit is retained only when all three domains have a
    # valid score on the same patient-date.
    visits = (
        np1.merge(np2, on=["PATNO", "INFODT"], how="inner")
        .merge(np3, on=["PATNO", "INFODT"], how="inner")
        .sort_values(["PATNO", "INFODT"])
        .reset_index(drop=True)
    )

    by_patient = {
        patient: group.reset_index(drop=True)
        for patient, group in visits.groupby("PATNO", sort=False)
    }

    examples = []
    sequence_records = []
    feature_cols = ["NP1RTOT", "NP2PTOT", "NP3TOT"]

    print("Building shared prediction examples...")

    for patient, group in by_patient.items():
        if len(group) < 3:
            continue

        for i in range(1, len(group) - 1):
            index_date = group.loc[i, "INFODT"]
            target_date = index_date + pd.DateOffset(
                months=TARGET_MONTHS
            )
            history_start = index_date - pd.DateOffset(
                months=HISTORY_MONTHS
            )

            history = group[
                (group["INFODT"] >= history_start)
                & (group["INFODT"] <= index_date)
            ].copy()

            if len(history) < MIN_HISTORY_VISITS:
                continue

            future = group[group["INFODT"] > index_date].copy()
            if future.empty:
                continue

            future["target_distance"] = (
                future["INFODT"] - target_date
            ).abs().dt.days

            candidates = future[
                future["target_distance"] <= TARGET_TOLERANCE_DAYS
            ]

            if candidates.empty:
                continue

            target = candidates.sort_values(
                ["target_distance", "INFODT"]
            ).iloc[0]

            example_id = (
                f"{patient}_{index_date.strftime('%Y%m%d')}_"
                f"{target['INFODT'].strftime('%Y%m%d')}"
            )

            row = {
                "EXAMPLE_ID": example_id,
                "PATNO": patient,
                "INDEX_DATE": index_date,
                "TARGET_DATE": target["INFODT"],
                "TARGET_GAP_DAYS": (
                    target["INFODT"] - index_date
                ).days,
                "NP3TOT_FUTURE": float(target["NP3TOT"]),
                "HISTORY_VISITS": len(history),
            }

            # Traditional ML representation: summary features.
            for column in feature_cols:
                values = history[column].astype(float)

                row[f"{column}_LAST"] = float(values.iloc[-1])
                row[f"{column}_MEAN"] = float(values.mean())
                row[f"{column}_MIN"] = float(values.min())
                row[f"{column}_MAX"] = float(values.max())
                row[f"{column}_SLOPE"] = slope(
                    history, column, index_date
                )

            examples.append(row)

            # GRU representation: one record per historical visit.
            for visit_number, (_, visit) in enumerate(
                history.iterrows(), start=1
            ):
                sequence_records.append({
                    "EXAMPLE_ID": example_id,
                    "PATNO": patient,
                    "INDEX_DATE": index_date,
                    "VISIT_NUMBER": visit_number,
                    "VISIT_DATE": visit["INFODT"],
                    "MONTHS_BEFORE_INDEX": (
                        index_date - visit["INFODT"]
                    ).days / 30.4375,
                    "MONTHS_SINCE_PREVIOUS_VISIT": (
                        np.nan if visit_number == 1
                        else (
                            visit["INFODT"]
                            - history.iloc[visit_number - 2]["INFODT"]
                        ).days / 30.4375
                    ),
                    "NP1RTOT": float(visit["NP1RTOT"]),
                    "NP2PTOT": float(visit["NP2PTOT"]),
                    "NP3TOT": float(visit["NP3TOT"]),
                })

    temporal = pd.DataFrame(examples)
    sequences = pd.DataFrame(sequence_records)

    if temporal.empty:
        raise ValueError(
            "No examples created. Check source files, dates, "
            "OFF-state filtering, and target tolerance."
        )

    # Remove examples with unavailable engineered features.
    temporal = temporal.replace([np.inf, -np.inf], np.nan)

    summary_cols = [
        col for col in temporal.columns
        if col not in {
            "EXAMPLE_ID", "PATNO", "INDEX_DATE",
            "TARGET_DATE", "NP3TOT_FUTURE",
        }
    ]

    temporal = temporal.dropna(
        subset=summary_cols + ["NP3TOT_FUTURE"]
    ).reset_index(drop=True)

    # Retain only sequence rows belonging to kept examples.
    valid_ids = set(temporal["EXAMPLE_ID"])
    sequences = sequences[
        sequences["EXAMPLE_ID"].isin(valid_ids)
    ].copy()

    # Create ONE patient-level split shared by all representations.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    train_idx, test_idx = next(
        splitter.split(
            temporal,
            temporal["NP3TOT_FUTURE"],
            groups=temporal["PATNO"],
        )
    )

    temporal["SPLIT"] = "train"
    temporal.loc[test_idx, "SPLIT"] = "test"

    # Give every GRU sequence the same assignment as its example.
    split_map = temporal.set_index("EXAMPLE_ID")["SPLIT"]
    sequences["SPLIT"] = sequences["EXAMPLE_ID"].map(split_map)

    examples_output = temporal[
        [
            "EXAMPLE_ID", "PATNO", "INDEX_DATE", "TARGET_DATE",
            "TARGET_GAP_DAYS", "NP3TOT_FUTURE",
            "HISTORY_VISITS", "SPLIT",
        ]
    ].copy()

    temporal.to_csv(OUT / "temporal_features.csv", index=False)
    sequences.to_csv(OUT / "gru_sequences.csv", index=False)
    examples_output.to_csv(OUT / "shared_examples.csv", index=False)

    train_patients = set(
        temporal.loc[temporal["SPLIT"] == "train", "PATNO"]
    )
    test_patients = set(
        temporal.loc[temporal["SPLIT"] == "test", "PATNO"]
    )

    print("\n========== SHARED DATASETS CREATED ==========")
    print(f"Matched visit rows: {len(visits)}")
    print(f"Prediction examples: {len(temporal)}")
    print(f"Unique patients: {temporal['PATNO'].nunique()}")
    print(f"Train examples: {(temporal['SPLIT'] == 'train').sum()}")
    print(f"Test examples: {(temporal['SPLIT'] == 'test').sum()}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")
    print(f"Mean history visits: {temporal['HISTORY_VISITS'].mean():.2f}")
    print(f"GRU visit rows: {len(sequences)}")
    print(f"Train/test patient overlap: {len(train_patients & test_patients)}")

    print("\nSaved:")
    print(OUT / "temporal_features.csv")
    print(OUT / "gru_sequences.csv")
    print(OUT / "shared_examples.csv")


if __name__ == "__main__":
    main()
