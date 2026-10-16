
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "usable_patient_ids.csv"


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_csv(dataset):
    aliases = {
        "np1": {"np1", "mdupsparti", "mdupspart1"},
        "np2": {"np2", "mdupspartii", "mdupspart2"},
        "np3": {"np3", "mdupspartiii", "mdupspart3"},
    }

    candidates = [
        p for p in ROOT.rglob("*.csv")
        if normalize(p.stem) in aliases[dataset]
        and not any(part.startswith(".") for part in p.parts)
        and "node_modules" not in p.parts
        and ".venv" not in p.parts
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find the {dataset.upper()} CSV under {ROOT}"
        )

    # Prefer the backend's synced filename when available.
    candidates.sort(
        key=lambda p: (
            p.stem.lower() != dataset,
            "backend" not in str(p).lower(),
            len(p.parts),
        )
    )

    print(f"{dataset.upper()} file: {candidates[0]}")
    if len(candidates) > 1:
        print(f"  Note: found {len(candidates)} matching files.")

    return candidates[0]


def load_csv(dataset):
    df = pd.read_csv(find_csv(dataset), low_memory=False)
    df.columns = df.columns.str.strip().str.upper()

    required = {"PATNO", "INFODT"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset.upper()} is missing columns: {sorted(missing)}"
        )

    df["PATNO"] = df["PATNO"].astype("string").str.strip()
    df["INFODT"] = pd.to_datetime(df["INFODT"], errors="coerce")

    df = df.dropna(subset=["PATNO", "INFODT"])
    df = df[df["PATNO"] != ""]

    return df


def main():
    np1 = load_csv("np1")
    np2 = load_csv("np2")
    np3 = load_csv("np3")

    for df, score, maximum in [
        (np1, "NP1RTOT", 24),
        (np2, "NP2PTOT", 52),
        (np3, "NP3TOT", 132),
    ]:
        if score not in df.columns:
            raise ValueError(f"Required score column missing: {score}")

        df[score] = pd.to_numeric(df[score], errors="coerce")
        valid = df[score].between(0, maximum)

        if score == "NP3TOT":
            valid &= df[score] != 101

        df.drop(df.index[~valid], inplace=True)

    # Use OFF-state NP3 records, matching the project pipeline.
    if "PDSTATE" in np3.columns:
        state = (
            np3["PDSTATE"].astype("string")
            .str.strip().str.upper()
        )
        np3 = np3[state.isin(["OFF", "OFF STATE", "OFF-STATE"])]
    elif "PAG_NAME" in np3.columns:
        np3 = np3[
            np3["PAG_NAME"].astype("string").str.strip().str.upper()
            == "NUPDR3OF"
        ]
    else:
        raise ValueError(
            "NP3 has neither PDSTATE nor PAG_NAME; "
            "cannot verify OFF-state eligibility."
        )

    # Average duplicate records for the same patient and date.
    np1 = np1.groupby(["PATNO", "INFODT"], as_index=False)["NP1RTOT"].mean()
    np2 = np2.groupby(["PATNO", "INFODT"], as_index=False)["NP2PTOT"].mean()
    np3 = np3.groupby(["PATNO", "INFODT"], as_index=False)["NP3TOT"].mean()

    # Only dates present in all three assessment domains.
    visits = (
        np1.merge(np2, on=["PATNO", "INFODT"], how="inner")
        .merge(np3, on=["PATNO", "INFODT"], how="inner")
        .sort_values(["PATNO", "INFODT"])
    )

    if visits.empty:
        raise RuntimeError(
            "No matching visits found. Check the CSV paths, dates, "
            "and PDSTATE values."
        )

    results = []

    for patient_id, group in visits.groupby("PATNO"):
        group = group.sort_values("INFODT")
        latest_date = group["INFODT"].max()

        # History window ending at the latest matched assessment.
        cutoff = latest_date - pd.DateOffset(months=12)
        history = group[
            (group["INFODT"] >= cutoff)
            & (group["INFODT"] <= latest_date)
        ]

        results.append({
            "PATNO": patient_id,
            "MATCHED_VISITS_TOTAL": len(group),
            "LATEST_MATCHED_DATE": latest_date.strftime("%Y-%m-%d"),
            "HISTORY_VISITS_LAST_12_MONTHS": len(history),
            "USABLE": len(history) >= 2,
        })

    report = pd.DataFrame(results)
    report = report.sort_values(
        ["USABLE", "HISTORY_VISITS_LAST_12_MONTHS"],
        ascending=[False, False],
    )

    report.to_csv(OUTPUT, index=False)

    usable = report[report["USABLE"]]

    print("\n========== RESULTS ==========")
    print(f"Patients with matched visits: {len(report)}")
    print(f"Patients with >= 2 visits in the last 12 months: {len(usable)}")
    print(f"\nSaved full report to:\n{OUTPUT}")

    if not usable.empty:
        print("\nFirst 30 usable participant IDs:")
        print(usable.head(30).to_string(index=False))
    else:
        print("\nNo participants met the two-visit criterion.")

    print(
        "\nNote: these IDs meet the matched-visit/history checks. "
        "The prediction API must still confirm that each ID is usable "
        "for its selected index date."
    )


if __name__ == "__main__":
    main()
