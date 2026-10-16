
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

np1 = pd.read_csv(DATA / "MDS-UPDRS_Part_I.csv", low_memory=False)
np2 = pd.read_csv(DATA / "MDS-UPDRS_Part_II.csv", low_memory=False)
np3 = pd.read_csv(DATA / "MDS-UPDRS_Part_III.csv", low_memory=False)

for df in (np1, np2, np3):
    df["INFODT"] = pd.to_datetime(df["INFODT"], errors="coerce")

np1 = np1[["PATNO", "INFODT", "NP1RTOT"]].copy()
np2 = np2[["PATNO", "INFODT", "NP2PTOT"]].copy()

np1["NP1RTOT"] = pd.to_numeric(np1["NP1RTOT"], errors="coerce")
np2["NP2PTOT"] = pd.to_numeric(np2["NP2PTOT"], errors="coerce")

np1.loc[~np1["NP1RTOT"].between(0, 24), "NP1RTOT"] = float("nan")
np2.loc[~np2["NP2PTOT"].between(0, 52), "NP2PTOT"] = float("nan")

np1 = np1.dropna(subset=["PATNO", "INFODT"])
np2 = np2.dropna(subset=["PATNO", "INFODT"])

np1 = np1.groupby(["PATNO", "INFODT"], as_index=False)["NP1RTOT"].mean()
np2 = np2.groupby(["PATNO", "INFODT"], as_index=False)["NP2PTOT"].mean()

np3["NP3TOT"] = pd.to_numeric(np3["NP3TOT"], errors="coerce")
np3.loc[
    ~np3["NP3TOT"].between(0, 132) | np3["NP3TOT"].eq(101),
    "NP3TOT"
] = float("nan")

state = np3["PDSTATE"].astype("string").str.strip().str.upper()
form = np3["PAG_NAME"].astype("string").str.strip().str.upper()
np3["STATE_CLEAN"] = state
np3.loc[
    np3["STATE_CLEAN"].isna() & form.eq("NUPDR3OF"),
    "STATE_CLEAN"
] = "OFF"

np3 = np3[
    np3["STATE_CLEAN"].eq("OFF")
    & np3["NP3TOT"].notna()
    & np3["PATNO"].notna()
    & np3["INFODT"].notna()
].copy()

np3 = np3.sort_values(["PATNO", "INFODT"])
np3 = np3.drop_duplicates(["PATNO", "INFODT"], keep="first")
np3 = np3[["PATNO", "INFODT", "NP3TOT"]]

visits = np1.merge(np2, on=["PATNO", "INFODT"], how="inner")
visits = visits.merge(np3, on=["PATNO", "INFODT"], how="inner")
visits = visits.dropna(
    subset=["PATNO", "INFODT", "NP1RTOT", "NP2PTOT", "NP3TOT"]
)
visits = visits.sort_values(["PATNO", "INFODT"]).reset_index(drop=True)

rows = []

for patno, group in visits.groupby("PATNO"):
    group = group.sort_values("INFODT").reset_index(drop=True)

    for i in range(1, len(group) - 1):
        current = group.iloc[i]
        previous = group.iloc[i - 1]

        target_date = current["INFODT"] + relativedelta(months=12)
        future = group.iloc[i + 1:].copy()
        future["DAY_DIFF"] = (future["INFODT"] - target_date).dt.days

        candidates = future[future["DAY_DIFF"].between(-30, 30)]
        if candidates.empty:
            continue

        target = candidates.iloc[candidates["DAY_DIFF"].abs().argmin()]

        rows.append({
            "PATNO": patno,
            "INDEX_DATE": current["INFODT"],
            "TARGET_DATE": target["INFODT"],
            "DAYS_FROM_12_MONTH_TARGET": target["DAY_DIFF"],
            "NP1RTOT": current["NP1RTOT"],
            "NP2PTOT": current["NP2PTOT"],
            "NP3TOT_CURRENT": current["NP3TOT"],
            "NP1RTOT_CHANGE": current["NP1RTOT"] - previous["NP1RTOT"],
            "NP2PTOT_CHANGE": current["NP2PTOT"] - previous["NP2PTOT"],
            "NP3TOT_CHANGE": current["NP3TOT"] - previous["NP3TOT"],
            "NP3TOT_FUTURE": target["NP3TOT"]
        })

result = pd.DataFrame(rows)

if result.empty:
    raise ValueError("No valid prediction examples found.")

result = result.dropna()

result.to_csv(OUT / "prediction_dataset.csv", index=False)

print("========== FINAL PREDICTION DATASET ==========")
print("Examples:", len(result))
print("Unique patients:", result["PATNO"].nunique())
print("Mean target offset in days:",
      round(result["DAYS_FROM_12_MONTH_TARGET"].mean(), 2))
print("\nTarget NP3 summary:")
print(result["NP3TOT_FUTURE"].describe().round(2))
print("\nSaved:", OUT / "prediction_dataset.csv")
