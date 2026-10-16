
from pathlib import Path

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_DIR / "uploads"

REQUIRED_DATASETS = {
    "np1": "MDS-UPDRS Part I",
    "np2": "MDS-UPDRS Part II",
    "np3": "MDS-UPDRS Part III",
    "demographics": "Demographics",
}


def validate_datasets():
    results = {}
    errors = []

    for key, label in REQUIRED_DATASETS.items():
        file_path = UPLOAD_DIR / f"{key}.csv"

        if not file_path.is_file():
            errors.append(f"{label}: file has not been uploaded.")
            results[key] = {
                "valid": False,
                "filename": None,
                "rows": 0,
                "columns": [],
                "error": "File not found.",
            }
            continue

        try:
            if file_path.stat().st_size == 0:
                raise ValueError("The CSV file is empty.")

            df = pd.read_csv(file_path, nrows=5)

            if len(df.columns) == 0:
                raise ValueError("No column headers were found.")

            if any(
                str(column).strip() == ""
                for column in df.columns
            ):
                raise ValueError("A column header is blank.")

            if df.columns.duplicated().any():
                raise ValueError("Duplicate column names were found.")

            row_count = sum(
                1 for _ in open(file_path, encoding="utf-8-sig")
            ) - 1

            if row_count <= 0:
                raise ValueError("The CSV contains no data rows.")

            results[key] = {
                "valid": True,
                "filename": file_path.name,
                "rows": row_count,
                "columns": df.columns.tolist(),
                "error": None,
            }

        except Exception as error:
            message = str(error)
            errors.append(f"{label}: {message}")

            results[key] = {
                "valid": False,
                "filename": file_path.name,
                "rows": 0,
                "columns": [],
                "error": message,
            }

    return {
        "valid": len(errors) == 0,
        "datasets": results,
        "errors": errors,
    }
