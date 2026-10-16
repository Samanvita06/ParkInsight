
from pathlib import Path
import shutil

BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
PROJECT_DIR = BACKEND_DIR.parent
ORIGINAL_DATA_DIR = PROJECT_DIR / "original_ml" / "data"

DATASET_MAPPING = {
    "np1": "MDS-UPDRS_Part_I.csv",
    "np2": "MDS-UPDRS_Part_II.csv",
    "np3": "MDS-UPDRS_Part_III.csv",
    "demographics": "Demographics.csv",
}


def sync_datasets():
    ORIGINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    missing = [
        key
        for key in DATASET_MAPPING
        if not (UPLOAD_DIR / f"{key}.csv").is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Upload all four datasets first. Missing: "
            + ", ".join(missing)
        )

    copied_files = {}

    for key, filename in DATASET_MAPPING.items():
        source = UPLOAD_DIR / f"{key}.csv"
        destination = ORIGINAL_DATA_DIR / filename
        shutil.copy2(source, destination)
        copied_files[key] = str(destination)

    return copied_files
