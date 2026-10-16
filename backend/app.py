
from pathlib import Path
from functools import lru_cache
import math
import re

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from dataset_bridge import sync_datasets
from dataset_validator import validate_datasets
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
from flask_cors import CORS

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173"
            ]
        }
    }
)
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
ARTIFACT_DIR = BACKEND_DIR / "artifacts"
DATA_DIR = PROJECT_DIR / "original_ml" / "data"
MODEL_PATH = (
    PROJECT_DIR
    / "original_ml"
    / "outputs"
    / "shared_temporal_rf.joblib"
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

DATASETS = {
    "np1": {
        "filename": "MDS-UPDRS_Part_I.csv",
        "total": "NP1RTOT",
        "maximum": 24,
    },
    "np2": {
        "filename": "MDS-UPDRS_Part_II.csv",
        "total": "NP2PTOT",
        "maximum": 52,
    },
    "np3": {
        "filename": "MDS-UPDRS_Part_III.csv",
        "total": "NP3TOT",
        "maximum": 132,
    },
    "demographics": {
        "filename": "Demographics.csv",
    },
}

HISTORY_MONTHS = 12
MIN_HISTORY_VISITS = 2
MONTH_DAYS = 30.4375

FEATURE_COLUMNS = [
    "HISTORY_VISITS",
    "NP1RTOT_LAST",
    "NP1RTOT_MEAN",
    "NP1RTOT_MIN",
    "NP1RTOT_MAX",
    "NP1RTOT_SLOPE",
    "NP2PTOT_LAST",
    "NP2PTOT_MEAN",
    "NP2PTOT_MIN",
    "NP2PTOT_MAX",
    "NP2PTOT_SLOPE",
    "NP3TOT_LAST",
    "NP3TOT_MEAN",
    "NP3TOT_MIN",
    "NP3TOT_MAX",
    "NP3TOT_SLOPE",
]


def json_error(message, status=400, details=None):
    response = {"success": False, "error": message}

    if details is not None:
        response["details"] = details

    return jsonify(response), status


def normalize_patient_id(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_patient_value(value):
    result = str(value).strip()

    if re.fullmatch(r"\d+\.0", result):
        result = result[:-2]

    return result


@lru_cache(maxsize=1)
def load_temporal_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained Temporal Random Forest not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise ValueError("The saved model bundle must be a dictionary.")

    if "model" not in bundle or "feature_cols" not in bundle:
        raise ValueError(
            "The model bundle must contain 'model' and 'feature_cols'."
        )

    model = bundle["model"]
    feature_cols = list(bundle["feature_cols"])
    target = bundle.get("target", "NP3TOT_FUTURE")

    if not hasattr(model, "predict"):
        raise ValueError("The loaded model does not support prediction.")

    if feature_cols != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature columns do not match the expected training "
            f"features. Model columns: {feature_cols}"
        )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "target": target,
    }


def load_domain(filename, total, maximum):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset is missing: {path}. "
            "Upload and sync the datasets first."
        )

    df = pd.read_csv(path, low_memory=False)

    required = {"PATNO", "INFODT", total}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: "
            f"{sorted(missing)}"
        )

    df["PATNO"] = normalize_patient_id(df["PATNO"])
    df["INFODT"] = pd.to_datetime(
        df["INFODT"], errors="coerce"
    ).dt.normalize()
    df[total] = pd.to_numeric(df[total], errors="coerce")

    df = df.dropna(subset=["PATNO", "INFODT", total])
    df = df[df[total].between(0, maximum)]

    return (
        df.groupby(
            ["PATNO", "INFODT"],
            as_index=False,
        )[total]
        .mean()
    )


def load_np3():
    filename = DATASETS["np3"]["filename"]
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset is missing: {path}. "
            "Upload and sync the datasets first."
        )

    df = pd.read_csv(path, low_memory=False)

    required = {"PATNO", "INFODT", "NP3TOT"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: "
            f"{sorted(missing)}"
        )

    df["PATNO"] = normalize_patient_id(df["PATNO"])
    df["INFODT"] = pd.to_datetime(
        df["INFODT"], errors="coerce"
    ).dt.normalize()
    df["NP3TOT"] = pd.to_numeric(df["NP3TOT"], errors="coerce")

    if "PDSTATE" in df.columns:
        state = (
            df["PDSTATE"]
            .astype("string")
            .str.strip()
            .str.upper()
        )
    else:
        state = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )

    if "PAG_NAME" in df.columns:
        form = (
            df["PAG_NAME"]
            .astype("string")
            .str.strip()
            .str.upper()
        )
    else:
        form = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )

    off = state.eq("OFF") | (
        state.isna() & form.eq("NUPDR3OF")
    )

    df = df[
        off
        & df["NP3TOT"].between(0, 132)
        & df["NP3TOT"].ne(101)
    ].dropna(subset=["PATNO", "INFODT", "NP3TOT"])

    return (
        df.groupby(
            ["PATNO", "INFODT"],
            as_index=False,
        )["NP3TOT"]
        .mean()
    )


def load_matched_visits():
    np1 = load_domain(
        DATASETS["np1"]["filename"],
        "NP1RTOT",
        DATASETS["np1"]["maximum"],
    )

    np2 = load_domain(
        DATASETS["np2"]["filename"],
        "NP2PTOT",
        DATASETS["np2"]["maximum"],
    )

    np3 = load_np3()

    visits = (
        np1.merge(
            np2,
            on=["PATNO", "INFODT"],
            how="inner",
        )
        .merge(
            np3,
            on=["PATNO", "INFODT"],
            how="inner",
        )
        .sort_values(["PATNO", "INFODT"])
        .reset_index(drop=True)
    )

    if visits.empty:
        raise ValueError(
            "No same-patient, same-date visits were found across "
            "NP1, NP2 and eligible OFF-state NP3 assessments."
        )

    return visits


def calculate_slope(history, column, index_date):
    x = (
        (history["INFODT"] - index_date)
        .dt.total_seconds()
        .to_numpy()
        / (MONTH_DAYS * 86400)
    )

    y = history[column].to_numpy(dtype=float)

    if len(np.unique(x)) < 2:
        return np.nan

    return float(np.polyfit(x, y, 1)[0])


def calculate_features(history, index_date):
    history = history.sort_values("INFODT").copy()

    if len(history) < MIN_HISTORY_VISITS:
        raise ValueError(
            f"At least {MIN_HISTORY_VISITS} matched historical visits "
            "within the previous 12 months are required."
        )

    if (history["INFODT"] > index_date).any():
        raise ValueError(
            "Future visits were detected in the prediction history."
        )

    row = {"HISTORY_VISITS": int(len(history))}

    for column in ["NP1RTOT", "NP2PTOT", "NP3TOT"]:
        values = history[column].astype(float)

        row[f"{column}_LAST"] = float(values.iloc[-1])
        row[f"{column}_MEAN"] = float(values.mean())
        row[f"{column}_MIN"] = float(values.min())
        row[f"{column}_MAX"] = float(values.max())
        row[f"{column}_SLOPE"] = calculate_slope(
            history,
            column,
            index_date,
        )

    features = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    features = features.replace([np.inf, -np.inf], np.nan)

    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise ValueError(
            "Some features could not be calculated from the available "
            f"history: {bad}. More historical visits may be needed."
        )

    return features


def build_patient_prediction(patient_id, requested_index_date=None):
    patient_id = normalize_patient_value(patient_id)
    visits = load_matched_visits()

    patient_visits = visits[
        visits["PATNO"] == patient_id
    ].copy()

    if patient_visits.empty:
        raise ValueError(
            f"No matched NP1, NP2 and eligible OFF-state NP3 visits "
            f"were found for patient {patient_id}."
        )

    if requested_index_date:
        index_date = pd.to_datetime(
            requested_index_date,
            errors="coerce",
        )

        if pd.isna(index_date):
            raise ValueError(
                "index_date must be a valid date such as YYYY-MM-DD."
            )

        index_date = index_date.normalize()

        matching_index = patient_visits[
            patient_visits["INFODT"] == index_date
        ]

        if matching_index.empty:
            available = (
                patient_visits["INFODT"]
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )

            raise ValueError(
                "The requested index date is not a matched assessment "
                "date for this patient. Choose a date with all three "
                "domains available."
            )
    else:
        index_date = patient_visits["INFODT"].max()

    history_start = index_date - pd.DateOffset(
        months=HISTORY_MONTHS
    )

    history = patient_visits[
        (patient_visits["INFODT"] >= history_start)
        & (patient_visits["INFODT"] <= index_date)
    ].copy()

    features = calculate_features(history, index_date)

    bundle = load_temporal_model()
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    prediction = float(
        model.predict(features[feature_cols])[0]
    )

    if not math.isfinite(prediction):
        raise ValueError("The model returned a non-finite prediction.")

    return {
        "patient_id": patient_id,
        "index_date": index_date.strftime("%Y-%m-%d"),
        "history_start_date": history_start.strftime("%Y-%m-%d"),
        "history_visits": int(len(history)),
        "history_visit_dates": (
            history["INFODT"].dt.strftime("%Y-%m-%d").tolist()
        ),
        "prediction_horizon_days": 365,
        "predicted_np3_total": prediction,
        "predicted_np3_total_rounded": round(prediction, 2),
        "target": bundle["target"],
        "features": features.iloc[0].to_dict(),
        "note": (
            "Research model estimate of future NP3 total, not a "
            "clinical diagnosis or a substitute for assessment "
            "by a qualified clinician."
        ),
    }


@app.errorhandler(413)
def request_too_large(_error):
    return json_error(
        "The uploaded file exceeds the 100 MB limit.",
        413,
    )


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "ParkInsight API is running.",
        "endpoints": [
            "/api/health",
            "/api/datasets",
            "/api/validate-datasets",
            "/api/sync-datasets",
            "/api/model-info",
            "/api/predict-temporal",
        ],
    })


@app.route("/api/health", methods=["GET"])
def health():
    try:
        bundle = load_temporal_model()
        model_available = True
        model_error = None
        model_type = type(bundle["model"]).__name__
    except Exception as exc:
        model_available = False
        model_error = str(exc)
        model_type = None

    return jsonify({
        "success": True,
        "status": "ok",
        "model_available": model_available,
        "model_type": model_type,
        "model_error": model_error,
    })


@app.route("/api/datasets", methods=["GET"])
def datasets_status():
    results = {}

    for key, info in DATASETS.items():
        uploaded_path = UPLOAD_DIR / f"{key}.csv"
        synced_path = DATA_DIR / info["filename"]

        results[key] = {
            "filename": info["filename"],
            "uploaded": uploaded_path.exists(),
            "synced": synced_path.exists(),
            "uploaded_size_bytes": (
                uploaded_path.stat().st_size
                if uploaded_path.exists()
                else None
            ),
            "synced_size_bytes": (
                synced_path.stat().st_size
                if synced_path.exists()
                else None
            ),
        }

    return jsonify({
        "success": True,
        "datasets": results,
    })


@app.route("/api/upload/<dataset_key>", methods=["POST"])
def upload_dataset(dataset_key):
    if dataset_key not in DATASETS:
        return json_error(
            f"Unknown dataset key: {dataset_key}",
            404,
        )

    uploaded_file = request.files.get("file")

    if uploaded_file is None or not uploaded_file.filename:
        return json_error(
            "Select a CSV file using the 'file' field."
        )

    filename = secure_filename(uploaded_file.filename)

    if not filename.lower().endswith(".csv"):
        return json_error("Only CSV files are accepted.")

    destination = UPLOAD_DIR / f"{dataset_key}.csv"

    try:
        uploaded_file.save(destination)

        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            return json_error("The uploaded file is empty.")

        return jsonify({
            "success": True,
            "dataset_key": dataset_key,
            "filename": DATASETS[dataset_key]["filename"],
            "uploaded_filename": filename,
            "size_bytes": destination.stat().st_size,
            "message": "Dataset uploaded successfully.",
        })
    except Exception as exc:
        return json_error(
            "Failed to save the uploaded dataset.",
            500,
            str(exc),
        )


@app.route("/api/validate-datasets", methods=["POST"])
def validate_uploaded_datasets():
    try:
        result = validate_datasets()
        status = 200 if result.get("valid") else 400

        return jsonify({
            "success": bool(result.get("valid")),
            "validation": result,
        }), status
    except Exception as exc:
        return json_error(
            "Dataset validation failed.",
            500,
            str(exc),
        )


@app.route("/api/sync-datasets", methods=["POST"])
def sync_uploaded_datasets():
    try:
        validation = validate_datasets()

        if not validation.get("valid"):
            return json_error(
                "Datasets failed validation and were not synced.",
                400,
                validation,
            )

        synced = sync_datasets()

        return jsonify({
            "success": True,
            "synced": synced,
            "message": "Validated datasets synced successfully.",
        })
    except FileNotFoundError as exc:
        return json_error(str(exc), 400)
    except Exception as exc:
        return json_error(
            "Dataset synchronization failed.",
            500,
            str(exc),
        )


@app.route("/api/model-info", methods=["GET"])
def model_info():
    try:
        bundle = load_temporal_model()

        return jsonify({
            "success": True,
            "available": True,
            "model_type": type(bundle["model"]).__name__,
            "target": bundle["target"],
            "feature_count": len(bundle["feature_cols"]),
            "feature_columns": bundle["feature_cols"],
            "prediction_horizon_days": 365,
            "model_path": str(MODEL_PATH),
        })
    except Exception as exc:
        return json_error(
            "The trained temporal model is unavailable.",
            503,
            str(exc),
        )


@app.route("/api/predict-temporal", methods=["POST"])
def predict_temporal():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return json_error(
            "Send a JSON object in the request body."
        )

    patient_id = payload.get("patient_id", payload.get("patno"))

    # Patient-history mode: calculate the 16 features from synced data.
    if patient_id is not None:
        try:
            result = build_patient_prediction(
                patient_id,
                payload.get("index_date"),
            )

            return jsonify({
                "success": True,
                "prediction": result,
            })
        except FileNotFoundError as exc:
            return json_error(str(exc), 400)
        except ValueError as exc:
            return json_error(str(exc), 400)
        except Exception as exc:
            app.logger.exception("Temporal prediction failed.")
            return json_error(
                "Temporal prediction failed.",
                500,
                str(exc),
            )

    # Backward-compatible mode: accept already-engineered features.
    supplied_features = payload.get("features")

    if not isinstance(supplied_features, dict):
        return json_error(
            "Provide 'patient_id' (or 'patno') and optionally "
            "'index_date', or provide a 'features' object containing "
            "all 16 model features."
        )

    bundle = None

    try:
        bundle = load_temporal_model()
        feature_cols = bundle["feature_cols"]

        missing = [
            col for col in feature_cols
            if col not in supplied_features
        ]

        if missing:
            return json_error(
                "Required model features are missing.",
                400,
                {"missing_features": missing},
            )

        values = {}

        for column in feature_cols:
            try:
                value = float(supplied_features[column])
            except (TypeError, ValueError):
                return json_error(
                    f"Feature '{column}' must be numeric."
                )

            if not math.isfinite(value):
                return json_error(
                    f"Feature '{column}' must be finite."
                )

            values[column] = value

        if values["HISTORY_VISITS"] < 1:
            return json_error(
                "HISTORY_VISITS must be at least 1."
            )

        feature_frame = pd.DataFrame(
            [values],
            columns=feature_cols,
        )

        prediction = float(
            bundle["model"].predict(feature_frame)[0]
        )

        if not math.isfinite(prediction):
            return json_error(
                "The model returned a non-finite prediction.",
                500,
            )

        return jsonify({
            "success": True,
            "prediction": {
                "predicted_np3_total": prediction,
                "predicted_np3_total_rounded": round(prediction, 2),
                "target": bundle["target"],
                "prediction_horizon_days": 365,
                "features": values,
                "note": (
                    "Research model estimate, not a clinical diagnosis."
                ),
            },
        })

    except Exception as exc:
        app.logger.exception("Temporal prediction failed.")
        return json_error(
            "Temporal prediction failed.",
            500,
            str(exc),
        )

@app.route("/api/predict-paper", methods=["POST"])
def predict_paper():
    try:
        root = Path(__file__).resolve().parent.parent
        outputs = root / "original_ml" / "outputs"

        model_path = outputs / "paper_method1_model.joblib"
        examples_path = outputs / "all_prediction_examples.csv"

        if not model_path.exists():
            return jsonify({
                "success": False,
                "error": "Method 1 model is not trained yet."
            }), 503

        if not examples_path.exists():
            return jsonify({
                "success": False,
                "error": "Method 1 prediction examples were not found."
            }), 503

        payload = request.get_json(silent=True) or {}
        patient_id = str(payload.get("patient_id", "")).strip()

        if not patient_id:
            return jsonify({
                "success": False,
                "error": "Enter a participant ID."
            }), 400

        examples = pd.read_csv(examples_path, low_memory=False)
        examples["PATNO"] = examples["PATNO"].astype(str).str.strip()
        examples["INDEX_DATE"] = pd.to_datetime(
            examples["INDEX_DATE"], errors="coerce"
        )

        matches = examples[
            (examples["PATNO"] == patient_id)
            & examples["INDEX_DATE"].notna()
        ]

        if "index_date" in payload and payload["index_date"]:
            requested_date = pd.to_datetime(
                payload["index_date"], errors="coerce"
            )

            if pd.isna(requested_date):
                return jsonify({
                    "success": False,
                    "error": "Invalid index date."
                }), 400

            matches = matches[
                matches["INDEX_DATE"] == requested_date
            ]

        if matches.empty:
            return jsonify({
                "success": False,
                "error": (
                    "No eligible Method 1 prediction example was found "
                    "for this participant and index date."
                )
            }), 404

        row = matches.sort_values("INDEX_DATE").iloc[-1]

        bundle = joblib.load(model_path)
        usable = bundle["usable_features"]
        selected = bundle["selected_features"]

        missing = [c for c in usable if c not in examples.columns]

        if missing:
            return jsonify({
                "success": False,
                "error": (
                    "Prediction examples are missing trained features. "
                    "Regenerate the examples and retrain Method 1."
                )
            }), 500

        raw = pd.DataFrame(
            [[row[c] for c in usable]],
            columns=usable,
        ).apply(pd.to_numeric, errors="coerce")

        transformed = pd.DataFrame(
            bundle["imputer"].transform(raw),
            columns=usable,
        )

        score = float(
            bundle["model"].predict(transformed[selected])[0]
        )

        if not np.isfinite(score):
            raise ValueError("The model returned a non-finite prediction.")

        return jsonify({
            "success": True,
            "prediction": {
                "predicted_np3_total": score,
                "predicted_np3_total_rounded": round(score, 2),
                "patient_id": patient_id,
                "index_date": row["INDEX_DATE"].strftime("%Y-%m-%d"),
                "prediction_horizon_days": 365,
                "model_name": bundle["model_name"],
                "note": (
                    "Method 1 current-visit model estimate; "
                    "not a clinical diagnosis."
                ),
            },
        })

    except Exception as error:
        app.logger.exception("Method 1 prediction failed")
        return jsonify({
            "success": False,
            "error": str(error),
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
