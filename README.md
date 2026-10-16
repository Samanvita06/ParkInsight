# ParkInsight 🧠
### Parkinson's Disease Progression Prediction Using Machine Learning

ParkInsight is a machine learning research project designed to explore Parkinson's disease progression using longitudinal clinical assessment data from the Parkinson's Progression Markers Initiative (PPMI).

The application provides a web interface for managing datasets, validating and synchronizing clinical records, exploring machine learning models, and predicting future motor symptom severity using the MDS-UPDRS Part III total score (NP3TOT).

> **Research disclaimer:** ParkInsight is an experimental research prototype. Its predictions are not medical diagnoses and should not be used as a substitute for professional medical advice.

---

## 📌 Project Overview

Parkinson's disease is a progressive neurological disorder that can affect movement and non-motor functions. Understanding how symptoms change over time can support research into disease progression.

ParkInsight investigates two approaches to predicting future motor assessment scores:

- **Method 1 — Paper-Inspired Prediction:** Uses eligible features from a current clinical assessment to estimate a future NP3TOT score.
- **Method 2 — Longitudinal Temporal Prediction:** Uses features summarizing a patient's historical clinical assessments to estimate a future NP3TOT score.

The project also includes model comparison and evaluation using regression metrics.

## 🎯 Objectives

- Process and validate clinical assessment datasets.
- Integrate demographic and MDS-UPDRS assessment data.
- Engineer features from clinical records and longitudinal histories.
- Predict future MDS-UPDRS Part III total scores.
- Compare traditional regression models and temporal learning approaches.
- Evaluate predictions using MAE, RMSE, and R².
- Provide an interactive interface for dataset management and model predictions.

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python, Flask |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Temporal modeling | Random Forest and other experimental model implementations |
| Deep learning experiments | GRU |
| Forecasting experiments | Chronos-T5 |
| Data source | PPMI clinical assessment data |
| Version control | Git and GitHub |

The exact dependencies are listed in `backend/requirements.txt` and `frontend/package.json`.

## 🧬 Dataset Description

ParkInsight works with clinical assessment data associated with the Parkinson's Progression Markers Initiative (PPMI).

The project uses four dataset categories:

| Dataset | Description |
|---|---|
| Demographics | Participant demographic information |
| NP1 | MDS-UPDRS Part I: Non-Motor Experiences of Daily Living |
| NP2 | MDS-UPDRS Part II: Motor Experiences of Daily Living |
| NP3 | MDS-UPDRS Part III: Motor Examination |

The primary prediction target is **NP3TOT**, the total score for MDS-UPDRS Part III.

The project aligns clinical assessments using participant identifiers and visit dates to construct training examples and future prediction targets.

**Data privacy:** The raw PPMI datasets and participant-related files are not included in this repository. Users must obtain authorized access to the data and provide compatible local files before running data-dependent workflows.

## ⚙️ Methodology

### Method 1: Paper-Inspired Prediction

This approach uses eligible features available at a current clinical assessment to predict a later NP3TOT score.

The workflow includes:

1. Preparing and aligning clinical assessment records.
2. Constructing current-visit features and future NP3TOT targets.
3. Splitting data at the participant level to reduce patient overlap between training and testing.
4. Performing feature selection using training data.
5. Training and comparing regression models.
6. Evaluating predictions on held-out participants.

The current-visit approach learns statistical relationships between present clinical characteristics and future motor assessment scores from historical examples. It does not explicitly model each participant's complete longitudinal trajectory.

### Method 2: Longitudinal Temporal Prediction

This approach summarizes information from multiple past clinical assessments to estimate a future NP3TOT score.

The temporal feature pipeline includes historical measurements and derived statistics such as:

- Number of historical visits
- Latest NP1, NP2, and NP3 scores
- Mean, minimum, and maximum historical scores
- Historical score slopes

The model uses eligible information available on or before the prediction index date. Future NP3TOT values are used as training targets and must not be included as input features.

The current implementation includes a shared temporal Random Forest model with a forecast horizon of approximately 365 days.

### Model Comparison

The project includes experiments with:

- Last-observed NP3 score baseline
- Linear Regression
- Ridge Regression
- Bayesian Ridge Regression
- Random Forest
- Gradient Boosting
- GRU
- Chronos-T5 Small

Not every model is necessarily available for live inference through the web application. Training scripts, saved artifacts, and evaluation outputs depend on the specific experiment.

## 📊 Evaluation Metrics

ParkInsight evaluates regression models using the following metrics.

| Metric | Meaning |
|---|---|
| MAE | Mean Absolute Error; the average absolute difference between predicted and actual scores |
| RMSE | Root Mean Squared Error; penalizes larger prediction errors more strongly |
| R² | Coefficient of determination; measures how much target variation is explained by the model relative to a baseline |

Lower MAE and RMSE indicate smaller prediction errors. Higher R² generally indicates better fit on the evaluated dataset.

Evaluation results should only be compared directly when models use the same test participants, target definition, prediction horizon, and evaluation rules.

## 🖥️ Application Features

- Dataset upload interface
- Dataset validation
- Dataset synchronization
- Current-visit future-score prediction
- Longitudinal future-score prediction
- Model comparison tables
- Regression evaluation metrics
- Flask API integration with a React frontend

## 📁 Project Structure

```text
ParkinsightApp/
│
├── backend/
│   ├── app.py
│   ├── dataset_bridge.py
│   ├── dataset_validator.py
│   ├── list_usable_patients.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── original_ml/
│   ├── build_shared_datasets.py
│   ├── compare_all_four.py
│   ├── compare_gru_temoral.py
│   ├── prepare_data.py
│   ├── run_paper_comparison.py
│   ├── train_chronos.py
│   ├── train_method1_model.py
│   └── train_models.py
│
├── .gitignore
├── README.md
└── package-lock.json
```

The structure above lists the main source files committed to GitHub. Local datasets, generated outputs, and trained artifacts are excluded by `.gitignore`.

## 🚀 Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or a compatible version supported by the project's dependencies
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Samanvita06/ParkInsight.git
cd ParkInsight
```

### 2. Set up the Python backend

From the project root, create a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
pip install -r backend/requirements.txt
```

### 3. Configure local datasets and model artifacts

Obtain the authorized PPMI data required by the project and place the compatible CSV files in the locations expected by the backend and machine learning scripts.

The original datasets and generated outputs are intentionally excluded from GitHub.

Some prediction endpoints require trained model artifacts. These must be generated by the appropriate training scripts or supplied through an authorized local workflow before those endpoints can perform inference.

Consult the relevant Python scripts for their expected input paths and configuration.

### 4. Start the Flask backend

From the project root, activate the virtual environment if necessary, then run:

```powershell
python backend/app.py
```

The frontend is configured to communicate with:

```text
http://127.0.0.1:5000
```

Keep the backend terminal running.

### 5. Start the React frontend

Open a second terminal and run:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in your terminal, typically:

```text
http://localhost:5173
```

The frontend and backend must both be running for API-dependent features to work.

## 🔌 Backend API

The Flask backend exposes endpoints for dataset management and prediction.

| Endpoint | Purpose |
|---|---|
| `GET /api/datasets` | Retrieve dataset status |
| `POST /api/upload/<key>` | Upload a dataset |
| `POST /api/validate-datasets` | Validate available datasets |
| `POST /api/sync-datasets` | Synchronize validated datasets |
| `POST /api/predict-temporal` | Generate a longitudinal prediction |
| `POST /api/predict-paper` | Generate a paper-inspired prediction |

Prediction requests require the inputs expected by the relevant backend endpoint. Dataset and model availability may affect whether a request succeeds.

## 🔐 Data Privacy and Reproducibility

This repository excludes:

- Raw clinical datasets
- Local dataset uploads
- Participant-ID files
- Environment files containing secrets
- Generated outputs and trained model artifacts

Do not commit private participant data, API keys, access tokens, or other confidential information.

Because datasets and trained artifacts are excluded, cloning the repository alone may not reproduce every prediction or experiment. Reproduction requires compatible authorized data, dependencies, configuration, and model artifacts.

## ⚠️ Limitations

- Predictions depend on data quality, feature availability, and the training population.
- Missing clinical assessments may affect sample construction and model performance.
- Prediction accuracy does not establish clinical usefulness.
- Future score prediction is not equivalent to identifying the biological rate of disease progression.
- Performance comparisons are meaningful only under consistent evaluation conditions.
- The project is a research prototype, not a clinically validated decision-support system.

## 🔮 Future Improvements

- More rigorous participant-level and chronological validation
- Improved handling of missing and irregularly timed assessments
- Broader evaluation across prediction horizons
- Model interpretability and feature importance analysis
- Reproducible experiment configurations and reporting
- Improved deployment and model artifact management
- Additional validation on appropriate independent datasets

## 👩‍💻 Authors

- **Samanvita Madi** — [@Samanvita06](https://github.com/Samanvita06)
- **[Teammate's Full Name]** — [@jaswika1997](https://github.com/jaswika1997)

Developed collaboratively as an academic machine learning research project.

## 📄 License

No license has been specified for this repository yet. Unless a license is added, standard copyright restrictions apply; reuse and redistribution are not automatically permitted.

---

*ParkInsight is developed for academic research and educational purposes in machine learning and Parkinson's disease progression analysis.*