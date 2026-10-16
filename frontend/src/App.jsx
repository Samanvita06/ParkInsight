
import { useEffect, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:5000";

const datasets = [
  { key: "np1", title: "MDS-UPDRS Part I", description: "Non-motor experiences of daily living" },
  { key: "np2", title: "MDS-UPDRS Part II", description: "Motor experiences of daily living" },
  { key: "np3", title: "MDS-UPDRS Part III", description: "Motor examination and prediction target" },
  { key: "demographics", title: "Demographics", description: "Participant demographic information" },
];

function App() {
  const [files, setFiles] = useState({});
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState({});
  const [validation, setValidation] = useState(null);
  const [patientId, setPatientId] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [predictionError, setPredictionError] = useState("");
  const [predicting, setPredicting] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [paperPatientId, setPaperPatientId] = useState("");
  const [paperPrediction, setPaperPrediction] = useState(null);
  const [paperPredictionError, setPaperPredictionError] = useState("");
  const [paperPredicting, setPaperPredicting] = useState(false);

  async function refreshDatasets() {
    const response = await fetch(`${API}/api/datasets`);
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Could not load datasets");
    }

    const uploadedFiles = {};
    for (const [key, dataset] of Object.entries(result.datasets || {})) {
      if (dataset.uploaded) uploadedFiles[key] = dataset.filename;
    }

    setFiles(uploadedFiles);
  }

  useEffect(() => {
    refreshDatasets().catch((error) => {
      setStatus(`Could not load dataset status: ${error.message}`);
    });
  }, []);

  async function uploadFile(key, file) {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setStatus("Please select a CSV file.");
      return;
    }

    setUploading((previous) => ({ ...previous, [key]: true }));
    setStatus(`Uploading ${file.name}...`);
    setValidation(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API}/api/upload/${key}`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Upload failed");

      await refreshDatasets();
      setStatus(result.message || `${file.name} uploaded successfully.`);
    } catch (error) {
      setStatus(`Upload failed: ${error.message}`);
    } finally {
      setUploading((previous) => ({ ...previous, [key]: false }));
    }
  }

  
  async function validateDatasets() {
    setLoading(true);
    setStatus("Validating CSV files...");
    setValidation(null);

    try {
      const response = await fetch(`${API}/api/validate-datasets`, {
        method: "POST",
      });

      const result = await response.json();
      const validationResult = result.validation ?? result;

      setValidation(validationResult);

      if (!response.ok || validationResult.valid !== true) {
        setStatus("Validation failed. Review the dataset results below.");
        return false;
      }

      setStatus("All four CSV files passed validation.");
      return true;
    } catch (error) {
      setStatus(`Validation request failed: ${error.message}`);
      return false;
    } finally {
      setLoading(false);
    }
  }


  async function syncDatasets() {
    const valid = await validateDatasets();
    if (!valid) return;

    setLoading(true);
    setStatus("Syncing validated datasets...");

    try {
      const response = await fetch(`${API}/api/sync-datasets`, {
        method: "POST",
      });

      const result = await response.json();
      if (!response.ok) {
        if (result.validation) setValidation(result.validation);
        throw new Error(result.error || "Dataset sync failed");
      }

      setStatus(result.message || "Datasets synced successfully.");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setLoading(false);
    }
  }

  const uploadedCount = Object.keys(files).length;
  
async function predictNP3() {
  if (!patientId.trim()) {
    setPredictionError("Enter a participant ID.");
    return;
  }

  setPredicting(true);
  setPrediction(null);
  setPredictionError("");

  try {
    const response = await fetch(`${API}/api/predict-temporal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId.trim() }),
    });

    const result = await response.json();

    console.log("Prediction API response:", JSON.stringify(result, null, 2));

    if (!response.ok || result.success === false) {
      throw new Error(result.error || "Prediction failed.");
    }

    
    const rawScore =
      result.prediction?.predicted_np3_total_rounded ??
      result.prediction?.predicted_np3_total ??
      result.prediction?.prediction ??
      result.predicted_np3_total ??
      result.predicted_score;


    const score = Number(rawScore);

    if (
      rawScore === undefined ||
      rawScore === null ||
      rawScore === "" ||
      !Number.isFinite(score)
    ) {
      throw new Error(
        `No valid numeric prediction returned. Response fields: ${
          Object.keys(result).join(", ")
        }. Check the browser Console for the full response.`
      );
    }

    setPrediction({
      ...result,
      displayPrediction: score,
    });
  } catch (error) {
    setPredictionError(error.message);
  } finally {
    setPredicting(false);
  }
}
async function predictPaper() {
  if (!paperPatientId.trim()) {
    setPaperPredictionError("Enter a participant ID.");
    return;
  }

  setPaperPredicting(true);
  setPaperPrediction(null);
  setPaperPredictionError("");

  try {
    const response = await fetch(`${API}/api/predict-paper`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_id: paperPatientId.trim(),
      }),
    });

    const result = await response.json();

    if (!response.ok || result.success !== true) {
      throw new Error(result.error || "Method 1 prediction failed.");
    }

    const details = result.prediction;
    const score = Number(
      details?.predicted_np3_total_rounded ??
      details?.predicted_np3_total
    );

    if (!Number.isFinite(score)) {
      throw new Error("Method 1 returned an invalid numeric prediction.");
    }

    setPaperPrediction({
      ...details,
      displayPrediction: score,
    });
  } catch (error) {
    setPaperPredictionError(error.message);
  } finally {
    setPaperPredicting(false);
  }
}
  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark">P</div>
        <h2>ParkInsight</h2>
        <p className="sidebar-subtitle">Parkinson&apos;s progression research</p>

        <nav>
          <a className="active" href="#prediction">▦ Predict Future NP3</a>
          <a href="#datasets">▤ 4 Datasets Upload</a>
          <a href="#models">⌁ Model Evaluation</a>
        </nav>

        <div className="privacy-note">
          <strong>Local data processing</strong>
          <p>Your CSV files are processed by the backend running on your computer.</p>
        </div>
      </aside>

      <section className="main-content" id="overview">
        <header className="topbar">
          <div>
            <p className="eyebrow">PARKINSIGHT · V2.0</p>
            <h1>Future MDS-UPDRS Part III Score Prediction</h1>
            <p className="muted">
              Prepare clinical data and evaluate your existing prediction pipelines.
            </p>
          </div>
          <span className="local-badge">Local workspace</span>
        </header>

        <section className="stats-grid">
          <article className="stat-card">
            <span>Required datasets</span>
            <strong>04</strong>
            <small>Clinical assessments and demographics</small>
          </article>

          <article className="stat-card">
            <span>Uploaded datasets</span>
            <strong>{uploadedCount}/4</strong>
            <small>Files currently stored by the backend</small>
          </article>

          <article className="stat-card">
            <span>Prediction target</span>
            <strong>NP3</strong>
            <small>Future MDS-UPDRS Part III total</small>
          </article>
        </section>

        <section className="panel" id="datasets">
          <div className="section-heading">
            <div>
              <p className="eyebrow">STEP 01</p>
              <h2>Clinical dataset setup</h2>
              <p className="muted">Upload the four CSV files required by your existing scripts.</p>
            </div>
          </div>

          <div className="dataset-grid">
            {datasets.map((dataset) => {
              const result = validation?.datasets?.[dataset.key];

              return (
                <article className="dataset-card" key={dataset.key}>
                  <div className="file-icon">CSV</div>
                  <h3>{dataset.title}</h3>
                  <p>{dataset.description}</p>

                  <label className="upload-button">
                    {uploading[dataset.key] ? "Uploading..." : "Choose CSV"}
                    <input
                      type="file"
                      accept=".csv,text/csv"
                      disabled={uploading[dataset.key] || loading}
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        uploadFile(dataset.key, file);
                        event.target.value = "";
                      }}
                    />
                  </label>

                  <div className="file-status">
                    {files[dataset.key] ? `✓ ${files[dataset.key]}` : "No file uploaded"}
                  </div>

                  {result && (
                    <div className="validation-result">
                      <strong>{result.valid ? "Validation passed" : "Validation failed"}</strong>
                      {result.filename && <p>File: {result.filename}</p>}
                      {result.valid && <p>Rows: {result.rows.toLocaleString()}</p>}
                      {result.valid && <p>Columns: {result.columns.length}</p>}
                      {result.error && <p className="validation-error">{result.error}</p>}
                    </div>
                  )}
                </article>
              );
            })}
          </div>

          <button
            className="primary-button"
            onClick={syncDatasets}
            disabled={loading || uploadedCount !== 4 || Object.values(uploading).some(Boolean)}
          >
            {loading ? "Please wait..." : "Validate and sync datasets"}
          </button>

          {uploadedCount !== 4 && (
            <p className="muted">Upload all four datasets to enable validation and syncing.</p>
          )}

          {status && <p className="status-message" role="status">{status}</p>}

          {validation?.errors?.length > 0 && (
            <div className="validation-error-list">
              <h3>Issues to fix</h3>
              <ul>
                {validation.errors.map((error, index) => (
                  <li key={`${index}-${error}`}>{error}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
        
        <section className="panel" id="method1">
          <p className="eyebrow">METHOD 1</p>
          <h2>Current-Visit Feature Prediction</h2>
          <p className="muted">
            Paper-inspired Random Forest using current-visit clinical
            features, with training-only imputation and feature selection.
          </p>

          <div className="prediction-form">
            <input
              type="text"
              value={paperPatientId}
              onChange={(e) => setPaperPatientId(e.target.value)}
              placeholder="Enter participant ID"
              aria-label="Method 1 participant ID"
            />

            <button
              className="primary-button"
              onClick={predictPaper}
              disabled={paperPredicting}
            >
              {paperPredicting ? "Predicting..." : "Predict with Method 1"}
            </button>
          </div>

          {paperPredictionError && (
            <p className="status-message">{paperPredictionError}</p>
          )}

          {paperPrediction && (
            <div className="prediction-result">
              <span>Predicted future NP3 total</span>
              <h3>{paperPrediction.displayPrediction.toFixed(2)}</h3>
              <p>
                Forecast horizon:{" "}
                {paperPrediction.prediction_horizon_days ?? 365} days
              </p>
              <p>Index date: {paperPrediction.index_date}</p>
              <p>Model: {paperPrediction.model_name}</p>
              <small>
                Research estimate only; not a clinical diagnosis.
              </small>
            </div>
          )}
        </section>

        <section className="panel" id="prediction">
          <p className="eyebrow">STEP 02</p>
          <h2>Method 2 — Longitudinal History</h2>
          <p className="muted">
            Predict a future MDS-UPDRS Part III total using eligible historical visits.
          </p>

          <div className="prediction-form">
            <input
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              placeholder="Enter participant ID"
              aria-label="Participant ID"
            />

            <button
              className="primary-button"
              onClick={predictNP3}
              disabled={predicting}
            >
              {predicting ? "Predicting..." : "Predict NP3"}
            </button>
          </div>

          {predictionError && (
            <p className="status-message">{predictionError}</p>
          )}

          {prediction && (
            <div className="prediction-result">
              <span>Predicted future NP3 total</span>
              <h3>
                {Number.isFinite(prediction.displayPrediction)
                  ? prediction.displayPrediction.toFixed(2)
                  : "Invalid prediction"}
              </h3>

              <p>
                Forecast horizon: {prediction.prediction_horizon_days ?? 365} days
              </p>

              {prediction.historical_visits != null && (
                <p>Historical visits: {prediction.historical_visits}</p>
              )}

              {prediction.index_date && (
                <p>Index date: {prediction.index_date}</p>
              )}

              <small>
                Research prediction only; not a clinical diagnosis.
              </small>
            </div>
          )}
        </section>
        <section className="panel" id="models">
          <p className="eyebrow">STEP 02</p>
          <h2>ML pipeline workspace</h2>
          <p className="muted">
            Your original ML scripts remain unchanged. Training controls will be connected after dataset setup.
          </p>
          <div className="model-comparison">
            <div className="comparison-heading">
              <h3>Method 1 vs Method 2</h3>
              <p>Side-by-side comparison of Parkinson's progression prediction models</p>
            </div>

            <div className="comparison-columns">
              <div className="comparison-method">
                <h4>Method 1 — Paper-Inspired & Temporal Models</h4>
                <div className="metrics-table-wrap">
                  <table className="metrics-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>MAE ↓</th>
                        <th>RMSE ↓</th>
                        <th>R² ↑</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Last NP3 Baseline</td>
                        <td>5.453</td>
                        <td>8.220</td>
                        <td>0.760</td>
                      </tr>
                      <tr>
                        <td>Linear Regression</td>
                        <td>4.444</td>
                        <td>6.202</td>
                        <td>0.863</td>
                      </tr>
                      <tr>
                        <td>Ridge</td>
                        <td>4.444</td>
                        <td>6.202</td>
                        <td>0.863</td>
                      </tr>
                      <tr>
                        <td>Bayesian Ridge</td>
                        <td>4.444</td>
                        <td>6.200</td>
                        <td>0.863</td>
                      </tr>
                      <tr>
                        <td>Paper-Inspired RF</td>
                        <td>4.666</td>
                        <td>6.568</td>
                        <td>0.847</td>
                      </tr>
                      <tr>
                        <td>Gradient Boosting</td>
                        <td>4.514</td>
                        <td>6.337</td>
                        <td>0.857</td>
                      </tr>
                      <tr className="best-model-row">
                        <td>Temporal Random Forest</td>
                        <td>3.823</td>
                        <td>5.568</td>
                        <td>0.890</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <p className="metrics-note">Test examples: 1,086</p>
              </div>

              <div className="comparison-method">
                <h4>Method 2 — Longitudinal Models</h4>
                <div className="metrics-table-wrap">
                  <table className="metrics-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>MAE ↓</th>
                        <th>RMSE ↓</th>
                        <th>R² ↑</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="best-model-row">
                        <td>Retuned Temporal RF</td>
                        <td>6.664</td>
                        <td>8.833</td>
                        <td>0.601</td>
                      </tr>
                      <tr>
                        <td>Previously Tuned RF</td>
                        <td>6.695</td>
                        <td>8.891</td>
                        <td>0.595</td>
                      </tr>
                      <tr>
                        <td>GRU</td>
                        <td>7.378</td>
                        <td>9.880</td>
                        <td>0.500</td>
                      </tr>
                      <tr>
                        <td>Chronos-T5 Small</td>
                        <td>7.645</td>
                        <td>9.993</td>
                        <td>0.489</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <p className="metrics-note">Test examples: 206</p>
              </div>
            </div>

            <p className="metrics-note">
              Lower MAE and RMSE are better; higher R² is better.
              The methods use different test-set sizes, so their scores are
              not a controlled direct comparison.
            </p>
          </div>
        </section>
      </section>
    </main>
  );
}

export default App;
