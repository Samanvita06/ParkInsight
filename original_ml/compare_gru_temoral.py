
from pathlib import Path
import copy
import random

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

SEED = 42
EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def metrics(name, actual, predicted):
    return {
        "Model": name,
        "Test_examples": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]


def collate_sequences(batch):
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    max_length = max(lengths).item()

    x = torch.zeros(len(sequences), max_length, 5)

    for i, sequence in enumerate(sequences):
        x[i, :len(sequence)] = torch.tensor(
            sequence, dtype=torch.float32
        )

    return (
        x.to(DEVICE),
        lengths,
        torch.tensor(targets, dtype=torch.float32).to(DEVICE),
    )


class GRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=5,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(1)


def main():
    features = pd.read_csv(OUT / "temporal_features.csv")
    sequences_df = pd.read_csv(OUT / "gru_sequences.csv")
    examples = pd.read_csv(OUT / "shared_examples.csv")

    examples["INDEX_DATE"] = pd.to_datetime(examples["INDEX_DATE"])
    features["INDEX_DATE"] = pd.to_datetime(features["INDEX_DATE"])
    sequences_df["VISIT_DATE"] = pd.to_datetime(
        sequences_df["VISIT_DATE"]
    )
    sequences_df["INDEX_DATE"] = pd.to_datetime(
        sequences_df["INDEX_DATE"]
    )

    # Use the saved shared patient split.
    train_ids = set(
        examples.loc[examples["SPLIT"] == "train", "EXAMPLE_ID"]
    )
    test_ids = set(
        examples.loc[examples["SPLIT"] == "test", "EXAMPLE_ID"]
    )

    train = features[features["EXAMPLE_ID"].isin(train_ids)].copy()
    test = features[features["EXAMPLE_ID"].isin(test_ids)].copy()

    assert set(train["PATNO"]).isdisjoint(set(test["PATNO"]))

    target = "NP3TOT_FUTURE"
    excluded = {
        "EXAMPLE_ID", "PATNO", "INDEX_DATE", "TARGET_DATE",
        "TARGET_GAP_DAYS", "SPLIT", target,
    }
    feature_cols = [
        c for c in features.columns if c not in excluded
    ]

    X_train = train[feature_cols].astype(float)
    y_train = train[target].astype(float)
    X_test = test[feature_cols].astype(float)
    y_test = test[target].astype(float)

    print("========== SHARED SPLIT ==========")
    print(f"Train examples: {len(train)}")
    print(f"Test examples: {len(test)}")
    print(f"Train patients: {train['PATNO'].nunique()}")
    print(f"Test patients: {test['PATNO'].nunique()}")

    # ------------------------------------------------------
    # 1. Tune and train Random Forest on training examples only.
    # ------------------------------------------------------
    print("\nTraining temporal-feature Random Forest...")

    cv_splits = min(4, train["PATNO"].nunique())
    cv = GroupKFold(n_splits=cv_splits)

    parameter_space = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_leaf": [2, 3, 5, 8, 12],
        "max_features": [0.7, 1.0],
    }

    search = RandomizedSearchCV(
        RandomForestRegressor(
            random_state=SEED,
            n_jobs=1,
        ),
        parameter_space,
        n_iter=15,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )

    search.fit(
        X_train,
        y_train,
        groups=train["PATNO"],
    )


    rf = search.best_estimator_
    rf_pred = rf.predict(X_test)

    joblib.dump(
        {
            "model": rf,
            "feature_cols": feature_cols,
            "target": target,
        },
        OUT / "shared_temporal_rf.joblib",
    )

    print("Best RF parameters:", search.best_params_)
    print(f"Grouped CV MAE: {-search.best_score_:.3f}")


    # ------------------------------------------------------
    # Additional model: Previously tuned Random Forest
    # ------------------------------------------------------
    tuned_rf = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=12,
        max_features=1.0,
        max_depth=15,
        random_state=SEED,
        n_jobs=-1,
    )

    tuned_rf.fit(X_train, y_train)
    tuned_rf_pred = tuned_rf.predict(X_test)

    tuned_rf_results = pd.DataFrame({
        "EXAMPLE_ID": test["EXAMPLE_ID"].to_numpy(),
        "PATNO": test["PATNO"].to_numpy(),
        "Actual": y_test.to_numpy(),
        "Tuned_RF": tuned_rf_pred,
    })

    tuned_rf_results.to_csv(
        OUT / "shared_tuned_rf_predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": tuned_rf,
            "feature_cols": feature_cols,
            "target": target,
        },
        OUT / "shared_tuned_rf.joblib",
    )

    print("Previously tuned RF predictions saved.")

    # ------------------------------------------------------
    # 2. Prepare chronological sequences for GRU.
    # Inputs: NP1, NP2, NP3, months before index, gap from prior visit.
    # Fit score scaling using training sequence rows only.
    # ------------------------------------------------------
    print("\nPreparing GRU sequences...")

    sequence_map = {}

    for example_id, group in sequences_df.groupby("EXAMPLE_ID"):
        group = group.sort_values("VISIT_DATE").copy()

        gap = group["MONTHS_SINCE_PREVIOUS_VISIT"].fillna(0).to_numpy()
        months_before = group["MONTHS_BEFORE_INDEX"].to_numpy()

        sequence_map[example_id] = np.column_stack([
            group["NP1RTOT"].to_numpy(),
            group["NP2PTOT"].to_numpy(),
            group["NP3TOT"].to_numpy(),
            months_before,
            gap,
        ]).astype(np.float32)

    train_examples = examples[
        examples["SPLIT"] == "train"
    ].copy()
    test_examples = examples[
        examples["SPLIT"] == "test"
    ].copy()

    train_examples = train_examples[
        train_examples["EXAMPLE_ID"].isin(sequence_map)
    ].reset_index(drop=True)
    test_examples = test_examples[
        test_examples["EXAMPLE_ID"].isin(sequence_map)
    ].reset_index(drop=True)

    # Fit normalization only on training sequence rows.
    scaler = StandardScaler()
    scaler.fit(np.concatenate([
        sequence_map[eid] for eid in train_examples["EXAMPLE_ID"]
    ], axis=0))

    for eid in sequence_map:
        sequence_map[eid] = scaler.transform(
            sequence_map[eid]
        ).astype(np.float32)

    train_sequences = [
        sequence_map[eid] for eid in train_examples["EXAMPLE_ID"]
    ]
    test_sequences = [
        sequence_map[eid] for eid in test_examples["EXAMPLE_ID"]
    ]

    train_targets = train_examples[target].to_numpy(dtype=np.float32)
    test_targets = test_examples[target].to_numpy(dtype=np.float32)

    # Keep a validation group of patients out of gradient updates.
    from sklearn.model_selection import GroupShuffleSplit

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=SEED,
    )
    fit_idx, val_idx = next(
        inner.split(
            train_examples,
            train_targets,
            groups=train_examples["PATNO"],
        )
    )

    fit_loader = DataLoader(
        SequenceDataset(
            [train_sequences[i] for i in fit_idx],
            train_targets[fit_idx],
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_sequences,
    )

    val_loader = DataLoader(
        SequenceDataset(
            [train_sequences[i] for i in val_idx],
            train_targets[val_idx],
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sequences,
    )

    model = GRURegressor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience_left = PATIENCE

    print(f"Training GRU on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()

        for x, lengths, y in fit_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

        model.eval()
        validation_losses = []

        with torch.no_grad():
            for x, lengths, y in val_loader:
                validation_losses.append(
                    loss_fn(model(x, lengths), y).item()
                )

        val_loss = float(np.mean(validation_losses))

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = PATIENCE
        else:
            patience_left -= 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: "
                f"validation MSE={val_loss:.3f}"
            )

        if patience_left == 0:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    model.eval()

    test_loader = DataLoader(
        SequenceDataset(test_sequences, test_targets),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sequences,
    )

    gru_pred = []

    with torch.no_grad():
        for x, lengths, _ in test_loader:
            gru_pred.extend(model(x, lengths).cpu().numpy())

    gru_pred = np.clip(gru_pred, 0, 132)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        OUT / "shared_gru.pt",
    )

    # ------------------------------------------------------
    # 3. Compare models on the common test examples.
    # ------------------------------------------------------
    rf_results = pd.DataFrame({
        "EXAMPLE_ID": test["EXAMPLE_ID"],
        "PATNO": test["PATNO"],
        "Actual": y_test.to_numpy(),
        "RandomForest": rf_pred,
    })

    gru_results = pd.DataFrame({
        "EXAMPLE_ID": test_examples["EXAMPLE_ID"],
        "PATNO": test_examples["PATNO"],
        "Actual": test_targets,
        "GRU": gru_pred,
    })

    common = rf_results.merge(
        gru_results,
        on=["EXAMPLE_ID", "PATNO"],
        suffixes=("_RF", "_GRU"),
    )

    results = pd.DataFrame([
        {
            "Model": "Temporal-feature Random Forest",
            "Test_examples": len(common),
            "Test_patients": common["PATNO"].nunique(),
            "MAE": mean_absolute_error(
                common["Actual_RF"], common["RandomForest"]
            ),
            "RMSE": np.sqrt(mean_squared_error(
                common["Actual_RF"], common["RandomForest"]
            )),
            "R2": r2_score(
                common["Actual_RF"], common["RandomForest"]
            ),
        },
        {
            "Model": "GRU sequence model",
            "Test_examples": len(common),
            "Test_patients": common["PATNO"].nunique(),
            "MAE": mean_absolute_error(
                common["Actual_RF"], common["GRU"]
            ),
            "RMSE": np.sqrt(mean_squared_error(
                common["Actual_RF"], common["GRU"]
            )),
            "R2": r2_score(
                common["Actual_RF"], common["GRU"]
            ),
        },
    ])

    common.to_csv(
        OUT / "shared_model_predictions.csv",
        index=False,
    )
    results.to_csv(
        OUT / "shared_model_comparison.csv",
        index=False,
    )

    print("\n========== FINAL COMPARISON ==========")
    print(results.round(3).to_string(index=False))
    print(f"\nCommon test examples: {len(common)}")
    print(f"Common test patients: {common['PATNO'].nunique()}")
    print(f"Saved results in: {OUT}")


if __name__ == "__main__":
    main()
