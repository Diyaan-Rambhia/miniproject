"""
Model 5: Fusion Scoring Head
==============================
Combines the three upstream signals into a single Threat Score (0-100):
  - Transformer classifier confidence (from model 2)
  - VAE anomaly score (from model 3)
  - DGA detector probability (from model 4)

This model does NOT touch raw traffic — it's trained on the *outputs*
of the other three models, paired with ground-truth labels.

Dataset expected: a CSV with one row per flow/event and these columns:
  - transformer_confidence : float, model 2's predicted-class confidence
  - vae_anomaly_score      : float, model 3's per-sample anomaly score
  - dga_probability        : float, model 4's malicious-domain probability
                              (use 0.0 if no domain was present for that flow)
  - label                  : int, 1 = actual attack/malicious, 0 = benign

You build this CSV by running models 2, 3, and 4 in inference mode over
your held-out evaluation set and writing their outputs + the true label
to one file. Paste that path into SIGNALS_CSV_PATH below.

Pipeline stages:
  PRE-TRAINING
    1. Load the signals CSV
    2. Scale the three signal columns
    3. Train/test split, wrap into PyTorch Dataset/DataLoader

  TRAINING
    4. Define fusion MLP
    5. Train loop with checkpointing every epoch

  POST-TRAINING
    6. Evaluate fused model: accuracy, precision/recall/F1, ROC-AUC
    7. Ablation study: each signal alone vs fused (this is your report centerpiece)
    8. Save final model + scaler
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ===================================================================
# CONFIG
# ===================================================================
SIGNALS_CSV_PATH = "PASTE_YOUR_SIGNALS_CSV_PATH_HERE.csv"
OUTPUT_DIR = "./outputs/fusion_head"
RANDOM_STATE = 42
TEST_SIZE = 0.2

BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
HIDDEN_DIM = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGNAL_COLUMNS = ["transformer_confidence", "vae_anomaly_score", "dga_probability"]


# ===================================================================
# PRE-TRAINING STAGE 1: LOAD
# ===================================================================
def load_signals(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"No file found at {csv_path}. Update SIGNALS_CSV_PATH — this file should contain "
            f"columns {SIGNAL_COLUMNS + ['label']}, built by running models 2/3/4 in inference "
            f"mode over your held-out evaluation set."
        )

    df = pd.read_csv(csv_path)
    missing = set(SIGNAL_COLUMNS + ["label"]) - set(df.columns)
    if missing:
        raise ValueError(f"Signals CSV is missing required columns: {missing}")

    print(f"Loaded {len(df):,} rows")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    return df


# ===================================================================
# PRE-TRAINING STAGE 2: SCALE
# ===================================================================
def scale_signals(df: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[SIGNAL_COLUMNS].values)
    y = df["label"].values.astype(np.int64)
    return X, y, scaler


# ===================================================================
# PRE-TRAINING STAGE 3: DATASET
# ===================================================================
class SignalsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===================================================================
# TRAINING STAGE 4: MODEL DEFINITION
# ===================================================================
class FusionMLP(nn.Module):
    """
    Small MLP over the 3 upstream signals.
    Input:  (batch, 3)  -> [transformer_confidence, vae_anomaly_score, dga_probability]
    Output: (batch, 2) logits (benign vs attack) — softmax(logits)[:,1] * 100 = Threat Score
    """
    def __init__(self, num_inputs=3, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)

    def threat_score(self, x):
        """Convenience method: returns 0-100 threat score instead of raw logits."""
        with torch.no_grad():
            probs = torch.softmax(self.forward(x), dim=1)
            return probs[:, 1] * 100.0


# ===================================================================
# TRAINING STAGE 5: TRAIN LOOP
# ===================================================================
def train_model(model, train_loader, val_loader, num_epochs, lr, output_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)
        val_acc, val_f1 = quick_evaluate(model, val_loader)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{num_epochs} | train_loss={epoch_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | time={elapsed:.2f}s"
        )

        torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_last.pt"))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_best.pt"))
            print(f"  -> new best model saved (val_f1={val_f1:.4f})")

    return model


def quick_evaluate(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, f1


# ===================================================================
# POST-TRAINING STAGE 6: FULL EVALUATION
# ===================================================================
def full_evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"\n{'='*60}\nFused Model — Final Evaluation\n{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("Confusion matrix:")
    print(cm)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


# ===================================================================
# POST-TRAINING STAGE 7: ABLATION STUDY
# ===================================================================
def ablation_study(X_train, y_train, X_test, y_test):
    """
    Compares each signal ALONE (via simple logistic regression, since a
    single scalar input doesn't need an MLP) against the fused model.
    This table is your strongest report/viva asset — it shows each
    signal's standalone contribution vs the combined system.
    """
    print(f"\n{'='*60}\nAblation Study — Each Signal Alone vs Fused\n{'='*60}")

    results = {}
    for i, col_name in enumerate(SIGNAL_COLUMNS):
        clf = LogisticRegression(class_weight="balanced")
        clf.fit(X_train[:, i].reshape(-1, 1), y_train)
        preds = clf.predict(X_test[:, i].reshape(-1, 1))
        probs = clf.predict_proba(X_test[:, i].reshape(-1, 1))[:, 1]

        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        results[col_name] = {"f1": f1, "auc": auc}
        print(f"{col_name:30s} alone -> F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    return results


# ===================================================================
# POST-TRAINING STAGE 8: SAVE ARTIFACTS
# ===================================================================
def save_artifacts(model, scaler, output_dir):
    torch.save(model.state_dict(), os.path.join(output_dir, "fusion_mlp_final.pt"))
    joblib.dump(scaler, os.path.join(output_dir, "signal_scaler.joblib"))
    print(f"\nSaved all artifacts to: {output_dir}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print(f"Using device: {DEVICE}")

    # ---- PRE-TRAINING ----
    df = load_signals(SIGNALS_CSV_PATH)
    X, y, scaler = scale_signals(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=RANDOM_STATE
    )

    train_loader = DataLoader(SignalsDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SignalsDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # ---- TRAINING ----
    model = FusionMLP(num_inputs=len(SIGNAL_COLUMNS), hidden_dim=HIDDEN_DIM).to(DEVICE)
    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, OUTPUT_DIR)

    best_path = os.path.join(OUTPUT_DIR, "checkpoint_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print("\nLoaded best checkpoint for final evaluation.")

    # ---- POST-TRAINING ----
    full_evaluate(model, X_test, y_test)
    ablation_study(X_train, y_train, X_test, y_test)
    save_artifacts(model, scaler, OUTPUT_DIR)


if __name__ == "__main__":
    main()