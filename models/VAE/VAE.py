"""
Model 3: VAE — Zero-Day / Anomaly Detector
============================================
Trained only on benign traffic. Learns to reconstruct "normal" flows;
anything that reconstructs poorly (high ELBO loss) is flagged as
anomalous at inference time.

Dataset expected: a folder containing all 8 CICIDS2017 CSV files.
Point DATA_DIR at that folder (same folder as the Transformer model).

Pipeline stages:
  PRE-TRAINING
    1. Load + merge all CSVs in DATA_DIR
    2. Clean (column names, NaNs, infinities, duplicates)
    3. Encode labels, scale numeric features
    4. Split into benign-only (for training) vs full set (for eval)
    5. Train/test split, wrap into PyTorch Dataset/DataLoader

  TRAINING
    6. Define VAE model (encoder -> latent -> decoder)
    7. Train loop (reconstruction + KL loss) with checkpointing every epoch

  POST-TRAINING
    8. Determine anomaly threshold from benign validation loss distribution
    9. Evaluate on held-out benign + attack flows: ROC-AUC, precision/recall at threshold
    10. Save final model + scaler + threshold + feature list
"""

import os
import glob
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve

# ===================================================================
# CONFIG
# ===================================================================
DATA_DIR = "/content/data"          # folder containing all 8 CSVs
OUTPUT_DIR = "./outputs/vae"
RANDOM_STATE = 42
TEST_SIZE = 0.2

BATCH_SIZE = 256
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3

LATENT_DIM = 16
HIDDEN_DIM = 64
KL_WEIGHT = 0.5          # weight on the KL term relative to reconstruction loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================================================================
# PRE-TRAINING STAGE 1: LOAD
# ===================================================================
def load_all_csvs(data_dir: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} CSV files. Loading...")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
        print(f"  loaded {os.path.basename(f)} -> {df.shape}")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape: {full_df.shape}")
    return full_df


# ===================================================================
# PRE-TRAINING STAGE 2: CLEAN
# ===================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]

    if "Label" not in df.columns:
        raise ValueError(f"Expected a 'Label' column, found: {list(df.columns)[:10]} ...")

    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with NaN/Inf")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    return df.reset_index(drop=True)


# ===================================================================
# PRE-TRAINING STAGE 3: ENCODE + SCALE
# ===================================================================
def encode_and_scale(df: pd.DataFrame):
    label_encoder = LabelEncoder()
    y_labels = df["Label"].values                    # keep raw string labels too
    y_encoded = label_encoder.fit_transform(y_labels)

    X = df.drop(columns=["Label"])
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    is_benign = (y_labels == "BENIGN")   # boolean mask, adjust if your label string differs

    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Benign flows: {is_benign.sum():,} / {len(is_benign):,} total")

    return X_scaled, y_encoded, is_benign, label_encoder, scaler, feature_names


# ===================================================================
# PRE-TRAINING STAGE 4/5: SPLIT + DATASET
# ===================================================================
class FlowDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def prepare_splits(X, is_benign, test_size, random_state):
    """
    Benign flows are split into train/val (for training the VAE, which
    only ever sees benign data). ALL flows (benign + attack) are kept
    aside as a separate evaluation set to test anomaly detection.
    """
    X_benign = X[is_benign]
    X_attack = X[~is_benign]

    X_benign_train, X_benign_eval = train_test_split(
        X_benign, test_size=test_size, random_state=random_state
    )

    print(f"Benign training set: {X_benign_train.shape[0]:,}")
    print(f"Benign eval set: {X_benign_eval.shape[0]:,}")
    print(f"Attack eval set: {X_attack.shape[0]:,}")

    return X_benign_train, X_benign_eval, X_attack


# ===================================================================
# TRAINING STAGE 6: MODEL DEFINITION
# ===================================================================
class VAE(nn.Module):
    """
    Simple feedforward VAE over flow feature vectors.
    Input:  (batch, num_features)
    Output: reconstruction (batch, num_features), mu, logvar
    """
    def __init__(self, num_features, hidden_dim, latent_dim):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, kl_weight):
    """
    ELBO = reconstruction loss + KL divergence.
    Returns total loss plus the two components separately (useful for
    per-sample anomaly scoring later).
    """
    recon_loss = torch.mean((recon_x - x) ** 2, dim=1)                      # per-sample MSE
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # per-sample KL

    total_per_sample = recon_loss + kl_weight * kl_div
    return total_per_sample.mean(), recon_loss, kl_div


# ===================================================================
# TRAINING STAGE 7: TRAIN LOOP
# ===================================================================
def train_vae(model, train_loader, val_loader, num_epochs, lr, kl_weight, output_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_x in train_loader:
            batch_x = batch_x.to(DEVICE)

            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss, _, _ = vae_loss(recon, batch_x, mu, logvar, kl_weight)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)

        val_loss = quick_evaluate(model, val_loader, kl_weight)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{num_epochs} | train_loss={epoch_loss:.4f} | "
            f"val_loss={val_loss:.4f} | time={elapsed:.1f}s"
        )

        # checkpoint every epoch
        torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_best.pt"))
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")

    return model


def quick_evaluate(model, data_loader, kl_weight):
    model.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x.to(DEVICE)
            recon, mu, logvar = model(batch_x)
            loss, _, _ = vae_loss(recon, batch_x, mu, logvar, kl_weight)
            total_loss += loss.item() * batch_x.size(0)
            n += batch_x.size(0)

    return total_loss / n


# ===================================================================
# POST-TRAINING STAGE 8: COMPUTE PER-SAMPLE ANOMALY SCORES
# ===================================================================
def compute_anomaly_scores(model, X: np.ndarray, kl_weight, batch_size=1024):
    """
    Runs the VAE over X and returns a per-sample anomaly score
    (reconstruction + KL loss, un-averaged across the batch).
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(DEVICE)
            recon, mu, logvar = model(batch)
            _, recon_loss, kl_div = vae_loss(recon, batch, mu, logvar, kl_weight)
            per_sample = (recon_loss + kl_weight * kl_div).cpu().numpy()
            scores.extend(per_sample)

    return np.array(scores)


# ===================================================================
# POST-TRAINING STAGE 9: THRESHOLD + EVALUATION
# ===================================================================
def determine_threshold(benign_val_scores: np.ndarray, percentile: float = 95.0):
    """
    Sets the anomaly threshold at a chosen percentile of the benign
    validation score distribution (e.g. 95th percentile means we
    expect ~5% false positive rate on benign traffic).
    """
    threshold = np.percentile(benign_val_scores, percentile)
    print(f"Anomaly threshold set at {percentile}th percentile of benign val scores: {threshold:.4f}")
    return threshold


def evaluate_anomaly_detector(benign_eval_scores, attack_scores, threshold):
    y_true = np.concatenate([
        np.zeros(len(benign_eval_scores)),   # 0 = normal
        np.ones(len(attack_scores)),          # 1 = anomaly
    ])
    y_scores = np.concatenate([benign_eval_scores, attack_scores])
    y_pred = (y_scores > threshold).astype(int)

    auc = roc_auc_score(y_true, y_scores)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'='*60}\nAnomaly Detection Evaluation\n{'='*60}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    return {"auc": auc, "precision": precision, "recall": recall, "f1": f1}


# ===================================================================
# POST-TRAINING STAGE 10: SAVE ARTIFACTS
# ===================================================================
def save_artifacts(model, scaler, threshold, feature_names, output_dir):
    torch.save(model.state_dict(), os.path.join(output_dir, "vae_final.pt"))
    joblib.dump(scaler, os.path.join(output_dir, "feature_scaler.joblib"))
    joblib.dump(threshold, os.path.join(output_dir, "anomaly_threshold.joblib"))
    joblib.dump(list(feature_names), os.path.join(output_dir, "feature_names.joblib"))
    print(f"\nSaved all artifacts to: {output_dir}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print(f"Using device: {DEVICE}")

    # ---- PRE-TRAINING ----
    df = load_all_csvs(DATA_DIR)
    df = clean_data(df)

    X, y_encoded, is_benign, label_encoder, scaler, feature_names = encode_and_scale(df)
    num_features = X.shape[1]

    X_benign_train, X_benign_eval, X_attack = prepare_splits(X, is_benign, TEST_SIZE, RANDOM_STATE)

    # further split benign_train into train/val for training-time monitoring
    X_train, X_val = train_test_split(X_benign_train, test_size=0.1, random_state=RANDOM_STATE)

    train_loader = DataLoader(FlowDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FlowDataset(X_val), batch_size=BATCH_SIZE, shuffle=False)

    # ---- TRAINING ----
    model = VAE(num_features=num_features, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")

    model = train_vae(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, KL_WEIGHT, OUTPUT_DIR)

    # reload best checkpoint
    best_path = os.path.join(OUTPUT_DIR, "checkpoint_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print("\nLoaded best checkpoint for final evaluation.")

    # ---- POST-TRAINING ----
    # threshold determined from benign VALIDATION scores (not train, not eval — avoids leakage)
    benign_val_scores = compute_anomaly_scores(model, X_val, KL_WEIGHT)
    threshold = determine_threshold(benign_val_scores, percentile=95.0)

    # final evaluation on held-out benign eval set + attack set
    benign_eval_scores = compute_anomaly_scores(model, X_benign_eval, KL_WEIGHT)
    attack_scores = compute_anomaly_scores(model, X_attack, KL_WEIGHT)

    evaluate_anomaly_detector(benign_eval_scores, attack_scores, threshold)
    save_artifacts(model, scaler, threshold, feature_names, OUTPUT_DIR)


if __name__ == "__main__":
    main()