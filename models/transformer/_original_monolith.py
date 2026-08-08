"""
Model 2: Transformer Encoder — Known-Attack Classifier
========================================================
Multi-class classifier over sequences of network flows, using a
Transformer encoder with self-attention.

Dataset expected: a folder containing all 8 CICIDS2017 CSV files.
Point DATA_DIR at that folder.

Pipeline stages:
  PRE-TRAINING
    1. Load + merge all CSVs in DATA_DIR
    2. Clean (column names, NaNs, infinities, duplicates)
    3. Encode labels, scale numeric features
    4. Build fixed-length sequences (sliding window over the flows)
    5. Train/test split
    6. Wrap into PyTorch Dataset/DataLoader

  TRAINING
    7. Define Transformer encoder model
    8. Train loop with checkpointing every epoch

  POST-TRAINING
    9. Evaluate: accuracy, per-class precision/recall/F1, confusion matrix
    10. Save final model + label encoder + scaler + feature list
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ===================================================================
# CONFIG
# ===================================================================
DATA_DIR = "/content/data"          # folder containing all 8 CSVs
OUTPUT_DIR = "./outputs/transformer"
RANDOM_STATE = 42
TEST_SIZE = 0.2

SEQ_LEN = 10          # number of flows per sequence (window size)
SEQ_STRIDE = 5         # step size between windows (overlap if < SEQ_LEN)

BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

D_MODEL = 128          # transformer hidden dim
NHEAD = 4              # attention heads
NUM_LAYERS = 3         # encoder layers
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

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
    y = label_encoder.fit_transform(df["Label"])

    X = df.drop(columns=["Label"])
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Classes: {list(label_encoder.classes_)}")

    return X_scaled, y, label_encoder, scaler, feature_names


# ===================================================================
# PRE-TRAINING STAGE 4: BUILD SEQUENCES (sliding window)
# ===================================================================
def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, stride: int):
    """
    Builds fixed-length sequences via a sliding window over the flow data.

    NOTE: this is a simple sequential windowing approach — it treats
    consecutive rows in the (cleaned, still time-ordered) dataframe as
    a sequence. If you want sequences grouped by source IP instead
    (more semantically meaningful "this host's behavior over time"),
    group the dataframe by source IP before calling this function and
    call it once per group, then concatenate the resulting sequences.
    This version is the general-purpose default.

    Sequence label = label of the LAST flow in the window (i.e. we're
    asking: "given the last seq_len flows, what is the current flow?").
    """
    num_rows, num_features = X.shape
    sequences = []
    labels = []

    for start in range(0, num_rows - seq_len + 1, stride):
        end = start + seq_len
        sequences.append(X[start:end])
        labels.append(y[end - 1])   # label of the last flow in the window

    sequences = np.stack(sequences)   # (num_sequences, seq_len, num_features)
    labels = np.array(labels)

    print(f"Built {sequences.shape[0]} sequences of shape ({seq_len}, {num_features})")
    return sequences, labels


# ===================================================================
# PRE-TRAINING STAGE 5/6: DATASET + DATALOADER
# ===================================================================
class FlowSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ===================================================================
# TRAINING STAGE 7: MODEL DEFINITION
# ===================================================================
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class FlowTransformerClassifier(nn.Module):
    """
    Transformer encoder over sequences of flow feature vectors.
    Input:  (batch, seq_len, num_features)
    Output: (batch, num_classes) logits
    """
    def __init__(self, num_features, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        x = self.input_proj(x)                 # -> (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        encoded = self.encoder(x)               # -> (batch, seq_len, d_model)

        # use the representation of the LAST position in the sequence
        # (corresponds to the flow we're classifying — see build_sequences)
        last_token_repr = encoded[:, -1, :]      # (batch, d_model)

        logits = self.classifier_head(last_token_repr)
        return logits


# ===================================================================
# TRAINING STAGE 8: TRAIN LOOP
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

        # quick validation pass
        val_acc, val_f1 = quick_evaluate(model, val_loader)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={epoch_loss:.4f} | val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f} | "
            f"time={elapsed:.1f}s"
        )

        # checkpoint every epoch (protects against session cutoffs on Kaggle/Colab)
        checkpoint_path = os.path.join(output_dir, "checkpoint_last.pt")
        torch.save(model.state_dict(), checkpoint_path)

        # also save best-so-far model separately
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_best.pt"))
            print(f"  -> new best model saved (val_macro_f1={val_f1:.4f})")

    return model


def quick_evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, macro_f1


# ===================================================================
# POST-TRAINING STAGE 9: FULL EVALUATION
# ===================================================================
def full_evaluate(model, data_loader, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\n{'='*60}\nFinal Test Evaluation\n{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")

    print("Per-class report:")
    print(
        classification_report(
            all_labels, all_preds,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm}


# ===================================================================
# POST-TRAINING STAGE 10: SAVE ARTIFACTS
# ===================================================================
def save_artifacts(model, label_encoder, scaler, feature_names, output_dir):
    torch.save(model.state_dict(), os.path.join(output_dir, "transformer_final.pt"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "feature_scaler.joblib"))
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

    X, y, label_encoder, scaler, feature_names = encode_and_scale(df)
    num_classes = len(label_encoder.classes_)
    num_features = X.shape[1]

    sequences, seq_labels = build_sequences(X, y, SEQ_LEN, SEQ_STRIDE)

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, seq_labels,
        test_size=TEST_SIZE,
        stratify=seq_labels,
        random_state=RANDOM_STATE,
    )

    train_dataset = FlowSequenceDataset(X_train, y_train)
    test_dataset = FlowSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- TRAINING ----
    model = FlowTransformerClassifier(
        num_features=num_features,
        num_classes=num_classes,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, OUTPUT_DIR)

    # ---- POST-TRAINING ----
    # reload best checkpoint before final evaluation
    best_path = os.path.join(OUTPUT_DIR, "checkpoint_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print("\nLoaded best checkpoint for final evaluation.")

    full_evaluate(model, test_loader, label_encoder)
    save_artifacts(model, label_encoder, scaler, feature_names, OUTPUT_DIR)


if __name__ == "__main__":
    main()