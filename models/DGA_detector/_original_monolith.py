"""
Model 4: DGA Detector — Character-Level LSTM
==============================================
Independent signal from the flow-based models — classifies domain
names as malicious (DGA-generated) vs legitimate, based purely on
the string itself (character-level).

Dataset expected: two plain files (paste your own paths below):
  - DGA_DOMAINS_PATH: one malicious/DGA domain per line (no header)
  - LEGIT_DOMAINS_PATH: one legitimate domain per line (e.g. from
    a Tranco/Alexa top-domains list), no header

Pipeline stages:
  PRE-TRAINING
    1. Load both domain lists, assign labels (1=DGA, 0=legit)
    2. Build character vocabulary from all domains seen
    3. Encode each domain as a padded sequence of character indices
    4. Train/test split, wrap into PyTorch Dataset/DataLoader

  TRAINING
    5. Define char-level embedding + LSTM classifier
    6. Train loop with checkpointing every epoch

  POST-TRAINING
    7. Evaluate: accuracy, precision/recall/F1, confusion matrix
    8. Save final model + vocabulary + max_len
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ===================================================================
# CONFIG
# ===================================================================
DGA_DOMAINS_PATH = "/content/Data/dga_data.csv"      # one domain per line
LEGIT_DOMAINS_PATH = "/content/Data/tranco_V3JQN.csv"   # one domain per line
OUTPUT_DIR = "./outputs/dga_detector"
RANDOM_STATE = 42
TEST_SIZE = 0.2

MAX_LEN = 64             # max characters per domain (truncate/pad to this)
BATCH_SIZE = 256
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3

EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================================================================
# PRE-TRAINING STAGE 1: LOAD + LABEL
# ===================================================================
def load_domain_lists(dga_path: str, legit_path: str):
    if not os.path.isfile(dga_path):
        raise FileNotFoundError(f"No file found at {dga_path}. Update DGA_DOMAINS_PATH.")
    if not os.path.isfile(legit_path):
        raise FileNotFoundError(f"No file found at {legit_path}. Update LEGIT_DOMAINS_PATH.")

    with open(dga_path, "r", encoding="utf-8", errors="ignore") as f:
        dga_domains = [line.strip().lower() for line in f if line.strip()]

    with open(legit_path, "r", encoding="utf-8", errors="ignore") as f:
        legit_domains = [line.strip().lower() for line in f if line.strip()]

    print(f"Loaded {len(dga_domains):,} DGA domains, {len(legit_domains):,} legit domains")

    domains = dga_domains + legit_domains
    labels = [1] * len(dga_domains) + [0] * len(legit_domains)

    df = pd.DataFrame({"domain": domains, "label": labels})
    df = df.drop_duplicates(subset="domain").reset_index(drop=True)
    print(f"After dedup: {len(df):,} total domains ({df['label'].sum():,} DGA, {(df['label']==0).sum():,} legit)")

    return df


# ===================================================================
# PRE-TRAINING STAGE 2: BUILD CHARACTER VOCABULARY
# ===================================================================
def build_vocab(domains: list):
    """
    Builds a char -> index mapping from all characters seen across
    the domain list. Index 0 reserved for padding, 1 for unknown.
    """
    chars = sorted(set("".join(domains)))
    vocab = {ch: idx + 2 for idx, ch in enumerate(chars)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    print(f"Vocabulary size: {len(vocab)} (includes PAD/UNK)")
    return vocab


def encode_domain(domain: str, vocab: dict, max_len: int) -> np.ndarray:
    """
    Converts a domain string into a fixed-length array of character
    indices, truncating or padding as needed.
    """
    indices = [vocab.get(ch, vocab["<UNK>"]) for ch in domain[:max_len]]
    if len(indices) < max_len:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return np.array(indices, dtype=np.int64)


def encode_all_domains(domains: list, vocab: dict, max_len: int) -> np.ndarray:
    return np.stack([encode_domain(d, vocab, max_len) for d in domains])


# ===================================================================
# PRE-TRAINING STAGE 3/4: DATASET
# ===================================================================
class DomainDataset(Dataset):
    def __init__(self, encoded_domains: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(encoded_domains, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===================================================================
# TRAINING STAGE 5: MODEL DEFINITION
# ===================================================================
class DGALSTMClassifier(nn.Module):
    """
    Char-level embedding -> LSTM -> classification head.
    Input:  (batch, max_len) of character indices
    Output: (batch, 2) logits (legit vs DGA)
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),   # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        # x: (batch, max_len)
        embedded = self.embedding(x)                     # (batch, max_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)        # h_n: (num_layers*2, batch, hidden_dim)

        # concatenate final forward + backward hidden states from the last layer
        last_forward = h_n[-2, :, :]
        last_backward = h_n[-1, :, :]
        final_repr = torch.cat([last_forward, last_backward], dim=1)   # (batch, hidden_dim*2)

        logits = self.classifier_head(final_repr)
        return logits


# ===================================================================
# TRAINING STAGE 6: TRAIN LOOP
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
            f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | time={elapsed:.1f}s"
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
# POST-TRAINING STAGE 7: FULL EVALUATION
# ===================================================================
def full_evaluate(model, data_loader):
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
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\n{'='*60}\nFinal Test Evaluation\n{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}  (false positives = blocking legit domains)")
    print(f"Recall:    {recall:.4f}  (false negatives = missed DGA domains)")
    print(f"F1:        {f1:.4f}\n")

    print(classification_report(all_labels, all_preds, target_names=["legit", "DGA"], zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}


# ===================================================================
# POST-TRAINING STAGE 8: SAVE ARTIFACTS
# ===================================================================
def save_artifacts(model, vocab, max_len, output_dir):
    torch.save(model.state_dict(), os.path.join(output_dir, "dga_lstm_final.pt"))
    joblib.dump(vocab, os.path.join(output_dir, "char_vocab.joblib"))
    joblib.dump(max_len, os.path.join(output_dir, "max_len.joblib"))
    print(f"\nSaved all artifacts to: {output_dir}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print(f"Using device: {DEVICE}")

    # ---- PRE-TRAINING ----
    df = load_domain_lists(DGA_DOMAINS_PATH, LEGIT_DOMAINS_PATH)

    vocab = build_vocab(df["domain"].tolist())
    X = encode_all_domains(df["domain"].tolist(), vocab, MAX_LEN)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # further split train into train/val for training-time monitoring
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=RANDOM_STATE
    )

    train_loader = DataLoader(DomainDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DomainDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(DomainDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # ---- TRAINING ----
    model = DGALSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, OUTPUT_DIR)

    # reload best checkpoint
    best_path = os.path.join(OUTPUT_DIR, "checkpoint_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print("\nLoaded best checkpoint for final evaluation.")

    # ---- POST-TRAINING ----
    full_evaluate(model, test_loader)
    save_artifacts(model, vocab, MAX_LEN, OUTPUT_DIR)


if __name__ == "__main__":
    main()