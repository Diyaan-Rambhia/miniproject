"""
Model 7: Adversarial Robustness Study (Hardened Transformer)
===============================================================
Not a new architecture — this takes the already-trained Transformer
classifier (model 2), attacks it with adversarial perturbations,
measures how much its performance drops, then retrains it on a mix
of clean + adversarial examples (adversarial training) to produce a
hardened second set of weights for the SAME architecture.

Dataset expected: same folder of CICIDS2017 CSVs used for model 2.
Also expects the trained model-2 checkpoint (checkpoint_best.pt) to
already exist — this script loads it as the baseline to attack.

Pipeline stages:
  PRE-TRAINING
    1. Load + clean data (same as model 2)
    2. Rebuild sequences, encode/scale, split
    3. Load the trained baseline Transformer (model 2's weights)

  ATTACK
    4. Craft adversarial examples via FGSM (fast) and PGD (stronger,
       multi-step) against the baseline model
    5. Evaluate baseline performance UNDER ATTACK (this is the "before
       defense" number — expect a real drop here)

  DEFENSE (TRAINING)
    6. Adversarial training: retrain the model on a mix of clean +
       adversarially-perturbed examples, checkpointing every epoch

  POST-TRAINING
    7. Re-evaluate the hardened model: clean accuracy, accuracy under
       FGSM attack, accuracy under PGD attack — this three-way table
       is your report's centerpiece for this section
    8. Save hardened model weights
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ===================================================================
# CONFIG
# ===================================================================
DATA_DIR = "./data/cicids2017"
BASELINE_CHECKPOINT_PATH = "./outputs/transformer/checkpoint_best.pt"   # model 2's trained weights
OUTPUT_DIR = "./outputs/adversarial"
RANDOM_STATE = 42
TEST_SIZE = 0.2

SEQ_LEN = 10
SEQ_STRIDE = 5

BATCH_SIZE = 256
NUM_EPOCHS = 15          # adversarial training epochs (retraining, so fewer needed)
LEARNING_RATE = 1e-3

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# attack strength
FGSM_EPSILON = 0.1        # perturbation magnitude (in scaled feature space)
PGD_EPSILON = 0.1
PGD_ALPHA = 0.02          # step size per PGD iteration
PGD_STEPS = 10

ADV_TRAIN_MIX_RATIO = 0.5   # fraction of each training batch replaced with adversarial examples

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================================================================
# PRE-TRAINING STAGE 1-2: LOAD, CLEAN, SEQUENCE (same as model 2)
# ===================================================================
def load_all_csvs(data_dir: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} CSV files. Loading...")
    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape: {full_df.shape}")
    return full_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    if "Label" not in df.columns:
        raise ValueError(f"Expected a 'Label' column, found: {list(df.columns)[:10]} ...")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def encode_and_scale(df: pd.DataFrame):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Label"])

    X = df.drop(columns=["Label"])
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature matrix shape: {X_scaled.shape} | Classes: {list(label_encoder.classes_)}")
    return X_scaled, y, label_encoder, scaler, feature_names


def build_sequences(X, y, seq_len, stride):
    num_rows, num_features = X.shape
    sequences, labels = [], []

    for start in range(0, num_rows - seq_len + 1, stride):
        end = start + seq_len
        sequences.append(X[start:end])
        labels.append(y[end - 1])

    sequences = np.stack(sequences)
    labels = np.array(labels)
    print(f"Built {sequences.shape[0]} sequences of shape ({seq_len}, {num_features})")
    return sequences, labels


class FlowSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ===================================================================
# MODEL DEFINITION (identical architecture to model 2 — must match
# exactly so the baseline checkpoint loads correctly)
# ===================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class FlowTransformerClassifier(nn.Module):
    def __init__(self, num_features, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        encoded = self.encoder(x)
        last_token_repr = encoded[:, -1, :]
        return self.classifier_head(last_token_repr)


# ===================================================================
# ATTACK STAGE 4: ADVERSARIAL EXAMPLE GENERATION
# ===================================================================
def fgsm_attack(model, x, y, epsilon, criterion):
    """
    Fast Gradient Sign Method — single-step perturbation in the
    direction that most increases the loss.
    """
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    perturbation = epsilon * x.grad.sign()
    x_adv = (x + perturbation).detach()
    return x_adv


def pgd_attack(model, x, y, epsilon, alpha, num_steps, criterion):
    """
    Projected Gradient Descent — stronger, multi-step version of FGSM.
    Perturbation is clipped to stay within an epsilon-ball of the
    original input at every step.
    """
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            perturbation = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = (x_orig + perturbation).detach()

    return x_adv


# ===================================================================
# ATTACK STAGE 5: EVALUATE UNDER ATTACK
# ===================================================================
def evaluate_clean(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def evaluate_under_attack(model, data_loader, attack_fn, criterion):
    """
    Same as evaluate_clean, but perturbs each batch with attack_fn
    before predicting. Requires gradients, so model stays in eval
    mode but we don't wrap this in torch.no_grad() at the top level.
    """
    model.eval()
    all_preds, all_labels = [], []

    for x, y in data_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = attack_fn(model, x, y)
        with torch.no_grad():
            logits = model(x_adv)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


# ===================================================================
# DEFENSE STAGE 6: ADVERSARIAL TRAINING
# ===================================================================
def adversarial_train(model, train_loader, val_loader, num_epochs, lr, epsilon, mix_ratio, output_dir):
    """
    Retrains the model on a mix of clean and FGSM-perturbed examples
    each batch, so the model learns to be robust to small perturbations
    rather than just memorizing clean data patterns.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            # split batch: part stays clean, part gets adversarially perturbed
            split_idx = int(batch_x.size(0) * mix_ratio)
            x_clean, y_clean = batch_x[split_idx:], batch_y[split_idx:]
            x_to_perturb, y_to_perturb = batch_x[:split_idx], batch_y[:split_idx]

            if split_idx > 0:
                x_adv = fgsm_attack(model, x_to_perturb, y_to_perturb, epsilon, criterion)
            else:
                x_adv = x_to_perturb

            x_combined = torch.cat([x_clean, x_adv], dim=0)
            y_combined = torch.cat([y_clean, y_to_perturb], dim=0)

            optimizer.zero_grad()
            logits = model(x_combined)
            loss = criterion(logits, y_combined)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_combined.size(0)

        epoch_loss /= len(train_loader.dataset)
        val_acc, val_f1 = evaluate_clean(model, val_loader)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{num_epochs} | adv_train_loss={epoch_loss:.4f} | "
            f"clean_val_acc={val_acc:.4f} | clean_val_f1={val_f1:.4f} | time={elapsed:.1f}s"
        )

        torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_last.pt"))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_best.pt"))
            print(f"  -> new best hardened model saved (val_f1={val_f1:.4f})")

    return model


# ===================================================================
# POST-TRAINING STAGE 7: THREE-WAY COMPARISON TABLE
# ===================================================================
def compare_robustness(model_before, model_after, test_loader, criterion):
    print(f"\n{'='*70}\nRobustness Comparison: Baseline vs Hardened\n{'='*70}")

    fgsm_fn = lambda m, x, y: fgsm_attack(m, x, y, FGSM_EPSILON, criterion)
    pgd_fn = lambda m, x, y: pgd_attack(m, x, y, PGD_EPSILON, PGD_ALPHA, PGD_STEPS, criterion)

    results = {}
    for name, model in [("Baseline (undefended)", model_before), ("Hardened (adv. trained)", model_after)]:
        clean_acc, clean_f1 = evaluate_clean(model, test_loader)
        fgsm_acc, fgsm_f1 = evaluate_under_attack(model, test_loader, fgsm_fn, criterion)
        pgd_acc, pgd_f1 = evaluate_under_attack(model, test_loader, pgd_fn, criterion)

        results[name] = {
            "clean_acc": clean_acc, "clean_f1": clean_f1,
            "fgsm_acc": fgsm_acc, "fgsm_f1": fgsm_f1,
            "pgd_acc": pgd_acc, "pgd_f1": pgd_f1,
        }

        print(f"\n{name}:")
        print(f"  Clean data:  acc={clean_acc:.4f}  macro_f1={clean_f1:.4f}")
        print(f"  Under FGSM:  acc={fgsm_acc:.4f}  macro_f1={fgsm_f1:.4f}  (drop: {clean_acc-fgsm_acc:.4f})")
        print(f"  Under PGD:   acc={pgd_acc:.4f}  macro_f1={pgd_f1:.4f}  (drop: {clean_acc-pgd_acc:.4f})")

    return results


# ===================================================================
# MAIN
# ===================================================================
def main():
    print(f"Using device: {DEVICE}")

    # ---- PRE-TRAINING: rebuild the same data pipeline as model 2 ----
    df = load_all_csvs(DATA_DIR)
    df = clean_data(df)
    X, y, label_encoder, scaler, feature_names = encode_and_scale(df)
    num_classes = len(label_encoder.classes_)
    num_features = X.shape[1]

    sequences, seq_labels = build_sequences(X, y, SEQ_LEN, SEQ_STRIDE)
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, seq_labels, test_size=TEST_SIZE, stratify=seq_labels, random_state=RANDOM_STATE
    )

    train_loader = DataLoader(FlowSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FlowSequenceDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # ---- LOAD BASELINE (model 2's trained weights) ----
    baseline_model = FlowTransformerClassifier(
        num_features=num_features, num_classes=num_classes,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
    ).to(DEVICE)

    if not os.path.exists(BASELINE_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {BASELINE_CHECKPOINT_PATH}. "
            f"Run model 2 (02_transformer_classifier.py) first."
        )
    baseline_model.load_state_dict(torch.load(BASELINE_CHECKPOINT_PATH, map_location=DEVICE))
    print("Loaded baseline Transformer weights.")

    criterion = nn.CrossEntropyLoss()

    # ---- ATTACK: evaluate baseline under attack (before defense) ----
    clean_acc, clean_f1 = evaluate_clean(baseline_model, test_loader)
    print(f"\nBaseline on clean test data: acc={clean_acc:.4f} macro_f1={clean_f1:.4f}")

    fgsm_fn = lambda m, x, y: fgsm_attack(m, x, y, FGSM_EPSILON, criterion)
    fgsm_acc, fgsm_f1 = evaluate_under_attack(baseline_model, test_loader, fgsm_fn, criterion)
    print(f"Baseline under FGSM attack:  acc={fgsm_acc:.4f} macro_f1={fgsm_f1:.4f} (drop: {clean_acc-fgsm_acc:.4f})")

    # ---- DEFENSE: adversarial training produces the hardened model ----
    # start from a fresh copy of the baseline weights, then adversarially retrain
    hardened_model = FlowTransformerClassifier(
        num_features=num_features, num_classes=num_classes,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
    ).to(DEVICE)
    hardened_model.load_state_dict(baseline_model.state_dict())

    hardened_model = adversarial_train(
        hardened_model, train_loader, test_loader,
        NUM_EPOCHS, LEARNING_RATE, FGSM_EPSILON, ADV_TRAIN_MIX_RATIO, OUTPUT_DIR
    )

    best_path = os.path.join(OUTPUT_DIR, "checkpoint_best.pt")
    if os.path.exists(best_path):
        hardened_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print("\nLoaded best hardened checkpoint for final comparison.")

    # ---- POST-TRAINING: three-way comparison ----
    compare_robustness(baseline_model, hardened_model, test_loader, criterion)

    torch.save(hardened_model.state_dict(), os.path.join(OUTPUT_DIR, "hardened_transformer_final.pt"))
    print(f"\nSaved hardened model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()