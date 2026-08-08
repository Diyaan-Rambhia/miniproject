"""
Phase: Pipeline Orchestration / Main Entrypoint
Expects: Imports from phase files in this directory (config, data_loader, preprocessing, dataset, model, train, evaluate, save_artifacts)
Outputs: Runs the complete end-to-end character-level LSTM DGA detector training and evaluation pipeline, saving artifacts to disk
"""

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.DGA_detector import config
from models.DGA_detector.data_loader import load_split_domain_csvs
from models.DGA_detector.preprocessing import build_vocab, encode_all_domains
from models.DGA_detector.dataset import prepare_dataloaders
from models.DGA_detector.model import DGALSTMClassifier
from models.DGA_detector.train import train_model
from models.DGA_detector.evaluate import full_evaluate
from models.DGA_detector.save_artifacts import save_artifacts


def main():
    print(f"Using device: {config.DEVICE}")

    # ---- PRE-TRAINING ----
    train_df, val_df, test_df = load_split_domain_csvs(config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV)

    vocab = build_vocab(train_df["domain"].astype(str).tolist())
    X_train = encode_all_domains(train_df["domain"].astype(str).tolist(), vocab, config.MAX_LEN)
    y_train = train_df["label"].values
    X_val   = encode_all_domains(val_df["domain"].astype(str).tolist(), vocab, config.MAX_LEN)
    y_val   = val_df["label"].values
    X_test  = encode_all_domains(test_df["domain"].astype(str).tolist(), vocab, config.MAX_LEN)
    y_test  = test_df["label"].values

    train_loader, val_loader, test_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=config.BATCH_SIZE,
    )

    # ---- TRAINING ----
    model = DGALSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LSTM_LAYERS,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)

    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(
        model, train_loader, val_loader,
        config.NUM_EPOCHS, config.LEARNING_RATE, config.OUTPUT_DIR, config.DEVICE
    )

    # reload best checkpoint
    best_path = os.path.join(config.OUTPUT_DIR, "checkpoint_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
        print("\nLoaded best checkpoint for final evaluation.")

    # ---- POST-TRAINING ----
    full_evaluate(model, test_loader, config.DEVICE)
    save_artifacts(model, vocab, config.MAX_LEN, config.OUTPUT_DIR, config.SAVED_WEIGHTS_DIR)


if __name__ == "__main__":
    main()
