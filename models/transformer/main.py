"""
Phase: Pipeline Orchestration / Main Entrypoint
Expects: Pre-split CSVs at config.TRAIN_CSV / VAL_CSV / TEST_CSV
Outputs: Trained FlowTransformerClassifier with weights in outputs/ and saved_weights/
"""

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.transformer import config
from models.transformer.data_loader import load_split_csvs
from models.transformer.preprocessing import clean_data, encode_and_scale, build_sequences
from models.transformer.dataset import prepare_dataloaders
from models.transformer.model import FlowTransformerClassifier
from models.transformer.train import train_model
from models.transformer.evaluate import full_evaluate
from models.transformer.save_artifacts import save_artifacts


def main():
    print(f"Using device: {config.DEVICE}")

    # ---- PRE-TRAINING ----
    train_df, val_df, test_df = load_split_csvs(config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV)

    # Fit scaler/encoder on train, apply to all splits
    train_df = clean_data(train_df)
    val_df   = clean_data(val_df)
    test_df  = clean_data(test_df)

    X_train, y_train, label_encoder, scaler, feature_names = encode_and_scale(train_df)
    num_classes = len(label_encoder.classes_)
    num_features = X_train.shape[1]

    # Apply fitted transforms to val/test
    import numpy as np
    X_val  = scaler.transform(val_df.drop(columns=["Label"]).select_dtypes(include=[np.number]))
    y_val  = label_encoder.transform(val_df["Label"])
    X_test = scaler.transform(test_df.drop(columns=["Label"]).select_dtypes(include=[np.number]))
    y_test = label_encoder.transform(test_df["Label"])

    train_seqs, train_labels = build_sequences(X_train, y_train, config.SEQ_LEN, config.SEQ_STRIDE)
    val_seqs,   val_labels   = build_sequences(X_val,   y_val,   config.SEQ_LEN, config.SEQ_STRIDE)
    test_seqs,  test_labels  = build_sequences(X_test,  y_test,  config.SEQ_LEN, config.SEQ_STRIDE)

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_seqs, train_labels,
        val_seqs,   val_labels,
        test_seqs,  test_labels,
        batch_size=config.BATCH_SIZE,
    )

    # ---- TRAINING ----
    transformer_model = FlowTransformerClassifier(
        num_features=num_features,
        num_classes=num_classes,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)

    print(f"\nModel parameter count: {sum(p.numel() for p in transformer_model.parameters()):,}")

    transformer_model = train_model(
        transformer_model, train_loader, val_loader,
        config.NUM_EPOCHS, config.LEARNING_RATE, config.OUTPUT_DIR, config.DEVICE
    )

    # ---- POST-TRAINING ----
    best_path = os.path.join(config.OUTPUT_DIR, "checkpoint_best.pt")
    if os.path.exists(best_path):
        transformer_model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
        print("\nLoaded best checkpoint for final evaluation.")

    full_evaluate(transformer_model, test_loader, label_encoder, config.DEVICE)
    save_artifacts(
        transformer_model, label_encoder, scaler, feature_names,
        config.OUTPUT_DIR, config.SAVED_WEIGHTS_DIR
    )


if __name__ == "__main__":
    main()
