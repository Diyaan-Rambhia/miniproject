"""
Phase: Pipeline Orchestration / Main Entrypoint
Expects: Pre-split CSVs at config.TRAIN_CSV / VAL_CSV / TEST_CSV
Outputs: Trained VAE model with weights in outputs/ and saved_weights/
"""

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.VAE import config
from models.VAE.data_loader import load_split_csvs
from models.VAE.preprocessing import clean_and_filter_benign, clean_data, encode_and_scale
from models.VAE.dataset import prepare_dataloaders
from models.VAE.model import VAE
from models.VAE.train import train_vae
from models.VAE.evaluate import compute_anomaly_scores, determine_threshold, evaluate_anomaly_detector
from models.VAE.save_artifacts import save_artifacts


def main():
    print(f"Using device: {config.DEVICE}")

    # ---- PRE-TRAINING ----
    train_df, val_df, test_df = load_split_csvs(config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV)
    
    # Fit scaler/encoder on train
    train_cleaned = clean_data(train_df)
    _, _, _, _, scaler, feature_names = encode_and_scale(train_cleaned)
    
    # Filter benign flows for VAE training & val thresholding
    train_benign = clean_and_filter_benign(train_df)
    val_benign   = clean_and_filter_benign(val_df)
    
    import numpy as np
    X_benign_train = scaler.transform(train_benign.drop(columns=["Label"]).select_dtypes(include=[np.number]))
    eval_benign    = scaler.transform(val_benign.drop(columns=["Label"]).select_dtypes(include=[np.number]))

    clean_test_df  = clean_data(test_df)
    attack_test_df = clean_test_df[clean_test_df["Label"] != "BENIGN"]
    eval_attack    = scaler.transform(attack_test_df.drop(columns=["Label"]).select_dtypes(include=[np.number]))

    num_features = X_benign_train.shape[1]

    train_loader, val_loader, _, _ = prepare_dataloaders(
        X_benign_train, batch_size=config.BATCH_SIZE, random_state=config.RANDOM_STATE
    )

    # ---- TRAINING ----
    model = VAE(num_features=num_features, hidden_dim=config.HIDDEN_DIM, latent_dim=config.LATENT_DIM).to(config.DEVICE)
    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")

    model = train_vae(
        model, train_loader, val_loader,
        config.NUM_EPOCHS, config.LEARNING_RATE, config.KL_WEIGHT,
        config.OUTPUT_DIR, config.DEVICE
    )

    best_path = os.path.join(config.OUTPUT_DIR, "checkpoint_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
        print("\nLoaded best checkpoint for final evaluation.")

    # ---- POST-TRAINING ----
    benign_val_scores = compute_anomaly_scores(model, eval_benign, config.KL_WEIGHT, config.DEVICE)
    threshold = determine_threshold(benign_val_scores, percentile=95.0)

    benign_eval_scores = compute_anomaly_scores(model, eval_benign, config.KL_WEIGHT, config.DEVICE)
    attack_scores      = compute_anomaly_scores(model, eval_attack, config.KL_WEIGHT, config.DEVICE)

    evaluate_anomaly_detector(benign_eval_scores, attack_scores, threshold)
    save_artifacts(model, scaler, threshold, feature_names, config.OUTPUT_DIR, config.SAVED_WEIGHTS_DIR)


if __name__ == "__main__":
    main()
