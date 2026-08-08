"""
Phase: Pipeline Orchestration / Main Entrypoint
Expects: Imports from phase files in this directory (config, data_loader, preprocessing, dataset, model, attack, train, evaluate, save_artifacts)
Outputs: Runs data loading, baseline checkpoint loading, baseline attack evaluation, adversarial training defense, robustness comparison, and artifact saving
"""

import os
import sys
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.Adverserial_robustness import config
from models.Adverserial_robustness.data_loader import load_split_csvs
from models.Adverserial_robustness.preprocessing import clean_data, encode_and_scale, build_sequences
from models.Adverserial_robustness.dataset import prepare_dataloaders
from models.Adverserial_robustness.model import FlowTransformerClassifier
from models.Adverserial_robustness.attack import fgsm_attack
from models.Adverserial_robustness.train import adversarial_train
from models.Adverserial_robustness.evaluate import evaluate_clean, evaluate_under_attack, compare_robustness
from models.Adverserial_robustness.save_artifacts import save_artifacts


def main():
    print(f"Using device: {config.DEVICE}")

    # ---- PRE-TRAINING ----
    train_df, test_df = load_split_csvs(config.TRAIN_CSV, config.TEST_CSV)
    train_df = clean_data(train_df)
    test_df  = clean_data(test_df)

    X_train, y_train, label_encoder, scaler, feature_names = encode_and_scale(train_df)
    num_classes = len(label_encoder.classes_)
    num_features = X_train.shape[1]

    import numpy as np
    X_test = scaler.transform(test_df.drop(columns=["Label"]).select_dtypes(include=[np.number]))
    y_test = label_encoder.transform(test_df["Label"])

    train_seqs, train_labels = build_sequences(X_train, y_train, config.SEQ_LEN, config.SEQ_STRIDE)
    test_seqs,  test_labels  = build_sequences(X_test,  y_test,  config.SEQ_LEN, config.SEQ_STRIDE)

    train_loader, test_loader = prepare_dataloaders(
        train_seqs, train_labels, test_seqs, test_labels, config.BATCH_SIZE
    )

    # ---- LOAD BASELINE ----
    baseline_model = FlowTransformerClassifier(
        num_features=num_features, num_classes=num_classes,
        d_model=config.D_MODEL, nhead=config.NHEAD, num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD, dropout=config.DROPOUT,
    ).to(config.DEVICE)

    if not os.path.exists(config.BASELINE_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {config.BASELINE_CHECKPOINT_PATH}. "
            f"Run model 2 transformer first."
        )
    baseline_model.load_state_dict(torch.load(config.BASELINE_CHECKPOINT_PATH, map_location=config.DEVICE))
    print("Loaded baseline Transformer weights.")

    criterion = nn.CrossEntropyLoss()

    # ---- ATTACK BASELINE ----
    clean_acc, clean_f1 = evaluate_clean(baseline_model, test_loader, config.DEVICE)
    print(f"\nBaseline on clean test data: acc={clean_acc:.4f} macro_f1={clean_f1:.4f}")

    fgsm_fn = lambda m, x, y: fgsm_attack(m, x, y, config.FGSM_EPSILON, criterion)
    fgsm_acc, fgsm_f1 = evaluate_under_attack(baseline_model, test_loader, fgsm_fn, criterion, config.DEVICE)
    print(f"Baseline under FGSM attack:  acc={fgsm_acc:.4f} macro_f1={fgsm_f1:.4f} (drop: {clean_acc-fgsm_acc:.4f})")

    # ---- DEFENSE (ADVERSARIAL TRAINING) ----
    hardened_model = FlowTransformerClassifier(
        num_features=num_features, num_classes=num_classes,
        d_model=config.D_MODEL, nhead=config.NHEAD, num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD, dropout=config.DROPOUT,
    ).to(config.DEVICE)
    hardened_model.load_state_dict(baseline_model.state_dict())

    hardened_model = adversarial_train(
        hardened_model, train_loader, test_loader,
        config.NUM_EPOCHS, config.LEARNING_RATE, config.FGSM_EPSILON,
        config.ADV_TRAIN_MIX_RATIO, config.OUTPUT_DIR, config.DEVICE
    )

    best_path = os.path.join(config.OUTPUT_DIR, "checkpoint_best.pt")
    if os.path.exists(best_path):
        hardened_model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
        print("\nLoaded best hardened checkpoint for final comparison.")

    # ---- POST-TRAINING COMPARISON ----
    compare_robustness(
        baseline_model, hardened_model, test_loader, criterion,
        config.FGSM_EPSILON, config.PGD_EPSILON, config.PGD_ALPHA, config.PGD_STEPS, config.DEVICE
    )

    save_artifacts(hardened_model, config.OUTPUT_DIR, config.SAVED_WEIGHTS_DIR)


if __name__ == "__main__":
    main()
