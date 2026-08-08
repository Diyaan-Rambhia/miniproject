"""
Pipeline 2: Adversarial Robustness Pipeline Orchestrator
Orchestrates: Load Baseline Transformer -> FGSM/PGD Attack Evaluation -> Adversarial Training -> 3-Way Comparison Table
"""

import os
import sys
import torch
import torch.nn as nn

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.Adverserial_robustness import config as adv_config
from models.Adverserial_robustness.data_loader import load_all_csvs
from models.Adverserial_robustness.preprocessing import clean_data, encode_and_scale, build_sequences
from models.Adverserial_robustness.dataset import prepare_dataloaders
from models.Adverserial_robustness.model import FlowTransformerClassifier
from models.Adverserial_robustness.attack import fgsm_attack
from models.Adverserial_robustness.train import adversarial_train
from models.Adverserial_robustness.evaluate import evaluate_clean, evaluate_under_attack, compare_robustness
from models.Adverserial_robustness.save_artifacts import save_artifacts


def main():
    print("=" * 80)
    print("      ADVERSARIAL ROBUSTNESS PIPELINE — END-TO-END ORCHESTRATION")
    print("=" * 80)

    # Step 1: Load Data and Build DataLoaders
    print("\n[Step 1/5] Preprocessing Data & Preparing DataLoaders...")
    train_df, test_df = load_split_csvs(adv_config.TRAIN_CSV, adv_config.TEST_CSV)
    train_df = clean_data(train_df)
    test_df  = clean_data(test_df)

    X_train, y_train, label_encoder, scaler, feature_names = encode_and_scale(train_df)
    num_classes = len(label_encoder.classes_)
    num_features = X_train.shape[1]

    import numpy as np
    X_test = scaler.transform(test_df.drop(columns=["Label"]).select_dtypes(include=[np.number]))
    y_test = label_encoder.transform(test_df["Label"])

    train_seqs, train_labels = build_sequences(X_train, y_train, adv_config.SEQ_LEN, adv_config.SEQ_STRIDE)
    test_seqs,  test_labels  = build_sequences(X_test,  y_test,  adv_config.SEQ_LEN, adv_config.SEQ_STRIDE)

    train_loader, test_loader = prepare_dataloaders(
        train_seqs, train_labels, test_seqs, test_labels, adv_config.BATCH_SIZE
    )

    # Step 2: Load Baseline Checkpoint
    print("\n[Step 2/5] Loading Baseline Transformer Model Checkpoint...")
    baseline_model = FlowTransformerClassifier(
        num_features=num_features, num_classes=num_classes,
        d_model=adv_config.D_MODEL, nhead=adv_config.NHEAD, num_layers=adv_config.NUM_LAYERS,
        dim_feedforward=adv_config.DIM_FEEDFORWARD, dropout=adv_config.DROPOUT,
    ).to(adv_config.DEVICE)

    baseline_ckpt = adv_config.BASELINE_CHECKPOINT_PATH
    if not os.path.exists(baseline_ckpt):
        # Fallback to transformer_final.pt if checkpoint_best.pt does not exist
        alt_ckpt = os.path.abspath(os.path.join(BASE_DIR, "transformer", "outputs", "transformer_final.pt"))
        if os.path.exists(alt_ckpt):
            baseline_ckpt = alt_ckpt
        else:
            raise FileNotFoundError(
                f"No baseline transformer checkpoint found at {adv_config.BASELINE_CHECKPOINT_PATH} "
                f"or {alt_ckpt}. Run Pipeline 1 (run_core_pipeline.py) first!"
            )

    baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=adv_config.DEVICE))
    print(f"Successfully loaded baseline weights from: {baseline_ckpt}")

    criterion = nn.CrossEntropyLoss()

    # Step 3: Attack Evaluation on Undefended Baseline Model
    print("\n[Step 3/5] Evaluating Baseline Model under Clean and FGSM Attack Conditions...")
    clean_acc, clean_f1 = evaluate_clean(baseline_model, test_loader, adv_config.DEVICE)
    print(f"  Baseline Clean Performance: Accuracy={clean_acc:.4f} | Macro F1={clean_f1:.4f}")

    fgsm_fn = lambda m, x_t, y_t: fgsm_attack(m, x_t, y_t, adv_config.FGSM_EPSILON, criterion)
    fgsm_acc, fgsm_f1 = evaluate_under_attack(baseline_model, test_loader, fgsm_fn, criterion, adv_config.DEVICE)
    print(f"  Baseline FGSM Attack:       Accuracy={fgsm_acc:.4f} | Macro F1={fgsm_f1:.4f} (Accuracy Drop: {clean_acc - fgsm_acc:.4f})")

    # Step 4: Adversarial Training (Hardened Model)
    print("\n[Step 4/5] Checking/Running Adversarial Training Defense...")
    hardened_ckpt = os.path.join(adv_config.OUTPUT_DIR, "hardened_transformer_final.pt")

    hardened_model = FlowTransformerClassifier(
        num_features=num_features, num_classes=num_classes,
        d_model=adv_config.D_MODEL, nhead=adv_config.NHEAD, num_layers=adv_config.NUM_LAYERS,
        dim_feedforward=adv_config.DIM_FEEDFORWARD, dropout=adv_config.DROPOUT,
    ).to(adv_config.DEVICE)

    if os.path.exists(hardened_ckpt):
        print(f"-> Hardened model checkpoint exists ({hardened_ckpt}). Skipping adversarial training.")
        hardened_model.load_state_dict(torch.load(hardened_ckpt, map_location=adv_config.DEVICE))
    else:
        print("-> Retraining model with FGSM adversarial examples...")
        hardened_model.load_state_dict(baseline_model.state_dict())
        hardened_model = adversarial_train(
            hardened_model, train_loader, test_loader,
            adv_config.NUM_EPOCHS, adv_config.LEARNING_RATE, adv_config.FGSM_EPSILON,
            adv_config.ADV_TRAIN_MIX_RATIO, adv_config.OUTPUT_DIR, adv_config.DEVICE
        )
        best_hardened = os.path.join(adv_config.OUTPUT_DIR, "checkpoint_best.pt")
        if os.path.exists(best_hardened):
            hardened_model.load_state_dict(torch.load(best_hardened, map_location=adv_config.DEVICE))
        save_artifacts(hardened_model, adv_config.OUTPUT_DIR, adv_config.SAVED_WEIGHTS_DIR)

    # Step 5: Final Three-Way Robustness Comparison
    print("\n[Step 5/5] Generating Three-Way Robustness Comparison Table...")
    robustness_results = compare_robustness(
        baseline_model, hardened_model, test_loader, criterion,
        adv_config.FGSM_EPSILON, adv_config.PGD_EPSILON, adv_config.PGD_ALPHA, adv_config.PGD_STEPS, adv_config.DEVICE
    )

    print("\n" + "=" * 80)
    print("               ROBUSTNESS EVALUATION COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model Variant':28s} | {'Clean Acc':10s} | {'FGSM Acc':10s} | {'PGD Acc':10s}")
    print("-" * 80)
    for model_name, metrics in robustness_results.items():
        print(f"{model_name:28s} | {metrics['clean_acc']:10.4f} | {metrics['fgsm_acc']:10.4f} | {metrics['pgd_acc']:10.4f}")
    print("=" * 80)

    print("\nAdversarial Robustness Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
