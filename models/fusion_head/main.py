"""
Phase: Pipeline Orchestration / Main Entrypoint
Expects: Imports from phase files in this directory (config, data_loader, preprocessing, dataset, model, train, evaluate, save_artifacts)
Outputs: Runs the complete end-to-end fusion head training, evaluation, and ablation pipeline, saving artifacts to disk
"""

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.fusion_head import config
from models.fusion_head.data_loader import load_signals
from models.fusion_head.preprocessing import scale_signals
from models.fusion_head.dataset import prepare_dataloaders
from models.fusion_head.model import FusionMLP
from models.fusion_head.train import train_model
from models.fusion_head.evaluate import full_evaluate, ablation_study
from models.fusion_head.save_artifacts import save_artifacts


def main():
    print(f"Using device: {config.DEVICE}")

    # ---- PRE-TRAINING ----
    df = load_signals(config.SIGNALS_CSV_PATH, config.SIGNAL_COLUMNS)
    X, y, scaler = scale_signals(df, config.SIGNAL_COLUMNS)

    (
        train_loader, val_loader,
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = prepare_dataloaders(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        batch_size=config.BATCH_SIZE,
    )

    # ---- TRAINING ----
    model = FusionMLP(num_inputs=len(config.SIGNAL_COLUMNS), hidden_dim=config.HIDDEN_DIM).to(config.DEVICE)
    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(
        model, train_loader, val_loader,
        config.NUM_EPOCHS, config.LEARNING_RATE, config.OUTPUT_DIR, config.DEVICE
    )

    best_path = os.path.join(config.OUTPUT_DIR, "checkpoint_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
        print("\nLoaded best checkpoint for final evaluation.")

    # ---- POST-TRAINING ----
    full_evaluate(model, X_test, y_test, config.DEVICE)
    ablation_study(X_train, y_train, X_test, y_test, config.SIGNAL_COLUMNS)
    save_artifacts(model, scaler, config.OUTPUT_DIR, config.SAVED_WEIGHTS_DIR)


if __name__ == "__main__":
    main()
