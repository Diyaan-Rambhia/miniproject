"""
Phase: Configuration and Setup
Expects: None
Outputs: Configuration constants (file paths, model hyper-parameters, device setup)
"""

import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Relative path placeholder for signals CSV; can be updated to point to generated signals
SIGNALS_CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "signals.csv"))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAVED_WEIGHTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_weights"))

RANDOM_STATE = 42
TEST_SIZE = 0.2

BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
HIDDEN_DIM = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGNAL_COLUMNS = ["transformer_confidence", "vae_anomaly_score", "dga_probability"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVED_WEIGHTS_DIR, exist_ok=True)
