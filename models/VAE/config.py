"""
Phase: Configuration and Setup
Expects: None
Outputs: Configuration constants (file paths, model hyper-parameters, device setup)
"""

import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

# Pre-split data files (produced by models/data/prepare_cicids2017.py)
TRAIN_CSV = os.path.join(DATA_DIR, "cicids2017_train.csv")
VAL_CSV   = os.path.join(DATA_DIR, "cicids2017_val.csv")
TEST_CSV  = os.path.join(DATA_DIR, "cicids2017_test.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAVED_WEIGHTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_weights"))

RANDOM_STATE = 42

BATCH_SIZE = 256
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3

LATENT_DIM = 16
HIDDEN_DIM = 64
KL_WEIGHT = 0.5          # weight on the KL term relative to reconstruction loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVED_WEIGHTS_DIR, exist_ok=True)
