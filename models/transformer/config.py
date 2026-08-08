"""
Phase: Configuration and Setup
Expects: None
Outputs: Configuration constants (file paths, preprocessing settings, hyper-parameters, device setup)
"""

import os
import torch

# Base paths (relative to models/transformer directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

# Pre-split data files (produced by models/data/prepare_cicids2017.py)
TRAIN_CSV = os.path.join(DATA_DIR, "cicids2017_train.csv")
VAL_CSV   = os.path.join(DATA_DIR, "cicids2017_val.csv")
TEST_CSV  = os.path.join(DATA_DIR, "cicids2017_test.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAVED_WEIGHTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_weights"))

RANDOM_STATE = 42

SEQ_LEN = 10          # number of flows per sequence (window size)
SEQ_STRIDE = 5         # step size between windows (overlap if < SEQ_LEN)

BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

D_MODEL = 128          # transformer hidden dim
NHEAD = 4              # attention heads
NUM_LAYERS = 3         # encoder layers
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVED_WEIGHTS_DIR, exist_ok=True)
