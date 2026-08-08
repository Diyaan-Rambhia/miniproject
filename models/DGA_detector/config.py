"""
Phase: Configuration and Setup
Expects: None
Outputs: Configuration constants (file paths, model hyper-parameters, device setup)
"""

import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
TRAIN_CSV = os.path.join(DATA_DIR, "domains_train.csv")
VAL_CSV   = os.path.join(DATA_DIR, "domains_val.csv")
TEST_CSV  = os.path.join(DATA_DIR, "domains_test.csv")
DGA_DOMAINS_PATH = os.path.abspath(os.path.join(DATA_DIR, "DGA", "dga_data.csv"))
LEGIT_DOMAINS_PATH = os.path.abspath(os.path.join(DATA_DIR, "Tranco legit domains", "tranco_V3JQN.csv"))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAVED_WEIGHTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_weights"))

RANDOM_STATE = 42
TEST_SIZE = 0.2

MAX_LEN = 64             # max characters per domain (truncate/pad to this)
BATCH_SIZE = 256
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3

EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVED_WEIGHTS_DIR, exist_ok=True)
