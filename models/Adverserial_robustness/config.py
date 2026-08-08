"""
Phase: Configuration and Setup
Expects: None
Outputs: Configuration parameters (file paths, model hyperparams, attack params, device setup)
"""

import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
TRAIN_CSV = os.path.join(DATA_DIR, "cicids2017_train.csv")
VAL_CSV   = os.path.join(DATA_DIR, "cicids2017_val.csv")
TEST_CSV  = os.path.join(DATA_DIR, "cicids2017_test.csv")
BASELINE_CHECKPOINT_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "transformer", "outputs", "checkpoint_best.pt"))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAVED_WEIGHTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_weights"))

RANDOM_STATE = 42
TEST_SIZE = 0.2

SEQ_LEN = 10
SEQ_STRIDE = 5

BATCH_SIZE = 256
NUM_EPOCHS = 15          # adversarial training epochs
LEARNING_RATE = 1e-3

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# attack strength
FGSM_EPSILON = 0.1        # perturbation magnitude (in scaled feature space)
PGD_EPSILON = 0.1
PGD_ALPHA = 0.02          # step size per PGD iteration
PGD_STEPS = 10

ADV_TRAIN_MIX_RATIO = 0.5   # fraction of each training batch replaced with adversarial examples

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVED_WEIGHTS_DIR, exist_ok=True)
