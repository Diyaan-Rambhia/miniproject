"""
Phase: Configuration and Setup
Expects: None
Outputs: Configuration parameters (checkpoint paths, feature paths, model params, device setup, SHAP availability flag)
"""

import os
import torch

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_CHECKPOINT = os.path.abspath(os.path.join(BASE_DIR, "..", "transformer", "outputs", "transformer_final.pt"))
FEATURE_NAMES_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "transformer", "outputs", "feature_names.joblib"))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Default transformer hyperparams (matching model 2)
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
