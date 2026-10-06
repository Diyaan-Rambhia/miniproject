"""
Backend Integration Configuration
"""

import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Transformer Artifacts
TRANSFORMER_CHECKPOINT = os.path.join(MODELS_DIR, "saved_weights", "transformer_final.pt")
TRANSFORMER_SCALER = os.path.join(MODELS_DIR, "saved_weights", "feature_scaler.joblib")
TRANSFORMER_LABEL_ENC = os.path.join(MODELS_DIR, "saved_weights", "label_encoder.joblib")
TRANSFORMER_FEAT_NAMES = os.path.join(MODELS_DIR, "saved_weights", "feature_names.joblib")

# VAE Artifacts
VAE_CHECKPOINT = os.path.join(MODELS_DIR, "VAE", "outputs", "vae_final.pt")
VAE_SCALER = os.path.join(MODELS_DIR, "VAE", "outputs", "feature_scaler.joblib")
VAE_THRESHOLD = os.path.join(MODELS_DIR, "VAE", "outputs", "anomaly_threshold.joblib")

# DGA Detector Artifacts
DGA_CHECKPOINT = os.path.join(MODELS_DIR, "DGA_detector", "outputs", "dga_lstm_final.pt")
DGA_VOCAB = os.path.join(MODELS_DIR, "DGA_detector", "outputs", "char_vocab.joblib")
DGA_MAX_LEN = os.path.join(MODELS_DIR, "DGA_detector", "outputs", "max_len.joblib")

# Fusion Head Artifacts
FUSION_CHECKPOINT = os.path.join(MODELS_DIR, "fusion_head", "outputs", "fusion_mlp_final.pt")
FUSION_SCALER = os.path.join(MODELS_DIR, "fusion_head", "outputs", "signal_scaler.joblib")
# Fusion MLP uses argmax on P(attack); threat_score = P(attack)*100 (see fusion_head/model.py).
FUSION_ATTACK_THRESHOLD = 50.0

ADVERSARIAL_CHECKPOINT = (
    os.path.join(MODELS_DIR, "saved_weights", "hardened_transformer_final.pt")
    if os.path.exists(os.path.join(MODELS_DIR, "saved_weights", "hardened_transformer_final.pt"))
    else os.path.join(MODELS_DIR, "Adverserial_robustness", "outputs", "hardened_transformer_final.pt")
)

# Database Path
DB_FILE = os.path.join(BASE_DIR, "events.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
