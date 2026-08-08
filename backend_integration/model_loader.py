"""
Model and Artifact Loader Container
Loads PyTorch model checkpoints and joblib scalers/encoders across Transformer, VAE, DGA, and Fusion models.
"""

import os
import joblib
import torch
import numpy as np

from backend_integration import config
from models.transformer.model import FlowTransformerClassifier
from models.VAE.model import VAE
from models.DGA_detector.model import DGALSTMClassifier
from models.fusion_head.model import FusionMLP


class ModelContainer:
    def __init__(self):
        self.transformer_model = None
        self.trans_scaler = None
        self.trans_label_encoder = None
        self.trans_feature_names = None

        self.vae_model = None
        self.vae_scaler = None
        self.vae_threshold = 0.5

        self.dga_model = None
        self.dga_vocab = {}
        self.dga_max_len = 64

        self.fusion_model = None
        self.fusion_scaler = None

        self.is_loaded = False

    def load_all(self):
        print("-> Loading model checkpoints and artifacts...")
        device = config.DEVICE

        # 1. Load Transformer Classifier
        if os.path.exists(config.TRANSFORMER_CHECKPOINT):
            self.trans_label_encoder = joblib.load(config.TRANSFORMER_LABEL_ENC)
            self.trans_scaler = joblib.load(config.TRANSFORMER_SCALER)
            self.trans_feature_names = joblib.load(config.TRANSFORMER_FEAT_NAMES)

            num_classes = len(self.trans_label_encoder.classes_)
            num_features = len(self.trans_feature_names)

            self.transformer_model = FlowTransformerClassifier(
                num_features=num_features, num_classes=num_classes,
                d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1
            ).to(device)
            self.transformer_model.load_state_dict(torch.load(config.TRANSFORMER_CHECKPOINT, map_location=device))
            self.transformer_model.eval()
            print("   [OK] Transformer Classifier loaded")
        else:
            print("   [WARN] Transformer checkpoint not found at", config.TRANSFORMER_CHECKPOINT)

        # 2. Load VAE Anomaly Detector
        if os.path.exists(config.VAE_CHECKPOINT):
            self.vae_scaler = joblib.load(config.VAE_SCALER)
            self.vae_threshold = float(joblib.load(config.VAE_THRESHOLD))
            num_features = len(joblib.load(config.TRANSFORMER_FEAT_NAMES)) if os.path.exists(config.TRANSFORMER_FEAT_NAMES) else 78

            self.vae_model = VAE(num_features=num_features, hidden_dim=64, latent_dim=16).to(device)
            self.vae_model.load_state_dict(torch.load(config.VAE_CHECKPOINT, map_location=device))
            self.vae_model.eval()
            print("   [OK] VAE Anomaly Detector loaded")
        else:
            print("   [WARN] VAE checkpoint not found at", config.VAE_CHECKPOINT)

        # 3. Load DGA Domain Detector
        if os.path.exists(config.DGA_CHECKPOINT):
            self.dga_vocab = joblib.load(config.DGA_VOCAB)
            self.dga_max_len = int(joblib.load(config.DGA_MAX_LEN))

            self.dga_model = DGALSTMClassifier(
                vocab_size=len(self.dga_vocab), embed_dim=32, hidden_dim=64, num_layers=2, dropout=0.2
            ).to(device)
            self.dga_model.load_state_dict(torch.load(config.DGA_CHECKPOINT, map_location=device))
            self.dga_model.eval()
            print("   [OK] DGA Domain Detector loaded")
        else:
            print("   [WARN] DGA checkpoint not found at", config.DGA_CHECKPOINT)

        # 4. Load Fusion MLP
        if os.path.exists(config.FUSION_CHECKPOINT):
            self.fusion_scaler = joblib.load(config.FUSION_SCALER)
            self.fusion_model = FusionMLP(num_inputs=3, hidden_dim=16).to(device)
            self.fusion_model.load_state_dict(torch.load(config.FUSION_CHECKPOINT, map_location=device))
            self.fusion_model.eval()
            print("   [OK] Fusion Head loaded")
        else:
            print("   [WARN] Fusion head checkpoint not found at", config.FUSION_CHECKPOINT)

        self.is_loaded = True


models_container = ModelContainer()
