"""
Phase: SHAP Feature Attribution
Expects: Model instance, background_sequences array, sequences_to_explain array, feature_names list, compute device
Outputs: Raw SHAP values array and top_shap_features mapping (timestep, feature_name, shap_value)
"""

import numpy as np
import torch
import torch.nn as nn
from config import SHAP_AVAILABLE

if SHAP_AVAILABLE:
    import shap


def compute_shap_values(model, background_sequences: np.ndarray, sequences_to_explain: np.ndarray, device):
    """
    Computes SHAP values for a small batch of flagged sequences using GradientExplainer.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("Install shap first: pip install shap")

    seq_len, num_features = background_sequences.shape[1], background_sequences.shape[2]

    class FlattenedWrapper(nn.Module):
        def __init__(self, base_model, seq_len, num_features):
            super().__init__()
            self.base_model = base_model
            self.seq_len = seq_len
            self.num_features = num_features

        def forward(self, x_flat):
            x = x_flat.view(-1, self.seq_len, self.num_features)
            return self.base_model(x)

    wrapped_model = FlattenedWrapper(model, seq_len, num_features).to(device)

    background_flat = torch.tensor(
        background_sequences.reshape(background_sequences.shape[0], -1), dtype=torch.float32
    ).to(device)
    explain_flat = torch.tensor(
        sequences_to_explain.reshape(sequences_to_explain.shape[0], -1), dtype=torch.float32
    ).to(device)

    explainer = shap.GradientExplainer(wrapped_model, background_flat)
    shap_values = explainer.shap_values(explain_flat)

    return shap_values


def top_shap_features(shap_values_for_sample: np.ndarray, feature_names: list, seq_len: int, top_k=5):
    """
    Given SHAP values for one flattened sample, maps back to
    (timestep, feature_name) pairs and returns top-k most influential ones.
    """
    num_features = len(feature_names)
    reshaped = shap_values_for_sample.reshape(seq_len, num_features)

    flat_abs = np.abs(reshaped).flatten()
    top_flat_indices = np.argsort(flat_abs)[::-1][:top_k]

    results = []
    for idx in top_flat_indices:
        timestep = idx // num_features
        feat_idx = idx % num_features
        results.append({
            "timestep": int(timestep),
            "feature": feature_names[feat_idx],
            "shap_value": float(reshaped[timestep, feat_idx]),
        })
    return results
