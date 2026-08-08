"""
Explainability Layer — SHAP + Attention Weights
==================================================
Two explainability channels for the Transformer classifier (model 2):

  1. SHAP — feature-level attribution. Runs the model many times on
     perturbed versions of an input to estimate which features drove
     a prediction. Slow (many forward passes per explanation), so
     this should only run on-demand (when a user drills into a
     flagged event), never on every flow automatically.

  2. Attention weights — comes for free from the Transformer's single
     forward pass, IF the encoder is modified to expose them. PyTorch's
     built-in nn.TransformerEncoder does NOT expose attention weights
     by default, so this file provides a custom encoder layer that
     captures and returns them.

Usage: import this module after training model 2, load its checkpoint,
and call explain_flagged_event() on a specific sequence.
"""

import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ===================================================================
# ATTENTION-CAPTURING TRANSFORMER (drop-in replacement for model 2's
# encoder — same architecture, but exposes attention weights)
# ===================================================================
class AttentionCapturingEncoderLayer(nn.Module):
    """
    Re-implements a standard Transformer encoder layer (self-attention
    + feedforward + residual/norm), but stores the attention weights
    from the last forward pass so they can be inspected afterward.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.last_attention_weights = None   # populated on each forward call

    def forward(self, x):
        attn_out, attn_weights = self.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
        self.last_attention_weights = attn_weights.detach()   # (batch, nhead, seq_len, seq_len)

        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class ExplainableFlowTransformer(nn.Module):
    """
    Same architecture as model 2's FlowTransformerClassifier, but built
    from AttentionCapturingEncoderLayer so attention weights are
    inspectable after a forward pass. Load model 2's saved weights
    into this — the parameter shapes match exactly.
    """
    def __init__(self, num_features, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            AttentionCapturingEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        last_token_repr = x[:, -1, :]
        return self.classifier_head(last_token_repr)

    def get_attention_weights(self):
        """Returns attention weights from every layer's last forward pass."""
        return [layer.last_attention_weights for layer in self.layers]


def load_explainable_model_from_checkpoint(checkpoint_path, num_features, num_classes,
                                            d_model, nhead, num_layers, dim_feedforward, dropout, device):
    """
    NOTE: nn.TransformerEncoderLayer and our custom AttentionCapturingEncoderLayer
    have DIFFERENT internal parameter names, so a model-2 checkpoint trained with
    the standard nn.TransformerEncoder will NOT load directly into this class.
    To use this cleanly: either (a) train model 2 directly with
    ExplainableFlowTransformer from the start, or (b) manually copy weights
    layer-by-layer (input_proj, pos_encoding, and per-layer self_attn/linear1/
    linear2/norm1/norm2 all have equivalent shapes). Option (a) is simpler —
    swap the model class in 02_transformer_classifier.py to this one if you
    want attention explainability without a manual weight-copy step.
    """
    model = ExplainableFlowTransformer(
        num_features, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


# ===================================================================
# ATTENTION EXPLANATION
# ===================================================================
def summarize_attention(model: ExplainableFlowTransformer, top_k=3):
    """
    After calling model(x) once, call this to get a human-readable
    summary of which positions in the sequence the model attended to
    most, averaged across heads, for the LAST layer (closest to the
    final decision).
    """
    all_layer_weights = model.get_attention_weights()
    last_layer_weights = all_layer_weights[-1]   # (batch, nhead, seq_len, seq_len)

    # average over heads, look at attention FROM the last token (the one being classified)
    avg_weights = last_layer_weights.mean(dim=1)          # (batch, seq_len, seq_len)
    attention_from_last_token = avg_weights[:, -1, :]      # (batch, seq_len)

    results = []
    for sample_weights in attention_from_last_token:
        top_indices = torch.topk(sample_weights, k=min(top_k, sample_weights.size(0))).indices.tolist()
        top_scores = sample_weights[top_indices].tolist()
        results.append(list(zip(top_indices, top_scores)))

    return results   # list of [(position_in_sequence, attention_score), ...] per sample


# ===================================================================
# SHAP EXPLANATION
# ===================================================================
def compute_shap_values(model, background_sequences: np.ndarray, sequences_to_explain: np.ndarray, device):
    """
    Computes SHAP values for a small batch of flagged sequences.

    background_sequences: a representative sample of "normal" inputs
        (e.g. 50-100 random training sequences) used as the SHAP baseline.
    sequences_to_explain: the specific flagged event(s) to explain
        (keep this small — SHAP is expensive per sample).

    NOTE: SHAP's DeepExplainer struggles with LayerNorm and other custom
    modules found in Transformers (documented SHAP limitation), so we
    use GradientExplainer instead, which is more robust for architectures
    like this one. It's slightly less exact but far more reliable here.
    Since our input is (seq_len, num_features), we wrap the model so
    SHAP sees a flattened input, then reshape internally.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("Install shap first: pip install shap --break-system-packages")

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

    return shap_values   # list (per class) of arrays (num_samples, seq_len*num_features)


def top_shap_features(shap_values_for_sample: np.ndarray, feature_names: list, seq_len: int, top_k=5):
    """
    Given SHAP values for one flattened sample, maps back to
    (timestep, feature_name) pairs and returns the top-k most
    influential ones by absolute SHAP value.
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


# ===================================================================
# COMBINED: EXPLAIN ONE FLAGGED EVENT
# ===================================================================
def explain_flagged_event(model, sequence: np.ndarray, background_sequences: np.ndarray,
                           feature_names: list, device, run_shap=True):
    """
    Full explanation for a single flagged sequence:
      - runs a forward pass to get the prediction + attention weights
      - optionally computes SHAP values (slow — only do this on-demand,
        e.g. when a user clicks into a flagged event on the dashboard)

    Returns a dict ready to hand off to the LLM explanation layer.
    """
    model.eval()
    x = torch.tensor(sequence[np.newaxis, :, :], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    # attention requires a forward pass WITH grad tracking disabled but
    # the hooks still populate during any forward call
    _ = model(x)
    attention_summary = summarize_attention(model, top_k=3)[0]

    explanation = {
        "predicted_class": pred_class,
        "confidence": confidence,
        "top_attended_timesteps": attention_summary,   # [(position, score), ...]
    }

    if run_shap and SHAP_AVAILABLE:
        shap_vals = compute_shap_values(
            model, background_sequences, sequence[np.newaxis, :, :], device
        )
        # SHAP's return format varies by version:
        #   older versions: list of arrays, one per class, each (num_samples, num_flat_features)
        #   newer versions: single array (num_samples, num_flat_features, num_classes)
        # Handle both here.
        if isinstance(shap_vals, list):
            sample_shap = shap_vals[pred_class][0]
        else:
            sample_shap = shap_vals[0, :, pred_class]

        explanation["top_shap_features"] = top_shap_features(
            sample_shap, feature_names, seq_len=sequence.shape[0], top_k=5
        )

    return explanation


if __name__ == "__main__":
    print("This module provides explainability functions — import it after training model 2.")
    print(f"SHAP available: {SHAP_AVAILABLE}")