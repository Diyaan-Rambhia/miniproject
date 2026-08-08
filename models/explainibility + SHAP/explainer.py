"""
Phase: Event Explanation Pipeline
Expects: Model instance, single sequence array, background sequences array, feature_names list, device, run_shap flag
Outputs: Comprehensive explanation dictionary (predicted_class, confidence, top_attended_timesteps, top_shap_features)
"""

import numpy as np
import torch
from config import SHAP_AVAILABLE
from attention_explainer import summarize_attention
from shap_explainer import compute_shap_values, top_shap_features


def explain_flagged_event(model, sequence: np.ndarray, background_sequences: np.ndarray,
                           feature_names: list, device, run_shap=True):
    """
    Full explanation for a single flagged sequence combining attention and SHAP attributions.
    """
    model.eval()
    x = torch.tensor(sequence[np.newaxis, :, :], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    _ = model(x)
    attention_summary = summarize_attention(model, top_k=3)[0]

    explanation = {
        "predicted_class": pred_class,
        "confidence": confidence,
        "top_attended_timesteps": attention_summary,
    }

    if run_shap and SHAP_AVAILABLE:
        shap_vals = compute_shap_values(
            model, background_sequences, sequence[np.newaxis, :, :], device
        )
        if isinstance(shap_vals, list):
            sample_shap = shap_vals[pred_class][0]
        else:
            sample_shap = shap_vals[0, :, pred_class]

        explanation["top_shap_features"] = top_shap_features(
            sample_shap, feature_names, seq_len=sequence.shape[0], top_k=5
        )

    return explanation
