"""
Phase: Anomaly Scoring, Threshold Determination, and Model Evaluation
Expects: Trained VAE model, feature matrices, kl_weight, device
Outputs: Per-sample anomaly score arrays, percentile threshold score, and anomaly detection evaluation metrics (ROC-AUC, Precision, Recall, F1)
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
try:
    from models.VAE.model import vae_loss
except ImportError:
    from model import vae_loss


def compute_anomaly_scores(model, X: np.ndarray, kl_weight, device, batch_size=1024):
    """
    Runs the VAE over X and returns a per-sample anomaly score (reconstruction + KL loss).
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            recon, mu, logvar = model(batch)
            _, recon_loss, kl_div = vae_loss(recon, batch, mu, logvar, kl_weight)
            per_sample = (recon_loss + kl_weight * kl_div).cpu().numpy()
            scores.extend(per_sample)

    return np.array(scores)


def determine_threshold(benign_val_scores: np.ndarray, percentile: float = 95.0):
    threshold = np.percentile(benign_val_scores, percentile)
    print(f"Anomaly threshold set at {percentile}th percentile of benign val scores: {threshold:.4f}")
    return threshold


def evaluate_anomaly_detector(benign_eval_scores, attack_scores, threshold):
    y_true = np.concatenate([
        np.zeros(len(benign_eval_scores)),   # 0 = normal
        np.ones(len(attack_scores)),          # 1 = anomaly
    ])
    y_scores = np.concatenate([benign_eval_scores, attack_scores])
    y_pred = (y_scores > threshold).astype(int)

    auc = roc_auc_score(y_true, y_scores)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'='*60}\nAnomaly Detection Evaluation\n{'='*60}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    return {"auc": auc, "precision": precision, "recall": recall, "f1": f1}
