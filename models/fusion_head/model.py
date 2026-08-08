"""
Phase: Model Definition
Expects: Architectural parameters (num_inputs, hidden_dim)
Outputs: FusionMLP PyTorch nn.Module class
"""

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """
    Small MLP over the 3 upstream signals.
    Input:  (batch, 3)  -> [transformer_confidence, vae_anomaly_score, dga_probability]
    Output: (batch, 2) logits (benign vs attack) — softmax(logits)[:,1] * 100 = Threat Score
    """
    def __init__(self, num_inputs=3, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)

    def threat_score(self, x):
        """Convenience method: returns 0-100 threat score instead of raw logits."""
        with torch.no_grad():
            probs = torch.softmax(self.forward(x), dim=1)
            return probs[:, 1] * 100.0
