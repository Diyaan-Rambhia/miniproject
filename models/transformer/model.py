"""
Phase: Model Definition
Expects: Model architecture hyperparameters (num_features, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout)
Outputs: PyTorch nn.Module classes (PositionalEncoding, FlowTransformerClassifier)
"""

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class FlowTransformerClassifier(nn.Module):
    """
    Transformer encoder over sequences of flow feature vectors.
    Input:  (batch, seq_len, num_features)
    Output: (batch, num_classes) logits
    """
    def __init__(self, num_features, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        x = self.input_proj(x)                 # -> (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        encoded = self.encoder(x)               # -> (batch, seq_len, d_model)

        # use the representation of the LAST position in the sequence
        last_token_repr = encoded[:, -1, :]      # (batch, d_model)

        logits = self.classifier_head(last_token_repr)
        return logits
