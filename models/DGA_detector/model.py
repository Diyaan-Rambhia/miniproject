"""
Phase: Model Definition
Expects: Architecture hyperparameters (vocab_size, embed_dim, hidden_dim, num_layers, dropout)
Outputs: PyTorch nn.Module class (DGALSTMClassifier)
"""

import torch
import torch.nn as nn


class DGALSTMClassifier(nn.Module):
    """
    Char-level embedding -> LSTM -> classification head.
    Input:  (batch, max_len) of character indices
    Output: (batch, 2) logits (legit vs DGA)
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),   # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        # x: (batch, max_len)
        embedded = self.embedding(x)                     # (batch, max_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)        # h_n: (num_layers*2, batch, hidden_dim)

        # concatenate final forward + backward hidden states from the last layer
        last_forward = h_n[-2, :, :]
        last_backward = h_n[-1, :, :]
        final_repr = torch.cat([last_forward, last_backward], dim=1)   # (batch, hidden_dim*2)

        logits = self.classifier_head(final_repr)
        return logits
