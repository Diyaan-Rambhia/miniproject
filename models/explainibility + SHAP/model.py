"""
Phase: Model Definition (Explainable Transformer Encoder)
Expects: Architectural hyperparams (d_model, nhead, dim_feedforward, dropout, num_features, num_classes)
Outputs: AttentionCapturingEncoderLayer, PositionalEncoding, ExplainableFlowTransformer, and checkpoint loading function
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    inspectable after a forward pass.
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
    model = ExplainableFlowTransformer(
        num_features, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model
