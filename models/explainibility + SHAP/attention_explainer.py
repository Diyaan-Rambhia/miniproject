"""
Phase: Attention Explanation
Expects: ExplainableFlowTransformer model instance after running a forward pass
Outputs: List of [(timestep_position, attention_score)] summarizing top-attended timesteps per sample
"""

import torch
from model import ExplainableFlowTransformer


def summarize_attention(model: ExplainableFlowTransformer, top_k=3):
    """
    After calling model(x) once, call this to get a human-readable
    summary of which positions in the sequence the model attended to
    most, averaged across heads, for the LAST layer.
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

    return results
