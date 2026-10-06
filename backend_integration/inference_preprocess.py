"""
Align live /score and /explain inputs with transformer training preprocessing.

Training path (models/transformer/preprocessing.py + main.py):
  clean numeric features -> StandardScaler (fit on train) -> build_sequences window
Inference must apply the same scaler to each flow row; feature columns must match
feature_names.joblib order (same as encode_and_scale column order).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_flow_sequence(
    flow_sequence: np.ndarray,
    scaler: StandardScaler | None,
    expected_num_features: int | None,
) -> np.ndarray:
    seq = np.asarray(flow_sequence, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError("flow_sequence must be 2D (seq_len, num_features)")

    if expected_num_features is not None and seq.shape[1] != expected_num_features:
        raise ValueError(
            f"Expected {expected_num_features} features per flow, got {seq.shape[1]}"
        )

    if scaler is not None:
        seq = scaler.transform(seq).astype(np.float32, copy=False)
    return seq
