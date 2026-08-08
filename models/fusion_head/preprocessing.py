"""
Phase: Preprocessing and Scaling
Expects: df (pandas DataFrame loaded via data_loader.py) and signal_columns list
Outputs: X (scaled numpy array), y (target label array), and fitted StandardScaler instance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_signals(df: pd.DataFrame, signal_columns: list):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[signal_columns].values)
    y = df["label"].values.astype(np.int64)
    return X, y, scaler
