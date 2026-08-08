"""
Phase: Preprocessing
Expects: full_df (pandas DataFrame loaded from raw CSVs via data_loader.py), and window parameters seq_len and stride from config.py
Outputs: sequences (numpy ndarray), seq_labels (numpy ndarray), label_encoder (sklearn LabelEncoder), scaler (sklearn StandardScaler), feature_names (pd.Index)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]

    if "Label" not in df.columns:
        raise ValueError(f"Expected a 'Label' column, found: {list(df.columns)[:10]} ...")

    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with NaN/Inf")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    return df.reset_index(drop=True)


def encode_and_scale(df: pd.DataFrame):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Label"])

    X = df.drop(columns=["Label"])
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Classes: {list(label_encoder.classes_)}")

    return X_scaled, y, label_encoder, scaler, feature_names


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, stride: int):
    """
    Builds fixed-length sequences via a sliding window over the flow data.
    Sequence label = label of the LAST flow in the window.
    """
    num_rows, num_features = X.shape
    sequences = []
    labels = []

    for start in range(0, num_rows - seq_len + 1, stride):
        end = start + seq_len
        sequences.append(X[start:end])
        labels.append(y[end - 1])   # label of the last flow in the window

    sequences = np.stack(sequences)   # (num_sequences, seq_len, num_features)
    labels = np.array(labels)

    print(f"Built {sequences.shape[0]} sequences of shape ({seq_len}, {num_features})")
    return sequences, labels
