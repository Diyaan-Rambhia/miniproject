"""
Phase: Preprocessing
Expects: Raw DataFrame loaded via data_loader.py
Outputs: X_scaled (numpy ndarray), y_encoded, is_benign (boolean mask), label_encoder, scaler, and feature_names
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
    y_labels = df["Label"].values                    # keep raw string labels too
    y_encoded = label_encoder.fit_transform(y_labels)

    X = df.drop(columns=["Label"])
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    is_benign = (y_labels == "BENIGN")   # boolean mask

    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Benign flows: {is_benign.sum():,} / {len(is_benign):,} total")

    return X_scaled, y_encoded, is_benign, label_encoder, scaler, feature_names


def clean_and_filter_benign(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    return df[df["Label"] == "BENIGN"].reset_index(drop=True)
