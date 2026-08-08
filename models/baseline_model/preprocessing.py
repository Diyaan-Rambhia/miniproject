"""
Phase: Preprocessing & Label Encoding
Expects: Loaded pandas DataFrame (from data_loader.py)
Outputs: Feature matrix DataFrame X, encoded target label array y, and fitted LabelEncoder instance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strips whitespace from column names, handles NaN/Inf, and removes duplicates.
    """
    df.columns = [c.strip() for c in df.columns]

    if "Label" not in df.columns:
        raise ValueError(f"Expected a 'Label' column, found: {list(df.columns)[:10]} ...")

    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Dropped {before - after} rows containing NaN/Inf ({before} -> {after})")

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Dropped {before - after} duplicate rows ({before} -> {after})")

    return df


def encode_and_split_features(df: pd.DataFrame):
    """
    Separates numeric features from the Label column and encodes string labels into integers.
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Label"])

    X = df.drop(columns=["Label"])
    X = X.select_dtypes(include=[np.number])

    print(f"Feature matrix shape: {X.shape}")
    print(f"Classes found: {list(label_encoder.classes_)}")

    return X, y, label_encoder
