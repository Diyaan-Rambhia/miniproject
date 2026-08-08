"""
Phase: Data Loading
Expects: train_csv, val_csv, test_csv path strings (pre-split files from models/data/)
Outputs: train_df, val_df, test_df (pandas DataFrames)
"""

import pandas as pd


def load_split_csvs(train_csv: str, val_csv: str, test_csv: str):
    """Loads pre-split CICIDS2017 CSV files."""
    print(f"Loading pre-split data files...")
    train_df = pd.read_csv(train_csv, low_memory=False)
    val_df   = pd.read_csv(val_csv,   low_memory=False)
    test_df  = pd.read_csv(test_csv,  low_memory=False)
    print(f"  Train: {train_df.shape}  Val: {val_df.shape}  Test: {test_df.shape}")
    return train_df, val_df, test_df
