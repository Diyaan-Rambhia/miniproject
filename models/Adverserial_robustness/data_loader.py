"""
Phase: Data Loading
Expects: data_dir path string pointing to folder on disk containing raw CSV files
Outputs: Combined raw pandas DataFrame
"""

import os
import glob
import pandas as pd


def load_split_csvs(train_csv: str, test_csv: str, val_csv: str = None):
    """Loads pre-split flow CSV files."""
    print(f"Loading pre-split flow files...")
    train_df = pd.read_csv(train_csv, low_memory=False)
    test_df  = pd.read_csv(test_csv,  low_memory=False)
    if val_csv and os.path.exists(val_csv):
        val_df = pd.read_csv(val_csv, low_memory=False)
        print(f"  Train: {train_df.shape}  Val: {val_df.shape}  Test: {test_df.shape}")
        return train_df, val_df, test_df
    print(f"  Train: {train_df.shape}  Test: {test_df.shape}")
    return train_df, test_df


def load_all_csvs(data_dir: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} CSV files. Loading...")
    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape: {full_df.shape}")
    return full_df
