"""
Phase: Data Loading
Expects: csv_path string (path to a single CICIDS2017 CSV file)
Outputs: Loaded pandas DataFrame
"""

import os
import pandas as pd


def load_cicids2017(csv_path: str) -> pd.DataFrame:
    """
    Loads a single CICIDS2017 CSV file.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"No file found at {csv_path}. Update CSV_PATH in config.py."
        )

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded shape: {df.shape}")
    return df
