"""
Phase: Data Loading
Expects: csv_path (path string to CSV containing signal outputs) and signal_columns list
Outputs: pandas DataFrame containing signal outputs and target label
"""

import os
import pandas as pd


def load_signals(csv_path: str, signal_columns: list) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"No file found at {csv_path}. Update SIGNALS_CSV_PATH — this file should contain "
            f"columns {signal_columns + ['label']}, built by running upstream models in inference mode."
        )

    df = pd.read_csv(csv_path)
    missing = set(signal_columns + ["label"]) - set(df.columns)
    if missing:
        raise ValueError(f"Signals CSV is missing required columns: {missing}")

    print(f"Loaded {len(df):,} rows")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    return df
