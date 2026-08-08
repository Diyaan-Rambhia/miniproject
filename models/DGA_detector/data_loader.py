"""
Phase: Data Loading
Expects: dga_path and legit_path (path strings to CSV files on disk)
Outputs: DataFrame containing domain strings and binary labels (1=DGA, 0=legit)
"""

import os
import pandas as pd


def load_split_domain_csvs(train_csv: str, val_csv: str, test_csv: str):
    """Loads pre-split domain CSV files."""
    print(f"Loading pre-split domain files...")
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)
    print(f"  Train: {train_df.shape}  Val: {val_df.shape}  Test: {test_df.shape}")
    return train_df, val_df, test_df


def load_dga_dataset(dga_path: str = None, legit_path: str = None) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    combined_path = os.path.abspath(os.path.join(base_dir, "..", "data", "domains_combined.csv"))
    if os.path.exists(combined_path):
        return pd.read_csv(combined_path)
    return load_domain_lists(dga_path, legit_path)


def load_domain_lists(dga_path: str, legit_path: str) -> pd.DataFrame:
    if not os.path.isfile(dga_path):
        raise FileNotFoundError(f"No file found at {dga_path}. Check DGA_DOMAINS_PATH.")
    if not os.path.isfile(legit_path):
        raise FileNotFoundError(f"No file found at {legit_path}. Check LEGIT_DOMAINS_PATH.")

    with open(dga_path, "r", encoding="utf-8", errors="ignore") as f:
        dga_domains = [line.strip().lower() for line in f if line.strip()]

    with open(legit_path, "r", encoding="utf-8", errors="ignore") as f:
        legit_domains = [line.strip().lower() for line in f if line.strip()]

    print(f"Loaded {len(dga_domains):,} DGA domains, {len(legit_domains):,} legit domains")

    domains = dga_domains + legit_domains
    labels = [1] * len(dga_domains) + [0] * len(legit_domains)

    df = pd.DataFrame({"domain": domains, "label": labels})
    df = df.drop_duplicates(subset="domain").reset_index(drop=True)
    print(f"After dedup: {len(df):,} total domains ({df['label'].sum():,} DGA, {(df['label']==0).sum():,} legit)")

    return df
