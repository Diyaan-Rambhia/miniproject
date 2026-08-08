"""
Phase: Dataset & DataLoader Wrapping
Expects: Feature matrices X_benign_train, X_benign_eval, X_attack (from preprocessing.py), batch_size
Outputs: train_loader, val_loader (DataLoaders for benign-only VAE training), plus X_val array for threshold
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class FlowDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def prepare_dataloaders(X_benign_train: np.ndarray, batch_size: int, random_state: int):
    """Split benign training data 90/10 for train/val monitoring; return loaders + val array."""
    X_train, X_val = train_test_split(X_benign_train, test_size=0.1, random_state=random_state)

    train_loader = DataLoader(FlowDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(FlowDataset(X_val),   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val


prepare_vae_dataloaders = prepare_dataloaders
