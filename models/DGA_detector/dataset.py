"""
Phase: Dataset & DataLoader Wrapping
Expects: Encoded domain array X, target labels y, split sizes, and batch size
Outputs: DomainDataset class and DataLoaders (train_loader, val_loader, test_loader)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class DomainDataset(Dataset):
    def __init__(self, encoded_domains: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(encoded_domains, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    batch_size: int,
):
    train_loader = DataLoader(DomainDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(DomainDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(DomainDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
