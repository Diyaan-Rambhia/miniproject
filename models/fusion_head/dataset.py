"""
Phase: Dataset & DataLoader Wrapping
Expects: Scaled feature matrix X and label vector y (from preprocessing.py), test size, random state, and batch size
Outputs: SignalsDataset PyTorch Dataset class, train_loader, val_loader, and raw arrays X_train, X_val, X_test, y_train, y_val, y_test
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SignalsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_dataloaders(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int, batch_size: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=random_state
    )

    train_loader = DataLoader(SignalsDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SignalsDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, X_test, y_train, y_val, y_test
