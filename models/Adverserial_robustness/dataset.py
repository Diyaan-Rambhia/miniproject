"""
Phase: Dataset & DataLoader Wrapping
Expects: Sequences array, labels array (from preprocessing.py), test_size, random_state, batch_size
Outputs: FlowSequenceDataset class, train_loader, test_loader
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class FlowSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def prepare_dataloaders(
    train_seqs: np.ndarray, train_labels: np.ndarray,
    test_seqs: np.ndarray,  test_labels: np.ndarray,
    batch_size: int,
):
    train_loader = DataLoader(FlowSequenceDataset(train_seqs, train_labels), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(FlowSequenceDataset(test_seqs,  test_labels),  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
