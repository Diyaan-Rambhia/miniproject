"""
Phase: Model Training
Expects: DGALSTMClassifier instance (from model.py), DataLoaders (train_loader, val_loader), training hyperparams, output directory, and compute device
Outputs: Trained PyTorch model and saved epoch checkpoints on disk
"""

import os
import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


def quick_evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, f1


def train_model(model, train_loader, val_loader, num_epochs, lr, output_dir, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)
        val_acc, val_f1 = quick_evaluate(model, val_loader, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{num_epochs} | train_loss={epoch_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | time={elapsed:.1f}s"
        )

        torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_last.pt"))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_best.pt"))
            print(f"  -> new best model saved (val_f1={val_f1:.4f})")

    return model
