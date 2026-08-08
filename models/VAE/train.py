"""
Phase: Model Training
Expects: VAE model instance, train_loader, val_loader, num_epochs, lr, kl_weight, output_dir, device
Outputs: Trained VAE model instance with epoch checkpoints saved on disk
"""

import os
import time
import torch
try:
    from models.VAE.model import vae_loss
except ImportError:
    from model import vae_loss


def quick_evaluate(model, data_loader, kl_weight, device):
    model.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x.to(device)
            recon, mu, logvar = model(batch_x)
            loss, _, _ = vae_loss(recon, batch_x, mu, logvar, kl_weight)
            total_loss += loss.item() * batch_x.size(0)
            n += batch_x.size(0)

    return total_loss / n


def train_vae(model, train_loader, val_loader, num_epochs, lr, kl_weight, output_dir, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_x in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss, _, _ = vae_loss(recon, batch_x, mu, logvar, kl_weight)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)

        val_loss = quick_evaluate(model, val_loader, kl_weight, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{num_epochs} | train_loss={epoch_loss:.4f} | "
            f"val_loss={val_loss:.4f} | time={elapsed:.1f}s"
        )

        torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_best.pt"))
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")

    return model
