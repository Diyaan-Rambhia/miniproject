"""
Phase: Adversarial Training (Defense)
Expects: FlowTransformerClassifier model, train_loader, val_loader, num_epochs, lr, epsilon, mix_ratio, output_dir, device
Outputs: Hardened adversarially-trained model instance with epoch checkpoints saved to disk
"""

import os
import time
import torch
import torch.nn as nn
try:
    from models.Adverserial_robustness.attack import fgsm_attack
    from models.Adverserial_robustness.evaluate import evaluate_clean
except ImportError:
    from attack import fgsm_attack
    from evaluate import evaluate_clean


def adversarial_train(model, train_loader, val_loader, num_epochs, lr, epsilon, mix_ratio, output_dir, device):
    """
    Retrains the model on a mix of clean and FGSM-perturbed examples each batch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            split_idx = int(batch_x.size(0) * mix_ratio)
            x_clean, y_clean = batch_x[split_idx:], batch_y[split_idx:]
            x_to_perturb, y_to_perturb = batch_x[:split_idx], batch_y[:split_idx]

            if split_idx > 0:
                x_adv = fgsm_attack(model, x_to_perturb, y_to_perturb, epsilon, criterion)
            else:
                x_adv = x_to_perturb

            x_combined = torch.cat([x_clean, x_adv], dim=0)
            y_combined = torch.cat([y_clean, y_to_perturb], dim=0)

            optimizer.zero_grad()
            logits = model(x_combined)
            loss = criterion(logits, y_combined)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_combined.size(0)

        epoch_loss /= len(train_loader.dataset)
        val_acc, val_f1 = evaluate_clean(model, val_loader, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{num_epochs} | adv_train_loss={epoch_loss:.4f} | "
            f"clean_val_acc={val_acc:.4f} | clean_val_f1={val_f1:.4f} | time={elapsed:.1f}s"
        )

        torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_last.pt"))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_best.pt"))
            print(f"  -> new best hardened model saved (val_f1={val_f1:.4f})")

    return model
