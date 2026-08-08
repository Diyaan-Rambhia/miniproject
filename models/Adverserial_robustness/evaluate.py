"""
Phase: Evaluation under Clean & Adversarial Conditions
Expects: Model instance, data_loader, attack_fn, criterion, device
Outputs: Evaluation accuracy/macro_f1 scores clean and under attack; three-way robustness comparison table
"""

import torch
from sklearn.metrics import accuracy_score, f1_score
try:
    from models.Adverserial_robustness.attack import fgsm_attack, pgd_attack
except ImportError:
    from attack import fgsm_attack, pgd_attack


def evaluate_clean(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def evaluate_under_attack(model, data_loader, attack_fn, criterion, device):
    model.eval()
    all_preds, all_labels = [], []

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack_fn(model, x, y)
        with torch.no_grad():
            logits = model(x_adv)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def compare_robustness(model_before, model_after, test_loader, criterion, fgsm_epsilon, pgd_epsilon, pgd_alpha, pgd_steps, device):
    print(f"\n{'='*70}\nRobustness Comparison: Baseline vs Hardened\n{'='*70}")

    fgsm_fn = lambda m, x, y: fgsm_attack(m, x, y, fgsm_epsilon, criterion)
    pgd_fn = lambda m, x, y: pgd_attack(m, x, y, pgd_epsilon, pgd_alpha, pgd_steps, criterion)

    results = {}
    for name, model in [("Baseline (undefended)", model_before), ("Hardened (adv. trained)", model_after)]:
        clean_acc, clean_f1 = evaluate_clean(model, test_loader, device)
        fgsm_acc, fgsm_f1 = evaluate_under_attack(model, test_loader, fgsm_fn, criterion, device)
        pgd_acc, pgd_f1 = evaluate_under_attack(model, test_loader, pgd_fn, criterion, device)

        results[name] = {
            "clean_acc": clean_acc, "clean_f1": clean_f1,
            "fgsm_acc": fgsm_acc, "fgsm_f1": fgsm_f1,
            "pgd_acc": pgd_acc, "pgd_f1": pgd_f1,
        }

        print(f"\n{name}:")
        print(f"  Clean data:  acc={clean_acc:.4f}  macro_f1={clean_f1:.4f}")
        print(f"  Under FGSM:  acc={fgsm_acc:.4f}  macro_f1={fgsm_f1:.4f}  (drop: {clean_acc-fgsm_acc:.4f})")
        print(f"  Under PGD:   acc={pgd_acc:.4f}  macro_f1={pgd_f1:.4f}  (drop: {clean_acc-pgd_acc:.4f})")

    return results
