"""
Phase: Model Evaluation
Expects: Trained model instance (from train.py or checkpoint), test DataLoader (from dataset.py), label_encoder (from preprocessing.py), and compute device
Outputs: Evaluation metrics dictionary (accuracy, macro F1, confusion matrix) and printed classification report
"""

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def full_evaluate(model, data_loader, label_encoder, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\n{'='*60}\nFinal Test Evaluation\n{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")

    print("Per-class report:")
    print(
        classification_report(
            all_labels, all_preds,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm}
