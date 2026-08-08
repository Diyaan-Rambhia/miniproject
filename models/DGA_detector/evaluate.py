"""
Phase: Model Evaluation
Expects: Trained model instance, test DataLoader (from dataset.py), and compute device
Outputs: Evaluation metrics dictionary (accuracy, precision, recall, f1, confusion matrix) and printed classification report
"""

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def full_evaluate(model, data_loader, device):
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
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\n{'='*60}\nFinal Test Evaluation\n{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}  (false positives = blocking legit domains)")
    print(f"Recall:    {recall:.4f}  (false negatives = missed DGA domains)")
    print(f"F1:        {f1:.4f}\n")

    print(classification_report(all_labels, all_preds, target_names=["legit", "DGA"], zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}
