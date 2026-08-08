"""
Phase: Model Evaluation and Ablation Study
Expects: Trained model instance, test dataset (X_test, y_test), train dataset (X_train, y_train), signal_columns list, device
Outputs: Evaluation metrics dict (accuracy, precision, recall, f1, auc, confusion matrix) and ablation study results dict
"""

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def full_evaluate(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"\n{'='*60}\nFused Model — Final Evaluation\n{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("Confusion matrix:")
    print(cm)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def ablation_study(X_train, y_train, X_test, y_test, signal_columns):
    """
    Compares each signal ALONE (via simple logistic regression) against the fused model.
    """
    print(f"\n{'='*60}\nAblation Study — Each Signal Alone vs Fused\n{'='*60}")

    results = {}
    for i, col_name in enumerate(signal_columns):
        clf = LogisticRegression(class_weight="balanced")
        clf.fit(X_train[:, i].reshape(-1, 1), y_train)
        preds = clf.predict(X_test[:, i].reshape(-1, 1))
        probs = clf.predict_proba(X_test[:, i].reshape(-1, 1))[:, 1]

        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        results[col_name] = {"f1": f1, "auc": auc}
        print(f"{col_name:30s} alone -> F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    return results
