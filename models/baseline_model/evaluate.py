"""
Phase: Model Evaluation
Expects: Trained model instance, test feature matrix X_test, target labels y_test, label_encoder, model_name string
Outputs: Dictionary of evaluation metrics (accuracy, macro F1, confusion matrix) and printed classification report
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_model(model, X_test, y_test, label_encoder, model_name: str):
    print(f"\n{'='*60}\nEvaluation: {model_name}\n{'='*60}")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")

    print("Per-class report:")
    print(
        classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm}
