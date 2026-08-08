"""
Phase: Pipeline Orchestration / Main Entrypoint
Expects: Imports from phase files in this directory (config, data_loader, preprocessing, dataset, model, evaluate, save_artifacts)
Outputs: Runs data loading, preprocessing, split/scaling, training of RF & LR, model evaluations, and artifact persistence
"""

import config
from data_loader import load_cicids2017
from preprocessing import clean_data, encode_and_split_features
from dataset import prepare_train_test
from model import train_random_forest, train_logistic_regression
from evaluate import evaluate_model
from save_artifacts import save_artifacts


def main():
    # 1. Load
    df = load_cicids2017(config.CSV_PATH)

    # 2. Clean
    df = clean_data(df)

    # 3. Encode + split features/labels
    X, y, label_encoder = encode_and_split_features(df)
    feature_names = X.columns

    # 4. Train/test split + scaling
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_train_test(
        X, y, config.TEST_SIZE, config.RANDOM_STATE
    )

    # 5. Train both baseline models
    rf_model = train_random_forest(X_train, y_train, config.RANDOM_STATE)
    lr_model = train_logistic_regression(X_train_scaled, y_train, config.RANDOM_STATE)

    # 6. Evaluate both
    rf_results = evaluate_model(rf_model, X_test, y_test, label_encoder, "Random Forest")
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, label_encoder, "Logistic Regression")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Random Forest        -> Accuracy: {rf_results['accuracy']:.4f} | Macro F1: {rf_results['macro_f1']:.4f}")
    print(f"Logistic Regression  -> Accuracy: {lr_results['accuracy']:.4f} | Macro F1: {lr_results['macro_f1']:.4f}")

    # 7. Save artifacts
    save_artifacts(rf_model, lr_model, scaler, label_encoder, feature_names, config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
