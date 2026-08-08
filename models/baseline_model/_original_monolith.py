"""
Model 1: Baseline Classical Model (sanity-check only)
======================================================
Purpose: verify the data pipeline and labels are correct before
investing time in the Transformer classifier. This model is NOT
part of the final production pipeline — it's a debugging step.

Dataset expected: a single CICIDS2017 CSV file (one day/scenario).
Just paste the path in CSV_PATH below. When you're ready to run on
the full multi-file dataset later, this same pipeline extends
naturally — just add a loop to load + concat multiple CSVs.

Pipeline stages:
  1. Load the CSV
  2. Clean (fix column names, drop/replace NaNs and infinities)
  3. Encode labels (multi-class: BENIGN vs each attack type)
  4. Train/test split
  5. Train Random Forest (primary) + Logistic Regression (secondary)
  6. Evaluate both: accuracy, per-class precision/recall/F1, confusion matrix
  7. Save trained model + label encoder + feature list to disk
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# -----------------------------------------------------------------
# CONFIG — edit these paths/values for your setup
# -----------------------------------------------------------------
CSV_PATH = "PASTE_YOUR_CSV_PATH_HERE.csv"   # e.g. "./data/Monday-WorkingHours.pcap_ISCX.csv"
OUTPUT_DIR = "./outputs/baseline"            # where model + artifacts get saved
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------
# STAGE 1: LOAD
# -----------------------------------------------------------------
def load_cicids2017(csv_path: str) -> pd.DataFrame:
    """
    Loads a single CICIDS2017 CSV file.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"No file found at {csv_path}. Update CSV_PATH at the top of this script."
        )

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded shape: {df.shape}")
    return df


# -----------------------------------------------------------------
# STAGE 2: CLEAN
# -----------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    CICIDS2017 CSVs are notoriously messy:
      - column names have leading/trailing whitespace
      - some columns contain Infinity / -Infinity
      - some rows have NaNs
    This function fixes all of that.
    """
    # strip whitespace from column names (a known CICIDS2017 quirk)
    df.columns = [c.strip() for c in df.columns]

    # the label column is typically named "Label" after stripping
    if "Label" not in df.columns:
        raise ValueError(f"Expected a 'Label' column, found: {list(df.columns)[:10]} ...")

    # replace inf/-inf with NaN, then drop rows with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Dropped {before - after} rows containing NaN/Inf ({before} -> {after})")

    # drop exact duplicate rows (common in this dataset)
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Dropped {before - after} duplicate rows ({before} -> {after})")

    return df


# -----------------------------------------------------------------
# STAGE 3: ENCODE LABELS + SELECT FEATURES
# -----------------------------------------------------------------
def encode_and_split_features(df: pd.DataFrame):
    """
    Separates features from the label column and encodes labels
    (BENIGN / DoS Hulk / PortScan / etc.) into integers.
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Label"])

    X = df.drop(columns=["Label"])

    # keep only numeric columns (CICIDS2017 feature columns are numeric;
    # drop anything non-numeric that slipped through, e.g. stray IDs)
    X = X.select_dtypes(include=[np.number])

    print(f"Feature matrix shape: {X.shape}")
    print(f"Classes found: {list(label_encoder.classes_)}")

    return X, y, label_encoder


# -----------------------------------------------------------------
# STAGE 4: TRAIN / TEST SPLIT + SCALING
# -----------------------------------------------------------------
def prepare_train_test(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # scale features — mainly benefits Logistic Regression;
    # Random Forest doesn't need it but it doesn't hurt to share the pipeline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


# -----------------------------------------------------------------
# STAGE 5: TRAIN MODELS
# -----------------------------------------------------------------
def train_random_forest(X_train, y_train, random_state):
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",   # important: CICIDS2017 classes are imbalanced
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf


def train_logistic_regression(X_train_scaled, y_train, random_state):
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    lr.fit(X_train_scaled, y_train)
    return lr


# -----------------------------------------------------------------
# STAGE 6: EVALUATE
# -----------------------------------------------------------------
def evaluate_model(model, X_test, y_test, label_encoder, model_name: str):
    print(f"\n{'='*60}\nEvaluation: {model_name}\n{'='*60}")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

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


# -----------------------------------------------------------------
# STAGE 7: SAVE ARTIFACTS
# -----------------------------------------------------------------
def save_artifacts(rf_model, lr_model, scaler, label_encoder, feature_names, output_dir):
    joblib.dump(rf_model, os.path.join(output_dir, "random_forest_baseline.joblib"))
    joblib.dump(lr_model, os.path.join(output_dir, "logistic_regression_baseline.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "feature_scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))
    joblib.dump(list(feature_names), os.path.join(output_dir, "feature_names.joblib"))
    print(f"\nSaved all artifacts to: {output_dir}")


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------
def main():
    # 1. Load
    df = load_cicids2017(CSV_PATH)

    # 2. Clean
    df = clean_data(df)

    # 3. Encode + split features/labels
    X, y, label_encoder = encode_and_split_features(df)
    feature_names = X.columns

    # 4. Train/test split + scaling
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_train_test(
        X, y, TEST_SIZE, RANDOM_STATE
    )

    # 5. Train both baseline models
    rf_model = train_random_forest(X_train, y_train, RANDOM_STATE)
    lr_model = train_logistic_regression(X_train_scaled, y_train, RANDOM_STATE)

    # 6. Evaluate both
    rf_results = evaluate_model(rf_model, X_test, y_test, label_encoder, "Random Forest")
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, label_encoder, "Logistic Regression")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Random Forest        -> Accuracy: {rf_results['accuracy']:.4f} | Macro F1: {rf_results['macro_f1']:.4f}")
    print(f"Logistic Regression  -> Accuracy: {lr_results['accuracy']:.4f} | Macro F1: {lr_results['macro_f1']:.4f}")
    print(
        "\nIf both models score reasonably (e.g. RF macro F1 well above random-guess baseline "
        "for the number of classes), your pipeline and labels are correct — proceed to the "
        "Transformer classifier. If scores look degenerate (e.g. always predicting BENIGN), "
        "debug preprocessing/label encoding before moving on."
    )

    # 7. Save artifacts
    save_artifacts(rf_model, lr_model, scaler, label_encoder, feature_names, OUTPUT_DIR)


if __name__ == "__main__":
    main()