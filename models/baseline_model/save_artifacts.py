"""
Phase: Saving Artifacts
Expects: rf_model, lr_model, scaler, label_encoder, feature_names list, output_dir path
Outputs: Saved artifact files on disk (random_forest_baseline.joblib, logistic_regression_baseline.joblib, feature_scaler.joblib, label_encoder.joblib, feature_names.joblib)
"""

import os
import joblib


def save_artifacts(rf_model, lr_model, scaler, label_encoder, feature_names, output_dir):
    joblib.dump(rf_model, os.path.join(output_dir, "random_forest_baseline.joblib"))
    joblib.dump(lr_model, os.path.join(output_dir, "logistic_regression_baseline.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "feature_scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))
    joblib.dump(list(feature_names), os.path.join(output_dir, "feature_names.joblib"))
    print(f"\nSaved all artifacts to: {output_dir}")
