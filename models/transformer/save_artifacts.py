"""
Phase: Saving Artifacts
Expects: Trained model, label_encoder, scaler, feature_names (from previous phases), output_dir, and saved_weights_dir
Outputs: Saved artifact files on disk in outputs/ and in saved_weights/
"""

import os
import shutil
import joblib
import torch


def save_artifacts(model, label_encoder, scaler, feature_names, output_dir, saved_weights_dir=None):
    torch.save(model.state_dict(), os.path.join(output_dir, "transformer_final.pt"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "feature_scaler.joblib"))
    joblib.dump(list(feature_names), os.path.join(output_dir, "feature_names.joblib"))
    print(f"\nSaved all artifacts to: {output_dir}")

    # Also copy final weights to central saved_weights/ directory
    if saved_weights_dir:
        os.makedirs(saved_weights_dir, exist_ok=True)
        dest = os.path.join(saved_weights_dir, "transformer_final.pt")
        shutil.copy2(os.path.join(output_dir, "transformer_final.pt"), dest)
        print(f"Copied final weights to: {dest}")
