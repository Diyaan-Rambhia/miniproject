"""
Phase: Saving Artifacts
Expects: Trained VAE model, scaler, threshold float, feature_names list, output_dir, saved_weights_dir
Outputs: Saved artifact files on disk in outputs/ and central saved_weights/
"""

import os
import shutil
import joblib
import torch


def save_artifacts(model, scaler, threshold, feature_names, output_dir, saved_weights_dir=None):
    torch.save(model.state_dict(), os.path.join(output_dir, "vae_final.pt"))
    joblib.dump(scaler, os.path.join(output_dir, "feature_scaler.joblib"))
    joblib.dump(threshold, os.path.join(output_dir, "anomaly_threshold.joblib"))
    joblib.dump(list(feature_names), os.path.join(output_dir, "feature_names.joblib"))
    print(f"\nSaved all artifacts to: {output_dir}")

    if saved_weights_dir:
        os.makedirs(saved_weights_dir, exist_ok=True)
        dest = os.path.join(saved_weights_dir, "vae_final.pt")
        shutil.copy2(os.path.join(output_dir, "vae_final.pt"), dest)
        print(f"Copied final weights to: {dest}")
