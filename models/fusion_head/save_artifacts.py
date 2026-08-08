"""
Phase: Saving Artifacts
Expects: Trained model instance, fitted StandardScaler instance, and output directory path
Outputs: Saved artifact files on disk (fusion_mlp_final.pt, signal_scaler.joblib)
"""

import os
import shutil
import joblib
import torch


def save_artifacts(model, scaler, output_dir, saved_weights_dir=None):
    torch.save(model.state_dict(), os.path.join(output_dir, "fusion_mlp_final.pt"))
    joblib.dump(scaler, os.path.join(output_dir, "signal_scaler.joblib"))
    print(f"\nSaved all artifacts to: {output_dir}")

    if saved_weights_dir:
        os.makedirs(saved_weights_dir, exist_ok=True)
        dest1 = os.path.join(saved_weights_dir, "fusion_head_final.pt")
        dest2 = os.path.join(saved_weights_dir, "fusion_mlp_final.pt")
        shutil.copy2(os.path.join(output_dir, "fusion_mlp_final.pt"), dest1)
        shutil.copy2(os.path.join(output_dir, "fusion_mlp_final.pt"), dest2)
        print(f"Copied final weights to: {dest1}")
