"""
Phase: Saving Artifacts
Expects: Hardened model instance and output directory path
Outputs: Saved hardened model checkpoint on disk (hardened_transformer_final.pt)
"""

import os
import shutil
import torch


def save_artifacts(model, output_dir, saved_weights_dir=None):
    torch.save(model.state_dict(), os.path.join(output_dir, "hardened_transformer_final.pt"))
    print(f"\nSaved hardened model to: {output_dir}")

    if saved_weights_dir:
        os.makedirs(saved_weights_dir, exist_ok=True)
        dest1 = os.path.join(saved_weights_dir, "adversarial_hardened_final.pt")
        dest2 = os.path.join(saved_weights_dir, "hardened_transformer_final.pt")
        shutil.copy2(os.path.join(output_dir, "hardened_transformer_final.pt"), dest1)
        shutil.copy2(os.path.join(output_dir, "hardened_transformer_final.pt"), dest2)
        print(f"Copied final weights to: {dest1}")
