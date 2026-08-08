"""
Phase: Saving Artifacts
Expects: Trained model, character vocabulary dict, max_len integer, and output directory path
Outputs: Saved artifact files on disk (dga_lstm_final.pt, char_vocab.joblib, max_len.joblib)
"""

import os
import shutil
import joblib
import torch


def save_artifacts(model, vocab, max_len, output_dir, saved_weights_dir=None):
    torch.save(model.state_dict(), os.path.join(output_dir, "dga_lstm_final.pt"))
    joblib.dump(vocab, os.path.join(output_dir, "char_vocab.joblib"))
    joblib.dump(max_len, os.path.join(output_dir, "max_len.joblib"))
    print(f"\nSaved all artifacts to: {output_dir}")

    if saved_weights_dir:
        os.makedirs(saved_weights_dir, exist_ok=True)
        dest1 = os.path.join(saved_weights_dir, "dga_detector_final.pt")
        dest2 = os.path.join(saved_weights_dir, "dga_lstm_final.pt")
        shutil.copy2(os.path.join(output_dir, "dga_lstm_final.pt"), dest1)
        shutil.copy2(os.path.join(output_dir, "dga_lstm_final.pt"), dest2)
        print(f"Copied final weights to: {dest1}")
