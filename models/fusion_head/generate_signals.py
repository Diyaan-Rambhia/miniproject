"""
Phase: Signals CSV Generation
Expects: Trained Transformer, VAE, and DGA models
Outputs: Saved signals.csv containing upstream model confidence/anomaly scores and ground-truth binary label
"""

import os
import numpy as np
import pandas as pd
import torch

from models.transformer import config as trans_config
from models.transformer.preprocessing import clean_data, encode_and_scale, build_sequences
from models.VAE import config as vae_config
from models.VAE.evaluate import compute_anomaly_scores
from models.DGA_detector import config as dga_config
from models.DGA_detector.preprocessing import build_char_vocab, encode_domains


def generate_signals_csv(transformer_model, vae_model, dga_model, output_path: str = None):
    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "signals.csv"))

    print(f"\n--- Generating Fusion Signals CSV ---")
    device = trans_config.DEVICE

    # 1. Load and process test flow dataset for Transformer and VAE
    df_flows = pd.read_csv(trans_config.TEST_CSV, low_memory=False)
    df_flows = clean_data(df_flows)
    X_scaled, y_flows, label_encoder, scaler, feature_names = encode_and_scale(df_flows)
    sequences, seq_labels = build_sequences(X_scaled, y_flows, trans_config.SEQ_LEN, trans_config.SEQ_STRIDE)

    # 2. Extract Transformer signals
    transformer_model.eval()
    transformer_model.to(device)
    transformer_probs = []

    # Identify BENIGN class index
    benign_idx = 0
    if "BENIGN" in label_encoder.classes_:
        benign_idx = int(np.where(label_encoder.classes_ == "BENIGN")[0][0])

    batch_size = trans_config.BATCH_SIZE
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_x = torch.tensor(sequences[start:start + batch_size], dtype=torch.float32).to(device)
            logits = transformer_model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            # Confidence in attack = 1.0 - P(BENIGN)
            attack_prob = 1.0 - probs[:, benign_idx]
            transformer_probs.extend(attack_prob)

    transformer_probs = np.array(transformer_probs)

    # 3. Extract VAE anomaly scores on the last flow of each sequence
    vae_model.eval()
    vae_model.to(device)
    last_flows = sequences[:, -1, :]
    vae_scores = compute_anomaly_scores(vae_model, last_flows, vae_config.KL_WEIGHT, device)

    # 4. Extract DGA probabilities
    df_dga = pd.read_csv(dga_config.TEST_CSV)
    vocab = build_char_vocab(df_dga["domain"].astype(str).tolist())
    X_dga, y_dga = encode_domains(df_dga["domain"].astype(str).tolist(), df_dga["label"].values, vocab, dga_config.MAX_LEN)

    dga_model.eval()
    dga_model.to(device)
    dga_probs_list = []
    with torch.no_grad():
        for start in range(0, len(X_dga), dga_config.BATCH_SIZE):
            batch_x = torch.tensor(X_dga[start:start + dga_config.BATCH_SIZE], dtype=torch.long).to(device)
            logits = dga_model(batch_x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            dga_probs_list.extend(probs)

    dga_probs_array = np.array(dga_probs_list)

    # Map / sample DGA probabilities to match sequence length
    if len(dga_probs_array) >= len(sequences):
        dga_probs = dga_probs_array[:len(sequences)]
    else:
        # Repeat/tile if fewer domain samples
        dga_probs = np.resize(dga_probs_array, len(sequences))

    # 5. Determine ground truth binary label (1 = Attack, 0 = Benign)
    binary_labels = (seq_labels != benign_idx).astype(int)

    signals_df = pd.DataFrame({
        "transformer_confidence": transformer_probs,
        "vae_anomaly_score": vae_scores,
        "dga_probability": dga_probs,
        "label": binary_labels,
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    signals_df.to_csv(output_path, index=False)
    print(f"Successfully generated and saved signals CSV ({len(signals_df):,} rows) to: {output_path}")

    return output_path
