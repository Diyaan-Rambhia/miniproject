"""
One-Command End-to-End Training Orchestrator

Runs all 5 model training pipelines in dependency order:
1. Transformer Classifier (Supervised Flow Detection)
2. VAE Anomaly Detector (Unsupervised Zero-Day Detection)
3. DGA Domain Detector (Char-level LSTM)
4. Signals CSV Generation + Fusion MLP Head
5. Adversarial Robustness Study (FGSM/PGD Defense)

Skips any model whose trained weights already exist in models/saved_weights/.
Prints timing and a final evaluation summary table.
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SAVED_WEIGHTS_DIR = os.path.join(BASE_DIR, "saved_weights")
os.makedirs(SAVED_WEIGHTS_DIR, exist_ok=True)

# Weight checkpoints to check for skipping
WEIGHT_FILES = {
    "Transformer": os.path.join(SAVED_WEIGHTS_DIR, "transformer_final.pt"),
    "VAE": os.path.join(SAVED_WEIGHTS_DIR, "vae_final.pt"),
    "DGA Detector": os.path.join(SAVED_WEIGHTS_DIR, "dga_detector_final.pt"),
    "Fusion Head": os.path.join(SAVED_WEIGHTS_DIR, "fusion_head_final.pt"),
    "Adversarial Hardened": os.path.join(SAVED_WEIGHTS_DIR, "adversarial_hardened_final.pt"),
}


def main():
    print("=" * 80)
    print("         NETWORK SECURITY PIPELINE — END-TO-END TRAINING ORCHESTRATOR")
    print("=" * 80)

    start_all = time.time()
    timings = {}

    # -------------------------------------------------------------------------
    # 1. Transformer Classifier
    # -------------------------------------------------------------------------
    print("\n[1/5] Transformer Classifier (Supervised Flow Detection)...")
    if os.path.exists(WEIGHT_FILES["Transformer"]):
        print(f"  -> Found existing weights at {WEIGHT_FILES['Transformer']}. Skipping training.")
        timings["Transformer Classifier"] = "Skipped (Cached)"
    else:
        t0 = time.time()
        from models.transformer.main import main as train_transformer
        train_transformer()
        t_elapsed = time.time() - t0
        timings["Transformer Classifier"] = f"done in {t_elapsed:.1f}s"
        print(f"  -> Training Transformer... done in {t_elapsed:.1f}s")

    # -------------------------------------------------------------------------
    # 2. VAE Anomaly Detector
    # -------------------------------------------------------------------------
    print("\n[2/5] VAE Anomaly Detector (Unsupervised Zero-Day Detection)...")
    if os.path.exists(WEIGHT_FILES["VAE"]):
        print(f"  -> Found existing weights at {WEIGHT_FILES['VAE']}. Skipping training.")
        timings["VAE Anomaly Detector"] = "Skipped (Cached)"
    else:
        t0 = time.time()
        from models.VAE.main import main as train_vae
        train_vae()
        t_elapsed = time.time() - t0
        timings["VAE Anomaly Detector"] = f"done in {t_elapsed:.1f}s"
        print(f"  -> Training VAE... done in {t_elapsed:.1f}s")

    # -------------------------------------------------------------------------
    # 3. DGA Domain Detector
    # -------------------------------------------------------------------------
    print("\n[3/5] DGA Domain Detector (Char-level LSTM)...")
    if os.path.exists(WEIGHT_FILES["DGA Detector"]):
        print(f"  -> Found existing weights at {WEIGHT_FILES['DGA Detector']}. Skipping training.")
        timings["DGA Domain Detector"] = "Skipped (Cached)"
    else:
        t0 = time.time()
        from models.DGA_detector.main import main as train_dga
        train_dga()
        t_elapsed = time.time() - t0
        timings["DGA Domain Detector"] = f"done in {t_elapsed:.1f}s"
        print(f"  -> Training DGA Detector... done in {t_elapsed:.1f}s")

    # -------------------------------------------------------------------------
    # 4. Signals CSV Generation + Fusion Head
    # -------------------------------------------------------------------------
    print("\n[4/5] Signals CSV Generation & Fusion Scoring Layer...")
    signals_csv = os.path.join(BASE_DIR, "data", "signals.csv")
    if not os.path.exists(signals_csv):
        print("  -> Generating signals.csv from upstream model outputs...")
        t0 = time.time()
        from models.transformer import config as trans_config, model as trans_mod
        from models.VAE import config as vae_config, model as vae_mod
        from models.DGA_detector import config as dga_config, model as dga_mod
        from models.fusion_head.generate_signals import generate_signals_csv

        # Load models to pass to generator
        # Load transformer
        t_model = trans_mod.FlowTransformerClassifier(
            num_features=78, num_classes=15,
            d_model=trans_config.D_MODEL, nhead=trans_config.NHEAD,
            num_layers=trans_config.NUM_LAYERS, dim_feedforward=trans_config.DIM_FEEDFORWARD,
            dropout=trans_config.DROPOUT
        ).to(trans_config.DEVICE)
        if os.path.exists(WEIGHT_FILES["Transformer"]):
            import torch
            t_model.load_state_dict(torch.load(WEIGHT_FILES["Transformer"], map_location=trans_config.DEVICE))

        v_model = vae_mod.VAE(78, vae_config.HIDDEN_DIM, vae_config.LATENT_DIM).to(vae_config.DEVICE)
        if os.path.exists(WEIGHT_FILES["VAE"]):
            import torch
            v_model.load_state_dict(torch.load(WEIGHT_FILES["VAE"], map_location=vae_config.DEVICE))

        d_model = dga_mod.DGALSTMClassifier(
            100, dga_config.EMBED_DIM, dga_config.HIDDEN_DIM,
            dga_config.NUM_LSTM_LAYERS, dga_config.DROPOUT
        ).to(dga_config.DEVICE)
        if os.path.exists(WEIGHT_FILES["DGA Detector"]):
            import torch
            d_model.load_state_dict(torch.load(WEIGHT_FILES["DGA Detector"], map_location=dga_config.DEVICE))

        generate_signals_csv(t_model, v_model, d_model, signals_csv)
        print(f"  -> Generated signals.csv in {time.time() - t0:.1f}s")

    if os.path.exists(WEIGHT_FILES["Fusion Head"]):
        print(f"  -> Found existing weights at {WEIGHT_FILES['Fusion Head']}. Skipping training.")
        timings["Fusion Head"] = "Skipped (Cached)"
    else:
        t0 = time.time()
        from models.fusion_head.main import main as train_fusion
        train_fusion()
        t_elapsed = time.time() - t0
        timings["Fusion Head"] = f"done in {t_elapsed:.1f}s"
        print(f"  -> Training Fusion Head... done in {t_elapsed:.1f}s")

    # -------------------------------------------------------------------------
    # 5. Adversarial Robustness Study
    # -------------------------------------------------------------------------
    print("\n[5/5] Adversarial Robustness Study (FGSM/PGD Defense)...")
    if os.path.exists(WEIGHT_FILES["Adversarial Hardened"]):
        print(f"  -> Found existing weights at {WEIGHT_FILES['Adversarial Hardened']}. Skipping training.")
        timings["Adversarial Robustness"] = "Skipped (Cached)"
    else:
        t0 = time.time()
        from models.Adverserial_robustness.main import main as train_adversarial
        train_adversarial()
        t_elapsed = time.time() - t0
        timings["Adversarial Robustness"] = f"done in {t_elapsed:.1f}s"
        print(f"  -> Training Adversarial Hardened model... done in {t_elapsed:.1f}s")

    total_time = time.time() - start_all

    # -------------------------------------------------------------------------
    # FINAL SUMMARY TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                END-TO-END TRAINING ORCHESTRATION SUMMARY")
    print("=" * 80)
    print(f"{'Component':35s} | {'Training Status / Timing':35s}")
    print("-" * 80)
    for component, status in timings.items():
        print(f"{component:35s} | {status:35s}")
    print("=" * 80)
    print(f"All model weights saved centrally to: {SAVED_WEIGHTS_DIR}")
    print(f"Total orchestration execution time: {total_time:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
