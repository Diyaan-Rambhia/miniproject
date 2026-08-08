"""
Pipeline 1: Core Detection System Orchestrator
Orchestrates: Transformer -> VAE -> DGA -> Signals CSV -> Fusion Head -> Explainability (SHAP+Attention) -> LLM Explanation
"""

import os
import sys
import joblib
import numpy as np
import torch

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Imports from phase modules
from models.transformer import config as trans_config
from models.transformer.data_loader import load_split_csvs as trans_load_csvs
from models.transformer.preprocessing import clean_data as trans_clean_data, encode_and_scale as trans_encode_scale, build_sequences as trans_build_seq
from models.transformer.dataset import prepare_dataloaders as trans_prepare_loaders
from models.transformer.model import FlowTransformerClassifier
from models.transformer.train import train_model as trans_train
from models.transformer.evaluate import full_evaluate as trans_evaluate
from models.transformer.save_artifacts import save_artifacts as trans_save

from models.VAE import config as vae_config
from models.VAE.data_loader import load_split_csvs as vae_load_csvs
from models.VAE.preprocessing import clean_and_filter_benign, encode_and_scale as vae_encode_scale
from models.VAE.dataset import prepare_vae_dataloaders
from models.VAE.model import VAE
from models.VAE.train import train_vae
from models.VAE.evaluate import compute_anomaly_scores, determine_threshold, evaluate_anomaly_detector
from models.VAE.save_artifacts import save_artifacts as vae_save

from models.DGA_detector import config as dga_config
from models.DGA_detector.data_loader import load_split_domain_csvs
from models.DGA_detector.preprocessing import build_vocab as build_char_vocab, encode_all_domains as encode_domains
from models.DGA_detector.dataset import prepare_dataloaders as dga_prepare_loaders
from models.DGA_detector.model import DGALSTMClassifier
from models.DGA_detector.train import train_model as dga_train
from models.DGA_detector.evaluate import full_evaluate as dga_evaluate
from models.DGA_detector.save_artifacts import save_artifacts as dga_save

from models.fusion_head import config as fusion_config
from models.fusion_head.generate_signals import generate_signals_csv
from models.fusion_head.data_loader import load_signals
from models.fusion_head.preprocessing import scale_signals
from models.fusion_head.dataset import prepare_dataloaders as fusion_prepare_loaders
from models.fusion_head.model import FusionMLP
from models.fusion_head.train import train_model as fusion_train
from models.fusion_head.evaluate import full_evaluate as fusion_evaluate, ablation_study
from models.fusion_head.save_artifacts import save_artifacts as fusion_save

import importlib.util

def load_explainability_module():
    exp_dir = os.path.abspath(os.path.join(BASE_DIR, "explainibility + SHAP"))
    
    # Load config
    spec_cfg = importlib.util.spec_from_file_location("exp_config", os.path.join(exp_dir, "config.py"))
    exp_config = importlib.util.module_from_spec(spec_cfg)
    spec_cfg.loader.exec_module(exp_config)
    sys.modules["config"] = exp_config
    
    # Load model
    spec_model = importlib.util.spec_from_file_location("exp_model", os.path.join(exp_dir, "model.py"))
    exp_model = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(exp_model)
    
    # Load explainer
    spec_exp = importlib.util.spec_from_file_location("exp_explainer", os.path.join(exp_dir, "explainer.py"))
    exp_explainer = importlib.util.module_from_spec(spec_exp)
    # inject dependencies into explainer module namespace
    spec_attn = importlib.util.spec_from_file_location("attention_explainer", os.path.join(exp_dir, "attention_explainer.py"))
    attn_mod = importlib.util.module_from_spec(spec_attn)
    spec_attn.loader.exec_module(attn_mod)
    sys.modules["attention_explainer"] = attn_mod
    
    spec_shap = importlib.util.spec_from_file_location("shap_explainer", os.path.join(exp_dir, "shap_explainer.py"))
    shap_mod = importlib.util.module_from_spec(spec_shap)
    spec_shap.loader.exec_module(shap_mod)
    sys.modules["shap_explainer"] = shap_mod
    
    spec_exp.loader.exec_module(exp_explainer)
    
    return exp_config, exp_model, exp_explainer

from models.llm_layer.llm_explanation import explain as generate_llm_explanation


def main():
    print("=" * 80)
    print("      CORE DETECTION PIPELINE — END-TO-END ORCHESTRATION")
    print("=" * 80)

    summary_results = {}

    # =========================================================================
    # STEP 1: Transformer Known-Attack Classifier
    # =========================================================================
    print("\n[Step 1/8] Checking Transformer Model...")
    trans_ckpt = os.path.join(trans_config.OUTPUT_DIR, "transformer_final.pt")
    
    # Load dataset for preprocessing & evaluation setup
    train_df, val_df, test_df = trans_load_csvs(trans_config.TRAIN_CSV, trans_config.VAL_CSV, trans_config.TEST_CSV)
    train_df = trans_clean_data(train_df)
    val_df   = trans_clean_data(val_df)
    test_df  = trans_clean_data(test_df)

    X_trans, y_trans, trans_label_enc, trans_scaler, trans_feat_names = trans_encode_scale(train_df)
    X_val  = trans_scaler.transform(val_df.drop(columns=["Label"]).select_dtypes(include=[np.number]))
    y_val  = trans_label_enc.transform(val_df["Label"])
    X_test = trans_scaler.transform(test_df.drop(columns=["Label"]).select_dtypes(include=[np.number]))
    y_test = trans_label_enc.transform(test_df["Label"])

    train_seqs, train_labels = trans_build_seq(X_trans, y_trans, trans_config.SEQ_LEN, trans_config.SEQ_STRIDE)
    val_seqs,   val_labels   = trans_build_seq(X_val,   y_val,   trans_config.SEQ_LEN, trans_config.SEQ_STRIDE)
    test_seqs,  test_labels  = trans_build_seq(X_test,  y_test,  trans_config.SEQ_LEN, trans_config.SEQ_STRIDE)

    trans_train_loader, trans_val_loader, trans_test_loader = trans_prepare_loaders(
        train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels, batch_size=trans_config.BATCH_SIZE
    )

    num_classes = len(trans_label_enc.classes_)
    num_features = X_trans.shape[1]

    transformer_model = FlowTransformerClassifier(
        num_features=num_features, num_classes=num_classes,
        d_model=trans_config.D_MODEL, nhead=trans_config.NHEAD,
        num_layers=trans_config.NUM_LAYERS, dim_feedforward=trans_config.DIM_FEEDFORWARD,
        dropout=trans_config.DROPOUT,
    ).to(trans_config.DEVICE)

    if os.path.exists(trans_ckpt):
        print(f"-> Transformer checkpoint exists ({trans_ckpt}). Skipping training.")
        transformer_model.load_state_dict(torch.load(trans_ckpt, map_location=trans_config.DEVICE))
    else:
        print("-> Training Transformer model from scratch...")
        transformer_model = trans_train(
            transformer_model, trans_train_loader, trans_test_loader,
            trans_config.NUM_EPOCHS, trans_config.LEARNING_RATE, trans_config.OUTPUT_DIR, trans_config.DEVICE
        )
        trans_save(transformer_model, trans_label_enc, trans_scaler, trans_feat_names, trans_config.OUTPUT_DIR, trans_config.SAVED_WEIGHTS_DIR)

    trans_metrics = trans_evaluate(transformer_model, trans_test_loader, trans_label_enc, trans_config.DEVICE)
    summary_results["Transformer Classifier"] = trans_metrics

    # =========================================================================
    # STEP 2: VAE Unknown/Zero-Day Detector
    # =========================================================================
    print("\n[Step 2/8] Checking VAE Anomaly Detector...")
    vae_ckpt = os.path.join(vae_config.OUTPUT_DIR, "vae_final.pt")
    
    train_vae_df, val_vae_df, test_vae_df = vae_load_csvs(vae_config.TRAIN_CSV, vae_config.VAL_CSV, vae_config.TEST_CSV)
    train_benign = clean_and_filter_benign(train_vae_df)
    val_benign   = clean_and_filter_benign(val_vae_df)
    X_benign_train = trans_scaler.transform(train_benign.drop(columns=["Label"]).select_dtypes(include=[np.number]))
    eval_benign    = trans_scaler.transform(val_benign.drop(columns=["Label"]).select_dtypes(include=[np.number]))
    
    clean_test_df = trans_clean_data(test_vae_df)
    attack_test_df = clean_test_df[clean_test_df["Label"] != "BENIGN"]
    eval_attack   = trans_scaler.transform(attack_test_df.drop(columns=["Label"]).select_dtypes(include=[np.number]))

    vae_train_loader, vae_val_loader, _, _ = prepare_vae_dataloaders(
        X_benign_train, batch_size=vae_config.BATCH_SIZE, random_state=vae_config.RANDOM_STATE
    )

    vae_model = VAE(num_features, vae_config.HIDDEN_DIM, vae_config.LATENT_DIM).to(vae_config.DEVICE)

    if os.path.exists(vae_ckpt):
        print(f"-> VAE checkpoint exists ({vae_ckpt}). Skipping training.")
        vae_model.load_state_dict(torch.load(vae_ckpt, map_location=vae_config.DEVICE))
        vae_thresh = joblib.load(os.path.join(vae_config.OUTPUT_DIR, "anomaly_threshold.joblib"))
    else:
        print("-> Training VAE model from scratch...")
        vae_model = train_vae(
            vae_model, vae_train_loader, vae_val_loader,
            vae_config.NUM_EPOCHS, vae_config.LEARNING_RATE, vae_config.KL_WEIGHT,
            vae_config.OUTPUT_DIR, vae_config.DEVICE
        )
        benign_val_scores = compute_anomaly_scores(vae_model, eval_benign, vae_config.KL_WEIGHT, vae_config.DEVICE)
        vae_thresh = determine_threshold(benign_val_scores)
        vae_save(vae_model, trans_scaler, vae_thresh, trans_feat_names, vae_config.OUTPUT_DIR, vae_config.SAVED_WEIGHTS_DIR)

    benign_eval_scores = compute_anomaly_scores(vae_model, eval_benign, vae_config.KL_WEIGHT, vae_config.DEVICE)
    attack_eval_scores = compute_anomaly_scores(vae_model, eval_attack, vae_config.KL_WEIGHT, vae_config.DEVICE)
    vae_metrics = evaluate_anomaly_detector(benign_eval_scores, attack_eval_scores, vae_thresh)
    summary_results["VAE Anomaly Detector"] = vae_metrics

    # =========================================================================
    # STEP 3: DGA Domain Detector
    # =========================================================================
    print("\n[Step 3/8] Checking DGA Domain Detector...")
    dga_ckpt = os.path.join(dga_config.OUTPUT_DIR, "dga_lstm_final.pt")

    train_dga_df, val_dga_df, test_dga_df = load_split_domain_csvs(dga_config.TRAIN_CSV, dga_config.VAL_CSV, dga_config.TEST_CSV)
    vocab = build_char_vocab(train_dga_df["domain"].astype(str).tolist())
    X_dga_train = encode_domains(train_dga_df["domain"].astype(str).tolist(), vocab, dga_config.MAX_LEN)
    y_dga_train = train_dga_df["label"].values
    X_dga_val   = encode_domains(val_dga_df["domain"].astype(str).tolist(), vocab, dga_config.MAX_LEN)
    y_dga_val   = val_dga_df["label"].values
    X_dga_test  = encode_domains(test_dga_df["domain"].astype(str).tolist(), vocab, dga_config.MAX_LEN)
    y_dga_test  = test_dga_df["label"].values

    dga_train_loader, dga_val_loader, dga_test_loader = dga_prepare_loaders(
        X_dga_train, y_dga_train, X_dga_val, y_dga_val, X_dga_test, y_dga_test, batch_size=dga_config.BATCH_SIZE
    )

    dga_model = DGALSTMClassifier(
        len(vocab), dga_config.EMBED_DIM, dga_config.HIDDEN_DIM,
        dga_config.NUM_LSTM_LAYERS, dga_config.DROPOUT
    ).to(dga_config.DEVICE)

    if os.path.exists(dga_ckpt):
        print(f"-> DGA detector checkpoint exists ({dga_ckpt}). Skipping training.")
        dga_model.load_state_dict(torch.load(dga_ckpt, map_location=dga_config.DEVICE))
    else:
        print("-> Training DGA detector from scratch...")
        dga_model = dga_train(
            dga_model, dga_train_loader, dga_test_loader,
            dga_config.NUM_EPOCHS, dga_config.LEARNING_RATE, dga_config.OUTPUT_DIR, dga_config.DEVICE
        )
        dga_save(dga_model, vocab, dga_config.MAX_LEN, dga_config.OUTPUT_DIR, dga_config.SAVED_WEIGHTS_DIR)

    dga_metrics = dga_evaluate(dga_model, dga_test_loader, dga_config.DEVICE)
    summary_results["DGA Domain Detector"] = dga_metrics

    # =========================================================================
    # STEP 4: Generate Signals CSV
    # =========================================================================
    print("\n[Step 4/8] Generating Signal CSV for Fusion Layer...")
    signals_csv_path = fusion_config.SIGNALS_CSV_PATH
    if os.path.exists(signals_csv_path):
        print(f"-> Signals CSV already exists ({signals_csv_path}). Using cached file.")
    else:
        generate_signals_csv(transformer_model, vae_model, dga_model, signals_csv_path)

    # =========================================================================
    # STEP 5: Fusion Head Pipeline
    # =========================================================================
    print("\n[Step 5/8] Training & Evaluating Fusion Scoring Layer...")
    fusion_ckpt = os.path.join(fusion_config.OUTPUT_DIR, "fusion_mlp_final.pt")

    df_signals = load_signals(signals_csv_path, fusion_config.SIGNAL_COLUMNS)
    X_fusion, y_fusion, fusion_scaler = scale_signals(df_signals, fusion_config.SIGNAL_COLUMNS)
    fusion_train_loader, fusion_test_loader, X_tr_f, X_te_f, y_tr_f, y_te_f = fusion_prepare_loaders(
        X_fusion, y_fusion, fusion_config.TEST_SIZE, fusion_config.RANDOM_STATE, fusion_config.BATCH_SIZE
    )

    fusion_model = FusionMLP(num_inputs=len(fusion_config.SIGNAL_COLUMNS), hidden_dim=fusion_config.HIDDEN_DIM).to(fusion_config.DEVICE)

    if os.path.exists(fusion_ckpt):
        print(f"-> Fusion head checkpoint exists ({fusion_ckpt}). Skipping training.")
        fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=fusion_config.DEVICE))
    else:
        print("-> Training Fusion MLP head from scratch...")
        fusion_model = fusion_train(
            fusion_model, fusion_train_loader, fusion_test_loader,
            fusion_config.NUM_EPOCHS, fusion_config.LEARNING_RATE, fusion_config.OUTPUT_DIR, fusion_config.DEVICE
        )
        fusion_save(fusion_model, fusion_scaler, fusion_config.OUTPUT_DIR, fusion_config.SAVED_WEIGHTS_DIR)

    fusion_metrics = fusion_evaluate(fusion_model, X_te_f, y_te_f, fusion_config.DEVICE)
    ablation_results = ablation_study(X_tr_f, y_tr_f, X_te_f, y_te_f, fusion_config.SIGNAL_COLUMNS)
    summary_results["Fusion Head"] = fusion_metrics
    summary_results["Ablation Study"] = ablation_results

    # =========================================================================
    # STEP 6: Explainability Layer (SHAP + Attention)
    # =========================================================================
    print("\n[Step 6/8] Wiring Dual-Channel Explainability Layer...")
    best_trans_ckpt = os.path.join(trans_config.OUTPUT_DIR, "checkpoint_best.pt")
    if not os.path.exists(best_trans_ckpt):
        best_trans_ckpt = trans_ckpt

    exp_config, exp_model, exp_explainer = load_explainability_module()

    explainable_model = exp_model.load_explainable_model_from_checkpoint(
        best_trans_ckpt, num_features, num_classes,
        trans_config.D_MODEL, trans_config.NHEAD, trans_config.NUM_LAYERS,
        trans_config.DIM_FEEDFORWARD, trans_config.DROPOUT, exp_config.DEVICE
    )

    sample_idx = 0
    sample_seq = test_seqs[sample_idx]
    bg_seqs = test_seqs[:20]

    raw_explanation = exp_explainer.explain_flagged_event(
        explainable_model, sample_seq, bg_seqs, list(trans_feat_names), exp_config.DEVICE, run_shap=exp_config.SHAP_AVAILABLE
    )

    # Compute Threat Score for the worked example using Fusion model
    sample_signal = X_te_f[0:1]
    sample_tensor = torch.tensor(sample_signal, dtype=torch.float32).to(fusion_config.DEVICE)
    worked_threat_score = float(fusion_model.threat_score(sample_tensor).cpu().numpy()[0])

    # =========================================================================
    # STEP 7: LLM Explanation Layer
    # =========================================================================
    print("\n[Step 7/8] Generating Plain-English Explanation via LLM Layer...")
    plain_english_text = generate_llm_explanation(raw_explanation, list(trans_label_enc.classes_))

    # =========================================================================
    # STEP 8: Final Summary & Worked Example Output
    # =========================================================================
    print("\n" + "=" * 80)
    print("                     FULL PIPELINE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"1. Transformer Classifier -> Accuracy: {trans_metrics['accuracy']:.4f} | Macro F1: {trans_metrics['macro_f1']:.4f}")
    print(f"2. VAE Anomaly Detector  -> ROC-AUC:  {vae_metrics['auc']:.4f} | F1: {vae_metrics['f1']:.4f}")
    print(f"3. DGA Domain Detector   -> Accuracy: {dga_metrics['accuracy']:.4f} | F1: {dga_metrics['f1']:.4f}")
    print(f"4. Fusion MLP Head       -> Accuracy: {fusion_metrics['accuracy']:.4f} | ROC-AUC: {fusion_metrics['auc']:.4f} | F1: {fusion_metrics['f1']:.4f}")

    print("\nAblation Study Table (Signal Alone vs Fused):")
    print("-" * 60)
    print(f"{'Signal Component':32s} | {'F1 Score':10s} | {'ROC-AUC':10s}")
    print("-" * 60)
    for sig_name, res in ablation_results.items():
        print(f"{sig_name:32s} | {res['f1']:10.4f} | {res['auc']:10.4f}")
    print(f"{'Fused MLP Model (All 3 Signals)':32s} | {fusion_metrics['f1']:10.4f} | {fusion_metrics['auc']:10.4f}")
    print("-" * 60)

    print("\nWORKED EXAMPLE OF A FLAGGED EVENT EXPLANATION:")
    print("=" * 80)
    print(plain_english_text)
    print("=" * 80)
    print("\nCore Detection Pipeline Orchestration Completed Successfully!")


if __name__ == "__main__":
    main()
