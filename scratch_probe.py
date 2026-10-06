import os, joblib, numpy as np, pandas as pd, torch
from models.transformer import config as trans_cfg
from models.transformer.preprocessing import clean_data, build_sequences
from models.VAE.model import VAE, vae_loss
from models.transformer.model import FlowTransformerClassifier

label_encoder = joblib.load(os.path.join(trans_cfg.OUTPUT_DIR, 'label_encoder.joblib'))
scaler = joblib.load(os.path.join(trans_cfg.OUTPUT_DIR, 'feature_scaler.joblib'))
feature_names = joblib.load(os.path.join(trans_cfg.OUTPUT_DIR, 'feature_names.joblib'))

df = pd.read_csv(trans_cfg.TEST_CSV, low_memory=False)
df = clean_data(df)
X = df.reindex(columns=feature_names)
X_scaled = scaler.transform(X)
label_names = list(label_encoder.classes_)

raw_labels = df['Label'].astype(str).to_numpy()
seq_true_names = []
for start in range(0, len(raw_labels)-trans_cfg.SEQ_LEN+1, trans_cfg.SEQ_STRIDE):
    end = start + trans_cfg.SEQ_LEN
    seq_true_names.append(raw_labels[end-1])
seq_true_names = np.array(seq_true_names)
seqs, _ = build_sequences(X_scaled, (df['Label'].to_numpy() != 'BENIGN').astype(np.int64), trans_cfg.SEQ_LEN, trans_cfg.SEQ_STRIDE)

model = FlowTransformerClassifier(num_features=X_scaled.shape[1], num_classes=len(label_names), d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1)
model.load_state_dict(torch.load(os.path.join(trans_cfg.OUTPUT_DIR, 'transformer_final.pt'), map_location='cpu'))
model.eval()
benign_idx = np.where(np.array(label_names) == 'BENIGN')[0][0]
transformer_probs = []
with torch.no_grad():
    for i in range(0, len(seqs), 256):
        batch = torch.tensor(seqs[i:i+256], dtype=torch.float32)
        logits = model(batch)
        transformer_probs.extend(torch.softmax(logits, dim=1).numpy())
transformer_probs = np.array(transformer_probs)

vae_model = VAE(num_features=X_scaled.shape[1], hidden_dim=64, latent_dim=16)
vae_model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(trans_cfg.OUTPUT_DIR)), 'VAE', 'outputs', 'vae_final.pt'), map_location='cpu'))
vae_model.eval()
vae_scores = []
for seq in seqs:
    x = torch.tensor(seq[-1:].copy(), dtype=torch.float32)
    with torch.no_grad():
        recon, mu, logvar = vae_model(x)
        _, recon_loss, kl_div = vae_loss(recon, x, mu, logvar, kl_weight=0.5)
        score = (recon_loss + 0.5 * kl_div).numpy()[0]
    vae_scores.append(float(score))
vae_scores = np.array(vae_scores)

rows = []
for i, true_name in enumerate(seq_true_names):
    p = transformer_probs[i]
    pred_idx = int(np.argmax(p))
    pred_name = label_names[pred_idx]
    if true_name in {'DoS Hulk', 'PortScan'} and pred_name == 'BENIGN':
        raw_attack_conf = float(1.0 - p[benign_idx])
        rows.append({
            'seq_index': i,
            'true_name': true_name,
            'pred_name': pred_name,
            'transformer_attack_confidence': raw_attack_conf,
            'transformer_best_prob': float(p[pred_idx]),
            'vae_score': float(vae_scores[i]),
            'heuristic_threat': float(np.clip(0.5 * raw_attack_conf + 0.25 * min(float(vae_scores[i]), 2.0), 0, 1) * 100.0),
        })

print('misclassified_total', len(rows))
for row in rows[:20]:
    print(row)

for target in ['DoS Hulk', 'PortScan']:
    subset = []
    for i, true_name in enumerate(seq_true_names):
        if true_name == target:
            p = transformer_probs[i]
            pred_idx = int(np.argmax(p))
            pred_name = label_names[pred_idx]
            subset.append((float(1.0 - p[benign_idx]), float(vae_scores[i]), float(p[pred_idx]), pred_name, i, true_name))
    subset = sorted(subset, reverse=True)
    print('TARGET', target)
    for item in subset[:5]:
        print(item)
