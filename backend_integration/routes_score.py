"""
Scoring API Endpoint Router (POST /score)
"""

import json
import uuid
import sqlite3
from datetime import datetime
import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException

from backend_integration import config
from backend_integration.db import get_db
from backend_integration.schemas import ScoreRequest, ScoreResponse
from backend_integration.model_loader import models_container
from models.VAE.model import vae_loss

router = APIRouter()


@router.post("/score", response_model=ScoreResponse)
def score_event(request: ScoreRequest, db: sqlite3.Connection = Depends(get_db)):
    if not models_container.is_loaded:
        models_container.load_all()

    device = config.DEVICE
    seq = np.array(request.flow_sequence, dtype=np.float32)

    if seq.ndim != 2:
        raise HTTPException(status_code=400, detail="flow_sequence must be a 2D matrix of shape (seq_len, num_features)")

    # 1. Transformer Classifier Signal
    transformer_confidence = 0.0
    pred_class_str = "BENIGN"
    if models_container.transformer_model is not None:
        seq_tensor = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = models_container.transformer_model(seq_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class_idx = int(np.argmax(probs))

        if models_container.trans_label_encoder is not None:
            pred_class_str = str(models_container.trans_label_encoder.classes_[pred_class_idx])
            benign_idx = 0
            if "BENIGN" in models_container.trans_label_encoder.classes_:
                benign_idx = int(np.where(models_container.trans_label_encoder.classes_ == "BENIGN")[0][0])
            transformer_confidence = float(1.0 - probs[benign_idx])
        else:
            transformer_confidence = float(probs[1]) if len(probs) > 1 else float(probs[0])

    # 2. VAE Anomaly Score Signal
    vae_score = 0.0
    if models_container.vae_model is not None:
        last_flow = seq[-1, :]
        flow_tensor = torch.tensor(last_flow[np.newaxis, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            recon, mu, logvar = models_container.vae_model(flow_tensor)
            _, recon_loss, kl_div = vae_loss(recon, flow_tensor, mu, logvar, kl_weight=0.5)
            per_sample = (recon_loss + 0.5 * kl_div).cpu().numpy()
            vae_score = float(per_sample[0])

    # 3. DGA Probability Signal
    dga_prob = 0.0
    if request.domain and models_container.dga_model is not None and models_container.dga_vocab:
        domain_str = request.domain.lower()
        max_len = models_container.dga_max_len
        vocab = models_container.dga_vocab

        encoded = [vocab.get(c, 0) for c in domain_str[:max_len]]
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        domain_tensor = torch.tensor([encoded], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = models_container.dga_model(domain_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            dga_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])

    # 4. Fusion Scoring Head -> Threat Score
    signals_raw = np.array([[transformer_confidence, vae_score, dga_prob]], dtype=np.float32)

    if models_container.fusion_model is not None and models_container.fusion_scaler is not None:
        signals_scaled = models_container.fusion_scaler.transform(signals_raw)
        sig_tensor = torch.tensor(signals_scaled, dtype=torch.float32).to(device)
        threat_score = float(models_container.fusion_model.threat_score(sig_tensor).cpu().numpy()[0])
    else:
        threat_score = float(np.clip((transformer_confidence * 0.5 + min(vae_score, 2.0) * 0.25 + dga_prob * 0.25) * 100.0, 0, 100))

    # Save to Database
    event_id = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO events (event_id, timestamp, threat_score, transformer_confidence, vae_anomaly_score, dga_probability, predicted_class, raw_sequence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (event_id, timestamp, threat_score, transformer_confidence, vae_score, dga_prob, pred_class_str, json.dumps(request.flow_sequence)))
    db.commit()

    return ScoreResponse(
        event_id=event_id,
        timestamp=timestamp,
        threat_score=threat_score,
        predicted_class=pred_class_str,
        transformer_confidence=transformer_confidence,
        vae_anomaly_score=vae_score,
        dga_probability=dga_prob,
    )
