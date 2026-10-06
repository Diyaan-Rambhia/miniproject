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
from backend_integration.inference_preprocess import scale_flow_sequence
from models.VAE.model import vae_loss

router = APIRouter()


def _fusion_predicted_label(threat_score: float, transformer_attack_type: str) -> str:
    """Final BENIGN/attack decision uses fused threat_score only (not Transformer argmax)."""
    if threat_score >= config.FUSION_ATTACK_THRESHOLD:
        if transformer_attack_type and transformer_attack_type != "BENIGN":
            return transformer_attack_type
        return "ATTACK"
    return "BENIGN"


@router.post("/score", response_model=ScoreResponse)
def score_event(request: ScoreRequest, db: sqlite3.Connection = Depends(get_db)):
    if not models_container.is_loaded:
        models_container.load_all()

    device = config.DEVICE
    raw_seq = np.array(request.flow_sequence, dtype=np.float32)

    if raw_seq.ndim != 2:
        raise HTTPException(status_code=400, detail="flow_sequence must be a 2D matrix of shape (seq_len, num_features)")

    expected_features = (
        len(models_container.trans_feature_names)
        if models_container.trans_feature_names is not None
        else None
    )
    try:
        transformer_seq = scale_flow_sequence(raw_seq, models_container.trans_scaler, expected_features)
        vae_seq = (
            scale_flow_sequence(raw_seq, models_container.vae_scaler, expected_features)
            if models_container.vae_model is not None and models_container.vae_scaler is not None
            else None
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    available_signals = {
        "transformer": models_container.transformer_model is not None,
        "vae": models_container.vae_model is not None and models_container.vae_scaler is not None,
        "dga": bool(request.domain and models_container.dga_model is not None and models_container.dga_vocab),
    }
    if not any(available_signals.values()):
        return ScoreResponse(
            status="insufficient_models",
            message="Insufficient models to score: no usable Transformer, VAE, or DGA model is available.",
        )

    # 1. Transformer Classifier Signal (attack type + non-benign confidence)
    transformer_confidence = 0.0
    transformer_predicted_class = "BENIGN"
    if models_container.transformer_model is not None:
        seq_tensor = torch.tensor(transformer_seq[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = models_container.transformer_model(seq_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class_idx = int(np.argmax(probs))

        if models_container.trans_label_encoder is not None:
            transformer_predicted_class = str(models_container.trans_label_encoder.classes_[pred_class_idx])
            benign_idx = 0
            if "BENIGN" in models_container.trans_label_encoder.classes_:
                benign_idx = int(np.where(models_container.trans_label_encoder.classes_ == "BENIGN")[0][0])
            transformer_confidence = float(1.0 - probs[benign_idx])
        else:
            transformer_predicted_class = "ATTACK" if pred_class_idx != 0 else "BENIGN"
            transformer_confidence = float(probs[1]) if len(probs) > 1 else float(probs[0])

    # 2. VAE Anomaly Score Signal, scaled with the VAE's own training scaler
    vae_score = 0.0
    if models_container.vae_model is not None and vae_seq is not None:
        last_flow = vae_seq[-1, :]
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

    # 4. Fusion Scoring Head -> Threat Score (authoritative malicious/benign decision)
    signals_raw = np.array([[transformer_confidence, vae_score, dga_prob]], dtype=np.float32)

    if all(available_signals.values()) and models_container.fusion_model is not None and models_container.fusion_scaler is not None:
        signals_scaled = models_container.fusion_scaler.transform(signals_raw)
        sig_tensor = torch.tensor(signals_scaled, dtype=torch.float32).to(device)
        threat_score = float(models_container.fusion_model.threat_score(sig_tensor).cpu().numpy()[0])
    else:
        weights = {"transformer": 0.5, "vae": 0.25, "dga": 0.25}
        values = {
            "transformer": transformer_confidence,
            "vae": min(vae_score, 2.0),
            "dga": dga_prob,
        }
        active_weight = sum(weights[name] for name, active in available_signals.items() if active)
        weighted_score = sum(
            weights[name] * values[name]
            for name, active in available_signals.items()
            if active
        ) / active_weight
        threat_score = float(np.clip(weighted_score * 100.0, 0, 100))

    predicted_class = _fusion_predicted_label(threat_score, transformer_predicted_class)

    event_id = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO events (
            event_id, timestamp, threat_score, transformer_confidence, vae_anomaly_score,
            dga_probability, predicted_class, transformer_predicted_class, raw_sequence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event_id, timestamp, threat_score, transformer_confidence, vae_score, dga_prob,
        predicted_class, transformer_predicted_class, json.dumps(request.flow_sequence),
    ))
    db.commit()

    return ScoreResponse(
        event_id=event_id,
        timestamp=timestamp,
        threat_score=threat_score,
        predicted_class=predicted_class,
        transformer_predicted_class=transformer_predicted_class,
        transformer_confidence=transformer_confidence,
        vae_anomaly_score=vae_score,
        dga_probability=dga_prob,
        status="scored" if all(available_signals.values()) else "partial",
        message=(
            "Fusion model unavailable; score uses the documented heuristic fallback."
            if all(available_signals.values()) and (models_container.fusion_model is None or models_container.fusion_scaler is None)
            else None if all(available_signals.values())
            else "Partial score uses only available models: " + ", ".join(
                name for name, active in available_signals.items() if active
            ) + "."
        ),
    )
