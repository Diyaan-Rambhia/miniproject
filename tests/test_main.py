import sqlite3

import numpy as np
import torch
from fastapi.testclient import TestClient

from backend_integration.db import get_db
from backend_integration.main import app
from backend_integration.model_loader import models_container
from backend_integration import routes_explain, routes_score
from backend_integration.schemas import ScoreRequest


def test_routes_respond_without_models(monkeypatch):
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            threat_score REAL NOT NULL,
            transformer_confidence REAL NOT NULL,
            vae_anomaly_score REAL NOT NULL,
            dga_probability REAL NOT NULL,
            predicted_class TEXT NOT NULL,
            raw_sequence TEXT
        )"""
    )
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("test-event", "2026-01-01 00:00:00", 0, 0, 0, 0, "BENIGN", "[[0.0]]"),
    )
    connection.commit()

    def override_get_db():
        yield connection

    app.dependency_overrides[get_db] = override_get_db
    for name in (
        "transformer_model", "vae_model", "dga_model", "fusion_model", "fusion_scaler",
        "trans_scaler", "trans_label_encoder", "trans_feature_names",
    ):
        monkeypatch.setattr(models_container, name, None)
    monkeypatch.setattr(models_container, "dga_vocab", {})
    monkeypatch.setattr(models_container, "is_loaded", True)
    monkeypatch.setattr(models_container, "load_all", lambda: None)
    monkeypatch.setattr(
        routes_explain,
        "llm_explain",
        lambda *_: (_ for _ in ()).throw(RuntimeError("LLM unavailable")),
    )

    try:
        with TestClient(app) as client:
            score = client.post("/score", json={"flow_sequence": [[0.0]]})
            assert score.status_code == 200
            assert score.json()["status"] == "insufficient_models"
            assert "no usable Transformer, VAE, or DGA model" in score.json()["message"]
            assert score.json()["threat_score"] is None
            assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1

            explain = client.post("/explain", json={"event_id": "test-event"})
            assert explain.status_code == 200
            assert "LLM service" in explain.json()["plain_english_explanation"]
            assert client.get("/events").status_code == 200
            assert client.get("/health").status_code == 200
    finally:
        app.dependency_overrides.pop(get_db, None)
        connection.close()


def test_score_uses_separate_transformer_and_vae_scalers(monkeypatch):
    class AffineScaler:
        def __init__(self, offset, divisor):
            self.offset = offset
            self.divisor = divisor

        def transform(self, values):
            return (np.asarray(values) - self.offset) / self.divisor

    transformer_inputs = []
    vae_inputs = []

    class Transformer:
        def __call__(self, values):
            transformer_inputs.append(values.detach().cpu().numpy())
            return torch.tensor([[2.0, 0.0]], device=values.device)

    class VAEModel:
        def __call__(self, values):
            vae_inputs.append(values.detach().cpu().numpy())
            latent = torch.zeros((values.shape[0], 16), device=values.device)
            return values, latent, latent

    transformer_scaler = AffineScaler(offset=1.0, divisor=2.0)
    vae_scaler = AffineScaler(offset=13.0, divisor=7.0)
    monkeypatch.setattr(models_container, "is_loaded", True)
    monkeypatch.setattr(models_container, "load_all", lambda: None)
    monkeypatch.setattr(models_container, "transformer_model", Transformer())
    monkeypatch.setattr(models_container, "trans_scaler", transformer_scaler)
    monkeypatch.setattr(models_container, "trans_label_encoder", type("Encoder", (), {"classes_": np.array(["BENIGN", "ATTACK"])})())
    monkeypatch.setattr(models_container, "trans_feature_names", [f"feature_{index}" for index in range(78)])
    monkeypatch.setattr(models_container, "vae_model", VAEModel())
    monkeypatch.setattr(models_container, "vae_scaler", vae_scaler)
    monkeypatch.setattr(models_container, "dga_model", None)
    monkeypatch.setattr(models_container, "dga_vocab", {})
    monkeypatch.setattr(models_container, "fusion_model", None)
    monkeypatch.setattr(models_container, "fusion_scaler", None)

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            threat_score REAL NOT NULL,
            transformer_confidence REAL NOT NULL,
            vae_anomaly_score REAL NOT NULL,
            dga_probability REAL NOT NULL,
            predicted_class TEXT NOT NULL,
            raw_sequence TEXT,
            transformer_predicted_class TEXT
        )"""
    )

    raw_sequence = np.arange(10 * 78, dtype=np.float32).reshape(10, 78)
    try:
        routes_score.score_event(ScoreRequest(flow_sequence=raw_sequence.tolist()), connection)
        np.testing.assert_allclose(
            transformer_inputs[0][0], transformer_scaler.transform(raw_sequence)
        )
        np.testing.assert_allclose(
            vae_inputs[0][0], vae_scaler.transform(raw_sequence)[-1]
        )
        assert not np.allclose(vae_inputs[0][0], transformer_inputs[0][0, -1])
    finally:
        connection.close()
