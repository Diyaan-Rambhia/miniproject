"""
Pydantic Payload Schemas for FastAPI API Endpoints
"""

from typing import List, Optional, Any, Union
from pydantic import BaseModel


class ScoreRequest(BaseModel):
    flow_sequence: List[List[float]]
    domain: Optional[str] = None


class ScoreResponse(BaseModel):
    event_id: Optional[str] = None
    timestamp: Optional[str] = None
    threat_score: Optional[float] = None
    # Authoritative detect/no-detect label from fused threat_score (see FUSION_ATTACK_THRESHOLD).
    predicted_class: Optional[str] = None
    # Transformer argmax attack type (informational; not the fusion decision).
    transformer_predicted_class: Optional[str] = None
    transformer_confidence: Optional[float] = None
    vae_anomaly_score: Optional[float] = None
    dga_probability: Optional[float] = None
    status: str = "scored"
    message: Optional[str] = None


class ExplainRequest(BaseModel):
    event_id: str


class ExplainResponse(BaseModel):
    event_id: str
    predicted_class: Union[str, int]
    confidence: float
    top_attended_timesteps: List[Any]
    top_shap_features: List[Any]
    plain_english_explanation: str


class EventResponse(BaseModel):
    event_id: str
    timestamp: str
    threat_score: float
    transformer_confidence: float
    vae_anomaly_score: float
    dga_probability: float
    predicted_class: str
    transformer_predicted_class: Optional[str] = None

    model_config = {"from_attributes": True}


class EventsListResponse(BaseModel):
    events: List[EventResponse]
