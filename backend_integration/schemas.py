"""
Pydantic Payload Schemas for FastAPI API Endpoints
"""

from typing import List, Optional, Any, Union
from pydantic import BaseModel


class ScoreRequest(BaseModel):
    flow_sequence: List[List[float]]
    domain: Optional[str] = None


class ScoreResponse(BaseModel):
    event_id: str
    timestamp: str
    threat_score: float
    predicted_class: str
    transformer_confidence: float
    vae_anomaly_score: float
    dga_probability: float


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

    model_config = {"from_attributes": True}


class EventsListResponse(BaseModel):
    events: List[EventResponse]
