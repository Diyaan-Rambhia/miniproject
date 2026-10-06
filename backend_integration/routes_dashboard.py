"""
Dashboard & Operational API Router (GET /health, GET /events, GET /events/{id}, GET /robustness)
"""

import sqlite3
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query

from backend_integration.db import get_db
from backend_integration.model_loader import models_container
from backend_integration.schemas import EventResponse, EventsListResponse

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Layered AI Defense Backend Integration Layer",
        "models": {
            "transformer": models_container.transformer_model is not None,
            "vae": models_container.vae_model is not None,
            "dga": models_container.dga_model is not None,
            "fusion": models_container.fusion_model is not None,
            "adversarial": models_container.adversarial_model is not None,
        },
    }


@router.get("/events", response_model=EventsListResponse)
def get_recent_events(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset))
    rows = cursor.fetchall()
    events = [dict(row) for row in rows]
    return EventsListResponse(events=events)


@router.get("/events/{event_id}", response_model=EventResponse)
def get_event_detail(event_id: str, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Event with ID '{event_id}' not found.")
    return dict(row)


@router.get("/robustness")
def get_robustness_results(db: sqlite3.Connection = Depends(get_db)):
    """Return stored adversarial robustness results; empty until training runs have populated the table."""
    cursor = db.cursor()
    cursor.execute(
        "SELECT run_id, timestamp, model_name, clean_acc, fgsm_acc, pgd_acc FROM robustness_results ORDER BY timestamp DESC"
    )
    rows = cursor.fetchall()
    results = []
    for row in rows:
        results.append({
            "run_id": row[0],
            "timestamp": row[1],
            "model_name": row[2],
            "variant": row[2],
            "clean_acc": row[3],
            "clean_f1": None,
            "fgsm_acc": row[4],
            "fgsm_f1": None,
            "pgd_acc": row[5],
            "pgd_f1": None,
        })
    return {"results": results}
