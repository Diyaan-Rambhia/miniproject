"""
FastAPI Application Entrypoint — Layered AI Defense Backend Service
"""

import os
import sys

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend_integration.db import init_db
from backend_integration.model_loader import models_container
from backend_integration import routes_score, routes_explain, routes_dashboard


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n--- Initializing Backend Integration Layer ---")
    init_db()
    models_container.load_all()
    print("--- Backend Service Initialized Successfully ---\n")
    yield


app = FastAPI(
    title="Layered AI Defense Pipeline — Backend API",
    description="FastAPI service serving scoring, dual-channel SHAP + attention explainability, and adversarial robustness telemetry.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS Middleware
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(routes_dashboard.router)
app.include_router(routes_score.router)
app.include_router(routes_explain.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_integration.main:app", host="127.0.0.1", port=8000, reload=True)
