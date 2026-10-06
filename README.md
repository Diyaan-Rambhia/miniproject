# Network Security — Layered AI Defense Pipeline

A layered, multi-signal network intrusion & anomaly detection system: a Transformer classifier for known attacks, a VAE for zero-day anomalies, a char-LSTM for malicious domains (DGA), a fusion layer combining all three into a Threat Score, SHAP + attention explainability, an LLM layer that explains flagged events in plain English, and an adversarial robustness study on the detector itself.

Full architecture, design decisions, and detailed documentation: see [`project_reference.md`](./project_reference.md). Real-time task progress: see [`TASK_PROGRESS.md`](./TASK_PROGRESS.md).

## Directory Structure

```text
miniproject/
├── .env                          # LLM_API_KEY (gitignored — see .env.example)
├── .env.example
├── .gitignore
├── project_reference.md          # Master architecture and design reference
├── README.md                     # Project overview and quickstart guide
├── TASK_PROGRESS.md              # Real-time task progress checklist
├── requirements.txt              # Unified Python dependencies
├── backend_integration/          # FastAPI backend service & SQLite store
│   ├── main.py                   # Application entrypoint & CORS config
│   ├── config.py                 # Paths, model checkpoints, threshold settings
│   ├── model_loader.py           # ModelContainer loader for detectors, Fusion, and robustness model
│   ├── inference_preprocess.py   # StandardScaler inference normalization
│   ├── routes_score.py           # POST /score endpoint (fusion threat scoring)
│   ├── routes_explain.py         # POST /explain endpoint (SHAP + attention + LLM)
│   ├── routes_dashboard.py       # GET /events, /events/{id}, /robustness, /health
│   ├── schemas.py                # Pydantic v2 request & response schemas
│   └── db.py                     # SQLite database initialization & helpers
├── mini_project_ui/              # Active React + Vite dashboard
├── mini/                         # Legacy duplicate frontend; not used by backend
├── docs/                         # SRS and Mermaid UML submission documents
├── tests/                        # Automated unit & integration tests
│   └── test_main.py              # Backend endpoint tests
└── models/
    ├── train_all.py              # One-command end-to-end training orchestrator
    ├── run_core_pipeline.py      # Core pipeline orchestrator (Transformer -> VAE -> DGA -> Fusion -> Explain)
    ├── run_adversarial_pipeline.py # Adversarial pipeline orchestrator (Attack -> Harden -> Compare)
    ├── data/                     # Raw datasets, prep scripts, & split CSVs
    │   ├── prepare_cicids2017.py # CICIDS2017 15-class stratified combiner & splitter
    │   └── prepare_domains.py    # DGA/Tranco domain combiner & splitter
    ├── saved_weights/            # Transformer artifacts and selected central checkpoints
    ├── transformer/              # Known-attack classifier (15 classes)
    ├── VAE/                      # Benign-only zero-day anomaly detector
    ├── DGA_detector/             # Bidirectional char-LSTM domain classifier
    ├── fusion_head/              # Threat score MLP & signals CSV generator
    ├── explainibility + SHAP/    # Attention & SHAP GradientExplainer
    ├── Adverserial_robustness/   # FGSM/PGD attacks & adversarial training defense
    └── llm_layer/                # Grounded prompt builder & OpenRouter client
```

## Setup & Installation

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Copy `.env.example` to `.env` at the project root and populate `LLM_API_KEY` (configured for OpenRouter by default):
   ```bash
   cp .env.example .env
   ```

4. **Data Preparation**:
   Run dataset combination and 70/10/20 stratified splitting scripts (includes 15-class coverage safeguards):
   ```bash
   python models/data/prepare_cicids2017.py
   python models/data/prepare_domains.py
   ```

## Running the System

### 1. Backend Service (FastAPI)
Launch the API backend for live inference, explanation generation, and event logging:
```bash
uvicorn backend_integration.main:app --reload --port 8000
```
Interactive API docs are available at `http://localhost:8000/docs`.

### 2. Frontend Dashboard (React + Vite)
In a separate terminal, start the UI dev server:
```bash
cd mini_project_ui
npm install
npm run dev
```
Access the dashboard at `http://localhost:5173`.

### 3. Model Training Pipelines
- **End-to-end training**:
  ```bash
  python models/train_all.py
  ```
  *(Automatically trains all upstream models, generates signals, and trains the Fusion head, skipping any model already saved in `models/saved_weights/`)*

- **Core pipeline test**:
  ```bash
  python models/run_core_pipeline.py
  ```

- **Adversarial robustness study**:
  ```bash
  python models/run_adversarial_pipeline.py
  ```

### 4. Running Tests
Run backend test coverage:
```bash
pytest tests/
```

## Current Status

- **Completed**:
  - Raw flow (CICIDS2017) combined (2.83M flows) and split 70/10/20 with verified coverage across all 15 classes.
  - Domain datasets (DGA + Tranco) combined (4.35M domains) and split 70/10/20.
  - Core models trained on corrected 15-class split: Transformer classifier, VAE anomaly detector, DGA char-LSTM, and Fusion MLP.
  - Trained checkpoints exist; backend loads Transformer artifacts centrally and VAE/DGA/Fusion from component output folders.
  - Signal generation and test-set scaler re-fitting bugs resolved.
  - FastAPI backend starts with all five model objects loaded; health, CORS, score, event-history, and one automated API test were verified.
  - Inference-time feature normalization (`inference_preprocess.py`) integrated.
  - Fusion-driven decision logic implemented (`predicted_class` determined by fusion threshold; `transformer_predicted_class` kept as type hint).
  - LLM explainer is configured for OpenRouter (`meta-llama/llama-3.1-8b-instruct:free`); current local key is placeholder-like, so a live provider call remains unverified.

- **Open Items / Risks**:
  - Validate VAE inference preprocessing: its saved scaler differs from the Transformer scaler used by `/score`.
  - Rebuild Fusion signals with event-aligned DGA data and leakage-free training/evaluation partitions before presenting Fusion metrics as independent holdout evidence.
  - Replace the placeholder-like `.env` LLM key and verify the live explanation provider.
  - Replace/mark synthetic fallback attributions and add production security controls before external exposure.

The adversarial evaluation is complete for the recorded run; results and accuracy values are in [`project_reference.md`](./project_reference.md). See Sections 4 and 11 there for evidence and remaining work.