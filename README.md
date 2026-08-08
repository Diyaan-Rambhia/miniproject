# Network Security — Layered AI Defense Pipeline

A layered, multi-signal network intrusion & anomaly detection system: a Transformer classifier for known attacks, a VAE for zero-day anomalies, a char-LSTM for malicious domains (DGA), a fusion layer combining all three into a Threat Score, SHAP + attention explainability, an LLM layer that explains flagged events in plain English, and an adversarial robustness study on the detector itself.

Full architecture, design decisions, and detailed documentation: see [`project_reference.md`](./project_reference.md). Real-time task progress: see [`TASK_PROGRESS.md`](./TASK_PROGRESS.md).

## Directory Structure

```text
miniproject/
├── .env                          # LLM_API_KEY (gitignored — see .env.example)
├── .env.example
├── .gitignore
├── project_reference.md
├── TASK_PROGRESS.md
├── requirements.txt
├── mini_project_ui/               # React dashboard (frontend)
└── models/
    ├── train_all.py               # One-command end-to-end training orchestrator
    ├── run_core_pipeline.py       # Core pipeline orchestrator (Transformer -> VAE -> DGA -> Fusion -> SHAP/Attn -> LLM)
    ├── run_adversarial_pipeline.py# Adversarial pipeline orchestrator (FGSM/PGD attacks -> Hardened Retraining -> 3-way table)
    ├── data/                      # Raw datasets, prep scripts (prepare_cicids2017.py, prepare_domains.py), & split CSVs
    ├── saved_weights/             # Central checkpoint store (transformer_final.pt, vae_final.pt, etc.)
    ├── transformer/               # Known-attack classifier (config.py, data_loader.py, preprocessing.py, dataset.py, model.py, train.py, evaluate.py, save_artifacts.py, main.py)
    ├── VAE/                       # Zero-day anomaly detector (config.py, data_loader.py, preprocessing.py, dataset.py, model.py, train.py, evaluate.py, save_artifacts.py, main.py, VAE.py)
    ├── DGA_detector/               # Malicious domain detector (config.py, data_loader.py, preprocessing.py, dataset.py, model.py, train.py, evaluate.py, save_artifacts.py, main.py)
    ├── fusion_head/                # Threat score MLP & signals CSV generator (config.py, data_loader.py, preprocessing.py, dataset.py, model.py, generate_signals.py, train.py, evaluate.py, save_artifacts.py, main.py)
    ├── explainibility + SHAP/        # Attention & SHAP explainers (config.py, model.py, explainer.py, attention_explainer.py, shap_explainer.py, main.py)
    ├── Adverserial_robustness/     # Attack & defense study (config.py, data_loader.py, preprocessing.py, dataset.py, model.py, attack.py, train.py, evaluate.py, save_artifacts.py, main.py)
    └── llm_layer/                  # LLM explanation layer (llm_config.py, llm_prompting.py, llm_explanation.py)
```

## Setup & Installation

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # Linux/macOS:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Copy `.env.example` to `.env` at the project root and populate `LLM_API_KEY` once your provider is chosen:
   ```bash
   cp .env.example .env
   ```

4. **Data Preparation**:
   Run dataset combination and 70/10/20 stratified splitting scripts:
   ```bash
   python models/data/prepare_cicids2017.py
   python models/data/prepare_domains.py
   ```

## Running the Training Pipeline

### Primary (One-Command End-to-End Training)
To train all models in dependency order, save checkpoints centrally to `models/saved_weights/`, and print an evaluation summary:

```bash
python models/train_all.py
```
*Note: `models/train_all.py` automatically skips any model whose trained checkpoint already exists in `models/saved_weights/`.*

### Secondary / Debugging (Individual Model Pipelines)
You can also run individual model pipelines directly:

```bash
python models/transformer/main.py
python models/VAE/main.py
python models/DGA_detector/main.py
python models/fusion_head/main.py         # Requires Transformer, VAE, and DGA trained first
python models/Adverserial_robustness/main.py   # Requires Transformer trained first
python models/run_core_pipeline.py        # Full detection + explainability pipeline
python models/run_adversarial_pipeline.py # Robustness study pipeline
```

Trained weights are saved locally in each model's `outputs/` directory and copied centrally to `models/saved_weights/`.

## Current Status

- **Completed**:
  - Raw flow (CICIDS2017) and domain (DGA/Tranco) datasets combined and split 70/10/20 into `models/data/`.
  - All model modules updated to use pre-split datasets via relative paths.
  - End-to-end training orchestrator (`models/train_all.py`) and pipeline scripts built.
  - Central weight-saving convention enabled in `models/saved_weights/`.
  - Unified project `requirements.txt` created.

- **Open Items (Pending)**:
  - Final decision on LLM `API_PROVIDER` (Anthropic vs OpenAI) and API key configuration.
  - FastAPI backend service implementation and path location decision.
  - Full model training execution on GPU compute cluster.

See [`project_reference.md`](./project_reference.md) Section 7 (Timeline) and Section 11 (Open Items) for full details.