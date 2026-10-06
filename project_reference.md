# Network Security — Layered AI Defense Pipeline — Project Reference

**Duration target:** ~9-12 weeks core build (Oct end date fixed)
**Team:** You (AI/ML + backend + adversarial study) + teammates (frontend/dashboard)
**Status (6 October 2026):** Core model checkpoints and serving artifacts are present; the FastAPI service starts and reports Transformer, VAE, DGA, Fusion, and adversarial models loaded. The merged React/Vite frontend builds, CORS preflight from `localhost:5173` succeeds, and its `/score` payload uses the backend's `flow_sequence` field. Transformer and VAE inference now apply their own saved scalers independently. Replaying the stored DoS Hulk and PortScan requests changed VAE anomaly scores from 3.05068994 to 1.34190059 and from 0.10061108 to 0.20533589, respectively. The database contains baseline and hardened robustness accuracy results. Fusion signal generation still pairs DGA test probabilities to flow sequences by row position/resizing and reuses test-derived signals in Fusion training/evaluation. The project-root `.env` contains a placeholder-like `LLM_API_KEY`, not a verified usable credential. Treat the system as a local research/demo integration, not production-ready.

---

## 1. Problem Statement

A **layered, multi-signal network defense pipeline** — multiple detection signals feeding a fusion layer, an explanation layer on top, and a study of adversarial robustness.

The system:
1. Classifies **known attacks** from network flow data (supervised).
2. Detects **unknown/zero-day** anomalies from flow data (unsupervised).
3. Flags **malicious domains** (DGA) as a separate, independent signal.
4. Fuses all signals into one **threat score**.
5. Provides **on-demand explanations** for stored events, with feature-level attributions and optional plain-English LLM text.
6. Is **tested against adversarial evasion**, with a defense applied and evaluated.

---

## 2. Architecture

```
                         Prepared Flow Sequence
                    │
      ┌─────────────┼──────────────────┐
      │              │                  │
10-flow windows   Optional domain
from ordered     supplied with request
numeric rows
      │              │
┌─────┴──────┐       │
│             │       │
Transformer   VAE   DGA Char-level
Encoder     (benign-  Detector
(known      only,    (malicious vs
 attacks)   anomaly   legit domains)
            score)
      │             │       │
      └──────┬──────┴───────┘
             │
     Fusion Scoring Layer
   (MLP on all 3 signals)
             │
       Threat Score (0-100)
             │
   ┌─────────┴──────────┐
   │                      │
 SHAP +               LLM Explanation
Attention             Layer (plain-English
Explainability        "why flagged")
   │                      │
   └─────────┬────────────┘
             │
       FastAPI Backend (`backend_integration/`)
             │
       React Dashboard (Frontend)


Parallel study (separate track, same detector):
       Baseline Transformer
             │
     Adversarial Perturbation
     (FGSM + PGD)
             │
     Evaluate detection drop
             │
     Adversarial Training (defense)
             │
     Re-evaluate: baseline vs hardened,
     clean vs FGSM vs PGD
```

---

## 3. Project Folder Structure

```
miniproject/
├── .env                         (LLM_API_KEY — gitignored)
├── .env.example
├── .gitignore
├── project_reference.md         (this file)
├── README.md
├── TASK_PROGRESS.md
├── requirements.txt
├── backend_integration/         (FastAPI backend service & SQLite store)
│   ├── main.py                  (FastAPI application entrypoint & lifespan)
│   ├── config.py                (paths, model artifacts, threshold configs)
│   ├── model_loader.py          (ModelContainer for detectors, Fusion, and adversarial model)
│   ├── inference_preprocess.py  (StandardScaler sequence preprocessing)
│   ├── routes_score.py          (POST /score endpoint with fusion decision logic)
│   ├── routes_explain.py        (POST /explain endpoint with SHAP/attention/LLM)
│   ├── routes_dashboard.py      (GET /events, /events/{id}, /robustness, /health)
│   ├── schemas.py               (Pydantic request & response models)
│   └── db.py                    (SQLite connection & initialization)
├── mini_project_ui/             (active React/Vite dashboard)
├── mini/                        (legacy duplicate frontend; not used by the backend)
├── docs/                        (implementation-based SRS and UML submission documents)
├── tests/                       (API and pipeline tests)
│   └── test_main.py
└── models/
    ├── train_all.py             (end-to-end training orchestrator)
    ├── run_core_pipeline.py     (pipeline 1: detection + explainability orchestrator)
    ├── run_adversarial_pipeline.py (pipeline 2: attack + hardened training orchestrator)
    ├── data/                    (raw + combined + split datasets, prep scripts)
      ├── saved_weights/           (Transformer artifacts and selected central checkpoints)
    ├── transformer/             (known-attack classifier)
    ├── VAE/                     (zero-day anomaly detector)
    ├── DGA_detector/            (malicious domain classifier)
    ├── fusion_head/             (combines the 3 signals -> Threat Score)
    ├── explainibility + SHAP/   (SHAP + attention explainability)
    ├── Adverserial_robustness/  (FGSM/PGD attack + adversarial training)
    └── llm_layer/               (grounded LLM prompt builder & OpenRouter caller)
```

Each model folder (except `data/`, `saved_weights/`, `llm_layer/`) is split into phase-based files: `config.py`, `data_loader.py`, `preprocessing.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `save_artifacts.py`, `main.py` — adjusted per folder's actual needs. `llm_layer/` is split into `llm_config.py`, `llm_prompting.py`, `llm_explanation.py` (canonical entry point).

---

## 4. Core Components

### 4.1 Data
- **Flows:** CICIDS2017 (Kaggle: chethuhn/network-intrusion-dataset — standard MachineLearningCSV version, 8 files, 2,830,743 flows, 78 numeric features)
- **Domains:** 4,347,492 unique domains combining a DGA domain list and Tranco (legitimate) top-domains list
- Both datasets have pre-split CSVs in `models/data/`; CICIDS2017 split coverage is enforced. Current Fusion signal provenance/alignment limitations are recorded in Sections 4.5 and 11.

### 4.2 Known-Attack Classifier — Transformer Encoder
- Input: sequence of numeric flow rows built with a sliding window (`SEQ_LEN=10`, `SEQ_STRIDE=5` by default); current preprocessing does not group by source IP.
- Architecture: small Transformer encoder — `NUM_LAYERS=3`, `D_MODEL=128`, `NHEAD=4`, `DIM_FEEDFORWARD=256`
- Output: multi-class classification across 15 classes (BENIGN + 14 attack classes). The serving class uses PyTorch's standard `nn.TransformerEncoder`; it does not return attention weights.
- **Class-coverage bug found and fixed:** Standard `train_test_split(..., stratify=...)` failed on ultra-rare classes (Heartbleed: 11 rows, Web Attack – SQL Injection: 21 rows, Infiltration: 36 rows), omitting them from validation or test splits and causing label encoder mismatches during evaluation and inference. An explicit coverage enforcement algorithm (`_ensure_class_coverage` & `verify_split_coverage` in `models/data/prepare_cicids2017.py`) was implemented to guarantee every one of the 15 classes appears in train, val, and test splits. The Transformer was retrained on this corrected 15-class split, producing verified weights (`transformer_final.pt` in `models/saved_weights/`) with full 15-class output mapping.
- **Architecture note:** `models/transformer/model.py` uses PyTorch's standard `nn.TransformerEncoder`. `models/explainibility + SHAP/model.py` defines a separate attention-capturing mirror and loads the Transformer checkpoint for explanation; the scoring instance itself does not expose attention. `models/Adverserial_robustness/model.py` maintains a duplicate Transformer definition for the robustness experiment.
- Measured (synthetic benchmark): ~416K params, ~136ms/batch (CPU, batch=256).

### 4.3 Unknown/Zero-Day Detector — VAE
- Trained **only on benign traffic** (~1.6M benign flows)
- Encoder (78 -> 64 -> 16) → latent (μ, σ) → sample z → decoder (16 -> 64 -> 78) → reconstruction
- Loss: reconstruction error + KL divergence (ELBO), `KL_WEIGHT=0.5`
- Anomaly score: high ELBO loss = doesn't look like normal traffic; threshold set from benign validation score distribution (95th percentile default)
- Checkpoint and artifacts are stored under `models/VAE/outputs/` and include `vae_final.pt`, `feature_scaler.joblib`, `anomaly_threshold.joblib`, and feature names. A copy of the checkpoint may also exist in `models/saved_weights/`; the backend loads the component output path. `/score` now applies the Transformer scaler for Transformer inference and the separately fitted VAE scaler for VAE inference. The same stored DoS Hulk and PortScan flow sequences were replayed; the API VAE scores matched direct inference with the VAE scaler (1.34190059 and 0.20533589) and differed from the Transformer-scaler replay.

### 4.4 DGA Domain Detector
- Input: domain name as a character sequence (char-level vocab built from data, `MAX_LEN=64`)
- Architecture: bidirectional LSTM (`EMBED_DIM=32`, `HIDDEN_DIM=64`, `NUM_LSTM_LAYERS=2`, `DROPOUT=0.2`)
- Output: malicious (DGA) vs legitimate — precision reported alongside recall since false positives block real domains
- Checkpoint and artifacts are loaded from `models/DGA_detector/outputs/` (`dga_lstm_final.pt`, `char_vocab.joblib`, `max_len.joblib`); central copies may exist, but the backend uses the component output path.

### 4.5 Fusion Scoring Layer
- Input: `[transformer_confidence, vae_anomaly_score, dga_probability]` for the same event
- Architecture: Small MLP (`HIDDEN_DIM=16`, `DROPOUT=0.1`) → Threat Score (0-100) via `P(attack) * 100.0`
- **Scaler / Checkpoint path bug found and fixed:** `models/fusion_head/generate_signals.py` previously called `encode_and_scale(df_flows)` on the test set, inadvertently fitting a new scaler and label encoder instead of using the pre-fitted `feature_scaler.joblib` and `label_encoder.joblib` from `transformer/outputs/`. This introduced distribution shift and corrupted upstream signals. Fixed by loading pre-fitted scalers and encoders.
- **Runtime artifacts:** `fusion_mlp_final.pt` and `signal_scaler.joblib` are loaded from `models/fusion_head/outputs/` by `backend_integration/model_loader.py`.
- **Evaluation validity is unresolved:** `generate_signals.py` derives flow signals from CICIDS test sequences and DGA signals from a separate domain test set, joining them by row position and truncating or resizing the DGA probability array. These are not demonstrated to describe the same event. The Fusion pipeline then splits this generated signal dataset into its own train/validation/test portions, so upstream held-out test examples contribute to Fusion training. Rebuild aligned, training-only Fusion signals with a leakage-free evaluation split and retrain/re-evaluate before presenting Fusion metrics as an independent holdout result.
- **Degraded fallback:** If the trained fusion model/scaler is unavailable, the backend computes a weighted heuristic over the active signals, normalizes by active weights, caps the VAE input at 2.0, and scales to 0–100. This is not a substitute for the trained Fusion model.
- Ablation code compares one-signal logistic regression baselines and the Fusion model. The quality of reported Fusion/ablation performance remains subject to the signal-alignment and split-leakage issues above.

### 4.6 Adversarial Robustness Study
- Not a new architecture — attacks and hardens a duplicate of the Transformer
- **Attack:** FGSM (single-step) and PGD (multi-step, `PGD_STEPS=10`, `PGD_ALPHA=0.02`) against the trained baseline
- **Defense:** adversarial training — retrain (from a copy of baseline weights) on a mix of clean + FGSM-perturbed batches (`ADV_TRAIN_MIX_RATIO=0.5`)
- **Verified current run:** baseline and hardened results are present in SQLite. Baseline accuracy: clean 0.9921, FGSM 0.6314, PGD 0.3902. Hardened accuracy: clean 0.9874, FGSM 0.9342, PGD 0.8817. The results endpoint/database currently preserve accuracy, not macro-F1; no claim about general robustness beyond this configured evaluation is implied.
- Uses its own duplicated data pipeline + model definition (matching `transformer/` exactly), kept self-contained.

### 4.7 Explainability
- **SHAP** (`GradientExplainer` — `DeepExplainer` failed on `LayerNorm`/custom modules, a known SHAP/Transformer incompatibility) — feature-level attribution, run on-demand via `POST /explain`
- **Attention weights** — captured by the separate explainability model implementation in `explainibility + SHAP/`, not by the serving Transformer instance.
- The route provides top timestep and feature attributions to the LLM prompt builder. If attribution code raises an exception, the current route returns hard-coded example attribution values; those are degraded placeholders, not computed explanations.

### 4.8 LLM Explanation Layer (`llm_layer/`)
- **Finalized as prompting-only** — no fine-tuning path
- Takes SHAP + attention output, builds a grounded prompt incorporating `ATTACK_SIGNATURES` for the detected class, and queries the LLM provider
- **Provider finalized to OpenRouter:** Configured in `models/llm_layer/llm_config.py` (`API_PROVIDER = "openrouter"`). Calls OpenRouter's OpenAI-compatible endpoint (`https://openrouter.ai/api/v1`) using `meta-llama/llama-3.1-8b-instruct:free`
- API key is loaded via `python-dotenv` from the project-root `.env` file (`LLM_API_KEY=...`). The checked-in `.env.example` contains an explicit placeholder.
- `.env` is gitignored. Current `.env` has a configured but placeholder-like/non-OpenRouter-format value; live provider generation is not verified and remains open.

### 4.9 Backend (`backend_integration/`)
- Fully built, debugged, and verified FastAPI backend at the project root. Provides:
  - `POST /score` — fast-path scoring over raw flow sequences and optional domain
  - `POST /explain` — slow-path on-demand SHAP + attention + OpenRouter explanation for any stored event
      - `GET /events` — paginated historical events list
  - `GET /events/{id}` — detailed single-event inspection
  - `GET /robustness` — robustness benchmark metrics (baseline vs hardened under clean/FGSM/PGD)
  - `GET /health` — service status and loaded model diagnostic summary
- **Per-model inference preprocessing:** `/score` applies the saved Transformer `StandardScaler` to the raw sequence for Transformer inference and independently applies the saved VAE scaler to that same raw sequence for VAE inference. Both artifacts use the same ordered 78-feature schema. `tests/test_main.py` verifies the models receive separately transformed inputs; stored DoS Hulk and PortScan fixtures verify the VAE API values match direct VAE-scaler inference.
- **Fusion-driven decision logic:**
  - `threat_score` (0-100) from the Fusion MLP is the authoritative decision metric.
  - **`predicted_class`** (BENIGN vs attack) is determined by `threat_score >= FUSION_ATTACK_THRESHOLD (50.0)`. If above threshold, it adopts `transformer_predicted_class` if non-benign, or defaults to "ATTACK". If below threshold, it is classified as "BENIGN".
  - **`transformer_predicted_class`** is preserved as an informational attack type hint from the Transformer argmax, explicitly separating the authoritative detection decision from class type labeling.
- **Graceful degradation:** Missing signal sources produce a partial heuristic score or, if none are usable, `insufficient_models`. Missing Fusion with all detector signals uses a documented fallback. Model loader dependency artifacts are not all independently guarded, so arbitrary partial/corrupt artifact sets still require testing.
- **Persistence:** SQLite (`events.db`) stores scored events and `robustness_results`.
- LLM explanations import directly from `models/llm_layer/` as the single source of truth.

### 4.10 Frontend (`mini_project_ui/`)
- Active React 19 + Vite dashboard with overview, pipeline visualization, live/demo console, event explanations, adversarial lab, and `/dashboard` operational view.
- Production build succeeds. The build reports a minified JavaScript chunk above 500 kB (currently about 729 kB); lint exits successfully with several unused-variable/fast-refresh warnings.
- Backend base defaults to `http://127.0.0.1:8000` and can be configured with `VITE_API_BASE` or the operational dashboard connection field. Vite proxies the backend routes on port 8000 in development. Direct API requests are supported by the configured localhost CORS origins; preflight from `http://localhost:5173` was verified.
- The dashboard score helper now sends `flow_sequence`, matching `ScoreRequest`. UI build and API contract were checked; full browser-based scoring/explanation interaction is not automated end-to-end.
- A separate `mini/` directory remains as a legacy duplicate and is not the active frontend.

### 4.11 Evaluation
- Per-class precision/recall/F1 for the Transformer across all 15 classes
- CICIDS2017 class imbalance: rare classes (Heartbleed: 11 rows, SQL Injection: 21 rows, Infiltration: 36 rows) have guaranteed split presence as a coverage safeguard
- ROC-AUC for the VAE anomaly detector
- Precision/recall for the DGA detector
- Ablation study code: Transformer alone vs VAE alone vs DGA alone vs fused; interpret Fusion claims only after the signal provenance and split concerns in Section 4.5 are resolved.
- Adversarial robustness: baseline and hardened clean/FGSM/PGD accuracy rows are persisted; current recorded values are in Section 4.6. Macro-F1 is calculated offline but not persisted or returned by `/robustness`.

---

## 5. Stretch Goals (only after core is fully working)
1. PCAP replay demo
2. BERT-style masked-flow pretraining
3. Encrypted traffic fingerprinting or C2 beaconing as an additional signal
4. GNN comparison model

---

## 6. Tech Stack

**AI/ML:** Python, PyTorch, scikit-learn, SHAP, custom FGSM/PGD implementation, joblib

**LLM:** OpenRouter API (`meta-llama/llama-3.1-8b-instruct:free` via OpenAI client), configured via `python-dotenv` from `.env`. Grounded prompting with known attack signatures; no fine-tuning.

**Backend:** FastAPI (`backend_integration/`), SQLite (`events.db`), Pydantic v2, Uvicorn

**Frontend:** React, Vite, plain CSS, fetch client

**Compute:** Local dev / College GPU cluster / Kaggle GPU tier

**Dev tooling:** AI coding agent with `TASK_PROGRESS.md` checkpointing

---

## 7. Timeline

| Phase | Goal | Status |
|---|---|---|
| Weeks 1-2 | Dataset acquisition + combine/split (flows + domains) | **Completed** (prepared datasets and rare-class coverage safeguard present) |
| Weeks 3-4 | Transformer trained + evaluated | **Completed** (checkpoint loads; corrected 15-class mapping present) |
| Weeks 5-6 | VAE + DGA detector trained + evaluated | **Artifacts and VAE inference scaling verified** (independent scaler replay and route regression pass; DGA artifact loads) |
| Week 7 | Fusion scoring layer built + evaluated | **Runtime integrated; research evaluation needs correction** (DGA/flow signal alignment and test-split leakage concerns) |
| Week 8 | SHAP + attention + LLM layer integrated | **Local explanation path present; provider unverified** (placeholder-like key; fallback attribution is synthetic) |
| Weeks 9-10 | Adversarial robustness study | **Completed for recorded run** (hardened checkpoint loads and both model variants have clean/FGSM/PGD accuracy rows in SQLite) |
| Week 11 | FastAPI backend + dashboard integration | **Operational checks passed** (all five models load; health and CORS pass; frontend build passes; score payload corrected) |
| Week 12 | Ablation study, polish, report, viva prep | **In progress** (status docs updated; resolve model-data validity risks and credential; complete end-to-end demo rehearsal) |

---

## 8. Design Decisions Log

- **VAE over plain autoencoder** — principled probabilistic (ELBO) threshold instead of arbitrary reconstruction-error cutoff.
- **Small Transformer, not BERT-scale** — task complexity (78 numeric features, 15 classes) doesn't warrant more capacity; measured ~1.36 sequences per parameter at current size, so scaling up risks overfitting without evidence it's needed.
- **Sliding-window sequencing, not strict per-source-IP grouping** — kept general-purpose for the first pass; per-IP grouping is a documented possible refinement, not required for the core deliverable.
- **SHAP `GradientExplainer` over `DeepExplainer`** — `DeepExplainer` doesn't reliably support `LayerNorm`/custom Transformer modules; confirmed via direct test.
- **Checkpoint-compatible Transformer mirror for explanations** — serving uses `nn.TransformerEncoder`; `explainibility + SHAP/` has a separate custom attention-capturing implementation with compatible checkpoint structure. Attention is not returned by the serving model.
- **SQLite over MySQL** — zero-setup, single-file portability fits local development and demo; avoids external database server dependencies.
- **LLM layer: prompting only, no fine-tuning** — dropped fine-tuning to constrain project scope; grounded prompts with attack signatures provide high quality explanations.
- **Historical rename:** the former `final_model_combined/` name was replaced by `llm_layer/`; the old name is not an active source path.
- **Adversarial folder duplicates the Transformer's data pipeline and model class rather than importing** — keeps the folder self-contained for the phase-split refactor; byte-aligned to `transformer/model.py`.
- **Stratified-split rare-class coverage bug and fix** — Standard `train_test_split` dropped rare classes (Heartbleed, SQL Injection, Infiltration) from val/test sets. Added deterministic coverage enforcement (`_ensure_class_coverage` and `verify_split_coverage` in `prepare_cicids2017.py`) ensuring all 15 classes exist in train, val, and test splits.
- **Per-model inference scaling** — `/score` starts from the raw flow sequence and independently applies the saved Transformer scaler and saved VAE scaler before each model. The VAE scaler statistics differ from the Transformer scaler; route-level regression and stored DoS Hulk/PortScan replays confirm the VAE path uses its own scaler.
- **Fusion-driven decision logic** — The Transformer argmax alone should not override the multi-signal detection decision. Decision logic was refactored so that `threat_score >= 50.0` from Fusion MLP authoritatively dictates `predicted_class` (BENIGN vs attack), while `transformer_predicted_class` acts strictly as an informational attack type hint.
- **OpenRouter as LLM provider** — Configured OpenRouter (`API_PROVIDER = "openrouter"`) in `llm_config.py` using `meta-llama/llama-3.1-8b-instruct:free` via the OpenAI client, unifying API access without proprietary vendor lock-in while keeping dev/demo usage free.

---

## 9. Dataset Preparation — Summary

- **CICIDS2017:** 8 raw CSVs combined into `cicids2017_combined.csv` (2,830,743 rows), stratified split 70/10/20 with random state 42. Rare-class coverage safeguard guarantees all 15 classes in train, val, and test splits.
- **DGA + Tranco:** 4,347,492 unique domains combined into `domains_combined.csv`, stratified split 70/10/20 with random state 123.
- **Artifact paths:** Transformer artifacts load from `models/saved_weights/`; backend configuration loads VAE, DGA, and Fusion artifacts from their component `outputs/` folders. The adversarial checkpoint is read from the central saved-weights path when present, otherwise from its component output path. `.gitignore` excludes model outputs and central weights, so checkpoints are local artifacts rather than portable repository contents.

---

## 10. Working With a Coding Agent — Conventions

- Long/multi-step prompts include a `TASK_PROGRESS.md` checklist protocol so a token/budget cutoff mid-task doesn't lose work.
- Every prompt asks for minimal-token responses.
- Before executing any multi-file change, the agent presents a planned file breakdown.
- `project_reference.md` (this file) is the single source of truth for architecture and state.

---

## 11. Open Items / Outstanding Tasks

1. **Fusion training/evaluation validity (high priority):** Align domain detector outputs to their corresponding network events rather than pairing separate test sets by index/resizing. Generate Fusion training signals from a training partition, preserve a truly untouched evaluation partition, retrain the scaler/MLP, and rerun ablation before presenting Fusion metrics as independent generalization evidence.
2. **OpenRouter credential:** Replace the placeholder-like project-root `LLM_API_KEY` with a valid secret locally, keep it uncommitted, and perform a live `/explain` provider round trip. The current value is not reported here.
3. **Attribution fallback integrity:** Replace or explicitly mark the current synthetic fallback attention/SHAP values so users cannot mistake examples for computed event explanations.
4. **Expanded test coverage:** The current suite passes (2 tests). Add score-threshold boundary cases, explainability fallback coverage, and a full browser/API explanation workflow with isolated persistence.
5. **Robustness reporting:** Current database/API stores accuracy only although the offline evaluator calculates macro-F1. Persist and expose macro-F1 if it is required in the academic comparison table.
6. **Production-only hardening:** Before exposure outside a trusted local demo, restrict wildcard CORS, add authentication/authorization, request-size limits, database access/retention controls, and deployment TLS. These are not currently implemented.
7. **Legacy frontend directory:** Decide whether to remove or archive the unused `mini/` duplicate after confirming no teammate depends on it; `mini_project_ui/` is the active app.
8. **Event domain persistence:** `/score` accepts an optional domain but SQLite and event response schemas do not store it; the dashboard event table therefore cannot show the domain for persisted events.

The adversarial run is no longer an open completion item: the hardened checkpoint loads and SQLite contains persisted clean/FGSM/PGD accuracy results for baseline and hardened variants. The values and scope limitations are stated in Section 4.6.