# Network Security — Layered AI Defense Pipeline — Project Reference

**Duration target:** ~9-12 weeks core build (Oct end date fixed)
**Team:** You (AI/ML + backend + adversarial study) + teammates (frontend/dashboard)
**Status:** Architecture finalized, code scaffolded across all model folders. Dataset prep + central weight-saving convention completed (see Section 9). Training not yet started.

---

## 1. Problem Statement

A **layered, multi-signal network defense pipeline** — multiple detection signals feeding a fusion layer, an explanation layer on top, and a study of adversarial robustness.

The system:
1. Classifies **known attacks** from network flow data (supervised).
2. Detects **unknown/zero-day** anomalies from flow data (unsupervised).
3. Flags **malicious domains** (DGA) as a separate, independent signal.
4. Fuses all signals into one **threat score**.
5. **Explains** every flagged event, both at the feature level and in plain English.
6. Is **tested against adversarial evasion**, with a defense applied and evaluated.

---

## 2. Architecture

```
                Traffic In
                    │
      ┌─────────────┼──────────────────┐
      │              │                  │
Flow Sequencing   Domain names     (from same
(per src IP,      extracted from    traffic capture)
time-windowed)    traffic
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
       FastAPI Backend
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
├── mini_project_ui/             (React dashboard — teammates)
└── models/
    ├── data/                    (raw + combined + split datasets)
    ├── saved_weights/           (central copy of every trained model's final weights)
    ├── transformer/             (known-attack classifier)
    ├── VAE/                     (zero-day anomaly detector)
    ├── DGA_detector/            (malicious domain classifier)
    ├── fusion_head/             (combines the 3 signals -> Threat Score)
    ├── explainibility+SHAP/     (SHAP + attention explainability)
    ├── Adverserial_robustness/  (FGSM/PGD attack + adversarial training)
    └── llm_layer/               (LLM explanation layer — renamed from final_model_combined;
                                   contains ONLY the LLM explainer, not the FastAPI backend)
```

Each model folder (except `data/`, `saved_weights/`, `llm_layer/`) is split into phase-based files: `config.py`, `data_loader.py`, `preprocessing.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `save_artifacts.py`, `main.py` — adjusted per folder's actual needs. `llm_layer/` is split into `llm_config.py`, `llm_prompting.py`, `llm_explanation.py` (entry point).

---

## 4. Core Components

### 4.1 Data
- **Flows:** CICIDS2017 (Kaggle: chethuhn/network-intrusion-dataset — standard MachineLearningCSV version, 8 files, ~2.83M flows, 78-80 features)
- **Domains:** a DGA domain list + a Tranco (legitimate) top-domains list
- See **Section 9** for current combine/split status of both.

### 4.2 Known-Attack Classifier — Transformer Encoder
- Input: sequence of flows (per source IP, time-windowed, `SEQ_LEN=10`, `SEQ_STRIDE=5` by default)
- Architecture: small Transformer encoder — `NUM_LAYERS=3`, `D_MODEL=128`, `NHEAD=4`, `DIM_FEEDFORWARD=256`
- Output: multi-class classification (benign / attack type) + attention weights
- **Architecture note (finalized):** `transformer/model.py` uses a custom `AttentionCapturingEncoderLayer` (not PyTorch's built-in `nn.TransformerEncoder`), so attention weights are exposed natively — one model, one checkpoint, used for both inference and explainability. `explainibility+SHAP/` imports this same class rather than duplicating it. `Adverserial_robustness/model_def.py` keeps its own duplicate copy (by design — see 4.6), must stay byte-identical to `transformer/model.py` for checkpoint loading to work.
- Measured (synthetic benchmark): ~416K params, ~136ms/batch (CPU, batch=256) → full dataset ≈1.7hr/20 epochs on CPU, ≈10-15min on GPU.

### 4.3 Unknown/Zero-Day Detector — VAE
- Trained **only on benign traffic**
- Encoder → latent (μ, σ) → sample z → decoder → reconstruction
- Loss: reconstruction error + KL divergence (ELBO), `KL_WEIGHT=0.5`
- Anomaly score: high ELBO loss = doesn't look like normal traffic; threshold set from benign validation score distribution (95th percentile default)
- Measured: ~21.6K params, ~2.7ms/batch (CPU) → full benign set ≈9.5min/30 epochs CPU, ≈2-3min GPU

### 4.4 DGA Domain Detector
- Input: domain name as a character sequence (char-level vocab built from data, `MAX_LEN=64`)
- Architecture: bidirectional LSTM (`EMBED_DIM=32`, `HIDDEN_DIM=64`, `NUM_LSTM_LAYERS=2`)
- Output: malicious (DGA) vs legitimate — precision reported alongside recall since false positives block real domains

### 4.5 Fusion Scoring Layer
- Input: `[transformer_confidence, vae_anomaly_score, dga_probability]` for the same event
- Small MLP (`HIDDEN_DIM=16`) → Threat Score (0-100)
- Requires a signals CSV built by running the 3 upstream models in inference mode over the *same* held-out test set and pairing outputs by event — see Section 9 for why the train/val/test split must be consistent across folders.
- Ablation study (each signal alone vs fused, via logistic regression per signal) is the report centerpiece for this component.

### 4.6 Adversarial Robustness Study
- Not a new architecture — attacks and hardens a duplicate of the Transformer
- **Attack:** FGSM (single-step) and PGD (multi-step, `PGD_STEPS=10`, `PGD_ALPHA=0.02`) against the trained baseline
- **Defense:** adversarial training — retrain (from a copy of baseline weights) on a mix of clean + FGSM-perturbed batches (`ADV_TRAIN_MIX_RATIO=0.5`)
- **Output:** three-way comparison table — baseline vs hardened, each evaluated clean / under FGSM / under PGD
- Uses its own duplicated data pipeline + model definition (matching `transformer/` exactly), not imports — deliberate choice to keep folders self-contained for the phase-split refactor.

### 4.7 Explainability
- **SHAP** (`GradientExplainer` — `DeepExplainer` was tried first and failed on `LayerNorm`/custom modules, a known SHAP/Transformer incompatibility) — feature-level attribution, on-demand only (expensive, many forward passes per explanation)
- **Attention weights** — free byproduct of the unified Transformer architecture (4.2)
- Both live in `explainibility+SHAP/`, importing the shared model class from `transformer/model.py`

### 4.8 LLM Explanation Layer (`llm_layer/`)
- **Finalized as prompting-only** — no fine-tuning path (dropped after being originally scoped as optional)
- Takes SHAP + attention output, builds a grounded prompt (includes known attack-signature descriptions per class), calls a single configured LLM provider
- `API_PROVIDER` is the one line to set in `llm_config.py` — **not yet decided**
- API key loaded via `python-dotenv` from a project-root `.env` file (`LLM_API_KEY=...`) — **not yet populated**, `.env` currently has placeholder only
- `.env` is gitignored; `.env.example` committed as a template

### 4.9 Backend (not yet built)
- Was originally scoped as part of `final_model_combined/`; needs re-scoping now that folder is `llm_layer/` and contains only the LLM piece
- **Still needed:** a FastAPI service with `/score` (fast path: Transformer+VAE+DGA+fusion), `/explain` (slow path: SHAP+attention+LLM), and dashboard read endpoints (`/events`, `/events/{id}`, `/robustness`, `/health`)
- **Open question:** which folder this lives in now — needs a decision (new `backend/` folder under `models/`, or elsewhere) before it can be built
- **Database:** SQLite (finalized — MySQL considered since it's a known tool, but SQLite kept for zero-setup, single-file portability; no server process needed for a solo/local demo)

### 4.10 Frontend (`mini_project_ui/`, teammates)
- React + JavaScript, minimal functional dashboard (no styling library, plain CSS)
- Sections: health indicator, events table (fetch `/events`, click row → `/explain`), robustness comparison table
- CORS needs to be enabled on the backend for whatever port the React dev server runs on

### 4.11 Evaluation
- Per-class precision/recall/F1 for the Transformer
- ROC-AUC for the VAE
- Precision/recall for the DGA detector
- Ablation study: Transformer alone vs VAE alone vs DGA alone vs fused
- Adversarial robustness: clean vs FGSM vs PGD, baseline vs hardened

---

## 5. Stretch Goals (only after core is fully working)
1. PCAP replay demo
2. BERT-style masked-flow pretraining
3. Encrypted traffic fingerprinting or C2 beaconing as an additional signal
4. GNN comparison model

---

## 6. Tech Stack

**AI/ML:** Python, PyTorch, scikit-learn, SHAP, custom FGSM/PGD implementation (not a third-party adversarial library), Weights & Biases (planned, not yet set up)

**LLM:** direct API call (Anthropic or OpenAI — undecided), via `python-dotenv` for key management. No fine-tuning.

**Backend:** FastAPI (not yet built — see 4.9), SQLite (finalized)

**Frontend:** React, plain CSS, no component library

**Compute:** College GPU cluster (primary, access request in progress) → Kaggle (30 GPU-hrs/week free tier, checkpoint every epoch to survive 12-hr session cap) → Colab (dev/debug only, backup)

**Dev tooling:** AI coding agent used for file-splitting/refactoring tasks, given detailed step-by-step prompts with a `TASK_PROGRESS.md` checkpoint protocol for long tasks (see Section 10)

---

## 7. Timeline

| Phase | Goal | Status |
|---|---|---|
| Weeks 1-2 | Dataset acquisition + combine/split (flows + domains) | **In progress** |
| Weeks 3-4 | Transformer trained + evaluated | Not started |
| Weeks 5-6 | VAE + DGA detector trained + evaluated | Not started |
| Week 7 | Fusion scoring layer built + evaluated | Not started |
| Week 8 | SHAP + attention + LLM layer integrated | Code scaffolded, provider undecided |
| Weeks 9-10 | Adversarial robustness study | Code scaffolded |
| Week 11 | FastAPI backend + dashboard integration | Backend not yet built (needs re-scoping) |
| Week 12 | Ablation study, polish, report, viva prep | Not started |

---

## 8. Design Decisions Log

- **VAE over plain autoencoder** — principled probabilistic (ELBO) threshold instead of arbitrary reconstruction-error cutoff.
- **Small Transformer, not BERT-scale** — task complexity (78 numeric features, ~15 classes) doesn't warrant more capacity; measured ~1.36 sequences per parameter even at current size, so scaling up risks overfitting without evidence it's needed.
- **Sliding-window sequencing, not strict per-source-IP grouping** — kept general-purpose for the first pass; per-IP grouping is a documented possible refinement, not required for the core deliverable.
- **SHAP `GradientExplainer` over `DeepExplainer`** — `DeepExplainer` doesn't reliably support `LayerNorm`/custom Transformer modules; confirmed via direct test.
- **Unified Transformer architecture** — `transformer/` and `explainibility+SHAP/` share one model class (attention-capturing by default) instead of two incompatible definitions, avoiding a manual weight-copy step.
- **SQLite over MySQL** — despite prior MySQL experience, SQLite's zero-setup/single-file nature fits a solo/local demo better; MySQL would be justified only for concurrent multi-writer production use, which doesn't apply here.
- **LLM layer: prompting only, no fine-tuning** — dropped fine-tuning entirely to reduce scope; a single `API_PROVIDER` variable is the only thing left to decide.
- **`final_model_combined/` renamed to `llm_layer/`** — folder turned out to contain only the LLM explainer, not the full backend as originally scoped; naming corrected to match actual contents. FastAPI backend now needs a new home (open question, Section 4.9).
- **Adversarial folder duplicates the Transformer's data pipeline and model class rather than importing** — keeps the folder self-contained for the phase-split refactor; must be kept in sync manually if the Transformer architecture changes.

---

## 9. Dataset Preparation — Current Task

**Goal:** combine each dataset's raw multi-file form into one file, then split once into train/val/test, so every model folder reads from the same prepared files instead of re-splitting independently.

- **CICIDS2017:** combine all 8 raw CSVs in `models/data/` into `cicids2017_combined.csv`, then stratified split into `cicids2017_train.csv` / `cicids2017_val.csv` / `cicids2017_test.csv` (~70/10/20), using the same random seed already used in `transformer/` and `VAE/` configs — required so the fusion head's signals CSV (Transformer + VAE outputs paired by event) lines up correctly.
- **DGA + Tranco:** combine into `domains_combined.csv` (`domain`, `label` columns, deduped), split into `domains_train.csv` / `domains_val.csv` / `domains_test.csv` (~70/10/20), independent random seed (different data source, no cross-model alignment needed).
- **Model folder updates:** every folder's `config.py`/data-loading code needs to point at the new split files in `models/data/` (relative paths, e.g. `../data/cicids2017_train.csv`) instead of loading and splitting raw data itself.
- **Central weights:** every model folder's training phase saves its final checkpoint to `models/saved_weights/` (e.g. `transformer_final.pt`, `vae_final.pt`, `dga_detector_final.pt`, `fusion_head_final.pt`, `adversarial_hardened_final.pt`) in addition to its own local `outputs/`, so anything downstream (backend, once built) has one reliable place to load from.

**Status:** Completed — datasets combined and split 70/10/20 in `models/data/`, all model configs and loaders updated to relative pre-split paths, central weight-saving enabled in `models/saved_weights/`, top-level `models/train_all.py` created, and `requirements.txt` populated.

**Live checklist:** see `TASK_PROGRESS.md` at the project root for real-time status of these four steps.

---

## 10. Working With a Coding Agent — Conventions

- Long/multi-step prompts include a `TASK_PROGRESS.md` checklist protocol so a token/budget cutoff mid-task doesn't lose work — the agent checks off each step as completed and can resume cleanly in a new session.
- Every prompt explicitly asks for minimal-token responses (no restating the prompt, no verbose commentary between steps).
- Before executing any multi-file change, the agent is asked to show its planned file breakdown / call sequence and wait for confirmation.
- `project_reference.md` (this file) is the source of truth the agent is told to read first on every new task.

---

## 11. Open Items / Not Yet Decided
- LLM `API_PROVIDER` (Anthropic vs OpenAI) — key not yet obtained
- Where the FastAPI backend now lives, since `final_model_combined/` → `llm_layer/` no longer holds it
- College GPU access — request status unconfirmed
- W&B account/project setup — not yet done