# Network Security — Layered AI Defense Pipeline — Project Reference

**Duration target:** ~9-12 weeks core build (plan for an August start given fixed October end date)
**Team:** You (AI/ML + backend + adversarial study) + teammates (frontend/dashboard)

---

## 1. Problem Statement

Build a **layered, multi-signal network defense pipeline** — not a single classifier, but a small system with multiple detection signals feeding a fusion layer, an explanation layer on top, and a study of how robust the system is to adversarial evasion. This mirrors how real security products use defense-in-depth rather than one model doing everything.

The system:
1. Classifies **known attacks** from network flow data (supervised).
2. Detects **unknown/zero-day** anomalies from flow data (unsupervised).
3. Flags **malicious domains** (DGA) as a separate, independent signal.
4. Fuses all signals into one **threat score**.
5. **Explains** every flagged event, both at the feature level and in plain English.
6. Is **tested against adversarial evasion** attempts, with a defense applied and evaluated.

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
   (MLP / logistic on all
     3 signals combined)
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
       Dashboard (Frontend)


Parallel study (separate track, same detector):
       Baseline Detector
             │
     Adversarial Perturbation
     (craft evasive flows)
             │
     Evaluate detection drop
             │
     Adversarial Training (defense)
             │
     Re-evaluate robustness
```

---

## 3. Core Components

### 3.1 Data
- **Dataset (flows):** CICIDS2017 (multiple CSVs split by day/attack type)
- **Dataset (domains):** public DGA dataset (malicious domains) + legitimate domain list (e.g. Alexa/Tranco top domains) for the DGA detector
- **Preprocessing (flows):** clean nulls/infs, normalize numeric features, encode categorical features, handle class imbalance
- **Flow Sequencing:** group flows by source IP into time-windowed sequences — enables the Transformer to learn temporal attack patterns, not just isolated points
- **Dev subsample:** stratified smaller subsample for fast iteration; full dataset only for final training runs

### 3.2 Known-Attack Classifier — Transformer Encoder
- Input: sequence of flows (per source IP, windowed)
- Architecture: small Transformer encoder — 2-4 layers, hidden dim 128-256
- Output: multi-class classification (benign / attack type)
- Attention weights double as a free explainability channel

### 3.3 Unknown/Zero-Day Detector — VAE
- Trained **only on benign traffic**
- Encoder → latent (μ, σ) → sample z → decoder → reconstruction
- Loss: reconstruction error + KL divergence (ELBO)
- Anomaly score: high ELBO loss = doesn't look like normal traffic
- VAE over plain AE: principled probabilistic threshold, better justified in report/viva

### 3.4 DGA Domain Detector
- Input: domain name as a character sequence
- Architecture: char-level LSTM or small Transformer (same family as your other sequence models — reuse patterns/code)
- Output: malicious (DGA-generated) vs legitimate domain
- Independent signal — doesn't depend on flow data, adds resilience if flow-based detection misses something

### 3.5 Fusion Scoring Layer
- Combine: Transformer classification confidence + VAE anomaly score + DGA detector output
- Small MLP or logistic regression on top of all three signals
- Output: single Threat Score (0–100)

### 3.6 Explainability (two layers)
- **SHAP** on the Transformer classifier — which input features drove a classification
- **Attention weight visualization** — which flows in the sequence mattered
- **LLM Explanation Layer** — takes the SHAP/attention output and threat score, generates a plain-English explanation of why an event was flagged (e.g. "Flagged due to high connection frequency and unusual destination port, consistent with SYN flood pattern"). Can be a fine-tuned small model or a well-prompted call to an existing LLM API — the value is in understanding and designing the prompting/fine-tuning, not just wiring an API call.

### 3.7 Adversarial Robustness Study
- Take the trained baseline detector (Transformer + VAE + fusion)
- Craft adversarially perturbed flows designed to evade detection (small feature perturbations that flip the model's decision)
- Measure how much detection performance drops under attack
- Apply a defense (adversarial training — retrain including perturbed examples)
- Re-evaluate: how much robustness is recovered, what's the accuracy/robustness tradeoff
- This becomes a dedicated section of your report/evaluation, not just a side note — it changes your project's narrative from "we detect attacks" to "we detect attacks, and we understand and test the limits of our own detector"

### 3.8 Backend
- **FastAPI** service exposing:
  - Inference endpoint (flow/sequence + domain in → threat score + explanation out)
  - Endpoint for historical scored data (dashboard charts)
  - Endpoint for explanation details per flagged event
  - Endpoint exposing adversarial robustness evaluation results (for a dedicated dashboard section/report figure)

### 3.9 Frontend (Team)
- Dashboard consuming the FastAPI backend
- Core views:
  - Live/recent threat score feed
  - Attack type breakdown (charts)
  - Top attacking IPs / flagged domains
  - Click into a flagged event → see SHAP + attention + LLM explanation
  - Timeline/heatmap of attack activity
  - Adversarial robustness summary view (before/after defense comparison)

### 3.10 Evaluation
- Per-class precision/recall/F1 for the Transformer classifier
- ROC-AUC for the VAE anomaly detector
- Precision/recall for the DGA detector
- Ablation study: Transformer alone vs VAE alone vs DGA alone vs fused
- Adversarial robustness: detection performance before vs after attack, before vs after defense

---

## 4. Stretch Goals (only after everything above is fully working)

In priority order:
1. PCAP replay demo — replay stored traffic through the trained pipeline for a "live" feel
2. BERT-style masked-flow pretraining before classification fine-tuning
3. Encrypted traffic fingerprinting or C2 beaconing detection as an additional signal
4. GNN-based model on host communication graph, compared against the Transformer

---

## 5. Tech Stack

**AI / ML**
- Python
- PyTorch (model implementation, training)
- PyTorch Lightning (optional, cleans up training loop)
- Scikit-learn (preprocessing, metrics, baseline comparisons)
- SHAP (feature-level explainability)
- Adversarial robustness library (e.g. Adversarial Robustness Toolbox / Foolbox) for the perturbation + defense study
- Weights & Biases (experiment tracking, free tier — survives interrupted GPU sessions)

**LLM Explanation Layer**
- Small fine-tuned model, or well-prompted call to an existing LLM API for generating plain-English explanations from SHAP/attention output

**Backend**
- FastAPI
- PostgreSQL or SQLite (store scored events, logs, robustness evaluation results)

**Frontend**
- React (or Streamlit for faster stand-up, React more presentable for final demo)
- Charting library (Recharts / Chart.js)

**Compute**
- Primary: college GPU cluster (push for reliable multi-hour session access via mentor)
- Backup: Kaggle (30 GPU-hrs/week free tier, 12-hr session cap — checkpoint every epoch)
- Dev/debug only: Colab free tier
- Coding assistants (Copilot/Codex/CC) for boilerplate, scaffolding, debugging — not a substitute for understanding training/evaluation results

**Deployment**
- Docker (containerize backend + model serving, optional but nice for demo reproducibility)

---

## 6. Compute Strategy

- Training is short-session friendly — no run needs to survive more than a few hours uninterrupted at this model scale
- Use mixed precision training (`torch.cuda.amp`)
- **Checkpoint every epoch, always**
- Log all runs to W&B
- Prototype/debug on subsampled dataset; scale to full dataset only once pipeline is verified
- Estimate ~15-35 GPU-hours for the core Transformer + VAE + DGA models combined (excluding adversarial training runs, which are typically fast since they reuse the trained baseline)

---

## 7. Timeline

| Phase | Goal |
|---|---|
| Weeks 1-2 | Dataset acquisition + preprocessing (flows + DGA domains), flow sequencing built |
| Weeks 3-4 | Transformer classifier trained + evaluated on subsample, then full dataset |
| Weeks 5-6 | VAE trained + evaluated, DGA detector trained + evaluated |
| Week 7 | Fusion scoring layer built and evaluated |
| Week 8 | SHAP + attention explainability, LLM explanation layer integrated |
| Weeks 9-10 | Adversarial robustness study: perturbation, evaluation, defense, re-evaluation |
| Week 11 | FastAPI backend complete, dashboard integration with team |
| Week 12 | Final ablation study, polish, report writing, viva prep, buffer |

*(This is ~12 weeks; if you start in August rather than immediately, expect the buffer at the end to shrink — plan your start date backward from October with this in mind.)*

---

## 8. Steps From Scratch

1. **Set up environment**
   - Python env, install PyTorch, scikit-learn, pandas, SHAP, an adversarial robustness library, FastAPI, W&B
   - Set up Kaggle/Colab backup compute; start the college GPU access conversation with your mentor early

2. **Get and explore the datasets**
   - Download CICIDS2017 (all daily CSVs)
   - Download/prepare a DGA dataset + legitimate domain list
   - EDA on both: class distribution, missing values, feature ranges

3. **Preprocessing pipeline (flows)**
   - Clean nulls/infinities, encode categorical fields, normalize numeric features
   - Handle class imbalance
   - Build stratified dev subsample

4. **Flow sequencing**
   - Group flows by source IP, build time-windowed sequences
   - Prepare sequence tensors + labels for the Transformer

5. **Baseline sanity check**
   - Train a quick classical model on the dev subsample to confirm pipeline/labels are correct before investing in the Transformer

6. **Transformer classifier**
   - Implement small Transformer encoder in PyTorch
   - Train on dev subsample, tune, then scale to full dataset
   - Evaluate: per-class precision/recall/F1, confusion matrix

7. **VAE anomaly detector**
   - Filter to benign-only flows
   - Implement encoder-decoder VAE, train, determine anomaly threshold
   - Evaluate: ROC-AUC

8. **DGA domain detector**
   - Preprocess domain strings into character sequences
   - Train char-level LSTM/Transformer classifier
   - Evaluate: precision/recall

9. **Fusion scoring layer**
   - Combine all three signals (Transformer confidence + VAE anomaly score + DGA output)
   - Train small MLP/logistic fusion head
   - Produce final 0–100 threat score

10. **Explainability**
    - Add SHAP on the Transformer classifier
    - Extract/visualize attention weights
    - Build the LLM explanation layer (fine-tune small model or design prompting) that turns SHAP/attention output into plain-English explanations

11. **Adversarial robustness study**
    - Craft adversarial perturbations against the trained baseline detector
    - Measure detection performance drop under attack
    - Apply adversarial training as a defense
    - Re-evaluate robustness, document the accuracy/robustness tradeoff

12. **Backend (FastAPI)**
    - Build inference endpoint (flow/sequence + domain → threat score + explanation)
    - Build endpoints for historical data, explanation lookup, and robustness evaluation results
    - Set up database to store scored events

13. **Frontend integration (with team)**
    - Connect dashboard to FastAPI endpoints
    - Build views: live feed, attack breakdown, top attackers/flagged domains, explanation drill-down, timeline/heatmap, adversarial robustness summary

14. **Evaluation & ablation study**
    - Document: Transformer-only vs VAE-only vs DGA-only vs fused performance
    - Document: pre-attack vs post-attack vs post-defense robustness results
    - Compile all metrics into tables/charts for the report

15. **Stretch goals (if time remains)**
    - PCAP replay demo
    - Masked-flow pretraining
    - Encrypted traffic fingerprinting or C2 beaconing as an additional signal
    - GNN comparison model

16. **Final steps**
    - Full-scale final training runs
    - Dockerize backend (optional but recommended)
    - Write project report
    - Prepare viva — be ready to explain: why Transformer over CNN/RF, why VAE over plain AE, why DGA detection matters as an independent signal, what the fusion layer does, what the adversarial study revealed, and how the explainability chain works end to end
