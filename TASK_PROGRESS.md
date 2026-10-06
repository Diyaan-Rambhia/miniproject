# Task Progress

## Verified Status (6 October 2026)
- Backend starts; `/health` reports Transformer, VAE, DGA, Fusion, and adversarial model objects loaded.
- Frontend production build passes; `localhost:5173` CORS preflight succeeds; `POST /score` with `flow_sequence` and event-history smoke pass using in-memory persistence.
- Backend tests: `pytest tests/test_main.py -q` passes (2 tests), including separate Transformer/VAE scaler coverage.
- VAE scaler verification: stored DoS Hulk score changed from 3.05068994 to 1.34190059; stored PortScan score changed from 0.10061108 to 0.20533589. The route values match direct inference with the saved VAE scaler.
- Adversarial run is complete: SQLite contains baseline and hardened clean/FGSM/PGD accuracy results. Metrics are recorded in `project_reference.md`.
- `.env` contains a configured placeholder-like value, not a verified valid OpenRouter API key.

## Phase 1: Data Preparation & Stratified Split Integrity
- [x] 1. CICIDS2017 combination — combined 8 CSVs (2,830,743 rows, 78 features) into `cicids2017_combined.csv`
- [x] 2. CICIDS2017 15-class stratified split — 70/10/20 train/val/test split with deterministic rare-class coverage enforcement (`_ensure_class_coverage` & `verify_split_coverage` in `prepare_cicids2017.py`) ensuring all 15 classes are preserved across splits
- [x] 3. DGA + Tranco domain datasets — combined 4,347,492 unique domains and stratified split 70/10/20 into train/val/test with seed 123
- [x] 4. Model data loaders update — updated all model folders (`transformer/`, `VAE/`, `DGA_detector/`, `fusion_head/`, `Adverserial_robustness/`) to read pre-split CSVs via relative paths
- [x] 5. Model artifact saving — Transformer artifacts are loaded centrally; VAE, DGA, and Fusion also maintain component `outputs/` artifacts used by backend configuration

## Phase 2: Core Model Training & Artifact Verification
- [x] 6. Transformer known-attack classifier — trained on corrected 15-class split; serving model and separate attention-capturing explanation mirror load compatible checkpoint weights; Transformer artifacts saved centrally
- [x] 7. VAE zero-day anomaly detector — trained on benign flow traffic; ELBO threshold calibration completed; component output artifacts present; `/score` applies the VAE's own scaler and replay/regression checks pass
- [x] 8. DGA domain detector — bidirectional char-LSTM trained on domain dataset; component output artifacts and checkpoint load in backend
- [x] 9. Fusion head training & bug fixes:
  - [x] Fixed signals CSV generation bug where `generate_signals.py` re-fitted a new scaler on the test set instead of loading `feature_scaler.joblib`
  - [x] Generated multi-signal CSV using upstream model outputs (signal event alignment and split provenance remain under review)
  - [x] Trained Fusion MLP (`fusion_mlp_final.pt`, `signal_scaler.joblib`) and verified runtime threat scoring (0-100); evaluation validity follow-up remains open
- [x] 10. End-to-end training orchestrator — corrected artifact skip names and verified all five existing central checkpoints are detected

## Phase 3: Explainability & LLM Integration
- [x] 11. SHAP + Attention explainers — integrated `GradientExplainer` and a separate attention-capturing Transformer mirror into `models/explainibility + SHAP/`
- [x] 12. Grounded LLM prompting — prompt builder grounds explanations in attack signatures, attention timesteps, and SHAP feature impacts
- [x] 13. LLM provider selection — finalized and configured to OpenRouter (`API_PROVIDER = "openrouter"` using `meta-llama/llama-3.1-8b-instruct:free` via OpenAI client) in `models/llm_layer/llm_config.py`
- [ ] 14. Populate active OpenRouter API key — current `.env` value is placeholder-like and not a valid OpenRouter token; add a valid local secret and verify a live `/explain` provider round trip

## Phase 4: Backend API & Service Integration
- [x] 15. FastAPI backend built (`backend_integration/`):
  - [x] `POST /score` — fast-path event evaluation
  - [x] `POST /explain` — slow-path on-demand SHAP + attention + LLM explanation
  - [x] `GET /events` & `GET /events/{id}` — paginated event retrieval and inspection
  - [x] `GET /robustness` — robustness benchmark retrieval
  - [x] `GET /health` — service health and loaded model diagnostics
- [x] 16. Backend bug fixes:
  - [x] Per-model inference-time scaling — `/score` independently applies the saved Transformer and VAE scalers to raw input
  - [x] Fusion decision logic bug — refactored `/score` so `threat_score >= 50.0` authoritatively determines `predicted_class` (BENIGN vs attack), while `transformer_predicted_class` acts as an informational attack type hint
  - [x] Checkpoint/scaler loading — verified real weights are loaded in `model_loader.py` with graceful fallback handling
- [x] 17. SQLite persistence — schema created and initialized for `events` and `robustness_results`
- [x] 18. Backend test coverage — `pytest tests/test_main.py -q` passes (2 tests); `pytest` and `httpx` are declared in requirements

## Phase 5: Adversarial Robustness Study
- [x] 19. Attack implementations — implemented FGSM (single-step) and PGD (multi-step) perturbations in `models/Adverserial_robustness/`
- [x] 20. Adversarial training defense — retraining loop implemented on clean + FGSM mixture (`ADV_TRAIN_MIX_RATIO = 0.5`)
- [x] 21. Adversarial training verification — hardened checkpoint loads and SQLite contains baseline and hardened clean/FGSM/PGD accuracy results (metrics in `project_reference.md` Section 4.6).

## Phase 6: Frontend Integration & Polish
- [x] 22. React dashboard integration — frontend production build and lint pass; health/CORS verified; score payload corrected to `flow_sequence`; valid score and event-history smoke passed using in-memory persistence
- [x] 23. Final documentation wrap-up — updated implementation reference, SRS, UML, README status, and this checklist

## Remaining Work / Risks
- [ ] 24. Correct Fusion signal provenance — align DGA predictions with corresponding flow events; current generator joins separate flow/domain test outputs by row position/resizing and reuses test-derived signals in Fusion training/evaluation. Rebuild, retrain, and report a clean held-out evaluation.
- [ ] 25. Replace synthetic explanation fallback attribution values or mark them explicitly unavailable; current hard-coded fallback can be mistaken for event-derived SHAP/attention.
- [ ] 26. Persist and expose macro-F1 for robustness results if required by the final report; the current SQLite/API contract stores accuracy only.
- [ ] 27. Add production security controls before external exposure: replace wildcard CORS, add authentication/authorization, bound request sizes, and define raw-event retention/access controls.
- [ ] 28. Review or archive the unused `mini/` frontend duplicate after confirming no teammate depends on it; `mini_project_ui/` is the active app.
- [ ] 29. Persist optional domains on scored events and include them in event history if the dashboard’s Domain column is expected to show submitted values.
