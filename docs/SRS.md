# Software Requirements Specification

## Layered AI Network Defense Pipeline

**Document status:** Implementation-based specification  
**System version:** Backend API 1.0.0  
**Prepared:** 6 October 2026

## Document Control

This Software Requirements Specification (SRS) describes the network-defense application represented by the current repository. It is based on the source code and the Design Decisions Log in [project_reference.md](../project_reference.md), not solely on the initial architecture proposal. Requirements marked **Current behavior** describe verified implementation behavior. Requirements marked **Acceptance target** state measurable quality goals that are proposed for evaluation and must not be mistaken for benchmark results already demonstrated by the project.

## Contents

1. [Introduction](#1-introduction)
   1. [Purpose](#11-purpose)
   2. [Scope](#12-scope)
   3. [Intended Audience](#13-intended-audience)
   4. [Definitions and Acronyms](#14-definitions-and-acronyms)
   5. [References](#15-references)
2. [Overall Description](#2-overall-description)
   1. [Product Perspective](#21-product-perspective)
   2. [Product Functions](#22-product-functions)
   3. [User Characteristics](#23-user-characteristics)
   4. [Constraints](#24-constraints)
   5. [Assumptions and Dependencies](#25-assumptions-and-dependencies)
3. [Specific Requirements](#3-specific-requirements)
   1. [Functional Requirements](#31-functional-requirements)
   2. [External Interface Requirements](#32-external-interface-requirements)
   3. [Non-Functional Requirements](#33-non-functional-requirements)
4. [Requirements Verification Summary](#4-requirements-verification-summary)
5. [Implementation Limitations and Known Integration Gaps](#5-implementation-limitations-and-known-integration-gaps)

# 1. Introduction

## 1.1 Purpose

This SRS defines the functional scope, interfaces, operating assumptions, quality requirements, and verification basis for a layered network-traffic defense application. The system combines supervised flow classification, unsupervised anomaly scoring, domain-generation-algorithm (DGA) classification, learned signal fusion, event persistence, feature-level and natural-language explanations, and an adversarial robustness evaluation workflow.

The document is intended to guide implementation review, academic assessment, integration testing, demonstrations, and future maintenance. It describes the implementation as it exists in the repository, including known limitations, rather than implying that proposed capabilities are complete or production-hardened.

## 1.2 Scope

The system accepts a sequence of numeric network-flow feature vectors and, optionally, a domain name. A FastAPI service preprocesses available input, obtains signals from the loaded detectors, combines the signals into a threat score when possible, persists scored events in SQLite, and provides operational endpoints for event history, explanations, health, and robustness results. A React/Vite dashboard presents runtime state, event data, explanations, and adversarial comparison data.

The model and study scope is:

- A Transformer encoder classifies a flow sequence into one of the trained traffic classes. The class mapping includes BENIGN and attack classes.
- A variational autoencoder (VAE), trained on benign traffic, supplies an anomaly score for the final flow in the submitted sequence.
- A character-level bidirectional LSTM supplies a malicious-domain probability when a domain and a usable DGA model are present.
- A Fusion MLP maps the three signals to an attack probability expressed on a 0–100 threat-score scale. The Fusion score, not the Transformer argmax, is authoritative for the final BENIGN/attack decision.
- When all three signal sources are usable but the Fusion checkpoint or scaler is missing, the backend applies the documented weighted heuristic. When only some signal sources are usable, it calculates a renormalized partial heuristic score. When none are usable, it returns an `insufficient_models` result.
- An on-demand explanation combines attention-derived timestep importance and optional SHAP feature attributions, then sends grounded inputs to the configured LLM provider for plain-English text. The endpoint has local fallbacks if attribution or LLM generation fails.
- An offline adversarial study evaluates baseline and adversarially trained Transformer models under clean, FGSM, and PGD conditions. Results can be persisted to the same SQLite database and retrieved by the dashboard API.

The application is a research and demonstration system. It is not specified as a network sensor, packet-capture processor, inline blocking appliance, or production security control. No packet-capture ingestion endpoint is present in the listed backend routes. Flow construction and domain extraction from PCAP or live traffic are outside the implemented API boundary.

## 1.3 Intended Audience

- **Academic reviewers and instructors:** assess requirements, architecture, implementation correspondence, and evaluation limitations.
- **Security analysts and demonstrators:** submit or inspect flow-scoring results, review event history, request explanations, and examine robustness results.
- **Developers and maintainers:** integrate the UI and API, operate model artifacts, test behavior, and extend the research pipeline.
- **System operators:** configure the backend, local database, model files, and optional LLM credential for a local demonstration.

## 1.4 Definitions and Acronyms

| Term | Definition |
|---|---|
| API | Application Programming Interface. |
| Attack type hint | The Transformer’s predicted class, retained as informative context; it does not control the final benign/attack decision. |
| Attention | Transformer self-attention weights summarized to indicate influential sequence timesteps. These weights are explanatory signals, not causal proof. |
| DGA | Domain Generation Algorithm; in this project, the detector classifies a domain string as legitimate or DGA-like. |
| Event | A scored request persisted to the SQLite `events` table, with a generated identifier, timestamp, signal values, decision, and raw sequence. |
| FGSM | Fast Gradient Sign Method, a one-step gradient-based perturbation used in the robustness study. |
| Flow sequence | A two-dimensional array of numeric feature rows representing an ordered sequence of flows. Training defaults use a sliding window of 10 rows and stride 5; the API does not enforce that exact length. |
| Fusion score / threat score | The Fusion MLP’s estimated attack probability multiplied by 100; if the trained Fusion path is unavailable, a documented weighted heuristic supplies the score. |
| LLM | Large Language Model. The configured explanation provider is OpenRouter, accessed through an OpenAI-compatible client. |
| PGD | Projected Gradient Descent, an iterative perturbation method used in robustness evaluation. |
| SHAP | SHapley Additive exPlanations, used here through `GradientExplainer` to produce feature attributions. |
| VAE | Variational Autoencoder, trained on benign flows and used to calculate a reconstruction-plus-KL anomaly score. |
| Zero-day / unknown anomaly | A flow whose feature pattern differs from the benign patterns represented during VAE training. The project’s anomaly score is not by itself the final application decision. |

## 1.5 References

- [Project reference and Design Decisions Log](../project_reference.md)
- [Backend application and route implementation](../backend_integration/main.py)
- [Pydantic API schemas](../backend_integration/schemas.py)
- [Model and artifact loader](../backend_integration/model_loader.py)
- [Flow inference preprocessing](../backend_integration/inference_preprocess.py)
- [Current React dashboard API client](../mini_project_ui/src/api.js)
- [Core model and evaluation pipelines](../models/run_core_pipeline.py) and [adversarial pipeline](../models/run_adversarial_pipeline.py)

# 2. Overall Description

## 2.1 Product Perspective

The application is a local, layered machine-learning defense prototype organized into three principal areas:

1. **`models/`** contains training and evaluation code, model definitions, explainability utilities, LLM prompting, datasets, and model artifacts. These include supervised flow classification, benign-only anomaly detection, DGA detection, Fusion, and a separate adversarial robustness experiment.
2. **`backend_integration/`** contains the FastAPI service, artifact-loading container, inference preprocessing, request/response schemas, SQLite operations, and the `/score`, `/explain`, dashboard, health, and robustness routes.
3. **`mini_project_ui/`** contains the React/Vite dashboard and browser-side API client. It consumes operational endpoints and has demonstration data/fallback behavior in the main experience.

The backend loads artifacts at application startup through a process-wide `ModelContainer`. Transformer artifacts are resolved from `models/saved_weights/`; VAE, DGA, and Fusion artifacts are resolved from their respective `outputs/` folders by the current backend configuration. The adversarial checkpoint is searched in the central saved-weights location and then its robustness output location. Missing or incompatible checkpoints are logged and represented as unavailable model components rather than preventing all service initialization.

The flow-scoring request path is sequential in the current implementation: preprocess the sequence, run the Transformer if loaded, run the VAE if loaded, run the DGA detector when a domain and DGA artifacts are available, then calculate the Fusion or fallback score and persist the event. The `/explain` route independently reloads an attention-capturing Transformer representation from the Transformer checkpoint for explanation. It is not the same runtime class object used for scoring, although it is designed to load compatible checkpoint weights.

The browser dashboard does not directly load model files. It calls the backend over HTTP. The dashboard’s current scoring request body does not match the backend’s request schema; this is a known integration issue described in Section 5.

## 2.2 Product Functions

The product provides or is intended to provide the following capabilities:

- Accept numeric flow sequences and optionally a domain name for scoring.
- Scale flow features with the saved training-time Transformer scaler and feature ordering when those artifacts are available.
- Produce independent Transformer, VAE, and DGA signals when the corresponding model inputs and artifacts are available.
- Produce a 0–100 threat score and derive the final BENIGN/attack decision from the 50.0 Fusion attack threshold.
- Preserve the Transformer’s predicted class separately as an attack-type hint.
- Record score results and the original flow sequence in SQLite for later explanation and history retrieval.
- Return paginated event history and individual stored-event details.
- Generate an on-demand explanation with attention, optional SHAP, and a grounded LLM response, with fallbacks for unavailable components.
- Report backend/model availability and retrieve stored adversarial robustness metrics.
- Show live or demonstration operational data through the dashboard.
- Run an offline baseline-versus-hardened model study under clean, FGSM, and PGD evaluation conditions.

## 2.3 User Characteristics

The primary human user is an analyst, student, or evaluator familiar with basic network-security terminology and the interpretation of model-generated risk indicators. The user is expected to provide valid numeric flow features in the model’s expected feature order. The application does not currently provide a packet-capture parser or a user-facing feature-name-to-value form, so raw API and sample-sequence use may require technical assistance.

Developers and operators are expected to understand Python, the project’s model artifacts, FastAPI, SQLite, and React/Vite configuration. Operators configuring the LLM explanation capability must manage the provider credential through the project-root `.env` file and must not expose it in browser code or version control.

## 2.4 Constraints

- **Input representation:** `/score` accepts a Pydantic `flow_sequence` list of numeric rows and an optional string `domain`. `scale_flow_sequence` requires a two-dimensional matrix and, when saved feature names are loaded, requires the corresponding feature count. The route does not itself enforce exactly 10 rows. The UI currently expects a 10 × 78 sequence, matching the project’s default Transformer window and present feature count, but that is a UI-side expectation rather than a complete API schema constraint.
- **Feature order:** flow columns must use the feature order represented by `feature_names.joblib` and the associated saved scaler. A numerically valid sequence in a different order can produce invalid model outputs without being recognized as a semantic ordering error.
- **Inference model inputs:** the DGA signal requires a domain and a loaded vocabulary/model. Without a domain, DGA is not counted as an available signal for the complete trained-fusion path.
- **Model availability:** runtime depends on compatible PyTorch checkpoints and associated `joblib` artifacts at configured paths. The backend can start with missing components but scoring accuracy and status degrade accordingly.
- **Storage:** event and robustness records use a local SQLite database. This is appropriate for a single-process demonstration and is not specified as a horizontally scaled or high-concurrency database architecture.
- **External service:** plain-English explanations depend on access to the configured LLM provider and a valid `LLM_API_KEY`; local attributions can still be returned if LLM generation fails.
- **Adversarial evaluation:** FGSM/PGD perturbations are generated against scaled model inputs in the research pipeline. Results are experimental metrics and do not constitute proof of robustness against real-world attacks.
- **Deployment:** the configuration selects CUDA when PyTorch reports it available and otherwise uses CPU. The present documentation assumes a local service deployment and does not define a production cloud deployment.

## 2.5 Assumptions and Dependencies

- Training and inference use compatible feature definitions, saved scalers, label mappings, vocabularies, and checkpoints.
- The backend is started from an environment with the packages declared by the repository’s requirements and the project root available on Python’s import path.
- SQLite is writable at `backend_integration/events.db`.
- The frontend is configured with the correct backend base URL and the browser can reach that backend.
- The OpenRouter-compatible LLM endpoint may be unavailable or rate-limited; the application must treat text generation as optional and retain the local explanation fields where available.
- The current reference records adversarial training verification and valid `.env` LLM-key population as open items. The corresponding artifacts and endpoint paths exist, but their presence alone does not establish that an evaluation run has completed or that a valid provider key is configured.
- Operational accuracy, fairness, real-world generalization, and robustness are not inferred from code structure; they require separate experimental evidence.

# 3. Specific Requirements

## 3.1 Functional Requirements

### 3.1.1 Flow Classification

**FR-FLOW-01.** The system shall accept a two-dimensional numeric flow sequence through `POST /score` using the `flow_sequence` field.

**FR-FLOW-02.** When Transformer model and preprocessing artifacts are available, the backend shall validate the matrix dimensionality and expected feature count, apply the saved feature scaler, and run the sequence through the Transformer classifier.

**FR-FLOW-03.** The Transformer shall produce class logits over the classes represented by its loaded label encoder. The backend shall derive the informational Transformer class from the maximum-probability class and derive `transformer_confidence` as one minus the BENIGN probability when BENIGN exists in the class mapping.

**FR-FLOW-04.** The backend shall preserve the Transformer class as `transformer_predicted_class` and shall not use that class alone to decide whether the event is BENIGN or an attack.

**Current behavior:** the Transformer is a three-layer encoder configuration in the deployed loader (`d_model=128`, `nhead=4`, feed-forward width 256, dropout 0.1), with sinusoidal positional encoding and a classifier applied to the last encoded sequence position. Its input feature count and output class count are derived from loaded artifacts.

### 3.1.2 Anomaly Detection

**FR-ANOM-01.** When the VAE and its checkpoint are available, the backend shall calculate an anomaly signal from the final flow row of the scaled flow sequence.

**FR-ANOM-02.** The VAE anomaly signal shall be calculated from per-sample reconstruction mean-squared error plus the configured weighted KL divergence (`KL_WEIGHT=0.5` in the scoring path).

**FR-ANOM-03.** The system shall expose the resulting numeric value as `vae_anomaly_score`. The anomaly value shall be treated as a Fusion input, not as the authoritative application-level BENIGN/attack label.

**Current behavior:** the VAE is a feed-forward encoder/latent sampler/decoder with a 16-dimensional latent representation and hidden width 64 in the backend loader. `/score` independently applies the saved VAE scaler to the raw sequence before VAE inference and applies the Transformer scaler separately before Transformer inference. The VAE threshold artifact is loaded but does not set the final class; Fusion remains authoritative. Regression coverage verifies separate model inputs, and stored DoS Hulk/PortScan replays confirm that API anomaly scores match direct inference with the VAE scaler.

### 3.1.3 DGA Detection

**FR-DGA-01.** The scoring request shall permit an optional domain string.

**FR-DGA-02.** If a domain, DGA classifier, and character vocabulary are available, the backend shall lowercase the domain, encode up to the configured maximum length, pad shorter inputs, run the DGA classifier, and expose the DGA probability as `dga_probability`.

**FR-DGA-03.** When no usable domain signal exists, the system shall mark scoring as partial and shall not represent the DGA probability as an independently observed prediction. The current response schema still serializes the initialized numeric default, so clients must interpret it together with `status` and `message`.

**Current behavior:** the DGA model is a character embedding, bidirectional LSTM, and binary classification head. The backend uses class index 1 as the DGA probability.

### 3.1.4 Fusion Scoring and Decision

**FR-FUSE-01.** When Transformer, VAE, and DGA signals are all available and both the trained Fusion checkpoint and signal scaler are loaded, the backend shall scale the signal vector and evaluate the Fusion MLP.

**FR-FUSE-02.** The trained Fusion threat score shall equal the MLP softmax probability of the attack class multiplied by 100, yielding a value in the range 0–100.

**FR-FUSE-03.** The backend shall determine the final class using `FUSION_ATTACK_THRESHOLD=50.0`: a score greater than or equal to the threshold is an attack decision; a lower score is BENIGN.

**FR-FUSE-04.** For an attack decision, the backend shall use the Transformer’s non-BENIGN class as an attack-type label when available; otherwise, it shall use `ATTACK`. For a score below threshold, it shall return `BENIGN`, irrespective of the Transformer argmax.

**FR-FUSE-05.** When model signals are available but a complete trained-Fusion path is unavailable, the backend shall use a documented heuristic and report the applicable degradation in the response. The full-signal fallback weights are Transformer 0.50, VAE 0.25, and DGA 0.25; the VAE input is capped at 2.0 and the weighted score is normalized to active weights and converted to a 0–100 range.

**FR-FUSE-06.** When only a subset of signal sources is available, the backend shall calculate a partial heuristic over active signals, renormalize by the active weights, and return `status="partial"` with a message identifying usable signal sources. When no Transformer, VAE, or usable DGA signal is available, it shall return `status="insufficient_models"` and no threat score.

**Current behavior:** the trained Fusion MLP is a two-hidden-layer ReLU network with hidden width 16. The decision rule is Fusion-driven; Transformer class is informational/labeling context only.

### 3.1.5 Explainability

**FR-EXP-01.** The system shall allow an analyst to request an explanation for a previously stored event by sending its `event_id` to `POST /explain`.

**FR-EXP-02.** If the event exists and has a stored raw sequence, the system shall parse and scale that sequence using the Transformer preprocessing artifacts when available.

**FR-EXP-03.** The explanation process shall return predicted class and confidence, a list of top attended timesteps, a list of top SHAP feature attributions when SHAP is available, and a plain-English explanation field.

**FR-EXP-04.** The system shall use the attention-capturing explanation model to summarize attention from the last sequence position, averaged across heads for the final encoder layer.

**FR-EXP-05.** When SHAP is available, the system shall use `GradientExplainer` and map influential feature values back to feature names and sequence timesteps. SHAP absence or failure shall not prevent the endpoint from attempting to return other explanation fields.

**FR-EXP-06.** The LLM layer shall construct a grounded prompt using the predicted class, confidence, attack-signature description, attention summary, and SHAP summary where available. The generated response shall not be represented as a guaranteed causal explanation.

**FR-EXP-07.** If the attribution pipeline fails, the current route shall return its local fallback attribution payload. If the external LLM call fails or credentials are absent, it shall return a plain-English unavailability message while preserving the other response fields when possible.

**Current behavior:** the explanation route rebuilds/imports explainability helpers at request time and loads an attention-capturing Transformer mirror from the configured Transformer checkpoint. Its synthetic attribution fallback is a continuity mechanism and must not be treated as a verified explanation of the input.

### 3.1.6 Adversarial Robustness Study

**FR-ROB-01.** The research pipeline shall support single-step FGSM perturbation and iterative PGD perturbation of Transformer inputs within configured perturbation limits.

**FR-ROB-02.** The adversarial training procedure shall begin from baseline Transformer weights and train on a configured mix of clean and FGSM-perturbed batches.

**FR-ROB-03.** The evaluation workflow shall compare baseline and hardened models on clean inputs, FGSM inputs, and PGD inputs, calculating accuracy and macro-F1 in the offline evaluation code.

**FR-ROB-04.** The evaluation workflow shall persist run identifier, timestamp, model name, and clean/FGSM/PGD accuracy to the SQLite `robustness_results` table.

**FR-ROB-05.** `GET /robustness` shall return the stored comparison data. If no evaluation data is present, it shall return an empty results list.

**Current behavior:** the database/API persists and exposes accuracy only. The route currently returns `clean_f1`, `fgsm_f1`, and `pgd_f1` as `null`, even though the offline evaluator calculates macro-F1. The project reference identifies complete final adversarial verification as pending.

### 3.1.7 Backend API and Persistence

**FR-API-01.** The backend shall expose health, scoring, explanation, event-history, event-detail, and robustness endpoints described in Section 3.2.

**FR-API-02.** On startup, the backend shall initialize the SQLite schema and attempt to load configured model artifacts. An unavailable individual model shall not, by itself, prevent the service from starting.

**FR-API-03.** A score request that produces a score shall receive a generated event identifier and timestamp and shall persist the score fields, final class, Transformer hint, and original sequence.

**FR-API-04.** The event-history endpoint shall accept bounded pagination using `limit` and `offset` query parameters.

**FR-API-05.** The backend shall return an HTTP 404 for an unknown event requested by event ID or explanation request. Invalid flow dimensionality/feature count shall be reported as HTTP 400 by `/score` or `/explain` where applicable.

### 3.1.8 Dashboard

**FR-UI-01.** The dashboard shall display backend reachability and model availability from `/health`.

**FR-UI-02.** The dashboard shall present paginated event history and permit a user to request an explanation for an event with an identifier.

**FR-UI-03.** The dashboard shall display robustness comparison results returned by `/robustness`, with unavailable accuracy values rendered as missing rather than fabricated values.

**FR-UI-04.** The dashboard shall permit manual flow-sequence JSON entry and optional domain entry for scoring, and shall send the backend-compatible `flow_sequence` field. The active UI now matches the FastAPI request schema; valid score and event-history requests were smoke-tested with isolated in-memory persistence.

**FR-UI-05.** The main dashboard experience may use demo event and robustness data when the backend is unavailable or when demo mode is selected. Demo values shall be visually distinguishable from live backend results.

## 3.2 External Interface Requirements

### 3.2.1 General API Conventions

- Base URL is configurable. The backend’s local development entry point uses `http://127.0.0.1:8000`.
- Request/response bodies use JSON where applicable.
- The FastAPI application declares version `1.0.0` and can expose generated OpenAPI documentation through the framework’s standard documentation routes.
- The API currently configures permissive CORS origins, including `*`; this is a development configuration, not an acceptable production security boundary.
- No authentication mechanism is implemented for the backend routes in the inspected code.

### 3.2.2 `POST /score`

**Purpose:** Score a numeric flow sequence and optional domain; persist a scored event when at least one usable signal exists.

**Request body:**

```json
{
  "flow_sequence": [
    [0.12, 4.8, 1.0],
    [0.18, 4.4, 0.0]
  ],
  "domain": "sample.example"
}
```

`flow_sequence` is required and has schema type `List[List[float]]`. `domain` is optional. The illustrative rows above are not the actual model’s feature width; runtime feature count must match loaded feature artifacts. The model training default is a sequence length of 10 with 78 numeric features, but the route validates two-dimensional shape and expected feature count rather than enforcing a fixed 10-row length.

**Successful score response (representative):**

```json
{
  "event_id": "a1b2c3d4",
  "timestamp": "2026-10-06 12:30:00",
  "threat_score": 73.4,
  "predicted_class": "PortScan",
  "transformer_predicted_class": "PortScan",
  "transformer_confidence": 0.81,
  "vae_anomaly_score": 0.72,
  "dga_probability": 0.13,
  "status": "scored",
  "message": null
}
```

The numeric values and event identifier are illustrative only. `predicted_class` reflects the Fusion threshold decision. When that decision is an attack and the Transformer’s class hint is BENIGN, the route returns `ATTACK` as the final class. Possible statuses are `scored`, `partial`, and `insufficient_models`. Partial and insufficient responses use the response model’s optional fields; an insufficient response has no score/event identifier and includes an explanatory message.

**Responses and errors:**

- `200 OK`: score result, partial score, or `insufficient_models` status. Lack of usable models is currently represented in the JSON response rather than an HTTP error.
- `400 Bad Request`: input is not a two-dimensional matrix or has the wrong expected feature count.
- `422 Unprocessable Entity`: request does not satisfy Pydantic’s body/type validation.

### 3.2.3 `POST /explain`

**Purpose:** Generate an explanation for an already persisted event.

**Request body:**

```json
{
  "event_id": "a1b2c3d4"
}
```

**Representative response:**

```json
{
  "event_id": "a1b2c3d4",
  "predicted_class": 4,
  "confidence": 0.81,
  "top_attended_timesteps": [[9, 0.42], [7, 0.31], [4, 0.18]],
  "top_shap_features": [
    {"timestep": 9, "feature": "Flow Duration", "shap_value": 0.42}
  ],
  "plain_english_explanation": "The sequence contains patterns associated with ..."
}
```

The class value may be an integer class index or string according to the response schema and fallback path. SHAP features may be empty when SHAP is unavailable or not computed. Attention entries are currently serialized as timestep/weight pairs. Exact output depends on available artifacts and providers.

**Responses and errors:** `200 OK` on a completed or degraded explanation; `404 Not Found` for an unknown event; `400 Bad Request` when a stored event lacks a raw sequence or the sequence is invalid for preprocessing; `500 Internal Server Error` if stored sequence JSON cannot be parsed.

### 3.2.4 `GET /events`

**Purpose:** Retrieve recent stored events in descending timestamp order.

**Query parameters:** `limit` defaults to 50 and must be from 1 through 500; `offset` defaults to 0 and must be non-negative.

**Response shape:**

```json
{
  "events": [
    {
      "event_id": "a1b2c3d4",
      "timestamp": "2026-10-06 12:30:00",
      "threat_score": 73.4,
      "transformer_confidence": 0.81,
      "vae_anomaly_score": 0.72,
      "dga_probability": 0.13,
      "predicted_class": "PortScan",
      "transformer_predicted_class": "PortScan"
    }
  ]
}
```

The event-list schema omits the raw sequence. It also does not persist or return the submitted domain in the present schema/database design.

### 3.2.5 `GET /events/{event_id}`

**Purpose:** Retrieve one event record for inspection.

**Response:** JSON event fields include `event_id`, `timestamp`, `threat_score`, `transformer_confidence`, `vae_anomaly_score`, `dga_probability`, `predicted_class`, and optional `transformer_predicted_class`. Raw input sequence is stored internally for explanation but is not part of the declared `EventResponse` fields. An unknown identifier returns `404 Not Found`.

### 3.2.6 `GET /health`

**Purpose:** Report service status and model availability.

**Representative response:**

```json
{
  "status": "ok",
  "service": "Layered AI Defense Backend Integration Layer",
  "models": {
    "transformer": true,
    "vae": true,
    "dga": true,
    "fusion": true,
    "adversarial": false
  }
}
```

The values above are illustrative. The endpoint reports `status="ok"` and booleans based on whether each model object is loaded; it does not perform a model inference or verify external LLM availability.

### 3.2.7 `GET /robustness`

**Purpose:** Retrieve metrics already stored by the offline robustness evaluator.

**Response shape:**

```json
{
  "results": [
    {
      "run_id": "baseline_undefended_1234567890",
      "timestamp": 1791289800.0,
      "model_name": "Baseline (undefended)",
      "variant": "Baseline (undefended)",
      "clean_acc": 0.91,
      "clean_f1": null,
      "fgsm_acc": 0.52,
      "fgsm_f1": null,
      "pgd_acc": 0.39,
      "pgd_f1": null
    }
  ]
}
```

Metrics are illustrative. The current route sets the F1 response fields to `null`; only accuracy values are stored in `robustness_results`. If no run results have been persisted, `results` is an empty array.

## 3.3 Non-Functional Requirements

### 3.3.1 Performance

**NFR-PERF-01 (Acceptance target).** On the project’s declared demonstration hardware, a warmed local `/score` request for the configured default sequence and optional domain should complete within 1 second at the 95th percentile, excluding model startup. This target is proposed for acceptance testing; no end-to-end latency measurement is asserted by this SRS.

**NFR-PERF-02 (Acceptance target).** A warmed `/explain` request should return within 30 seconds under normal local and provider conditions, or return the endpoint’s documented degraded/fallback response within that limit. The dashboard client currently allows up to 30 seconds for this slow path.

**NFR-PERF-03.** Performance reporting shall identify hardware, device (CPU/GPU), warm/cold state, sequence dimensions, model availability, and whether SHAP/LLM calls are included. Batch throughput measurements shall not be represented as end-to-end request latency.

**Evidence and limitation:** the project reference reports a synthetic Transformer benchmark of approximately 136 ms per batch of 256 on CPU. It is not a measured `/score` latency and does not include preprocessing, VAE, DGA, Fusion, SQLite, or network overhead. The current route runs detector calls sequentially.

### 3.3.2 Scalability

**NFR-SCALE-01 (Target).** The service should support repeated local analyst requests without requiring model reload for every score request after application startup.

**NFR-SCALE-02 (Target).** Any future multi-worker or horizontally scaled deployment shall replace or configure persistence and artifact-loading behavior for concurrent workers; the current SQLite single-file storage and in-process model container are demonstration-scale components, not evidence of tested horizontal scalability.

**NFR-SCALE-03.** Event retrieval shall remain bounded by pagination parameters, with the current API allowing at most 500 records per `/events` request.

### 3.3.3 Reliability and Graceful Degradation

**NFR-REL-01.** Failure to load one model shall be logged and shall not prevent the backend from initializing if the application and database can otherwise start.

**NFR-REL-02.** Scoring shall distinguish a complete result, partial result, and no-usable-model result through the response `status` and message fields.

**NFR-REL-03.** A missing trained Fusion artifact shall not cause the service to silently present a heuristic result as a trained Fusion result; the response shall include the current fallback message when all signals exist but the trained Fusion path is unavailable.

**NFR-REL-04.** A missing LLM credential, SHAP installation, or reachable LLM provider shall not be represented as a successful provider-generated explanation. The service should return available local explanation content and an explicit text-generation fallback.

**NFR-REL-05.** Event scoring shall not write a scored event when no usable model signal exists. When a score is produced, the route shall persist the score and raw sequence before returning the event identifier.

**Current behavior caveat:** startup attempts to load artifacts through the model container. Artifact dependency files are not uniformly guarded before deserialization in every loading branch; some incomplete artifact sets may still raise during startup. Graceful degradation is implemented for missing checkpoints in several paths but should be validated against each partial-artifact combination.

### 3.3.4 Security

**NFR-SEC-01.** The LLM API credential shall be read from a server-side environment variable (`LLM_API_KEY`) or project-root `.env`; it shall not be embedded in frontend assets, committed as a real credential, or returned by an API endpoint.

**NFR-SEC-02.** The `.env` file containing an active provider key shall remain excluded from version control; a non-secret example file may document the required variable name.

**NFR-SEC-03 (Production requirement; not implemented).** A deployment exposed beyond a trusted local environment shall restrict CORS origins to approved clients, use TLS at the deployment boundary, and add appropriate API authentication/authorization and request controls. The current backend has wildcard CORS and does not implement API authentication.

**NFR-SEC-04 (Target).** Production deployments should apply request-size and schema limits to flow sequences, domains, and event identifiers to reduce resource-exhaustion and malformed-input risk. The API currently performs dimensionality/feature checks but does not define a comprehensive maximum sequence length or request-size policy.

**NFR-SEC-05.** Because raw sequences are persisted to support explanations, operators shall treat the SQLite database as potentially sensitive traffic-derived data, restrict filesystem access, and define appropriate retention and deletion practices before real traffic is used.

### 3.3.5 Usability

**NFR-USE-01.** The dashboard shall show backend reachability and loaded model states distinctly from event predictions.

**NFR-USE-02.** The interface shall distinguish threat score (0–100), Transformer confidence, VAE anomaly score, and DGA probability; these values have different meanings and units.

**NFR-USE-03.** The interface shall identify demo data as demonstration content and shall not present it as live inference or persisted event history.

**NFR-USE-04 (Target).** The score-entry workflow should display the expected sequence dimensions and feature ordering requirements and should provide a backend-compatible payload or a clear validation error before sending data.

**NFR-USE-05.** Explanation text shall be presented as model-generated assistance, not as definitive proof of malicious intent or causal attribution.

### 3.3.6 Maintainability and Portability

**NFR-MAINT-01.** Model artifact paths, score threshold, and compute device selection shall remain centrally configurable rather than duplicated in API route code.

**NFR-MAINT-02.** The trained Transformer scaler and feature-name order used at inference shall correspond to those used to prepare training data; inference shall not fit a new scaler on each request.

**NFR-MAINT-03.** Documentation and diagrams shall reflect the separate training and serving responsibilities and identify duplicated or mirrored model definitions when they affect checkpoint/explanation interpretation.

**NFR-PORT-01.** The local demonstration shall be operable on CPU where CUDA is unavailable, subject to performance limits. The code selects CUDA when available and CPU otherwise.

# 4. Requirements Verification Summary

| Requirement area | Primary verification evidence |
|---|---|
| Flow validation and scaling | Unit/API tests for valid 2D input, wrong dimensionality, wrong feature width, and saved-scaler parity with training preprocessing. |
| Transformer class and confidence | Test known input shape, class-index mapping, BENIGN index lookup, and separation of Transformer hint from final decision. |
| VAE anomaly signal | Test scoring uses the final scaled row and the same reconstruction-plus-weighted-KL calculation as the training/evaluation path. |
| DGA signal | Test absent domain, loaded-model case, lowercase/length/padding behavior, and class-1 probability mapping. |
| Fusion decision | Test values just below, equal to, and above 50.0; confirm Fusion controls final class even when Transformer argmax disagrees. |
| Fallback scoring | Test full-signal heuristic, missing Fusion artifacts, one or more missing signal sources, and no usable signals. Verify `status`, message, normalized score, and persistence behavior. |
| Explainability | Test stored and unknown event IDs, missing/invalid raw sequence, SHAP disabled/failure, attention output, and LLM unavailable behavior. Treat fallback attribution as degraded output. |
| Persistence and event API | Test event insertion, descending-time event listing, pagination bounds, event detail, and raw-sequence exclusion from response schemas. |
| Robustness API | Insert known robustness rows, verify accuracy response values and null F1 fields, and verify empty-state response. |
| Dashboard integration | Build the production frontend and send a valid `flow_sequence` payload to `/score`; verify the response and resulting event-history entry. This was smoke-tested with an in-memory database. |
| Security configuration | Verify `.env` is ignored, no secret is included in frontend build artifacts, and production CORS/auth controls are documented and tested when added. |
| Performance | Benchmark warmed end-to-end API requests separately from model-only batch inference and report p50/p95, machine, device, and sample shape. |

# 5. Implementation Limitations and Known Integration Gaps

1. **Fusion signal provenance and evaluation leakage.** `generate_signals.py` obtains flow signals from CICIDS test sequences and domain signals from a separate DGA test set, pairing by row position and truncating/resizing rather than joining corresponding events. The Fusion pipeline then splits these test-derived signals into its own training and evaluation partitions. Rebuild aligned training-only signals and retain a truly untouched Fusion evaluation set before making independent holdout claims.
2. **Sequential scoring.** Transformer, VAE, and DGA calls are sequential in `routes_score.py`; parallel detector execution is not currently implemented.
3. **Separate explanation model implementation.** The scoring class in `models/transformer/model.py` uses PyTorch’s `nn.TransformerEncoder`. The explanation module defines an attention-capturing Transformer mirror and loads the Transformer checkpoint into it; the scoring model itself does not expose captured attention.
4. **API sequence length validation.** The scoring route validates two-dimensional input and expected feature count, but does not enforce exactly 10 time steps. Ten rows are the training/UI default, not an endpoint validator.
5. **Optional DGA input changes status.** Because DGA is counted as available only when a non-empty domain and its artifacts are present, omitting the optional domain generally makes the result `partial`, even when Transformer and VAE both run.
6. **Fallback attribution is synthetic.** If the explanation pipeline raises an exception, current code returns hard-coded example attention/SHAP values. These values maintain response shape but are not computed explanations and must be identified as degraded output.
7. **Robustness F1 not exposed.** Offline evaluation calculates macro-F1, but database persistence and the `/robustness` route expose accuracy only; response F1 values are currently null.
8. **LLM key/provider dependency.** The current `.env` value is placeholder-like and does not match a valid OpenRouter token format. Plain-English provider output remains unverified until a valid local credential and provider round trip are available.
9. **Security hardening is incomplete.** The service is configured with permissive wildcard CORS and has no backend authentication. It should remain a trusted local demonstration unless deployment controls are added.
10. **Historical planning versus runtime paths.** Current `backend_integration/config.py` resolves VAE, DGA, and Fusion checkpoints from their model-folder `outputs/` locations and Transformer artifacts from `models/saved_weights/`; the configured source paths are authoritative.
11. **No event-domain persistence.** The request accepts a domain, but the event schema and insert statement do not store the domain; event history cannot reliably display the submitted domain from backend persistence.
12. **Threshold artifact use.** A VAE anomaly threshold is loaded by the model container but is not applied in `/score` to make the final decision. Fusion remains authoritative.

