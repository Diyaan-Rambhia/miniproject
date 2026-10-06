# Sentinel: frontend for the Layered AI Defense Pipeline

React + Vite + Three.js. Talks to the existing FastAPI backend using the same endpoints as the old `App.jsx`:

| Endpoint | Used for |
|---|---|
| `GET /health` | LIVE / DEMO badge |
| `GET /events?limit=50&offset=0` | events table, hero stats, gauge |
| `POST /explain` `{event_id}` | SHAP, attention, LLM text |
| `GET /robustness` | adversarial lab (array, `{results: [...]}` or `{variant: metrics}` all work) |

If `/health` fails, the UI switches to **DEMO REPLAY** (clearly labelled, invented numbers) so the demo never breaks.

## Functional dashboard

Open `/dashboard` for the API-backed health, flow scoring, event history, explanations, and robustness view. Set `VITE_API_BASE` at build time or change the Backend URL in the dashboard; the latter is saved in this browser.

The active FastAPI `/score` contract accepts `{ "flow_sequence": number[][], "domain"?: string }`. The dashboard form currently validates a 10 × 78 sequence; each row must follow the order in the backend's saved `feature_names.joblib` artifact. The frontend does not bundle model artifacts.

## Run it

```bash
npm install
npm run dev          # http://localhost:5173
```

Backend on a different port? Create `.env` with `VITE_API_BASE=http://127.0.0.1:8000`.

This `mini_project_ui/` directory is the active integrated frontend. The separate `mini/` directory is a legacy duplicate; do not copy it over this application.

## Presenter keys

`↓ / → / Space` next stop · `↑ / ←` previous · `1-4` sections · `D` toggle demo replay

## Files

- `src/api.js` backend calls + response normalising
- `src/demoData.js` demo events / explanations / robustness (placeholder numbers)
- `src/three/hero.js`, `src/three/pipeline.js` the two 3D scenes
- `src/components/` Console, Explain (SHAP + attention), Lab, Gauge, Pipeline, Hero
