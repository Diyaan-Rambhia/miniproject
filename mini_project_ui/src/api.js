// Talks to the FastAPI backend (same endpoints the original App.jsx used):
//   GET  /health
//   GET  /events?limit=50&offset=0     -> { events: [...] } or [...]
//   POST /explain  { event_id }        -> explanation object
//   GET  /robustness                   -> { results: [...] } | [...] | { variant: metrics }
// Every call has a short timeout so the UI can fall back to demo data quickly.

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

async function request(path, options = {}, timeoutMs = 3000, base = API_BASE) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(base.replace(/\/$/, '') + path, { ...options, signal: ctrl.signal });
    let data = null;
    try { data = await res.json(); } catch { /* an empty response is still useful for status reporting */ }
    if (!res.ok) {
      const error = new Error(`HTTP ${res.status}`);
      error.status = res.status;
      error.data = data;
      throw error;
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

const num = (v, d = 0) => (v === undefined || v === null || Number.isNaN(Number(v)) ? d : Number(v));

export function normalizeEvent(e) {
  return {
    event_id: e.event_id ?? e.id ?? e.timestamp,
    timestamp: e.timestamp ?? '',
    predicted_class: e.predicted_class ?? 'UNKNOWN',
    threat_score: num(e.threat_score),
    transformer_confidence: num(e.transformer_confidence),
    vae_anomaly_score: num(e.vae_anomaly_score),
    dga_probability: num(e.dga_probability),
  };
}

export async function checkHealth() {
  try {
    await request('/health', {}, 2000);
    return true;
  } catch {
    return false;
  }
}

export async function fetchEvents(limit = 50) {
  const data = await request(`/events?limit=${limit}&offset=0`);
  const list = Array.isArray(data) ? data : data.events || [];
  return list.map(normalizeEvent);
}

export async function fetchExplanation(eventId) {
  const d = await request(
    '/explain',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ event_id: eventId }) },
    30000, // SHAP + LLM is the slow path
  );
  return normalizeExplanation(d);
}

export function normalizeExplanation(d) {
  return {
    predicted_class: d.predicted_class ?? 'UNKNOWN',
    confidence: num(d.confidence),
    plain_english_explanation: d.plain_english_explanation || '',
    top_shap_features: (d.top_shap_features || []).map((s) => ({
      feature: s.feature,
      timestep: num(s.timestep),
      shap_value: num(s.shap_value),
    })),
    top_attended_timesteps: (d.top_attended_timesteps || []).map((a) =>
      Array.isArray(a) ? { timestep: num(a[0]), weight: num(a[1]) } : { timestep: num(a.timestep), weight: num(a.weight) },
    ),
  };
}

export async function fetchRobustness() {
  const d = await request('/robustness');
  const raw = d && d.results !== undefined ? d.results : d;
  const rows = Array.isArray(raw)
    ? raw.map((r) => ({ variant: r.variant || r.name, ...r }))
    : Object.entries(raw || {}).map(([variant, m]) => ({ variant, ...m }));
  return rows.map((r) => ({
    variant: String(r.variant),
    clean_acc: num(r.clean_acc), clean_f1: num(r.clean_f1),
    fgsm_acc: num(r.fgsm_acc), fgsm_f1: num(r.fgsm_f1),
    pgd_acc: num(r.pgd_acc), pgd_f1: num(r.pgd_f1),
  }));
}

export function fetchDashboardHealth(base) {
  return request('/health', {}, 5000, base);
}

export async function fetchDashboardEvents({ limit = 20, offset = 0, base = API_BASE } = {}) {
  const data = await request(`/events?limit=${limit}&offset=${offset}`, {}, 10000, base);
  const list = Array.isArray(data) ? data : data?.events ?? data?.items ?? data?.results ?? [];
  return {
    events: list.map((event) => ({
      ...event,
      event_id: event.event_id ?? event.id ?? event.timestamp,
      timestamp: event.timestamp ?? '',
      predicted_class: event.predicted_class ?? 'UNKNOWN',
      threat_score: num(event.threat_score),
      domain: event.domain ?? event.domain_name ?? '',
    })),
    total: Array.isArray(data) ? null : num(data?.total, null),
  };
}

export function scoreFlow(flowSequence, domain, base = API_BASE) {
  const payload = { flow_sequence: flowSequence };
  if (domain.trim()) payload.domain = domain.trim();
  return request('/score', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }, 30000, base);
}

export function explainDashboardEvent(eventId, base = API_BASE) {
  return request('/explain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event_id: eventId }),
  }, 30000, base);
}

export function fetchDashboardRobustness(base = API_BASE) {
  return request('/robustness', {}, 10000, base);
}
