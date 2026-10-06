import React, { useEffect, useMemo, useState } from 'react';
import {
  API_BASE,
  explainDashboardEvent,
  fetchDashboardEvents,
  fetchDashboardHealth,
  fetchDashboardRobustness,
  scoreFlow,
} from '../api.js';
import './dashboard.css';

const PAGE_SIZE = 20;
const MODEL_NAMES = ['transformer', 'vae', 'dga', 'fusion', 'adversarial'];

function readSavedBase() {
  try { return window.localStorage.getItem('sentinel-api-base') || API_BASE; }
  catch { return API_BASE; }
}

function readModelState(data, model) {
  const source = data?.models ?? data?.model_status ?? data?.signals ?? data ?? {};
  const aliases = [model, `${model}_loaded`, `${model}_available`, `${model}_active`];
  const key = aliases.find((name) => Object.hasOwn(source, name));
  if (!key) return null;
  const value = source[key];
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') return /^(true|loaded|active|available|ready)$/i.test(value);
  if (value && typeof value === 'object') {
    for (const field of ['loaded', 'available', 'active', 'ready']) {
      if (typeof value[field] === 'boolean') return value[field];
    }
  }
  return false;
}

function errorText(error) {
  const detail = error?.data?.detail ?? error?.data?.message ?? error?.data?.error;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg || JSON.stringify(item)).join('; ');
  return error?.message || 'The request failed. Check the backend and try again.';
}

function isInsufficientModels(value) {
  const detail = value?.detail ?? value?.data?.detail;
  const code = value?.code ?? value?.data?.code;
  return value?.insufficient_models === true
    || value?.data?.insufficient_models === true
    || String(code || '').toLowerCase() === 'insufficient_models'
    || /insufficient.models/i.test(typeof detail === 'string' ? detail : '');
}

function normalizeExplanation(data) {
  const source = data?.explanation && typeof data.explanation === 'object' ? data.explanation : data || {};
  const shap = source.top_shap_features ?? source.shap_features ?? [];
  const attention = source.top_attended_timesteps ?? source.attention_weights ?? [];
  return {
    text: source.plain_english_explanation ?? source.llm_explanation ?? (typeof source.explanation === 'string' ? source.explanation : undefined) ?? source.text ?? '',
    shap: Array.isArray(shap) ? shap.map((item) => ({
      feature: item.feature ?? item.name ?? 'Unknown feature',
      impact: Number(item.shap_value ?? item.impact ?? item.value),
    })) : [],
    attention: Array.isArray(attention) ? attention.map((item) => Array.isArray(item)
      ? { timestep: Number(item[0]), weight: Number(item[1]) }
      : { timestep: Number(item.timestep), weight: Number(item.weight) }) : [],
  };
}

function normalizeRobustness(data) {
  const raw = data?.results ?? data?.metrics ?? data;
  const rows = Array.isArray(raw)
    ? raw
    : raw && typeof raw === 'object'
      ? Object.entries(raw).filter(([, metrics]) => metrics && typeof metrics === 'object').map(([variant, metrics]) => ({ variant, ...metrics }))
      : [];
  return rows.map((row) => ({
    variant: row.variant ?? row.name ?? 'Model',
    clean: row.clean_acc ?? row.clean_accuracy ?? row.clean?.accuracy,
    fgsm: row.fgsm_acc ?? row.fgsm_accuracy ?? row.fgsm?.accuracy,
    pgd: row.pgd_acc ?? row.pgd_accuracy ?? row.pgd?.accuracy,
  }));
}

function formatAccuracy(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`;
}

function PanelState({ state, empty, children }) {
  if (state.status === 'loading') return <p className="dash-state" role="status">Loading…</p>;
  if (state.status === 'error') return <p className="dash-state dash-error" role="alert">{state.error}</p>;
  if (empty) return <p className="dash-state">{empty}</p>;
  return children;
}

export default function Dashboard() {
  const [base, setBase] = useState(readSavedBase);
  const [baseInput, setBaseInput] = useState(readSavedBase);
  const [offset, setOffset] = useState(0);
  const [healthReload, setHealthReload] = useState(0);
  const [eventsReload, setEventsReload] = useState(0);
  const [robustReload, setRobustReload] = useState(0);
  const [health, setHealth] = useState({ status: 'loading', data: null, error: '' });
  const [events, setEvents] = useState({ status: 'loading', rows: [], total: null, error: '' });
  const [robustness, setRobustness] = useState({ status: 'loading', rows: [], error: '' });
  const [flowJson, setFlowJson] = useState('');
  const [domain, setDomain] = useState('');
  const [score, setScore] = useState({ status: 'idle', data: null, error: '' });
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [explanation, setExplanation] = useState({ status: 'idle', data: null, error: '' });

  useEffect(() => {
    let current = true;
    setHealth({ status: 'loading', data: null, error: '' });
    fetchDashboardHealth(base)
      .then((data) => { if (current) setHealth({ status: 'success', data, error: '' }); })
      .catch((error) => { if (current) setHealth({ status: 'error', data: null, error: errorText(error) }); });
    return () => { current = false; };
  }, [base, healthReload]);

  useEffect(() => {
    let current = true;
    setEvents((previous) => ({ ...previous, status: 'loading', error: '' }));
    fetchDashboardEvents({ limit: PAGE_SIZE, offset, base })
      .then((result) => {
        if (!current) return;
        const rows = result.events.slice().sort((a, b) => (Date.parse(b.timestamp) || 0) - (Date.parse(a.timestamp) || 0));
        setEvents({ status: 'success', rows, total: result.total, error: '' });
      })
      .catch((error) => { if (current) setEvents((previous) => ({ ...previous, status: 'error', error: errorText(error) })); });
    return () => { current = false; };
  }, [base, offset, eventsReload]);

  useEffect(() => {
    let current = true;
    setRobustness({ status: 'loading', rows: [], error: '' });
    fetchDashboardRobustness(base)
      .then((data) => { if (current) setRobustness({ status: 'success', rows: normalizeRobustness(data), error: '' }); })
      .catch((error) => { if (current) setRobustness({ status: 'error', rows: [], error: errorText(error) }); });
    return () => { current = false; };
  }, [base, robustReload]);

  const activeModels = useMemo(() => MODEL_NAMES.filter((model) => readModelState(health.data, model) === true), [health.data]);
  const hasNextPage = events.total !== null
    ? offset + PAGE_SIZE < events.total
    : events.rows.length === PAGE_SIZE;

  const connect = (event) => {
    event.preventDefault();
    const nextBase = baseInput.trim().replace(/\/$/, '');
    if (!nextBase) return;
    try { window.localStorage.setItem('sentinel-api-base', nextBase); } catch { /* this tab can still use the selected base */ }
    setBase(nextBase);
  };

  const refreshAll = () => {
    setHealthReload((value) => value + 1);
    setEventsReload((value) => value + 1);
    setRobustReload((value) => value + 1);
  };

  const submitScore = async (event) => {
    event.preventDefault();
    let flows;
    try { flows = JSON.parse(flowJson); }
    catch { setScore({ status: 'error', data: null, error: 'Enter the flow sequence as valid JSON.' }); return; }
    if (!Array.isArray(flows) || flows.length !== 10 || flows.some((row) => !Array.isArray(row) || row.length !== 78 || row.some((value) => typeof value !== 'number' || !Number.isFinite(value)))) {
      setScore({ status: 'error', data: null, error: 'Expected exactly 10 flow rows, each containing 78 finite numeric features.' });
      return;
    }

    setScore({ status: 'loading', data: null, error: '' });
    try {
      const response = await scoreFlow(flows, domain, base);
      if (isInsufficientModels(response)) {
        setScore({ status: 'insufficient', data: null, error: errorText({ data: response }) });
        return;
      }
      const result = response?.result && typeof response.result === 'object' ? response.result : response;
      const threatScore = Number(result?.threat_score);
      if (!Number.isFinite(threatScore)) throw new Error('The score response did not include a numeric threat_score.');
      setScore({
        status: 'success',
        data: {
          ...result,
          threat_score: threatScore,
          predicted_class: result.predicted_class ?? (threatScore >= 50 ? 'ATTACK' : 'BENIGN'),
          transformer_predicted_class: result.transformer_predicted_class ?? result.signals?.transformer_predicted_class ?? null,
          event_id: result.event_id ?? result.id ?? null,
          domain: result.domain ?? domain,
        },
        error: '',
      });
    } catch (error) {
      setScore({ status: isInsufficientModels(error) ? 'insufficient' : 'error', data: null, error: errorText(error) });
    }
  };

  const explainEvent = async (eventRecord) => {
    setSelectedEvent(eventRecord);
    setExplanation({ status: 'loading', data: null, error: '' });
    if (eventRecord.event_id === undefined || eventRecord.event_id === null || eventRecord.event_id === '') {
      setExplanation({ status: 'unavailable', data: null, error: 'This event has no event_id, so the explanation endpoint cannot be called.' });
      return;
    }
    try {
      const response = await explainDashboardEvent(eventRecord.event_id, base);
      setExplanation({ status: 'success', data: normalizeExplanation(response), error: '' });
    } catch (error) {
      setExplanation({ status: 'unavailable', data: null, error: errorText(error) });
    }
  };

  const modelStates = MODEL_NAMES.map((model) => [model, readModelState(health.data, model)]);
  const signals = score.data?.signals ?? score.data ?? {};
  const displayedThreatScore = score.data ? Math.max(0, Math.min(100, score.data.threat_score)) : 0;
  const signalValues = [
    ['Transformer confidence', signals.transformer_confidence],
    ['VAE anomaly score', signals.vae_anomaly_score],
    ['DGA probability', signals.dga_probability],
  ].filter(([, value]) => value !== undefined && value !== null);

  return (
    <div className="dashboard-root">
      <header className="dashboard-header">
        <a className="dashboard-brand" href="/" aria-label="Sentinel overview"><span className="brand-mark">S</span><span>SENTINEL <small>DEFENSE CONSOLE</small></span></a>
        <div className="backend-form">
          <label htmlFor="backend-url">Backend URL</label>
          <input id="backend-url" type="url" value={baseInput} onChange={(event) => setBaseInput(event.target.value)} />
          <button type="button" onClick={connect}>Connect</button>
        </div>
        <button className="dash-button secondary" type="button" onClick={refreshAll}>Refresh all</button>
      </header>

      <main className="dashboard-main">
        <div className="dashboard-title">
          <div><p className="dash-kicker">Operational view</p><h1>Detection dashboard</h1></div>
          <div className={`connection-state ${health.status === 'success' ? 'connected' : health.status === 'loading' ? 'pending' : 'disconnected'}`}>
            <span />{health.status === 'success' ? 'Backend reachable' : health.status === 'loading' ? 'Checking backend' : 'Backend unavailable'}
          </div>
        </div>

        <section className="dash-section health-section" aria-labelledby="health-title">
          <div className="section-heading"><div><p className="dash-kicker">01 / Runtime</p><h2 id="health-title">Health and active models</h2></div><span className="active-count">{activeModels.length} / 5 active</span></div>
          {health.status === 'loading' ? <p className="dash-state" role="status">Checking GET /health…</p> : null}
          {health.status === 'error' ? <p className="dash-state dash-error" role="alert">GET /health failed: {health.error}</p> : null}
          {health.status === 'success' && <>
            <div className="model-list">{modelStates.map(([model, state]) => <div className={`model-state ${state === true ? 'is-active' : state === false ? 'is-inactive' : 'is-unknown'}`} key={model}><span className="model-dot" /><span>{model}</span><b>{state === true ? 'LIVE' : state === false ? 'OFFLINE' : 'UNKNOWN'}</b></div>)}</div>
            <p className="active-signals">Active now: {activeModels.length ? activeModels.join(', ') : 'none reported'}</p>
          </>}
        </section>

        <section className="dash-section score-section" aria-labelledby="score-title">
          <div className="section-heading"><div><p className="dash-kicker">02 / Inference</p><h2 id="score-title">Score a flow sequence</h2></div><span className="contract-note">Backend API</span></div>
          <div className="score-layout">
            <form className="score-form" onSubmit={submitScore}>
              <label htmlFor="flow-json">Flow sequence JSON</label>
              <textarea id="flow-json" value={flowJson} onChange={(event) => setFlowJson(event.target.value)} placeholder="[[78 numeric features], ... 10 rows total]" spellCheck="false" />
              <p className="field-note">Request: <code>{'{ "flow_sequence": number[10][78], "domain"?: string }'}</code>. Feature order must match <code>feature_names.joblib</code>.</p>
              <div className="score-fields">
                <label htmlFor="domain-name">Domain <span>optional</span><input id="domain-name" value={domain} onChange={(event) => setDomain(event.target.value)} placeholder="example.test" /></label>
                <button className="dash-button secondary" type="button" onClick={() => setFlowJson(JSON.stringify(Array.from({ length: 10 }, () => Array(78).fill(0))))}>Insert zero template</button>
                <button className="dash-button primary" type="submit" disabled={score.status === 'loading'}>{score.status === 'loading' ? 'Scoring…' : 'Score sequence'}</button>
              </div>
              {score.status === 'error' && <p className="dash-state dash-error" role="alert">{score.error}</p>}
              {score.status === 'insufficient' && <div className="notice notice-warning" role="alert"><b>Insufficient models</b><span>{score.error || 'The backend does not have enough trained models to score this sequence.'}</span></div>}
            </form>

            <div className="score-result" aria-live="polite">
              {score.status === 'idle' && <div className="result-placeholder"><span>—</span><p>Submit a valid 10 × 78 sequence to see its fused result.</p></div>}
              {score.status === 'loading' && <p className="dash-state" role="status">Waiting for POST /score…</p>}
              {score.status === 'success' && score.data && <>
                <p className="dash-kicker">FUSION decision</p>
                <div className="score-readout"><strong>{displayedThreatScore.toFixed(1)}</strong><span>/ 100</span></div>
                <div className="score-track"><i style={{ width: `${displayedThreatScore}%` }} /></div>
                <p className={`decision ${String(score.data.predicted_class).toUpperCase() === 'BENIGN' ? 'benign' : 'attack'}`}>{score.data.predicted_class}</p>
                {score.data.transformer_predicted_class && <p className="transformer-hint"><span>Transformer hint</span><b>{score.data.transformer_predicted_class}</b><small>Attack-type guess only; the FUSION decision above is authoritative.</small></p>}
                {signalValues.length > 0 && <div className="signal-list">{signalValues.map(([label, value]) => <div key={label}><span>{label}</span><b>{Number.isFinite(Number(value)) ? Number(value).toFixed(4) : String(value)}</b></div>)}</div>}
                {score.data.domain && <p className="result-domain">Domain: {score.data.domain}</p>}
                {score.data.event_id !== null && <button className="dash-button secondary explain-score" type="button" onClick={() => explainEvent({ ...score.data, timestamp: score.data.timestamp ?? new Date().toISOString() })}>Explain this event</button>}
                {score.data.event_id === null && <p className="field-note">No event_id returned; POST /explain cannot be requested for this result.</p>}
              </>}
            </div>
          </div>
        </section>

        <div className="dashboard-columns">
          <section className="dash-section events-section" aria-labelledby="events-title">
            <div className="section-heading"><div><p className="dash-kicker">03 / Activity</p><h2 id="events-title">Event history</h2></div><button className="dash-button secondary" type="button" onClick={() => setEventsReload((value) => value + 1)} disabled={events.status === 'loading'}>Refresh events</button></div>
            <PanelState state={events} empty={events.rows.length === 0 ? 'No events returned.' : null}>
              <div className="event-table-wrap"><table className="event-table"><thead><tr><th>Timestamp</th><th>Threat</th><th>Class</th><th>Domain</th></tr></thead><tbody>
                {events.rows.map((event, index) => <tr key={event.event_id ?? `${event.timestamp}-${index}`}><td><button className="event-link" type="button" onClick={() => explainEvent(event)}>{event.timestamp || '—'}</button></td><td>{Number(event.threat_score).toFixed(1)}</td><td><span className={`class-tag ${String(event.predicted_class).toUpperCase() === 'BENIGN' ? 'class-benign' : 'class-threat'}`}>{event.predicted_class}</span></td><td>{event.domain || '—'}</td></tr>)}
              </tbody></table></div>
              <div className="pagination"><span>{offset + 1}–{offset + events.rows.length}{events.total !== null ? ` of ${events.total}` : ''}</span><div><button className="dash-button secondary" type="button" disabled={offset === 0 || events.status === 'loading'} onClick={() => setOffset((value) => Math.max(0, value - PAGE_SIZE))}>Previous</button><button className="dash-button secondary" type="button" disabled={!hasNextPage || events.status === 'loading'} onClick={() => setOffset((value) => value + PAGE_SIZE)}>Next</button></div></div>
            </PanelState>
          </section>

          <section className="dash-section robustness-section" aria-labelledby="robustness-title">
            <div className="section-heading"><div><p className="dash-kicker">04 / Evaluation</p><h2 id="robustness-title">Robustness comparison</h2></div><button className="dash-button secondary" type="button" onClick={() => setRobustReload((value) => value + 1)} disabled={robustness.status === 'loading'}>Refresh</button></div>
            <PanelState state={robustness} empty={robustness.rows.length === 0 ? 'Not yet available. Adversarial evaluation may not have been run.' : null}>
              <div className="robust-table-wrap"><table className="robust-table"><thead><tr><th>Model</th><th>Clean</th><th>FGSM</th><th>PGD</th></tr></thead><tbody>
                {robustness.rows.map((row, index) => <tr key={`${row.variant}-${index}`}><th scope="row">{row.variant}</th><td>{formatAccuracy(row.clean)}</td><td>{formatAccuracy(row.fgsm)}</td><td>{formatAccuracy(row.pgd)}</td></tr>)}
              </tbody></table></div>
              <p className="field-note">Accuracy. Missing metrics are shown as —.</p>
            </PanelState>
          </section>
        </div>

        {(selectedEvent || explanation.status !== 'idle') && <section className="dash-section explanation-section" aria-labelledby="explanation-title">
          <div className="section-heading"><div><p className="dash-kicker">05 / Explainability</p><h2 id="explanation-title">Event explanation</h2></div>{selectedEvent?.event_id && <span className="event-id">Event {selectedEvent.event_id}</span>}</div>
          {explanation.status === 'loading' && <p className="dash-state" role="status">Loading POST /explain…</p>}
          {explanation.status === 'unavailable' && <div className="notice notice-warning" role="status"><b>Explanation unavailable</b><span>{explanation.error || 'The backend returned no explanation.'}</span></div>}
          {explanation.status === 'success' && explanation.data && <>
            {explanation.data.text ? <p className="plain-explanation">{explanation.data.text}</p> : <div className="notice notice-warning"><b>Explanation unavailable</b><span>No plain-English explanation was returned. LLM credentials may be unset; available attribution data is shown below.</span></div>}
            <div className="explanation-lists"><div><h3>Top attended timesteps</h3>{explanation.data.attention.length ? <ol>{explanation.data.attention.map((item, index) => <li key={`${item.timestep}-${index}`}>t{item.timestep}<b>{Number.isFinite(item.weight) ? item.weight.toFixed(5) : '—'}</b></li>)}</ol> : <p className="field-note">No attention weights returned.</p>}</div><div><h3>Top SHAP features</h3>{explanation.data.shap.length ? <ol>{explanation.data.shap.map((item, index) => <li key={`${item.feature}-${index}`}><span>{item.feature}</span><b>{Number.isFinite(item.impact) ? item.impact.toFixed(5) : '—'}</b></li>)}</ol> : <p className="field-note">No SHAP features returned.</p>}</div></div>
          </>}
        </section>}

        <footer className="dashboard-footer">Backend contract is provisional until the FastAPI service is implemented and verified.</footer>
      </main>
    </div>
  );
}