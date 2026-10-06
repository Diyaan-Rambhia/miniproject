import React, { useEffect, useRef, useState } from 'react';
import Gauge from './Gauge.jsx';
import Explain from './Explain.jsx';
import { level } from '../hooks.js';

const clock = (ts) => (String(ts).length >= 19 ? String(ts).slice(11, 19) : String(ts));

export default function Console({ events, mode, onExplain, stats }) {
  const [sel, setSel] = useState(null);
  const [ex, setEx] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const token = useRef(0);
  const auto = useRef(false);

  const select = async (evt) => {
    const my = ++token.current;
    setSel(evt); setEx(null); setErr(''); setLoading(true);
    try {
      const d = await onExplain(evt);
      if (my === token.current) setEx(d);
    } catch (e) {
      if (my === token.current) setErr('Could not load the explanation from the backend.');
    } finally {
      if (my === token.current) setLoading(false);
    }
  };

  // open the first high-threat event once events arrive, so the panel is never empty
  useEffect(() => {
    if (!auto.current && events.length) {
      auto.current = true;
      select(events.find((e) => e.threat_score >= 70) || events[0]);
    }
  }, [events]); // eslint-disable-line react-hooks/exhaustive-deps

  const s = sel;
  return (
    <section id="console" className="section console">
      <header className="sec-head">
        <div>
          <div className="eyebrow">Live console</div>
          <h2>Flagged events, and why.</h2>
        </div>
        <span className={`pill mode-${mode}`}><i />{mode === 'live' ? 'LIVE BACKEND' : 'DEMO REPLAY'}</span>
      </header>

      <div className="kpis">
        <div><span>Events</span><b>{stats.total}</b></div>
        <div><span>High threat</span><b className="hot">{stats.high}</b></div>
        <div><span>Mean score</span><b>{stats.mean.toFixed(1)}</b></div>
        <div><span>Most common attack</span><b className="txt">{stats.top}</b></div>
      </div>

      <div className="console-grid">
        <div className="panel events">
          <div className="thead"><span>Time</span><span>Class</span><span>Score</span><span>T</span><span>VAE</span><span>DGA</span></div>
          <div className="tbody" role="listbox" aria-label="Recent flagged events">
            {events.length === 0 && <p className="dim pad">No events yet.</p>}
            {events.map((e) => (
              <button key={e.event_id} role="option" aria-selected={s && s.event_id === e.event_id} className={`row lv-${level(e.threat_score)} ${s && s.event_id === e.event_id ? 'sel' : ''}`} onClick={() => select(e)}>
                <span className="mono">{clock(e.timestamp)}</span>
                <span className="cls">{e.predicted_class}</span>
                <span className="sc"><i style={{ width: Math.min(100, e.threat_score) + '%' }} /><b>{e.threat_score.toFixed(1)}</b></span>
                <span className="mono dimv">{(e.transformer_confidence * 100).toFixed(0)}%</span>
                <span className="mono dimv">{e.vae_anomaly_score.toFixed(2)}</span>
                <span className="mono dimv">{(e.dga_probability * 100).toFixed(0)}%</span>
              </button>
            ))}
          </div>
        </div>

        <div className="panel detail">
          {s ? (
            <>
              <div className="d-top">
                <Gauge value={s.threat_score} size={190} />
                <div className="d-meta">
                  <div className="tag">Predicted class</div>
                  <div className="d-class">{ex ? ex.predicted_class : s.predicted_class}</div>
                  <div className="d-id mono">{String(s.event_id)} · {s.timestamp}</div>
                  <div className="sigbars">
                    <label style={{ '--c': 'var(--cyan)' }}><span>Transformer confidence</span><i><u style={{ width: s.transformer_confidence * 100 + '%' }} /></i><b>{(s.transformer_confidence * 100).toFixed(1)}%</b></label>
                    <label style={{ '--c': 'var(--violet)' }}><span>VAE anomaly score</span><i><u style={{ width: Math.min(100, (s.vae_anomaly_score / 3) * 100) + '%' }} /></i><b>{s.vae_anomaly_score.toFixed(3)}</b></label>
                    <label style={{ '--c': 'var(--amber)' }}><span>DGA probability</span><i><u style={{ width: s.dga_probability * 100 + '%' }} /></i><b>{(s.dga_probability * 100).toFixed(1)}%</b></label>
                  </div>
                </div>
              </div>
              {err ? <p className="err">{err}</p> : <Explain data={ex} loading={loading} />}
            </>
          ) : <p className="dim pad">Select an event.</p>}
        </div>
      </div>
    </section>
  );
}
