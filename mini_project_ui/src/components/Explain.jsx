import React from 'react';
import { useTypewriter } from '../hooks.js';

export default function Explain({ data, loading }) {
  const text = useTypewriter(data ? data.plain_english_explanation : '', 110);
  if (loading) {
    return (
      <div className="ex-loading" role="status">
        <div className="scanbar" />
        <span>Running SHAP and attention, then asking the LLM to explain…</span>
      </div>
    );
  }
  if (!data) return <p className="dim">Select an event to see why it was flagged.</p>;

  const shap = data.top_shap_features.slice(0, 7);
  const maxAbs = Math.max(0.0001, ...shap.map((s) => Math.abs(s.shap_value)));
  const steps = Math.max(10, ...data.top_attended_timesteps.map((a) => a.timestep + 1));
  const wByT = {};
  data.top_attended_timesteps.forEach((a) => { wByT[a.timestep] = a.weight; });
  const maxW = Math.max(0.0001, ...data.top_attended_timesteps.map((a) => a.weight));

  return (
    <div className="ex">
      <div className="ex-text">
        <div className="tag">Plain-English assessment</div>
        <p>{text}<span className="caret" /></p>
      </div>

      <div className="ex-cols">
        <div>
          <div className="tag">SHAP · what pushed the decision</div>
          <ul className="shap">
            {shap.length === 0 && <li className="dim">No SHAP features returned.</li>}
            {shap.map((s, i) => {
              const w = (Math.abs(s.shap_value) / maxAbs) * 50;
              const pos = s.shap_value >= 0;
              return (
                <li key={i}>
                  <span className="fn" title={s.feature}>{s.feature}<em>t{s.timestep}</em></span>
                  <span className="bar"><i className={pos ? 'pos' : 'neg'} style={{ width: w + '%', [pos ? 'left' : 'right']: '50%' }} /></span>
                  <span className="sv">{s.shap_value > 0 ? '+' : ''}{s.shap_value.toFixed(3)}</span>
                </li>
              );
            })}
          </ul>
          <div className="legend"><i className="neg" /> pushes toward benign <i className="pos" /> pushes toward attack</div>
        </div>

        <div>
          <div className="tag">Attention · which flows the model watched</div>
          <div className="attn" style={{ gridTemplateColumns: `repeat(${steps}, 1fr)` }}>
            {Array.from({ length: steps }, (_, t) => {
              const w = wByT[t] || 0;
              return <div key={t} className="cell" style={{ '--a': (w / maxW).toFixed(3) }} title={`timestep ${t}: ${w.toFixed(4)}`}><b>{w ? w.toFixed(2) : ''}</b><span>t{t}</span></div>;
            })}
          </div>
          <div className="legend">brighter = more attention. t0 is the oldest flow in the window.</div>
        </div>
      </div>
    </div>
  );
}
