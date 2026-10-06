import React, { useEffect, useRef } from 'react';
import { createPipeline } from '../three/pipeline.js';
import Gauge from './Gauge.jsx';

export const STAGES = [
  { key: 'OVERVIEW', color: 'var(--cyan)', title: 'One event. Three independent questions.', body: 'Each window of flows and each domain name goes through three specialist models in parallel. No single model decides alone.', stat: '3 detectors, 1 fusion head' },
  { key: 'TRANSFORMER', color: 'var(--cyan)', title: 'Is this a known attack?', body: 'A 3-layer Transformer encoder reads 10 consecutive flows and separates benign traffic from each attack type. Its attention weights are kept for the explanation.', stat: 'D_MODEL 128 · 4 heads · ~416K parameters' },
  { key: 'VAE', color: 'var(--violet)', title: 'Does it look like normal traffic at all?', body: 'Trained on benign traffic only. Traffic it cannot rebuild scores high, which is how zero-day behaviour shows up without any attack labels.', stat: 'ELBO anomaly score · 95th-percentile threshold' },
  { key: 'DGA', color: 'var(--amber)', title: 'Was that domain written by a human?', body: 'A bidirectional character-level LSTM reads a domain one letter at a time and flags names produced by domain generation algorithms.', stat: '159,234 parameters · 42-character vocabulary' },
  { key: 'FUSION', color: 'var(--red)', title: 'One threat score, 0 to 100.', body: 'A small MLP weighs the three signals into a single score. The ablation study compares every detector alone against the fused result.', stat: '3 signals → MLP(16) → threat score' },
];

export default function Pipeline({ stage, sample, trackRef, stageRef }) {
  const cv = useRef(null);
  useEffect(() => {
    let p;
    try { p = createPipeline(cv.current); } catch (e) { /* no WebGL */ }
    return () => p && p.dispose();
  }, []);
  const S = STAGES[stage];
  return (
    <section id="pipeline" className="track" ref={trackRef}>
      <div className="pstage" ref={stageRef}>
        <canvas ref={cv} className="pipe-canvas" aria-label="Exploded view of the four models in the pipeline" />
        <div className="pipe-copy">
          {STAGES.map((s, i) => (
            <article key={s.key} className={`pscene ${i === stage ? 'on' : ''}`} aria-hidden={i !== stage} style={{ '--c': s.color }}>
              <div className="eyebrow" style={{ color: s.color }}>{s.key}</div>
              <h2>{s.title}</h2>
              <p>{s.body}</p>
              <div className="spec">{s.stat}</div>
            </article>
          ))}
        </div>
        <div className={`pipe-out ${stage === 4 ? 'on' : ''}`}>
          <Gauge value={stage === 4 && sample ? sample.threat_score : 0} size={190} />
          {sample && (
            <ul className="sigs">
              <li style={{ '--c': 'var(--cyan)' }}><span>Transformer</span><b>{(sample.transformer_confidence * 100).toFixed(1)}%</b></li>
              <li style={{ '--c': 'var(--violet)' }}><span>VAE anomaly</span><b>{sample.vae_anomaly_score.toFixed(3)}</b></li>
              <li style={{ '--c': 'var(--amber)' }}><span>DGA prob.</span><b>{(sample.dga_probability * 100).toFixed(1)}%</b></li>
            </ul>
          )}
        </div>
        <div className="pdots" role="presentation">
          {STAGES.map((s, i) => <i key={s.key} className={i === stage ? 'on' : ''} />)}
        </div>
      </div>
    </section>
  );
}
