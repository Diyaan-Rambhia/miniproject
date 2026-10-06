import React, { useEffect, useRef } from 'react';
import { createHero } from '../three/hero.js';

export default function Hero({ stats, mode, onStart }) {
  const ref = useRef(null);
  useEffect(() => {
    let h;
    try { h = createHero(ref.current); } catch (e) { /* no WebGL: text still shows */ }
    return () => h && h.dispose();
  }, []);
  return (
    <section id="hero" className="hero">
      <canvas ref={ref} className="hero-canvas" aria-label="Animated network of hosts with packets flowing between them" />
      <div className="hero-copy">
        <div className="eyebrow">Layered AI defense pipeline</div>
        <h1>Every packet gets three second opinions.</h1>
        <p>A Transformer for known attacks, a VAE for zero-days and an LSTM for malicious domains, fused into one threat score and explained in plain English.</p>
        <div className="hero-actions">
          <button className="btn primary" onClick={onStart}>Open the live console</button>
          <a className="btn dashboard-link" href="/dashboard">Open dashboard</a>
          <span className={`pill mode-${mode}`}><i />{mode === 'live' ? 'LIVE BACKEND' : mode === 'demo' ? 'DEMO REPLAY' : 'CONNECTING'}</span>
        </div>
        <dl className="hero-stats">
          <div><dt>Events analysed</dt><dd>{stats.total}</dd></div>
          <div><dt>High threat</dt><dd className="hot">{stats.high}</dd></div>
          <div><dt>Mean score</dt><dd>{stats.mean.toFixed(1)}</dd></div>
        </dl>
      </div>
      <div className="scroll-cue"><i />Scroll or press &darr;</div>
    </section>
  );
}
