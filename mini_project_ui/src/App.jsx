import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { store } from './store.js';
import * as api from './api.js';
import { DEMO_POOL, demoExplanation, DEMO_ROBUSTNESS } from './demoData.js';
import Hero from './components/Hero.jsx';
import Pipeline from './components/Pipeline.jsx';
import Console from './components/Console.jsx';
import Lab from './components/Lab.jsx';
import Dashboard from './components/Dashboard.jsx';

const NAV = [['hero', 'Overview'], ['pipeline', 'Pipeline'], ['console', 'Console'], ['lab', 'Adversarial lab']];

export default function App() {
  const route = window.location.pathname.replace(/\/$/, '');
  return route === '/dashboard' ? <Dashboard /> : <Landing />;
}

function Landing() {
  const [mode, setMode] = useState('checking'); // checking | live | demo
  const [forceDemo, setForceDemo] = useState(false);
  const [events, setEvents] = useState([]);
  const [robust, setRobust] = useState({ rows: [], source: 'demo' });
  const [stage, setStage] = useState(0);
  const track = useRef(null);
  const pstage = useRef(null);
  const cursor = useRef(24);

  // ---------- data: live backend, else demo replay ----------
  useEffect(() => {
    let dead = false;
    let timer;
    (async () => {
      const ok = !forceDemo && (await api.checkHealth());
      if (dead) return;
      if (ok) {
        setMode('live');
        const load = async () => { try { const e = await api.fetchEvents(50); if (!dead) setEvents(e); } catch { /* keep last */ } };
        await load();
        timer = setInterval(load, 6000);
        try { const rows = await api.fetchRobustness(); if (!dead && rows.length) setRobust({ rows, source: 'live' }); else if (!dead) setRobust({ rows: DEMO_ROBUSTNESS, source: 'demo' }); }
        catch { if (!dead) setRobust({ rows: DEMO_ROBUSTNESS, source: 'demo' }); }
      } else {
        setMode('demo');
        setRobust({ rows: DEMO_ROBUSTNESS, source: 'demo' });
        cursor.current = 24;
        setEvents(DEMO_POOL.slice(0, 24).reverse());
        timer = setInterval(() => {
          const i = cursor.current++;
          const base = DEMO_POOL[i % DEMO_POOL.length];
          const evt = { ...base, event_id: base.event_id + '-' + Math.floor(i / DEMO_POOL.length), timestamp: new Date().toISOString().replace('T', ' ').slice(0, 19) };
          setEvents((prev) => [evt, ...prev].slice(0, 60));
        }, 2600);
      }
    })();
    return () => { dead = true; clearInterval(timer); };
  }, [forceDemo]);

  const stats = useMemo(() => {
    const total = events.length;
    const high = events.filter((e) => e.threat_score >= 70).length;
    const mean = total ? events.reduce((a, e) => a + e.threat_score, 0) / total : 0;
    const counts = {};
    events.forEach((e) => { if (e.predicted_class !== 'BENIGN') counts[e.predicted_class] = (counts[e.predicted_class] || 0) + 1; });
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    return { total, high, mean, top: top ? top[0] : 'none' };
  }, [events]);

  useEffect(() => { store.threat = Math.min(1, events.slice(0, 20).reduce((a, e) => a + e.threat_score, 0) / Math.max(1, Math.min(20, events.length)) / 100); }, [events]);
  const sample = useMemo(() => events.find((e) => e.threat_score >= 70) || events[0] || null, [events]);

  const explain = useCallback(async (evt) => {
    if (mode === 'live') return api.fetchExplanation(evt.event_id);
    await new Promise((r) => setTimeout(r, 900));
    return demoExplanation(evt);
  }, [mode]);

  // ---------- scroll: pipeline progress, visibility, mouse ----------
  useEffect(() => {
    let raf;
    const loop = () => {
      const t = track.current;
      if (t) {
        const max = t.offsetHeight - window.innerHeight;
        const p = max > 0 ? Math.min(1, Math.max(0, -t.getBoundingClientRect().top / max)) : 0;
        store.pipe = p * 4;
        const st = Math.round(store.pipe);
        setStage((prev) => (prev === st ? prev : st));
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const io = new IntersectionObserver((ents) => {
      ents.forEach((en) => { if (en.target.id === 'hero') store.heroOn = en.isIntersecting; if (en.target.id === 'pipeline') store.pipeOn = en.isIntersecting; });
    }, { threshold: 0 });
    ['hero', 'pipeline'].forEach((id) => { const el = document.getElementById(id); if (el) io.observe(el); });
    const mm = (e) => { store.mx = (e.clientX / window.innerWidth - 0.5) * 2; store.my = (e.clientY / window.innerHeight - 0.5) * 2; };
    window.addEventListener('pointermove', mm);
    return () => { cancelAnimationFrame(raf); io.disconnect(); window.removeEventListener('pointermove', mm); };
  }, []);

  // ---------- presenter navigation ----------
  const stops = useCallback(() => {
    const y = (id) => (document.getElementById(id) ? document.getElementById(id).offsetTop : 0);
    const t = track.current;
    const out = [{ id: 'hero', y: 0 }];
    if (t) { const max = t.offsetHeight - window.innerHeight; for (let s = 0; s < 5; s++) out.push({ id: 'p' + s, y: t.offsetTop + (s / 4) * max + (s === 0 ? 2 : 0) }); }
    out.push({ id: 'console', y: y('console') }, { id: 'lab', y: y('lab') });
    return out;
  }, []);
  const go = useCallback((dir) => {
    const list = stops();
    const cur = window.scrollY;
    let idx = 0;
    list.forEach((s, i) => { if (Math.abs(s.y - cur) < Math.abs(list[idx].y - cur)) idx = i; });
    const next = Math.min(list.length - 1, Math.max(0, idx + dir));
    window.scrollTo({ top: list[next].y, behavior: store.reduce ? 'auto' : 'smooth' });
  }, [stops]);
  const jump = (id) => {
    const list = stops();
    const map = { hero: 'hero', pipeline: 'p0', console: 'console', lab: 'lab' };
    const s = list.find((x) => x.id === map[id]);
    if (s) window.scrollTo({ top: s.y, behavior: store.reduce ? 'auto' : 'smooth' });
  };
  useEffect(() => {
    const onKey = (e) => {
      if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
      if (['ArrowDown', 'ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); go(1); }
      else if (['ArrowUp', 'ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); go(-1); }
      else if (e.key === 'd' || e.key === 'D') setForceDemo((v) => !v);
      else if (/^[1-4]$/.test(e.key)) jump(NAV[+e.key - 1][0]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [go]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <div className="fx-scan" aria-hidden="true" />
      <header className="nav">
        <a className="brand" href="#hero" onClick={(e) => { e.preventDefault(); jump('hero'); }}><i />SENTINEL</a>
        <nav>{NAV.map(([id, label], i) => <button key={id} onClick={() => jump(id)}><em>{i + 1}</em>{label}</button>)}</nav>
        <button className={`pill mode-${mode}`} onClick={() => setForceDemo((v) => !v)} title="Press D to toggle demo replay">
          <i />{mode === 'live' ? 'LIVE' : mode === 'demo' ? 'DEMO' : '…'}
        </button>
      </header>

      <main>
        <Hero stats={stats} mode={mode} onStart={() => jump('console')} />
        <Pipeline stage={stage} sample={sample} trackRef={track} stageRef={pstage} />
        <Console events={events} mode={mode} onExplain={explain} stats={stats} />
        <Lab rows={robust.rows} source={robust.source} />
      </main>
      <footer className="foot">Sentinel · Layered AI Defense Pipeline · keys: ↑ ↓ step · 1–4 sections · D demo replay</footer>
    </>
  );
}
