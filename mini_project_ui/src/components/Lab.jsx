import React, { useEffect, useRef, useState } from 'react';

const HARD = /hard|adv|robust|defen/i;

function Card({ row, metric, phase, hardened }) {
  const val = (k) => row[`${k}_${metric}`];
  const cols = [
    { k: 'clean', label: 'Clean', c: 'var(--cyan)', show: true },
    { k: 'fgsm', label: 'FGSM', c: 'var(--amber)', show: phase >= 1 },
    { k: 'pgd', label: 'PGD', c: 'var(--red)', show: phase >= 2 },
  ];
  return (
    <div className={`panel labcard ${hardened ? 'hardened' : 'base'}`}>
      <div className="lc-head"><b>{row.variant}</b><span className="tagp">{hardened ? 'ADVERSARIALLY TRAINED' : 'STANDARD TRAINING'}</span></div>
      <div className="bars">
        {cols.map((c) => {
          const v = c.show ? val(c.k) : val('clean');
          return (
            <div key={c.k} className="bcol" style={{ '--c': c.c }}>
              <b className="bv">{(val(c.k) * 100).toFixed(1)}</b>
              <div className="btrack"><i style={{ height: Math.max(2, v * 100) + '%', opacity: c.show ? 1 : 0.25 }} /></div>
              <span>{c.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function Lab({ rows, source }) {
  const [metric, setMetric] = useState('acc');
  const [phase, setPhase] = useState(3); // complete at rest
  const timers = useRef([]);
  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  const launch = () => {
    timers.current.forEach(clearTimeout);
    setPhase(0);
    timers.current = [setTimeout(() => setPhase(1), 500), setTimeout(() => setPhase(2), 1500), setTimeout(() => setPhase(3), 2500)];
  };

  const hardened = rows.find((r) => HARD.test(r.variant));
  const base = rows.find((r) => r !== hardened) || rows[0];
  const shown = rows.length > 1 && hardened ? [base, hardened] : rows;
  const gain = hardened && base && hardened !== base ? (hardened[`pgd_${metric}`] - base[`pgd_${metric}`]) * 100 : null;

  return (
    <section id="lab" className="section lab">
      <header className="sec-head">
        <div>
          <div className="eyebrow">Adversarial lab</div>
          <h2>What happens when someone attacks the detector.</h2>
        </div>
        <div className="lab-tools">
          {source === 'demo' && <span className="pill mode-demo"><i />DEMO DATA</span>}
          <div className="seg" role="group" aria-label="Metric">
            <button className={metric === 'acc' ? 'on' : ''} onClick={() => setMetric('acc')}>Accuracy</button>
            <button className={metric === 'f1' ? 'on' : ''} onClick={() => setMetric('f1')}>F1</button>
          </div>
          <button className="btn primary" onClick={launch}>Launch attack</button>
        </div>
      </header>

      {rows.length === 0 ? <p className="dim">No robustness results loaded yet.</p> : (
        <>
          <div className="labgrid">
            {shown.map((r) => <Card key={r.variant} row={r} metric={metric} phase={phase} hardened={r === hardened} />)}
          </div>
          {gain !== null && (
            <p className="lab-note">Under PGD, adversarial training holds {metric === 'acc' ? 'accuracy' : 'F1'} <b>{gain >= 0 ? '+' : ''}{gain.toFixed(1)} points</b> higher than the standard model.</p>
          )}
          <div className="tablewrap">
            <table>
              <thead><tr><th>Model</th><th>Clean acc</th><th>Clean F1</th><th>FGSM acc</th><th>FGSM F1</th><th>PGD acc</th><th>PGD F1</th></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.variant}><td>{r.variant}</td>{['clean_acc', 'clean_f1', 'fgsm_acc', 'fgsm_f1', 'pgd_acc', 'pgd_f1'].map((k) => <td key={k} className="mono">{r[k].toFixed(4)}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
