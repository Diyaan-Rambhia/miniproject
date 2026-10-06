import React from 'react';
import { useTween, level } from '../hooks.js';

// 270-degree arc gauge, 0..100.
export default function Gauge({ value = 0, size = 220, label = 'THREAT SCORE' }) {
  const v = useTween(value, 800);
  const r = 46, C = 2 * Math.PI * r, arc = C * 0.75;
  const off = arc * (1 - Math.min(100, Math.max(0, v)) / 100);
  return (
    <div className={`gauge lv-${level(value)}`} style={{ width: size, height: size }}>
      <svg viewBox="0 0 120 120" width={size} height={size} role="img" aria-label={`${label} ${Math.round(value)} out of 100`}>
        <circle className="g-track" cx="60" cy="60" r={r} strokeDasharray={`${arc} ${C}`} transform="rotate(135 60 60)" />
        <circle className="g-fill" cx="60" cy="60" r={r} strokeDasharray={`${arc} ${C}`} strokeDashoffset={off} transform="rotate(135 60 60)" />
        {[0, 25, 50, 75, 100].map((t) => {
          const a = ((135 + (t / 100) * 270) * Math.PI) / 180;
          return <line key={t} className="g-tick" x1={60 + Math.cos(a) * 54} y1={60 + Math.sin(a) * 54} x2={60 + Math.cos(a) * 58} y2={60 + Math.sin(a) * 58} />;
        })}
      </svg>
      <div className="g-read">
        <b>{Math.round(v)}</b>
        <span>{label}</span>
      </div>
    </div>
  );
}
