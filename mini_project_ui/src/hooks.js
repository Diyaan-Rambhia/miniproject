import { useEffect, useRef, useState } from 'react';

// Animates a number toward `target` (count-up / smooth gauge).
export function useTween(target, ms = 700) {
  const [v, setV] = useState(target);
  const from = useRef(target);
  const cur = useRef(target);
  useEffect(() => {
    const start = performance.now();
    const a = cur.current;
    from.current = a;
    let raf;
    const step = (now) => {
      const t = Math.min(1, (now - start) / ms);
      const e = 1 - Math.pow(1 - t, 3);
      cur.current = a + (target - a) * e;
      setV(cur.current);
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

// Reveals `text` one chunk at a time.
export function useTypewriter(text, cps = 90) {
  const [n, setN] = useState(0);
  useEffect(() => {
    setN(0);
    if (!text) return undefined;
    const t0 = performance.now();
    let raf;
    const step = (now) => {
      const k = Math.min(text.length, Math.floor(((now - t0) / 1000) * cps));
      setN(k);
      if (k < text.length) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [text, cps]);
  return text ? text.slice(0, n) : '';
}

export const level = (s) => (s >= 70 ? 'high' : s >= 40 ? 'med' : 'low');
