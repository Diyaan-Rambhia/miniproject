import * as THREE from 'three';
import { store } from '../store.js';

// Pipeline story. Four plates: Transformer, VAE, DGA (parallel detectors) and Fusion (below).
// store.pipe goes 0..4: 0 = stacked, 1 = Transformer in focus, 2 = VAE, 3 = DGA, 4 = Fusion + score.

const COLORS = [0x22e3ff, 0x9b6bff, 0xffb020, 0xff3355];
const CSS = ['#22e3ff', '#9b6bff', '#ffb020', '#ff3355'];

function plateTexture(kind, color) {
  const S = 512;
  const c = document.createElement('canvas');
  c.width = c.height = S;
  const x = c.getContext('2d');
  const bg = x.createLinearGradient(0, 0, S, S);
  bg.addColorStop(0, 'rgba(6,16,28,.96)');
  bg.addColorStop(1, 'rgba(4,10,20,.96)');
  x.fillStyle = bg; x.fillRect(0, 0, S, S);
  x.strokeStyle = color + '33'; x.lineWidth = 1;
  for (let i = 0; i <= S; i += 32) { x.beginPath(); x.moveTo(i, 0); x.lineTo(i, S); x.stroke(); x.beginPath(); x.moveTo(0, i); x.lineTo(S, i); x.stroke(); }
  x.fillStyle = color; x.strokeStyle = color;
  if (kind === 0) { // attention matrix
    for (let i = 0; i < 10; i++) for (let j = 0; j < 10; j++) {
      const w = Math.exp(-Math.abs(i - j) * 0.45) * (0.4 + 0.6 * ((i * 7 + j * 13) % 10) / 10);
      x.globalAlpha = 0.12 + w * 0.85;
      x.fillRect(96 + j * 32, 96 + i * 32, 28, 28);
    }
  } else if (kind === 1) { // bell curve + outliers
    x.globalAlpha = 0.9; x.lineWidth = 4; x.beginPath();
    for (let i = 0; i <= 320; i++) { const px = 96 + i, z = (i - 160) / 55, py = 400 - Math.exp(-z * z / 2) * 230; i ? x.lineTo(px, py) : x.moveTo(px, py); }
    x.stroke();
    x.globalAlpha = 1; x.fillStyle = '#ff3355';
    [[420, 380], [446, 392], [402, 396]].forEach(([px, py]) => { x.beginPath(); x.arc(px, py, 7, 0, 7); x.fill(); });
  } else if (kind === 2) { // domain characters
    x.globalAlpha = 0.9; x.font = '700 40px monospace';
    ['google.com', 'xkq7rt9zpl.net', 'github.com', 'a8fj2kd0qz.ru'].forEach((s, i) => {
      x.fillStyle = i % 2 ? '#ff3355' : color; x.fillText(s, 70, 150 + i * 66);
    });
  } else { // fusion: three inputs into one ring
    x.globalAlpha = 0.9; x.lineWidth = 6;
    x.beginPath(); x.arc(S / 2, S / 2, 130, Math.PI * 0.75, Math.PI * 2.25); x.stroke();
    x.globalAlpha = 0.5; x.lineWidth = 3;
    [[90, 120], [90, 256], [90, 392]].forEach(([px, py]) => { x.beginPath(); x.moveTo(px, py); x.lineTo(S / 2 - 120, S / 2); x.stroke(); });
  }
  x.globalAlpha = 1;
  return new THREE.CanvasTexture(c);
}

function dotTex() {
  const c = document.createElement('canvas'); c.width = c.height = 32;
  const x = c.getContext('2d'); const g = x.createRadialGradient(16, 16, 0, 16, 16, 16);
  g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0, 0, 32, 32);
  return new THREE.CanvasTexture(c);
}

export function createPipeline(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 60);
  camera.position.set(0, 3.6, 9.2);
  const root = new THREE.Group();
  scene.add(root);

  // plate sizes: [w, d]
  const SIZE = [[2.1, 1.6], [2.1, 1.6], [2.1, 1.6], [3.0, 2.0]];
  const SPREAD = [new THREE.Vector3(-2.45, 1.5, 0), new THREE.Vector3(0, 1.5, 0), new THREE.Vector3(2.45, 1.5, 0), new THREE.Vector3(0, -1.5, 0.4)];
  const STACK = [0, 1, 2, 3].map((i) => new THREE.Vector3(0, 0.75 - i * 0.5, 0));

  const plates = SIZE.map(([w, d], i) => {
    const g = new THREE.Group();
    const geo = new THREE.BoxGeometry(w, 0.06, d);
    const face = new THREE.MeshBasicMaterial({ map: plateTexture(i, CSS[i]), transparent: true });
    const mesh = new THREE.Mesh(geo, [
      new THREE.MeshBasicMaterial({ color: COLORS[i], transparent: true, opacity: 0.4 }),
      new THREE.MeshBasicMaterial({ color: COLORS[i], transparent: true, opacity: 0.4 }),
      face, // +y
      new THREE.MeshBasicMaterial({ color: 0x02050a, transparent: true }),
      new THREE.MeshBasicMaterial({ color: COLORS[i], transparent: true, opacity: 0.4 }),
      new THREE.MeshBasicMaterial({ color: COLORS[i], transparent: true, opacity: 0.4 }),
    ]);
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: COLORS[i], transparent: true }));
    g.add(mesh); g.add(edge);
    const glow = new THREE.Mesh(new THREE.PlaneGeometry(w * 1.5, d * 1.5), new THREE.MeshBasicMaterial({ color: COLORS[i], transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending, depthWrite: false }));
    glow.rotation.x = -Math.PI / 2; glow.position.y = -0.05; g.add(glow);
    root.add(g);
    return { g, mesh, edge, glow, face, op: 1, sc: 1, mats: mesh.material };
  });

  // beams from each detector down to the fusion plate, with travelling dots
  const beams = [0, 1, 2].map((i) => {
    const pts = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: COLORS[i], transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending }));
    root.add(line);
    const dots = new THREE.Points(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.PointsMaterial({ color: COLORS[i], size: 0.22, map: dotTex(), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    root.add(dots);
    return { line, dots, pts, geo };
  });

  // faint dust
  const DUST = 260, dp = new Float32Array(DUST * 3);
  for (let i = 0; i < DUST; i++) { dp[i * 3] = (Math.random() - 0.5) * 22; dp[i * 3 + 1] = (Math.random() - 0.5) * 12; dp[i * 3 + 2] = -7 + Math.random() * 8; }
  const dg = new THREE.BufferGeometry(); dg.setAttribute('position', new THREE.BufferAttribute(dp, 3));
  scene.add(new THREE.Points(dg, new THREE.PointsMaterial({ size: 0.05, map: dotTex(), color: 0x5fb6d0, transparent: true, opacity: 0.5, depthWrite: false, blending: THREE.AdditiveBlending })));

  const HOT = new THREE.Color(0xff3355);
const ease = (a, b, x) => { x = Math.min(1, Math.max(0, (x - a) / (b - a))); return x * x * (3 - 2 * x); };
  let raf = 0, last = performance.now(), running = true, aspect = 1;
  let spread = 0, focusX = 0, focusY = 0, rotY = 0.35;
    const tmpV = new THREE.Vector3();

  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    renderer.setSize(w, h, false);
    aspect = w / h; camera.aspect = aspect; camera.updateProjectionMatrix();
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (!running || !store.pipeOn) { last = now; return; }
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const t = now / 1000;
    const s = store.pipe;
    const k = store.reduce ? 1 : 1 - Math.exp(-dt * 5);

    const spreadT = ease(0.15, 0.85, s);              // stack -> spread during the first step
    spread += (spreadT - spread) * k;
    const focus = s < 0.9 ? -1 : Math.min(3, Math.round(s) - 1);
    const focusPos = focus < 0 ? new THREE.Vector3(0, 0, 0) : SPREAD[focus];
    const wide = aspect > 1.1;
    const fit = wide ? Math.min(1, aspect / 1.7) * 0.76 : Math.max(0.36, aspect * 0.5);
    const baseX = wide ? 2.3 * Math.min(1, aspect / 1.7) : 0;
    const baseY = wide ? 0 : -1.3;
    const fx = focus < 0 ? 0 : -focusPos.x * 0.3 * spread;
    focusX += (fx - focusX) * k;
    const fy = focus < 0 ? 0 : -focusPos.y * 0.3 * spread;
    focusY += (fy - focusY) * k;
    root.position.set(baseX + focusX * fit, baseY + focusY * fit, 0);
    root.scale.setScalar(fit);
    rotY += ((focus < 0 ? 0.55 : 0.12) + (store.reduce ? 0 : Math.sin(t * 0.3) * 0.06) - rotY) * k;
    root.rotation.y = rotY + store.mx * 0.18;
    root.rotation.x = store.my * 0.06;

    plates.forEach((p, i) => {
      tmpV.copy(STACK[i]).lerp(SPREAD[i], spread);
      p.g.position.copy(tmpV);
      if (focus < 0) p.g.position.y += Math.sin(t * 1.2 + i) * 0.03;
      const isFocus = focus < 0 || focus === i || focus === 3;
      const targetOp = isFocus ? 1 : 0.16;
      const targetSc = focus === i ? 1.12 : 1;
      p.op += (targetOp - p.op) * k;
      p.sc += (targetSc - p.sc) * k;
      p.g.scale.setScalar(p.sc);
      p.mats.forEach((m, mi) => { m.opacity = (mi === 2 || mi === 3 ? 1 : 0.4) * p.op; });
      p.edge.material.opacity = 0.35 + 0.65 * p.op;
      p.glow.material.opacity = 0.05 + 0.13 * p.op;
    });
    // fusion plate warms toward red as the score lands
    const hot = ease(3.3, 4, s);
    plates[3].edge.material.color.setHex(0xffffff).lerp(HOT, hot);
    plates[3].glow.material.opacity = 0.05 + 0.25 * hot;

    const bo = ease(0.5, 1.6, s);
    beams.forEach((b, i) => {
      const a = plates[i].g.position, z = plates[3].g.position;
      b.pts[0].set(a.x, a.y - 0.05, a.z);
      b.pts[2].set(z.x + (i - 1) * 0.5, z.y + 0.05, z.z + (i - 1) * 0.15);
      b.pts[1].set((a.x + b.pts[2].x) / 2, Math.min(a.y, z.y) + 0.4 - 0.8 * (1 - spread), (a.z + b.pts[2].z) / 2 + 0.7);
      const curve = new THREE.QuadraticBezierCurve3(b.pts[0], b.pts[1], b.pts[2]);
      b.geo.setFromPoints(curve.getPoints(28));
      const dimmed = focus >= 0 && focus !== i && focus !== 3 ? 0.15 : 1;
      b.line.material.opacity = 0.65 * bo * dimmed;
      b.dots.material.opacity = bo * dimmed;
      const pos = b.dots.geometry.attributes.position;
      for (let j = 0; j < 3; j++) { const u = (t * 0.35 + j / 3 + i * 0.13) % 1; curve.getPoint(u, tmpV); pos.setXYZ(j, tmpV.x, tmpV.y, tmpV.z); }
      pos.needsUpdate = true;
    });

    camera.position.y = 3.6 - spread * 0.5;
    camera.lookAt(0, 0.2, 0);
    renderer.render(scene, camera);
  }

  resize();
  raf = requestAnimationFrame(frame);
  window.addEventListener('resize', resize);
  return {
    resize,
    dispose() { running = false; cancelAnimationFrame(raf); window.removeEventListener('resize', resize); renderer.dispose(); },
  };
}
