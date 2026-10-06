import * as THREE from 'three';
import { store } from '../store.js';

// Hero: a spherical network of hosts. Packets travel along the links; attack packets are red and
// flash the node they land on. store.threat raises the share of red packets.

function glowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const x = c.getContext('2d');
  const g = x.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.3, 'rgba(255,255,255,.55)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g;
  x.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

export function createHero(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 50);
  camera.position.set(0, 0.2, 8.5);
  const group = new THREE.Group();
  scene.add(group);
  const tex = glowTexture();

  // ---- nodes on a jittered fibonacci sphere ----
  const N = 78;
  const P = [];
  const ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < N; i++) {
    const y = 1 - (i / (N - 1)) * 2;
    const rad = Math.sqrt(1 - y * y);
    const th = ga * i;
    const R = 2.5 + (Math.random() - 0.5) * 0.5;
    P.push(new THREE.Vector3(Math.cos(th) * rad * R, y * R, Math.sin(th) * rad * R));
  }

  // ---- edges: each node to its 3 nearest ----
  const edgeSet = new Set();
  const edges = [];
  P.forEach((p, i) => {
    const near = P.map((q, j) => ({ j, d: p.distanceToSquared(q) })).filter((o) => o.j !== i).sort((a, b) => a.d - b.d).slice(0, 3);
    near.forEach(({ j }) => {
      const key = i < j ? i + '-' + j : j + '-' + i;
      if (!edgeSet.has(key)) { edgeSet.add(key); edges.push([i, j]); }
    });
  });
  const lp = new Float32Array(edges.length * 6);
  edges.forEach(([a, b], k) => { P[a].toArray(lp, k * 6); P[b].toArray(lp, k * 6 + 3); });
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.BufferAttribute(lp, 3));
  const lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x1aa9c4, transparent: true, opacity: 0.32, blending: THREE.AdditiveBlending, depthWrite: false }));
  group.add(lines);

  // ---- node points (per-node colour so hits can flash) ----
  const nPos = new Float32Array(N * 3);
  const nCol = new Float32Array(N * 3);
  const flash = new Float32Array(N);
  P.forEach((p, i) => p.toArray(nPos, i * 3));
  const ng = new THREE.BufferGeometry();
  ng.setAttribute('position', new THREE.BufferAttribute(nPos, 3));
  ng.setAttribute('color', new THREE.BufferAttribute(nCol, 3));
  const nodes = new THREE.Points(ng, new THREE.PointsMaterial({ size: 0.24, map: tex, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  group.add(nodes);

  // ---- core ----
  const core = new THREE.Mesh(new THREE.IcosahedronGeometry(0.62, 1), new THREE.MeshBasicMaterial({ color: 0x22e3ff, wireframe: true, transparent: true, opacity: 0.85 }));
  group.add(core);
  const coreGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, color: 0x22e3ff, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false }));
  coreGlow.scale.setScalar(3.4);
  group.add(coreGlow);
  const shell = new THREE.Mesh(new THREE.SphereGeometry(2.05, 32, 20), new THREE.MeshBasicMaterial({ color: 0x0e4a5c, wireframe: true, transparent: true, opacity: 0.09 }));
  group.add(shell);

  // ---- packets ----
  const M = 170;
  const pk = Array.from({ length: M }, () => ({ e: 0, t: 0, v: 0, dir: 1, atk: false }));
  const pPos = new Float32Array(M * 3);
  const pCol = new Float32Array(M * 3);
  const pg = new THREE.BufferGeometry();
  pg.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  pg.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
  const packets = new THREE.Points(pg, new THREE.PointsMaterial({ size: 0.2, map: tex, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  group.add(packets);
  const respawn = (p) => {
    p.e = Math.floor(Math.random() * edges.length);
    p.t = 0;
    p.dir = Math.random() < 0.5 ? 1 : -1;
    p.atk = Math.random() < 0.06 + store.threat * 0.5;
    p.v = (p.atk ? 0.7 : 0.35) * (0.6 + Math.random() * 0.8);
  };
  pk.forEach((p) => { respawn(p); p.t = Math.random(); });

  // ---- background particles ----
  const DUST = 400;
  const dp = new Float32Array(DUST * 3);
  for (let i = 0; i < DUST; i++) { dp[i * 3] = (Math.random() - 0.5) * 24; dp[i * 3 + 1] = (Math.random() - 0.5) * 14; dp[i * 3 + 2] = -6 + Math.random() * 8; }
  const dg = new THREE.BufferGeometry();
  dg.setAttribute('position', new THREE.BufferAttribute(dp, 3));
  const dust = new THREE.Points(dg, new THREE.PointsMaterial({ size: 0.05, map: tex, color: 0x5fb6d0, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }));
  scene.add(dust);

  const CY = new THREE.Color(0x22e3ff), RD = new THREE.Color(0xff3355), tmp = new THREE.Color();
  let raf = 0, last = performance.now(), running = true, aspect = 1;

  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    renderer.setSize(w, h, false);
    aspect = w / h;
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    const wide = aspect > 1.1;
    group.position.set(wide ? Math.min(2.6, aspect * 1.3) : 0, wide ? 0 : -1.4, 0);
    group.scale.setScalar(wide ? Math.min(1, aspect / 1.7) * 1.02 : Math.max(0.5, aspect * 0.85));
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (!running || !store.heroOn) { last = now; return; }
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const t = now / 1000;
    const still = store.reduce;

    group.rotation.y = still ? 0.5 : t * 0.12 + store.mx * 0.35;
    group.rotation.x = 0.18 + store.my * 0.15;
    core.rotation.y = t * 0.6; core.rotation.x = t * 0.4;
    const beat = 1 + Math.sin(t * 2.4) * 0.05 + store.threat * 0.12;
    core.scale.setScalar(beat);
    coreGlow.material.color.copy(CY).lerp(RD, Math.min(1, store.threat * 1.4));
    core.material.color.copy(coreGlow.material.color);
    dust.rotation.y = t * 0.01;

    for (let i = 0; i < M; i++) {
      const p = pk[i];
      p.t += dt * p.v;
      if (p.t >= 1) {
        const end = p.dir === 1 ? edges[p.e][1] : edges[p.e][0];
        flash[end] = p.atk ? 1 : Math.max(flash[end], 0.35);
        respawn(p);
      }
      const [a, b] = edges[p.e];
      const s = p.dir === 1 ? p.t : 1 - p.t;
      pPos[i * 3] = P[a].x + (P[b].x - P[a].x) * s;
      pPos[i * 3 + 1] = P[a].y + (P[b].y - P[a].y) * s;
      pPos[i * 3 + 2] = P[a].z + (P[b].z - P[a].z) * s;
      if (p.atk) { pCol[i * 3] = 1; pCol[i * 3 + 1] = 0.2; pCol[i * 3 + 2] = 0.32; }
      else { pCol[i * 3] = 0.13; pCol[i * 3 + 1] = 0.89; pCol[i * 3 + 2] = 1; }
    }
    for (let i = 0; i < N; i++) {
      flash[i] = Math.max(0, flash[i] - dt * 1.6);
      tmp.copy(CY).multiplyScalar(0.55).lerp(RD, Math.min(1, flash[i] * 1.2)).multiplyScalar(0.7 + flash[i] * 1.3);
      nCol[i * 3] = tmp.r; nCol[i * 3 + 1] = tmp.g; nCol[i * 3 + 2] = tmp.b;
    }
    pg.attributes.position.needsUpdate = true; pg.attributes.color.needsUpdate = true; ng.attributes.color.needsUpdate = true;

    camera.position.x += (store.mx * 0.5 - camera.position.x) * 0.04;
    camera.position.y += (0.2 - store.my * 0.4 - camera.position.y) * 0.04;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }

  resize();
  raf = requestAnimationFrame(frame);
  window.addEventListener('resize', resize);
  return {
    resize,
    dispose() {
      running = false;
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
      renderer.dispose();
    },
  };
}
