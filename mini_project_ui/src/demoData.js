// Demo replay data. Used only when the backend is unreachable (or when forced with the D key).
// Numbers are invented placeholders shaped like the real API responses. The UI labels them DEMO.

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const ATTACK_CLASSES = [
  'DDoS', 'PortScan', 'DoS Hulk', 'Bot', 'FTP-Patator', 'SSH-Patator',
  'DoS slowloris', 'Web Attack - Brute Force', 'Web Attack - XSS', 'Infiltration',
];

const FEATURES = [
  'Flow Duration', 'Total Fwd Packets', 'Flow Bytes/s', 'Flow Packets/s', 'SYN Flag Count',
  'Fwd Packet Length Max', 'Bwd Packet Length Mean', 'Init_Win_bytes_forward', 'Flow IAT Mean',
  'PSH Flag Count', 'Destination Port', 'Average Packet Size',
];

const CLASS_HINT = {
  'DDoS': 'a very high packet rate from many sources, with tiny forward packets and a burst of SYN flags',
  'PortScan': 'short flows sweeping many destination ports with almost no payload',
  'DoS Hulk': 'a flood of large HTTP requests with unusually high bytes per second',
  'Bot': 'regular, low-volume beaconing at steady intervals',
  'FTP-Patator': 'repeated short login attempts against port 21',
  'SSH-Patator': 'repeated short login attempts against port 22',
  'DoS slowloris': 'connections held open for a long time with very slow, partial requests',
  'Web Attack - Brute Force': 'many similar HTTP requests to a login page',
  'Web Attack - XSS': 'HTTP requests with unusual payload sizes to a web form',
  'Infiltration': 'a rare long-lived flow with a strange upload/download ratio',
};

const r = rng(2026);
const pick = (arr) => arr[Math.floor(r() * arr.length)];

function makeEvent(i, tBase) {
  const isAttack = r() < 0.46;
  const cls = isAttack ? pick(ATTACK_CLASSES) : 'BENIGN';
  const dga = isAttack && r() < 0.18 ? 0.7 + r() * 0.29 : r() * (isAttack ? 0.35 : 0.08);
  const vae = isAttack ? 0.6 + r() * 2.6 : r() * 0.5;
  const conf = isAttack ? 0.72 + r() * 0.27 : 0.9 + r() * 0.099;
  let score = isAttack ? 52 + r() * 46 : r() * 24;
  if (dga > 0.7) score = Math.min(99, score + 8);
  return {
    event_id: 'evt-' + String(1000 + i),
    timestamp: new Date(tBase + i * 2300).toISOString().replace('T', ' ').slice(0, 19),
    predicted_class: cls,
    threat_score: Math.round(score * 10) / 10,
    transformer_confidence: Math.round(conf * 1000) / 1000,
    vae_anomaly_score: Math.round(vae * 10000) / 10000,
    dga_probability: Math.round(dga * 1000) / 1000,
  };
}

const T0 = Date.UTC(2026, 8, 29, 9, 0, 0);
export const DEMO_POOL = Array.from({ length: 120 }, (_, i) => makeEvent(i, T0));

export function demoExplanation(evt) {
  const rr = rng(String(evt.event_id).split('').reduce((a, c) => a + c.charCodeAt(0), 7));
  const attack = evt.predicted_class !== 'BENIGN';
  const feats = [...FEATURES].sort(() => rr() - 0.5).slice(0, 6).map((f, k) => ({
    feature: f,
    timestep: Math.floor(rr() * 10),
    shap_value: (attack ? 1 : -1) * (0.42 - k * 0.055) * (k % 3 === 2 ? -1 : 1) * (0.8 + rr() * 0.4),
  }));
  const w = Array.from({ length: 10 }, (_, t) => Math.pow(rr(), 2.2) + (t > 6 ? 0.25 : 0));
  const sum = w.reduce((a, b) => a + b, 0);
  const attn = w.map((x, t) => ({ timestep: t, weight: x / sum })).sort((a, b) => b.weight - a.weight).slice(0, 4);
  const text = attack
    ? `Flagged as ${evt.predicted_class} with a threat score of ${evt.threat_score.toFixed(0)}. The flow sequence shows ${CLASS_HINT[evt.predicted_class] || 'behaviour far from the normal pattern'}. ` +
      `${evt.dga_probability > 0.6 ? 'The domain name also looks algorithmically generated, which raised the score further. ' : ''}` +
      `The model paid most attention to the most recent flows in the window, and ${feats[0].feature} pushed the decision furthest toward attack.`
    : `This traffic looks normal. The flow sequence matches typical benign behaviour, the reconstruction error is low, and the domain looks human-made. No single feature pushed the score up.`;
  return {
    predicted_class: evt.predicted_class,
    confidence: evt.transformer_confidence,
    plain_english_explanation: text,
    top_shap_features: feats,
    top_attended_timesteps: attn,
  };
}

export const DEMO_ROBUSTNESS = [
  { variant: 'Baseline', clean_acc: 0.9921, clean_f1: 0.9468, fgsm_acc: 0.6314, fgsm_f1: 0.5527, pgd_acc: 0.3902, pgd_f1: 0.3184 },
  { variant: 'Adversarially hardened', clean_acc: 0.9874, clean_f1: 0.9391, fgsm_acc: 0.9342, fgsm_f1: 0.9026, pgd_acc: 0.8817, pgd_f1: 0.8342 },
];
