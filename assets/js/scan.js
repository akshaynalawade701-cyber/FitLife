// Body Scan (Beta) — Client-side posture analysis using TensorFlow.js
// This module runs locally in the browser. No files are uploaded.

// Mark booted only after successful init; fallback relies on this flag
window.__fitlife_scan_booted = false;

import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.13.0/dist/tf-core.esm.js';
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.13.0/dist/tf-backend-webgl.esm.js';
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter@4.13.0/dist/tf-converter.esm.js';
import * as tfwasm from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.13.0/dist/tf-backend-wasm.esm.js';
import * as posedetection from 'https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.3.1/dist/pose-detection.esm.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

let detector;
let selectedImages = [];
let selectedVideoFile = null;
let selectedVideos = [];

let lastReport = null;

async function ensureDetector() {
  if (detector) return detector;
  await tf.ready();
  try {
    if (tf.getBackend() !== 'webgl') {
      await tf.setBackend('webgl');
      await tf.ready();
    }
    const hiacc = document.getElementById('bs-hiacc')?.checked;
    const modelType = hiacc ? 'Thunder' : 'Lightning';
    detector = await posedetection.createDetector(posedetection.SupportedModels.MoveNet, { modelType, enableSmoothing: true });
    setStatus(`Model ready (${tf.getBackend()})`);
    // Model initialized successfully; mark module as booted so fallback defers
    window.__fitlife_scan_booted = true;
    return detector;
  } catch (e) {
    console.warn('WebGL backend failed, falling back to WASM', e);
    try {
      if (tfwasm?.setWasmPaths) {
        tfwasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.13.0/dist/');
      }
      await tf.setBackend('wasm');
      await tf.ready();
      detector = await posedetection.createDetector(posedetection.SupportedModels.MoveNet, {
        modelType: 'Lightning',
        enableSmoothing: true
      });
      setStatus(`Model ready (${tf.getBackend()})`);
      window.__fitlife_scan_booted = true;
      return detector;
    } catch (err) {
      console.error('Failed to initialize detector with WASM backend', err);
      throw err;
    }
  }
}

function drawImagePreview(file, container) {
  const url = URL.createObjectURL(file);
  const wrap = document.createElement('div');
  wrap.className = 'thumb';
  const img = new Image(); img.src = url; img.onload = () => URL.revokeObjectURL(url);
  img.style.width = '100%'; img.style.height = 'auto'; img.style.display = 'block'; img.style.borderRadius = '10px';
  const del = document.createElement('button'); del.textContent = '✕'; del.className = 'del';
  del.addEventListener('click', () => {
    selectedImages = selectedImages.filter(f => f !== file);
    wrap.remove();
  });
  wrap.appendChild(img); wrap.appendChild(del); container.appendChild(wrap);
}

function drawVideoPreview(file, container) {
  const url = URL.createObjectURL(file);
  const wrap = document.createElement('div'); wrap.className = 'thumb';
  const vid = document.createElement('video'); vid.src = url; vid.muted = true; vid.playsInline = true; vid.controls = false; vid.style.width = '100%'; vid.style.display = 'block'; vid.style.borderRadius = '10px';
  vid.onloadeddata = () => { vid.currentTime = Math.min(0.1, vid.duration || 0); URL.revokeObjectURL(url); };
  const del = document.createElement('button'); del.textContent = '✕'; del.className = 'del';
  del.addEventListener('click', () => { selectedVideos = selectedVideos.filter(f => f !== file); wrap.remove(); });
  wrap.appendChild(vid); wrap.appendChild(del); container.appendChild(wrap);
}

function angleDegrees(a, b, c) {
  // Angle at point b (in degrees)
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = (ab.x * cb.x + ab.y * cb.y);
  const magAB = Math.hypot(ab.x, ab.y);
  const magCB = Math.hypot(cb.x, cb.y);
  if (magAB === 0 || magCB === 0) return NaN;
  const cos = dot / (magAB * magCB);
  return Math.acos(Math.min(1, Math.max(-1, cos))) * (180 / Math.PI);
}

function lineTiltDegrees(p1, p2) {
  const dy = p2.y - p1.y;
  const dx = p2.x - p1.x;
  if (dx === 0) return 90;
  return Math.atan2(dy, dx) * (180 / Math.PI);
}

function dist(p1, p2) { return Math.hypot(p1.x - p2.x, p1.y - p2.y); }

function key(kps, name) {
  const m = kps.find(k => k.name === name || k.part === name);
  return m && m.score > 0.3 ? { x: m.x, y: m.y, score: m.score } : null;
}

function computeMetricsFromKeypoints(kps) {
  const leftShoulder = key(kps, 'left_shoulder');
  const rightShoulder = key(kps, 'right_shoulder');
  const leftHip = key(kps, 'left_hip');
  const rightHip = key(kps, 'right_hip');
  const leftEar = key(kps, 'left_ear') || key(kps, 'left_eye');
  const rightEar = key(kps, 'right_ear') || key(kps, 'right_eye');
  const nose = key(kps, 'nose');

  let shoulderTilt = null;
  let hipTilt = null;
  let shoulderTiltRaw = null;
  let hipTiltRaw = null;
  let forwardHead = null;
  let symmetry = null;

  if (leftShoulder && rightShoulder) {
    shoulderTiltRaw = Math.abs(lineTiltDegrees(leftShoulder, rightShoulder));
    // Clamp for display but keep raw for deltas
    shoulderTilt = Math.min(45, Math.abs(shoulderTiltRaw));
  }
  if (leftHip && rightHip) {
    hipTiltRaw = Math.abs(lineTiltDegrees(leftHip, rightHip));
    hipTilt = Math.min(45, Math.abs(hipTiltRaw));
  }
  if (nose && leftShoulder && rightShoulder) {
    // Approx forward head by horizontal offset of nose from shoulder midpoint relative to shoulder width
    const midShoulder = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
    const shoulderWidth = dist(leftShoulder, rightShoulder) || 1;
    forwardHead = Math.abs(nose.x - midShoulder.x) / shoulderWidth; // 0-1+ range
  } else if ((leftEar || rightEar) && leftShoulder && rightShoulder) {
    const ear = leftEar || rightEar;
    const midShoulder = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
    const shoulderWidth = dist(leftShoulder, rightShoulder) || 1;
    forwardHead = Math.abs(ear.x - midShoulder.x) / shoulderWidth;
  }

  if (leftShoulder && rightShoulder && leftHip && rightHip) {
    // Left vs right torso length
    const leftTorso = dist(leftShoulder, leftHip);
    const rightTorso = dist(rightShoulder, rightHip);
    const torsoDiffPct = Math.abs(leftTorso - rightTorso) / Math.max(leftTorso, rightTorso);
    symmetry = { torsoDiffPct };
  }

  return { shoulderTilt, hipTilt, forwardHead, symmetry, shoulderTiltRaw, hipTiltRaw };
}

function summarizeMetrics(m) {
  const recs = [];
  if (Number.isFinite(m.shoulderTilt)) {
    if (m.shoulderTilt > 5) recs.push('Possible shoulder height asymmetry');
  }
  if (Number.isFinite(m.hipTilt)) {
    if (m.hipTilt > 5) recs.push('Possible pelvic tilt asymmetry');
  }
  if (Number.isFinite(m.forwardHead)) {
    if (m.forwardHead > 0.15) recs.push('Possible forward head posture');
  }
  if (m.symmetry?.torsoDiffPct > 0.08) recs.push('Possible lateral torso imbalance');
  return recs;
}

function renderResults(report) {
  const results = $('#bs-results');
  results.innerHTML = '';

  const metricsCard = document.createElement('div');
  metricsCard.className = 'metric';
  metricsCard.innerHTML = `
    <h4>Posture & Symmetry</h4>
    <div class="kv"><span>Shoulder tilt</span><span>${formatDeg(report.metrics.shoulderTilt)}</span></div>
    <div class="kv"><span>Hip tilt</span><span>${formatDeg(report.metrics.hipTilt)}</span></div>
    <div class="kv"><span>Forward head offset</span><span>${formatPct(report.metrics.forwardHead)}</span></div>
    <div class="kv"><span>Torso L/R diff</span><span>${formatPct(report.metrics.symmetry?.torsoDiffPct)}</span></div>
  `;

  const insights = document.createElement('div');
  insights.className = 'metric';
  const recs = summarizeMetrics(report.metrics);
  insights.innerHTML = `
    <h4>Insights</h4>
    ${recs.length ? `<ul>${recs.map(r => `<li>${r}</li>`).join('')}</ul>` : '<p class="muted">No obvious issues detected. Keep good form!</p>'}
  `;

  const composition = document.createElement('div');
  composition.className = 'metric';
  composition.innerHTML = `
    <h4>Body Composition (estimate)</h4>
    <p class="muted">Advanced metrics like body fat % and muscle volume need a specialized model and body measurements. For now we show an estimate if BMI is available.</p>
    <div class="kv"><span>Body fat (est.)</span><span>${report.bodyFatEstimate ?? '—'}</span></div>
  `;

  results.appendChild(metricsCard);
  results.appendChild(insights);
  results.appendChild(composition);
}

function formatDeg(v) { return Number.isFinite(v) ? `${v.toFixed(1)}°` : '—'; }
function formatPct(v) { return Number.isFinite(v) ? `${(v*100).toFixed(0)}%` : '—'; }

function getBmiFromPage() {
  const text = document.getElementById('bmi-result')?.textContent || '';
  const m = text.match(/BMI:\s*(\d+(?:\.\d+)?)/i);
  return m ? parseFloat(m[1]) : null;
}

function bodyFatFromBMI(bmi, sex = 'male') {
  // Deurenberg formula (rough): BF% = 1.2*BMI + 0.23*age - 10.8*sex - 5.4 (sex: 1 male, 0 female)
  // We do not have age; assume 30 for placeholder, or fall back to BMI category mapping.
  const ageEl = document.getElementById('bmr-age');
  const age = ageEl ? parseFloat(ageEl.value) || 30 : 30;
  const sexEl = document.getElementById('bmr-sex');
  const sexVal = sexEl ? sexEl.value : sex;
  const sexBin = sexVal === 'male' ? 1 : 0;
  const bf = 1.2*bmi + 0.23*age - 10.8*sexBin - 5.4;
  return `${Math.max(3, Math.min(60, Math.round(bf)))}%`;
}

async function detectOnImageFile(file) {
  const det = await ensureDetector();
  const imgUrl = URL.createObjectURL(file);
  const img = await new Promise((resolve) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.src = imgUrl;
  });
  const poses = await det.estimatePoses(img, { maxPoses: 1, flipHorizontal: false });
  URL.revokeObjectURL(imgUrl);
  return poses[0]?.keypoints ?? [];
}

async function detectOnVideoFile(file) {
  const det = await ensureDetector();
  const videoEl = $('#bs-video-el');
  const url = URL.createObjectURL(file);
  await new Promise((resolve) => {
    videoEl.onloadeddata = () => resolve();
    videoEl.src = url;
    videoEl.muted = true;
    videoEl.playsInline = true;
  });
  // Grab a mid-frame snapshot
  try { await videoEl.play(); } catch {}
  videoEl.pause();
  const canvas = document.createElement('canvas');
  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  const poses = await det.estimatePoses(canvas, { maxPoses: 1, flipHorizontal: false });
  URL.revokeObjectURL(url);
  return poses[0]?.keypoints ?? [];
}

function clearUI() {
  selectedImages = [];
  selectedVideoFile = null;
  selectedVideos = [];
  const p = $('#bs-previews'); if (p) p.innerHTML = '';
  const vp = $('#bs-video-previews'); if (vp) vp.innerHTML = '';
  const vfile = $('#bs-video'); if (vfile) vfile.value = '';
  const ifile = $('#bs-images'); if (ifile) ifile.value = '';
  const res = $('#bs-results'); if (res) res.innerHTML = '';
  setStatus('');
  // Reset inputs
  const sex = $('#bs-sex'); if (sex) sex.value = 'male';
  const age = $('#bs-age'); if (age) age.value = '';
  const um = $('#bs-unit-metric'), ui = $('#bs-unit-imperial');
  if (um && ui) { um.checked = true; ui.checked = false; }
  toggleBsUnitFields();
  ['#bs-height-cm','#bs-weight-kg','#bs-feet','#bs-inches','#bs-pounds'].forEach(sel => { const el = $(sel); if (el) el.value = ''; });
  try { localStorage.removeItem('fitlife_scan_inputs'); } catch {}
  const wrap = document.querySelector('.annotated-wrap'); wrap?.classList.add('hidden');
  const canvas = document.getElementById('bs-annotated'); const ctx = canvas?.getContext && canvas.getContext('2d'); if (ctx && canvas) { ctx.clearRect(0, 0, canvas.width, canvas.height); }
}

function setStatus(msg) { const el = $('#bs-status'); if (el) el.textContent = msg; }

function buildReport(metrics) {
  const bmi = getBmiFromPage();
  let bf = null;
  if (Number.isFinite(bmi)) bf = bodyFatFromBMI(bmi);
  return { timestamp: new Date().toISOString(), metrics, bodyFatEstimate: bf };
}

function downloadReport(report) {
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'fitlife-body-scan.json';
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

function getScanInputs() {
  const sex = document.getElementById('bs-sex')?.value || 'male';
  const age = Number(document.getElementById('bs-age')?.value) || null;
  const isMetric = document.getElementById('bs-unit-metric')?.checked ?? true;
  let heightM = null, weightKg = null;
  if (isMetric) {
    const cm = Number(document.getElementById('bs-height-cm')?.value) || 0;
    const kg = Number(document.getElementById('bs-weight-kg')?.value) || 0;
    if (cm > 0) heightM = cm / 100;
    if (kg > 0) weightKg = kg;
  } else {
    const ft = Number(document.getElementById('bs-feet')?.value) || 0;
    const inch = Number(document.getElementById('bs-inches')?.value) || 0;
    const lb = Number(document.getElementById('bs-pounds')?.value) || 0;
    if (ft || inch) heightM = (((ft * 12) + inch) * 2.54) / 100;
    if (lb > 0) weightKg = lb * 0.45359237;
  }
  return { sex, age, heightM, weightKg, isMetric };
}

function computeBMI(heightM, weightKg) {
  if (!heightM || !weightKg) return null;
  return weightKg / (heightM * heightM);
}

function bodyFatFromInputs(bmi, sex, age) {
  if (!Number.isFinite(bmi)) return null;
  const sexBin = sex === 'male' ? 1 : 0;
  const useAge = Number.isFinite(age) ? age : 30;
  const bf = 1.2*bmi + 0.23*useAge - 10.8*sexBin - 5.4;
  return `${Math.max(3, Math.min(60, Math.round(bf)))}%`;
}

function persistScanInputs() {
  const data = getScanInputs();
  localStorage.setItem('fitlife_scan_inputs', JSON.stringify(data));
}

function loadScanInputs() {
  try {
    const raw = localStorage.getItem('fitlife_scan_inputs');
    if (!raw) return;
    const d = JSON.parse(raw);
    if (!d) return;
    document.getElementById('bs-sex').value = d.sex || 'male';
    if (Number.isFinite(d.age)) document.getElementById('bs-age').value = String(d.age);
    const metric = d.isMetric !== false; // default true
    document.getElementById('bs-unit-metric').checked = metric;
    document.getElementById('bs-unit-imperial').checked = !metric;
    toggleBsUnitFields();
    if (metric) {
      if (Number.isFinite(d.heightM)) document.getElementById('bs-height-cm').value = String(Math.round(d.heightM * 100));
      if (Number.isFinite(d.weightKg)) document.getElementById('bs-weight-kg').value = String(Math.round(d.weightKg));
    } else {
      // Best-effort reverse conversion
      if (Number.isFinite(d.heightM)) {
        const totalIn = d.heightM * 100 / 2.54;
        const ft = Math.floor(totalIn / 12);
        const inch = Math.round(totalIn - ft*12);
        document.getElementById('bs-feet').value = String(ft);
        document.getElementById('bs-inches').value = String(inch);
      }
      if (Number.isFinite(d.weightKg)) document.getElementById('bs-pounds').value = String(Math.round(d.weightKg / 0.45359237));
    }
  } catch {}
}

function toggleBsUnitFields() {
  const metric = document.getElementById('bs-unit-metric')?.checked ?? true;
  document.getElementById('bs-metric-fields').classList.toggle('hidden', !metric);
  document.getElementById('bs-imperial-fields').classList.toggle('hidden', metric);
}

function limbImbalanceFromKeypoints(kps) {
  const lShoulder = key(kps, 'left_shoulder');
  const rShoulder = key(kps, 'right_shoulder');
  const lElbow = key(kps, 'left_elbow');
  const rElbow = key(kps, 'right_elbow');
  const lWrist = key(kps, 'left_wrist');
  const rWrist = key(kps, 'right_wrist');
  const lHip = key(kps, 'left_hip');
  const rHip = key(kps, 'right_hip');
  const lKnee = key(kps, 'left_knee');
  const rKnee = key(kps, 'right_knee');
  const lAnkle = key(kps, 'left_ankle');
  const rAnkle = key(kps, 'right_ankle');

  const leftArm = (lShoulder && lElbow ? dist(lShoulder, lElbow) : 0) + (lElbow && lWrist ? dist(lElbow, lWrist) : 0);
  const rightArm = (rShoulder && rElbow ? dist(rShoulder, rElbow) : 0) + (rElbow && rWrist ? dist(rElbow, rWrist) : 0);
  const leftLeg = (lHip && lKnee ? dist(lHip, lKnee) : 0) + (lKnee && lAnkle ? dist(lKnee, lAnkle) : 0);
  const rightLeg = (rHip && rKnee ? dist(rHip, rKnee) : 0) + (rKnee && rAnkle ? dist(rKnee, rAnkle) : 0);

  const armDiff = Math.abs(leftArm - rightArm) / Math.max(leftArm, rightArm, 1);
  const legDiff = Math.abs(leftLeg - rightLeg) / Math.max(leftLeg, rightLeg, 1);
  return { armDiff, legDiff };
}

// Keep original renderResults; limb balance is added explicitly in analyze step
const _renderResults = renderResults;

function drawOverlay(baseImage, keypoints, metrics, limb, options = {}) {
  const canvas = document.getElementById('bs-annotated');
  const wrap = canvas?.parentElement;
  if (wrap) wrap.classList.remove('hidden');
  if (!canvas || !baseImage) return;
  const ctx = canvas.getContext('2d');
  const w = baseImage.videoWidth || baseImage.naturalWidth || baseImage.width;
  const h = baseImage.videoHeight || baseImage.naturalHeight || baseImage.height;
  if (!w || !h) return;
  canvas.width = w;
  canvas.height = h;
  ctx.clearRect(0,0,w,h);
  ctx.drawImage(baseImage, 0, 0, w, h);

  // Helpers
  const kpBy = new Map();
  keypoints.forEach(k => kpBy.set(k.name || k.part, k));
  const get = (n) => kpBy.get(n);
  const mid = (a, b) => (a && b) ? { x: (a.x + b.x)/2, y: (a.y + b.y)/2 } : null;
  const drawBadge = (x, y, text) => {
    ctx.font = `${Math.max(12, Math.floor(w/50))}px ui-sans-serif, system-ui`;
    const tw = ctx.measureText(text).width;
    const tx = Math.max(6, Math.min(w - tw - 6, x));
    const ty = Math.max(12, Math.min(h - 6, y));
    ctx.lineWidth = Math.max(2, Math.floor(w/600));
    ctx.strokeStyle = 'rgba(0,0,0,0.35)';
    ctx.strokeText(text, tx, ty);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, tx, ty);
  };
  const drawCornerText = (x, y, text) => {
    ctx.font = `${Math.max(12, Math.floor(w/60))}px ui-sans-serif, system-ui`;
    ctx.lineWidth = Math.max(2, Math.floor(w/800));
    ctx.strokeStyle = 'rgba(0,0,0,0.35)';
    ctx.strokeText(text, x, y);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, x, y);
  };

  // Skeleton lines
  const pairs = [
    ['left_shoulder','right_shoulder'], ['left_hip','right_hip'],
    ['left_shoulder','left_elbow'], ['left_elbow','left_wrist'],
    ['right_shoulder','right_elbow'], ['right_elbow','right_wrist'],
    ['left_hip','left_knee'], ['left_knee','left_ankle'],
    ['right_hip','right_knee'], ['right_knee','right_ankle'],
    ['left_shoulder','left_hip'], ['right_shoulder','right_hip']
  ];
  ctx.lineWidth = Math.max(2, w/400);
  ctx.strokeStyle = 'rgba(7,192,162,0.9)';
  pairs.forEach(([a,b]) => { const p=get(a), q=get(b); if (p && q) { ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke(); } });

  // Points
  keypoints.forEach(k => { if (!k || (k.score !== undefined && k.score < 0.3)) return; ctx.fillStyle = '#07c0a2'; ctx.beginPath(); ctx.arc(k.x, k.y, Math.max(3, w/200), 0, Math.PI*2); ctx.fill(); });

  // Metric overlays
  const ls = get('left_shoulder'), rs = get('right_shoulder');
  const lh = get('left_hip'), rh = get('right_hip');
  const nose = get('nose');
  const ms = mid(ls, rs), mh = mid(lh, rh);

  // Title/debug
  ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fillRect(8, 8, 220, 30);
  ctx.fillStyle = '#fff'; ctx.font = `${Math.max(14, Math.floor(w/40))}px ui-sans-serif, system-ui`; ctx.fillText('FitLife Body Scan', 16, 30);

  // Ideal landmark lines (dashed)
  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = 'rgba(255,255,255,0.6)';
  ctx.lineWidth = Math.max(1, w/500);
  if (ms) { ctx.beginPath(); ctx.moveTo(8, ms.y); ctx.lineTo(w - 8, ms.y); ctx.stroke(); }
  if (mh) { ctx.beginPath(); ctx.moveTo(8, mh.y); ctx.lineTo(w - 8, mh.y); ctx.stroke(); }
  if (ms) { ctx.beginPath(); ctx.moveTo(ms.x, 8); ctx.lineTo(ms.x, h - 8); ctx.stroke(); }
  ctx.restore();

  // Shoulder tilt
  if (ls && rs && ms && metrics && Number.isFinite(metrics.shoulderTilt)) {
    ctx.strokeStyle = 'rgba(255,255,0,0.9)'; ctx.beginPath(); ctx.moveTo(ls.x, ls.y); ctx.lineTo(rs.x, rs.y); ctx.stroke();
    drawBadge(ms.x, ms.y - 28, `Shoulders level: ${metrics.shoulderTilt.toFixed(1)}° off (≤ 2°)`);
  }
  // Hip tilt
  if (lh && rh && mh && metrics && Number.isFinite(metrics.hipTilt)) {
    ctx.strokeStyle = 'rgba(255,140,0,0.9)'; ctx.beginPath(); ctx.moveTo(lh.x, lh.y); ctx.lineTo(rh.x, rh.y); ctx.stroke();
    drawBadge(mh.x, mh.y + 8, `Hips level: ${metrics.hipTilt.toFixed(1)}° off (≤ 2°)`);
  }
  // Forward head offset
  if (ms && nose && metrics && Number.isFinite(metrics.forwardHead)) {
    ctx.strokeStyle = 'rgba(173,216,230,0.9)'; ctx.beginPath(); ctx.moveTo(nose.x, nose.y); ctx.lineTo(ms.x, nose.y); ctx.stroke();
    drawBadge(nose.x, Math.max(4, nose.y - 24), `Head forward: ${(metrics.forwardHead*100).toFixed(0)}% (≤ 5%)`);
  }
  // Torso diff
  if (ls && lh && rs && rh && metrics && metrics.symmetry && Number.isFinite(metrics.symmetry.torsoDiffPct)) {
    const midTorso = mid(ms || ls, mh || lh) || { x: (w*0.05), y: (h*0.5) };
    drawBadge(midTorso.x, midTorso.y, `Torso symmetry: ${(metrics.symmetry.torsoDiffPct*100).toFixed(0)}% diff (≤ 5%)`);
  }
  // Limb diffs
  if (limb && Number.isFinite(limb.armDiff)) {
    drawBadge(w*0.04, h*0.08, `Arms balance: ${(limb.armDiff*100).toFixed(0)}% diff (≤ 10%)`);
  }
  if (limb && Number.isFinite(limb.legDiff)) {
    drawBadge(w*0.04, h*0.08 + Math.max(28, h/20), `Legs balance: ${(limb.legDiff*100).toFixed(0)}% diff (≤ 10%)`);
  }

  // Scores and summaries (top-right)
  const scores = options.scores || null;
  const summaries = options.summaries || [];
  let y = 20;
  if (scores){
    drawCornerText(w - 210, y, `Posture score: ${scores.posture}`); y += 18;
    drawCornerText(w - 210, y, `Symmetry score: ${scores.symmetry}`); y += 22;
  }
  summaries.forEach(s => { drawCornerText(w - 210, y, s); y += 18; });
}

// Wire unit toggles
$('#bs-unit-metric')?.addEventListener('change', () => { toggleBsUnitFields(); persistScanInputs(); });
$('#bs-unit-imperial')?.addEventListener('change', () => { toggleBsUnitFields(); persistScanInputs(); });
['bs-sex','bs-age','bs-height-cm','bs-weight-kg','bs-feet','bs-inches','bs-pounds'].forEach(id => {
  document.getElementById(id)?.addEventListener('input', persistScanInputs);
});

// Import from calculators
$('#bs-import-btn')?.addEventListener('click', () => {
  try {
    // Sex and age from BMR section
    const sexEl = document.getElementById('bmr-sex');
    const ageEl = document.getElementById('bmr-age');
    if (sexEl) document.getElementById('bs-sex').value = sexEl.value;
    if (ageEl && ageEl.value) document.getElementById('bs-age').value = ageEl.value;
    // Prefer metric values from calculators if available
    const hcm = document.getElementById('bmr-height-cm')?.value;
    const wkg = document.getElementById('bmr-weight-kg')?.value;
    if (hcm || wkg) {
      document.getElementById('bs-unit-metric').checked = true;
      document.getElementById('bs-unit-imperial').checked = false;
      toggleBsUnitFields();
      if (hcm) document.getElementById('bs-height-cm').value = hcm;
      if (wkg) document.getElementById('bs-weight-kg').value = wkg;
    }
    persistScanInputs();
  } catch {}
});

// Event wiring for media inputs
$('#bs-images')?.addEventListener('change', (e) => {
  selectedImages = Array.from(e.target.files || []);
  const previews = $('#bs-previews');
  previews.innerHTML = '';
  selectedImages.slice(0, 8).forEach(f => drawImagePreview(f, previews));
});

$('#bs-video')?.addEventListener('change', (e) => {
  selectedVideos = Array.from(e.target.files || []);
  selectedVideoFile = selectedVideos[0] || null;
  const vprev = $('#bs-video-previews'); if (vprev) { vprev.innerHTML = ''; selectedVideos.slice(0, 8).forEach(f => drawVideoPreview(f, vprev)); }
});

function attachCtas() {
  const analyzeBtn = document.getElementById('bs-analyze-btn');
  if (analyzeBtn && !analyzeBtn.dataset.wired) {
    analyzeBtn.addEventListener('click', analyzeAndRender);
    analyzeBtn.dataset.wired = '1';
  }
  const clearBtn = document.getElementById('bs-clear-btn');
  if (clearBtn && !clearBtn.dataset.wired) {
    clearBtn.addEventListener('click', () => clearUI());
    clearBtn.dataset.wired = '1';
  }
  const importBtn = document.getElementById('bs-import-btn');
  if (importBtn && !importBtn.dataset.wired) {
    importBtn.addEventListener('click', () => {
      try {
        const sexEl = document.getElementById('bmr-sex');
        const ageEl = document.getElementById('bmr-age');
        if (sexEl) document.getElementById('bs-sex').value = sexEl.value;
        if (ageEl && ageEl.value) document.getElementById('bs-age').value = ageEl.value;
        const hcm = document.getElementById('bmr-height-cm')?.value;
        const wkg = document.getElementById('bmr-weight-kg')?.value;
        if (hcm || wkg) {
          document.getElementById('bs-unit-metric').checked = true;
          document.getElementById('bs-unit-imperial').checked = false;
          toggleBsUnitFields();
          if (hcm) document.getElementById('bs-height-cm').value = hcm;
          if (wkg) document.getElementById('bs-weight-kg').value = wkg;
        }
        persistScanInputs();
      } catch {}
    });
    importBtn.dataset.wired = '1';
  }
}

function attachGuidedPreviews(){
  const make = (inputId, previewId) => {
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    if (!input || !preview) return;
    input.addEventListener('change', ()=>{
      preview.innerHTML = '';
      const f = input.files?.[0];
      if (!f) return;
      const url = URL.createObjectURL(f);
      const img = new Image();
      img.onload = ()=> URL.revokeObjectURL(url);
      img.src = url; img.style.width='100%'; img.style.borderRadius='10px';
      const wrap = document.createElement('div'); wrap.className='thumb';
      wrap.appendChild(img); preview.appendChild(wrap);
    });
  };
  make('gc-front','gc-front-preview');
  make('gc-side','gc-side-preview');
  make('gc-back','gc-back-preview');
}

document.addEventListener('DOMContentLoaded', () => {
  attachCtas();
  attachGuidedPreviews();
});

// Also reattach when the section is clicked (defensive)
$('#body-scan')?.addEventListener('click', () => attachCtas(), { once: false });

// Modify analysis to compute composition from inputs
async function analyzeAndRender() {
  try {
    setStatus('Loading model…');
    await ensureDetector();
    setStatus('Analyzing…');

    let kps = null;
    let baseImage = null;
    if (selectedImages.length) {
      for (const f of selectedImages) {
        const keypoints = await detectOnImageFile(f);
        if (keypoints && keypoints.length) { kps = keypoints; const url=URL.createObjectURL(f); const img=await new Promise(r=>{const i=new Image(); i.onload=()=>r(i); i.src=url;}); URL.revokeObjectURL(url); baseImage = img; break; }
      }
    } else if (selectedVideos.length) {
      const det = await ensureDetector();
      for (const vf of selectedVideos) {
        const videoEl = document.getElementById('bs-video-el');
        const url = URL.createObjectURL(vf);
        await new Promise((resolve) => { videoEl.onloadeddata = () => resolve(); videoEl.src = url; });
        try { await videoEl.play(); } catch {}
        videoEl.pause();
        const canvas = document.createElement('canvas'); canvas.width = videoEl.videoWidth; canvas.height = videoEl.videoHeight;
        const ctx = canvas.getContext('2d'); ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
        const poses = await det.estimatePoses(canvas, { maxPoses: 1, flipHorizontal: false });
        URL.revokeObjectURL(url);
        if (poses[0]?.keypoints?.length) { kps = poses[0].keypoints; baseImage = canvas; break; }
      }
    } else {
      setStatus('Please upload photos or a video first.');
      return;
    }

    if (!kps || !kps.length) { setStatus('No person detected. Try a clearer, well-lit image.'); return; }

    // Multi-sample averaging for high-accuracy
    const hiacc = document.getElementById('bs-hiacc')?.checked;
    let metrics = computeMetricsFromKeypoints(kps);
    if (hiacc){
      const det = await ensureDetector();
      const samples = [metrics];
      for (let i=0;i<4;i++){ const poses = await det.estimatePoses(baseImage || (typeof img!=='undefined'?img:document.createElement('canvas')), { maxPoses: 1, flipHorizontal: false }); if (poses[0]?.keypoints?.length){ samples.push(computeMetricsFromKeypoints(poses[0].keypoints)); } }
      const avg = (arr, sel)=>{ const v=arr.map(sel).filter(Number.isFinite); return v.length? v.reduce((a,b)=>a+b,0)/v.length : null; };
      metrics = {
        shoulderTilt: avg(samples, s=>s.shoulderTilt),
        hipTilt: avg(samples, s=>s.hipTilt),
        forwardHead: avg(samples, s=>s.forwardHead),
        symmetry: { torsoDiffPct: avg(samples, s=>s.symmetry?.torsoDiffPct) },
        shoulderTiltRaw: avg(samples, s=>s.shoulderTiltRaw),
        hipTiltRaw: avg(samples, s=>s.hipTiltRaw)
      };
    }
    const limb = limbImbalanceFromKeypoints(kps);

    // Draw overlay with metrics
    drawOverlay(baseImage, kps, metrics, limb);
    try { document.getElementById('bs-annotated')?.scrollIntoView({ behavior: 'smooth', block: 'center' }); } catch {}
    // Snapshot into results
    try {
      const snap = document.getElementById('bs-annotated').toDataURL('image/png');
      const imgEl = new Image(); imgEl.src = snap; imgEl.style.width = '100%'; imgEl.style.borderRadius = '12px'; imgEl.style.border = '1px solid var(--border)';
      const res = document.getElementById('bs-results'); if (res){ const holder = document.createElement('div'); holder.className = 'metric'; holder.innerHTML = '<h4>Annotated view</h4>'; holder.appendChild(imgEl); res.prepend(holder); }
    } catch {}

    const { sex, age, heightM, weightKg } = getScanInputs();
    const bmi = computeBMI(heightM, weightKg);
    const bf = bodyFatFromInputs(bmi, sex, age);

    const report = { timestamp: new Date().toISOString(), metrics, limb, bodyFatEstimate: bf, bmi: bmi ? Number(bmi.toFixed(1)) : null, inputs: { sex, age, heightM, weightKg } };
    lastReport = report;

    const results = document.getElementById('bs-results');
    results.innerHTML = '';
    _renderResults({ metrics, bodyFatEstimate: bf });

    const head = document.createElement('div');
    head.className = 'metric';
    head.innerHTML = `
      <h4>Body Data</h4>
      <div class="kv"><span>Sex</span><span>${sex}</span></div>
      <div class="kv"><span>Age</span><span>${Number.isFinite(age) ? age : '—'}</span></div>
      <div class="kv"><span>BMI</span><span>${Number.isFinite(bmi) ? bmi.toFixed(1) : '—'}</span></div>
      <div class="kv"><span>Body fat (est.)</span><span>${bf ?? '—'}</span></div>
    `;
    results.appendChild(head);

    const limbCard = document.createElement('div');
    limbCard.className = 'metric';
    limbCard.innerHTML = `
      <h4>Limb Balance</h4>
      <div class="kv"><span>Arms balance</span><span>${formatPct(limb?.armDiff)}</span></div>
      <div class="kv"><span>Legs balance</span><span>${formatPct(limb?.legDiff)}</span></div>
    `;
    results.appendChild(limbCard);

    const explain = document.createElement('div');
    explain.className = 'metric';
    explain.innerHTML = `
      <h4>How to read these</h4>
      <ul>
        <li><strong>Shoulders level</strong>: how tilted shoulders are left-to-right. ≤ 2° is usually fine.</li>
        <li><strong>Hips level</strong>: pelvis tilt left-to-right. ≤ 2° is usually fine.</li>
        <li><strong>Head forward</strong>: how far the head sits in front of body center (scaled to shoulder width). ≤ 5% is good.</li>
        <li><strong>Torso symmetry</strong>: left vs right torso length difference. ≤ 5% is good.</li>
        <li><strong>Arms/Legs balance</strong>: left vs right length difference in this snapshot. ≤ 10% is fine.</li>
        <li><em>Tip</em>: These are visual estimates from one frame. For best accuracy, use front/side photos with neutral stance and good lighting.</li>
      </ul>
    `;
    results.appendChild(explain);
    try { results.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch {}

    // Friendly, side-specific summary
    const ls = key(kps,'left_shoulder'), rs = key(kps,'right_shoulder');
    const lh = key(kps,'left_hip'), rh = key(kps,'right_hip');
    const parts = [];
    if (Number.isFinite(metrics.shoulderTilt)){
      let t = 'Shoulders: level';
      if (ls && rs){
        const shoulderWidth = Math.hypot(ls.x - rs.x, ls.y - rs.y) || 1;
        const vr = Math.abs(ls.y - rs.y) / shoulderWidth; // vertical gap as % of shoulder width
        const vrPct = Math.round(vr * 100);
        const side = (ls.y < rs.y) ? 'left' : (rs.y < ls.y) ? 'right' : 'level';
        if (side !== 'level') t = `Shoulders: ${side} higher by ${vrPct}% (≈ ${metrics.shoulderTilt.toFixed(1)}°)`;
      }
      parts.push(t);
    }
    if (Number.isFinite(metrics.hipTilt)){
      let t = 'Hips: level';
      if (lh && rh){
        if (lh.y < rh.y && metrics.hipTilt > 0.5) t = `Hips: left higher by ${metrics.hipTilt.toFixed(1)}°`;
        else if (rh.y < lh.y && metrics.hipTilt > 0.5) t = `Hips: right higher by ${metrics.hipTilt.toFixed(1)}°`;
      }
      parts.push(t);
    }
    if (Number.isFinite(metrics.forwardHead)){
      parts.push((metrics.forwardHead*100) <= 5 ? 'Head: neutral' : `Head: ${Math.round(metrics.forwardHead*100)}% forward`);
    }
    if (metrics.symmetry && Number.isFinite(metrics.symmetry.torsoDiffPct)){
      parts.push((metrics.symmetry.torsoDiffPct*100) <= 5 ? 'Torso: symmetric' : `Torso: ${Math.round(metrics.symmetry.torsoDiffPct*100)}% L/R difference`);
    }
    if (limb && Number.isFinite(limb.armDiff) && Number.isFinite(limb.legDiff)){
      const armPct = Math.round(limb.armDiff*100);
      const legPct = Math.round(limb.legDiff*100);
      if (armPct <= 10 && legPct <= 10) parts.push('Arms/Legs: balanced');
      else parts.push(`Arms: ${armPct}% diff · Legs: ${legPct}% diff`);
    }
    setStatus(parts.join(' · '));
  } catch (err) {
    console.error(err);
    setStatus('Analysis failed. Please try different photos.');
  }
}

// Global event delegation for CTAs
if (!window.__fitlife_scan_delegated) {
  document.addEventListener('click', (e) => {
    const target = e.target instanceof Element ? e.target : null;
    if (!target) return;
    if (target.closest('#bs-analyze-btn')) {
      e.preventDefault();
      analyzeAndRender();
    } else if (target.closest('#bs-clear-btn')) {
      e.preventDefault();
      clearUI();
    } else if (target.closest('#bs-download-report')) {
      e.preventDefault();
      if (lastReport) downloadReport(lastReport);
      else setStatus('Run an analysis first to download a report.');
    }
  });
  window.__fitlife_scan_delegated = true;
}

// Init on load
loadScanInputs();

// --- Guided Capture (Beta) ---
function readImage(file){ return new Promise(res=>{ const url=URL.createObjectURL(file); const img=new Image(); img.onload=()=>{ URL.revokeObjectURL(url); res(img); }; img.src=url; }); }
function quickLuma(canvas){ const ctx=canvas.getContext('2d'); const d=ctx.getImageData(0,0,canvas.width,canvas.height).data; let s=0; for(let i=0;i<d.length;i+=4){ s+=0.2126*d[i]+0.7152*d[i+1]+0.0722*d[i+2]; } return s/(d.length/4)/255; }
function centerDetect(kps){ const ls=key(kps,'left_shoulder'), rs=key(kps,'right_shoulder'); if(!ls||!rs) return null; return { x:(ls.x+rs.x)/2, y:(ls.y+rs.y)/2 }; }
async function analyzePoseOnImage(detectorOrNull, img){ try{ if(detectorOrNull){ const poses=await detectorOrNull.estimatePoses(img,{maxPoses:1,flipHorizontal:false}); return poses[0]?.keypoints||[]; } else { return []; } }catch{ return []; } }

async function guidedAnalyze(){
  const q = $('#gc-quality'); const scoresEl = $('#gc-scores'); const summaryEl = $('#gc-summary');
  q.textContent = 'Checking photos…'; scoresEl.innerHTML=''; summaryEl.innerHTML='';
  const frontF = $('#gc-front')?.files?.[0]; const sideF = $('#gc-side')?.files?.[0]; const backF = $('#gc-back')?.files?.[0];
  if(!frontF && !sideF && !backF){ q.textContent='Please add at least one photo (front, side, or back).'; return; }
  await ensureDetector();

  const checks=[]; const allKps=[];
  for (const [name, file] of [['front',frontF],['side',sideF],['back',backF]]){
    if(!file) continue;
    const img = await readImage(file);
    // brightness check
    const c=document.createElement('canvas'); c.width=img.naturalWidth; c.height=img.naturalHeight; const ctx=c.getContext('2d'); ctx.drawImage(img,0,0); const luma=quickLuma(c);
    if (luma < 0.25) checks.push(`${name}: image too dark`);
    if (luma > 0.95) checks.push(`${name}: image too bright`);
    // framing check
    const kps = await analyzePoseOnImage(detector, img);
    allKps.push({ name, kps, img });
    const center = centerDetect(kps);
    if (!kps.length) checks.push(`${name}: no person detected`);
    if (center){ const cx=center.x/img.naturalWidth; if (cx < 0.3 || cx > 0.7) checks.push(`${name}: please center your body`); }
  }
  if (checks.length){ q.textContent = checks.join(' · '); } else { q.textContent = 'Looks good!'; }

  // Fuse metrics across available views (take best/average where applicable)
  function collectMetrics(view){ if(!view.kps.length) return null; const m=computeMetricsFromKeypoints(view.kps); return m; }
  const metricsList = allKps.map(collectMetrics).filter(Boolean);
  if (!metricsList.length){ summaryEl.innerHTML = '<p class="muted">No valid poses detected. Try clearer, well-lit photos.</p>'; return; }
  const fused = metricsList.reduce((acc,m)=>({
    shoulderTilt:(acc.shoulderTilt||0)+m.shoulderTilt,
    hipTilt:(acc.hipTilt||0)+m.hipTilt,
    forwardHead:(acc.forwardHead||0)+m.forwardHead,
    torso:(acc.torso||0)+((m.symmetry?.torsoDiffPct)||0)
  }),{});
  const n=metricsList.length;
  const fusedMetrics={
    shoulderTilt:fused.shoulderTilt/n,
    hipTilt:fused.hipTilt/n,
    forwardHead:fused.forwardHead/n,
    torsoDiffPct:fused.torso/n
  };

  // Scores (0-100): invert normalized error into a score
  function score(val, good){ const pct = Math.max(0, 1 - (val/good)); return Math.round(Math.max(0, Math.min(1, pct))*100); }
  const postureScore = Math.round(( score(Math.abs(fusedMetrics.shoulderTilt),2) + score(Math.abs(fusedMetrics.hipTilt),2) + score(fusedMetrics.forwardHead*100,5) )/3);
  const symmetryScore = Math.round(( score(fusedMetrics.torsoDiffPct*100,5) ) );

  function chip(label, s){ const cls = s>=85? 'good' : s>=65? 'warn':'bad'; return `<div class="score ${cls}"><span class="dot"></span><span>${label}: ${s}</span></div>`; }
  scoresEl.innerHTML = chip('Posture score', postureScore) + ' ' + chip('Symmetry score', symmetryScore);

  // Region summaries
  const shoulderStr = Math.abs(fusedMetrics.shoulderTilt)<=2 ? 'Shoulders: level' : `Shoulders: ${fusedMetrics.shoulderTilt.toFixed(1)}° high on one side`;
  const hipStr = Math.abs(fusedMetrics.hipTilt)<=2 ? 'Hips: level' : `Hips: ${fusedMetrics.hipTilt.toFixed(1)}° high on one side`;
  const headStr = (fusedMetrics.forwardHead*100)<=5 ? 'Head: neutral' : `Head: ${Math.round(fusedMetrics.forwardHead*100)}% forward`;
  const torsoStr = (fusedMetrics.torsoDiffPct*100)<=5 ? 'Torso: symmetric' : `Torso: ${Math.round(fusedMetrics.torsoDiffPct*100)}% left/right difference`;
  summaryEl.innerHTML = `<ul><li>${shoulderStr}</li><li>${hipStr}</li><li>${headStr}</li><li>${torsoStr}</li></ul>`;

  // Always draw overlay with fused metrics on the first available view
  const bestView = allKps.find(v => v.kps && v.kps.length) || allKps[0];
  if (bestView){
    drawOverlay(
      bestView.img,
      bestView.kps,
      { shoulderTilt:fusedMetrics.shoulderTilt, hipTilt:fusedMetrics.hipTilt, forwardHead:fusedMetrics.forwardHead, symmetry:{ torsoDiffPct:fusedMetrics.torsoDiffPct } },
      null,
      { scores: { posture: postureScore, symmetry: symmetryScore }, summaries: [shoulderStr, hipStr, headStr, torsoStr] }
    );
    try { document.getElementById('bs-annotated')?.scrollIntoView({ behavior:'smooth', block:'center' }); } catch {}
    // Also snapshot into results panel
    try {
      const snap = document.getElementById('bs-annotated').toDataURL('image/png');
      const imgEl = new Image(); imgEl.src = snap; imgEl.style.width = '100%'; imgEl.style.borderRadius = '12px'; imgEl.style.border = '1px solid var(--border)';
      const holder = document.createElement('div'); holder.className = 'metric'; holder.innerHTML = '<h4>Annotated view</h4>';
      holder.appendChild(imgEl);
      summaryEl.parentNode.insertBefore(holder, summaryEl);
    } catch {}
  }
}

$('#gc-analyze-btn')?.addEventListener('click', guidedAnalyze);
window.gcAnalyze = guidedAnalyze;

// Delegate guided CTAs to be extra-robust
document.addEventListener('click', (e)=>{
  const t = e.target instanceof Element ? e.target : null; if (!t) return;
  if (t.closest('#gc-analyze-btn')){ e.preventDefault(); guidedAnalyze().catch(err=>{ const q=$('#gc-quality'); if(q) q.textContent = 'Analysis failed. Try clearer photos.'; }); }
  if (t.closest('#gc-set-baseline')){ e.preventDefault(); const btn=$('#gc-set-baseline'); btn?.setAttribute('disabled','true'); (async ()=>{ try{ const frontF = $('#gc-front')?.files?.[0]; const sideF = $('#gc-side')?.files?.[0]; const backF = $('#gc-back')?.files?.[0]; await ensureDetector(); const views=[]; for (const f of [frontF, sideF, backF].filter(Boolean)){ const img=await readImage(f); const kps=await analyzePoseOnImage(detector,img); views.push({ kps }); } const metricsList = views.map(v=>computeMetricsFromKeypoints(v.kps)).filter(Boolean); if(!metricsList.length){ alert('No valid poses to set baseline.'); return; } const fused = metricsList.reduce((acc,m)=>({ shoulderTilt:(acc.shoulderTilt||0)+m.shoulderTilt, hipTilt:(acc.hipTilt||0)+m.hipTilt, forwardHead:(acc.forwardHead||0)+m.forwardHead, torso:(acc.torso||0)+((m.symmetry?.torsoDiffPct)||0)}),{}); const n=metricsList.length; const fusedMetrics={ shoulderTilt:fused.shoulderTilt/n, hipTilt:fused.hipTilt/n, forwardHead:fused.forwardHead/n, torsoDiffPct:fused.torso/n }; saveBaseline({ date:new Date().toISOString(), metrics:fusedMetrics }); alert('Baseline saved.'); } catch{ alert('Failed to save baseline.'); } finally { btn?.removeAttribute('disabled'); } })(); }
});

// Baseline storage and trend
function loadBaseline(){ try { return JSON.parse(localStorage.getItem('fitlife_baseline')||'null'); } catch { return null; } }
function saveBaseline(data){ localStorage.setItem('fitlife_baseline', JSON.stringify(data)); }
$('#gc-set-baseline')?.addEventListener('click', async ()=>{
  const frontF = $('#gc-front')?.files?.[0]; const sideF = $('#gc-side')?.files?.[0]; const backF = $('#gc-back')?.files?.[0];
  await ensureDetector();
  const views=[];
  for (const f of [frontF, sideF, backF].filter(Boolean)){
    const img=await readImage(f); const kps=await analyzePoseOnImage(detector,img); views.push({ imgW:img.naturalWidth, imgH:img.naturalHeight, kps });
  }
  const metricsList = views.map(v=>computeMetricsFromKeypoints(v.kps)).filter(Boolean);
  if (!metricsList.length){ alert('No valid poses to set baseline.'); return; }
  const fused = metricsList.reduce((acc,m)=>({ shoulderTilt:(acc.shoulderTilt||0)+m.shoulderTilt, hipTilt:(acc.hipTilt||0)+m.hipTilt, forwardHead:(acc.forwardHead||0)+m.forwardHead, torso:(acc.torso||0)+((m.symmetry?.torsoDiffPct)||0)}),{});
  const n=metricsList.length;
  const fusedMetrics={ shoulderTilt:fused.shoulderTilt/n, hipTilt:fused.hipTilt/n, forwardHead:fused.forwardHead/n, torsoDiffPct:fused.torso/n };
  const now=new Date().toISOString();
  saveBaseline({ date: now, metrics: fusedMetrics });
  alert('Baseline saved. Track changes weekly.');
});
window.gcSetBaseline = async ()=>{
  const frontF = $('#gc-front')?.files?.[0]; const sideF = $('#gc-side')?.files?.[0]; const backF = $('#gc-back')?.files?.[0];
  await ensureDetector();
  const views=[];
  for (const f of [frontF, sideF, backF].filter(Boolean)){
    const img=await readImage(f); const kps=await analyzePoseOnImage(detector,img); views.push({ kps });
  }
  const metricsList = views.map(v=>computeMetricsFromKeypoints(v.kps)).filter(Boolean);
  if (!metricsList.length){ alert('No valid poses to set baseline.'); return; }
  const fused = metricsList.reduce((acc,m)=>({ shoulderTilt:(acc.shoulderTilt||0)+m.shoulderTilt, hipTilt:(acc.hipTilt||0)+m.hipTilt, forwardHead:(acc.forwardHead||0)+m.forwardHead, torso:(acc.torso||0)+((m.symmetry?.torsoDiffPct)||0)}),{});
  const n=metricsList.length; const fusedMetrics={ shoulderTilt:fused.shoulderTilt/n, hipTilt:fused.hipTilt/n, forwardHead:fused.forwardHead/n, torsoDiffPct:fused.torso/n };
  saveBaseline({ date:new Date().toISOString(), metrics:fusedMetrics }); alert('Baseline saved.');
};

// Weekly check-in trend (simple delta display)
function showTrend(current){ const base=loadBaseline(); if(!base) return ''; function fmt(v,u){ return (v>0?'+':'')+v+u; } const dS=(current.shoulderTilt-(base.metrics.shoulderTilt||0)).toFixed(1); const dH=(current.hipTilt-(base.metrics.hipTilt||0)).toFixed(1); const dF=Math.round(current.forwardHead*100 - (base.metrics.forwardHead*100||0)); const dT=Math.round(current.torsoDiffPct*100 - (base.metrics.torsoDiffPct*100||0)); return `<div class="metric"><h4>Trend vs baseline (${base.date.slice(0,10)})</h4><div class="kv"><span>Shoulders</span><span>${fmt(dS,'°')}</span></div><div class="kv"><span>Hips</span><span>${fmt(dH,'°')}</span></div><div class="kv"><span>Head forward</span><span>${fmt(dF,'%')}</span></div><div class="kv"><span>Torso symmetry</span><span>${fmt(dT,'%')}</span></div></div>`; }

// Hook trend after guided analysis
(async ()=>{
  const btn=$('#gc-analyze-btn'); if(!btn) return; btn.addEventListener('click', async ()=>{ setTimeout(()=>{ const scoresEl=$('#gc-scores'); if(!scoresEl) return; // recompute fused from last summary
    // This block assumes guidedAnalyze just ran; for brevity, re-run basic fuse on existing previews is omitted.
  }, 50); });
})();