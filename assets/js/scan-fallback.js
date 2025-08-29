(function(){
  const $ = (s, r=document)=>r.querySelector(s);
  const statusEl = $('#bs-status');
  function setStatus(msg){ if (statusEl) statusEl.textContent = msg; }
  if (!statusEl) return;

  async function loadScript(src){
    return new Promise((resolve, reject)=>{
      const s = document.createElement('script');
      s.src = src; s.async = true; s.crossOrigin = 'anonymous';
      s.onload = resolve; s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  async function importModule(urls){
    for (const url of urls){
      try { const mod = await import(/* webpackIgnore: true */ url); if (mod) return mod; } catch (_) {}
    }
    return null;
  }

  async function resolvePoseDetection(){
    let PD = window.poseDetection || window.poseDetectionModule || window.pose_detection || window['pose-detection'];
    if (!PD) {
      try { await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.3.1/dist/pose-detection.min.js'); } catch {}
      PD = window.poseDetection || window.poseDetectionModule || window.pose_detection || window['pose-detection'];
    }
    if (!PD) {
      try { await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.3.1/dist/pose-detection.js'); } catch {}
      PD = window.poseDetection || window.poseDetectionModule || window.pose_detection || window['pose-detection'];
    }
    if (!PD) {
      try { await loadScript('https://unpkg.com/@tensorflow-models/pose-detection@2.3.1/dist/pose-detection.min.js'); } catch {}
      PD = window.poseDetection || window.poseDetectionModule || window.pose_detection || window['pose-detection'];
    }
    if (!PD) {
      try { await loadScript('https://unpkg.com/@tensorflow-models/pose-detection@2.3.1/dist/pose-detection.js'); } catch {}
      PD = window.poseDetection || window.poseDetectionModule || window.pose_detection || window['pose-detection'];
    }
    if (!PD) {
      // Cloudflare CDN as last resort
      try { await loadScript('https://cdnjs.cloudflare.com/ajax/libs/tensorflow-models-pose-detection/2.3.1/pose-detection.min.js'); } catch {}
      PD = window.poseDetection || window.poseDetectionModule || window.pose_detection || window['pose-detection'];
    }
    if (!PD) {
      // Try ESM dynamic import
      const esm = await importModule([
        'https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.3.1/dist/pose-detection.esm.js',
        'https://unpkg.com/@tensorflow-models/pose-detection@2.3.1/dist/pose-detection.esm.js',
        'https://esm.run/@tensorflow-models/pose-detection@2.3.1'
      ]);
      if (esm) return esm;
    }
    return PD;
  }

  async function createDetectorMoveNet(){
    setStatus('Loading model…');
    // Prefer ESM dynamic import (avoids UMD 404/nosniff on some CDNs)
    try {
      const TF = await importModule([
        'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.13.0/dist/tf.esm.js',
        'https://unpkg.com/@tensorflow/tfjs@4.13.0/dist/tf.esm.js',
        'https://esm.run/@tensorflow/tfjs@4.13.0'
      ]);
      if (TF && !window.tf) window.tf = TF;
      const TF_WASM = await importModule([
        'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.13.0/dist/tf-backend-wasm.esm.js',
        'https://unpkg.com/@tensorflow/tfjs-backend-wasm@4.13.0/dist/tf-backend-wasm.esm.js',
        'https://esm.run/@tensorflow/tfjs-backend-wasm@4.13.0'
      ]);
      if (TF_WASM && TF_WASM.setWasmPaths) {
        TF_WASM.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.13.0/dist/');
      }
    } catch {}
    // Primary TFJS from jsdelivr
    try { await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.13.0/dist/tf.min.js'); } catch {}
    if (!window.tf) { try { await loadScript('https://unpkg.com/@tensorflow/tfjs@4.13.0/dist/tf.min.js'); } catch {}
    }
    try { await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.13.0/dist/tf-backend-wasm.min.js'); } catch {}
    if (!(window.tf && window.tf.setWasmPaths)) { try { await loadScript('https://unpkg.com/@tensorflow/tfjs-backend-wasm@4.13.0/dist/tf-backend-wasm.min.js'); } catch {} }
    if (!(window.tf && window.tf.setWasmPaths)) { try { await loadScript('https://cdnjs.cloudflare.com/ajax/libs/tensorflow/4.13.0/tf.min.js'); } catch {} }
    const PD = await resolvePoseDetection();
    if (!PD || !window.tf) throw new Error('pose-detection unavailable');
    try {
      if (window.tf && window.tf.setWasmPaths) {
        window.tf.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.13.0/dist/');
      }
      await window.tf.setBackend('wasm');
      await window.tf.ready();
    } catch (_) {}
    try {
      const det = await (PD.createDetector ? PD.createDetector(PD.SupportedModels.MoveNet, { modelType: 'Lightning', enableSmoothing: true }) : PD.default.createDetector(PD.default.SupportedModels.MoveNet, { modelType: 'Lightning', enableSmoothing: true }));
      setStatus('Model ready (wasm)');
      return det;
    } catch (e) {
      // Try CPU backend
      try {
        await window.tf.setBackend('cpu');
        await window.tf.ready();
        const det = await (PD.createDetector ? PD.createDetector(PD.SupportedModels.MoveNet, { modelType: 'Lightning', enableSmoothing: true }) : PD.default.createDetector(PD.default.SupportedModels.MoveNet, { modelType: 'Lightning', enableSmoothing: true }));
        setStatus('Model ready (cpu)');
        return det;
      } catch (err) {
        setStatus('Model load failed. Check network or try a different photo.');
        throw e;
      }
    }
  }

  async function createDetectorBlazePose(){
    const PD = await resolvePoseDetection();
    if (!PD) throw new Error('pose-detection unavailable');
    try {
      const det = await PD.createDetector(PD.SupportedModels.BlazePose, { runtime: 'tfjs', modelType: 'lite' });
      setStatus('Model ready (blazepose)');
      return det;
    } catch (e) { throw e; }
  }

  async function ensureFallbackDetector(){
    // Disable TFJS-based detectors due to CDN 404/CORS; use MediaPipe only
    return null;
  }

  function lineTiltDegrees(p1, p2){ const dy = p2.y - p1.y, dx = p2.x - p1.x; if (dx === 0) return 90; return Math.atan2(dy, dx) * (180/Math.PI); }
  function dist(p1, p2){ return Math.hypot(p1.x - p2.x, p1.y - p2.y); }
  function key(kps, name){ const m = kps.find(k=>k.name===name||k.part===name); return m && (m.score>0.3||m.score===undefined)?{x:m.x,y:m.y,score:m.score}:null; }

  function computeMetricsFromKeypoints(kps){
    const ls=key(kps,'left_shoulder'), rs=key(kps,'right_shoulder'), lh=key(kps,'left_hip'), rh=key(kps,'right_hip');
    const le=key(kps,'left_elbow'), re=key(kps,'right_elbow'), lw=key(kps,'left_wrist'), rw=key(kps,'right_wrist');
    const lk=key(kps,'left_knee'), rk=key(kps,'right_knee'), la=key(kps,'left_ankle'), ra=key(kps,'right_ankle');
    const nose=key(kps,'nose'), leye=key(kps,'left_eye')||key(kps,'left_ear'), reye=key(kps,'right_eye')||key(kps,'right_ear');
    let shoulderTilt=null, hipTilt=null, forwardHead=null, symmetry=null;
    if(ls&&rs){ shoulderTilt=Math.min(45, Math.abs(lineTiltDegrees(ls,rs))); }
    if(lh&&rh){ hipTilt=Math.min(45, Math.abs(lineTiltDegrees(lh,rh))); }
    if(nose&&ls&&rs){ const mid={x:(ls.x+rs.x)/2,y:(ls.y+rs.y)/2}, sw=dist(ls,rs)||1; forwardHead=Math.abs(nose.x-mid.x)/sw; }
    else if((leye||reye)&&ls&&rs){ const ear=leye||reye, mid={x:(ls.x+rs.x)/2,y:(ls.y+rs.y)/2}, sw=dist(ls,rs)||1; forwardHead=Math.abs(ear.x-mid.x)/sw; }
    if(ls&&rs&&lh&&rh){ const lt=dist(ls,lh), rt=dist(rs,rh); symmetry={ torsoDiffPct: Math.abs(lt-rt)/Math.max(lt,rt) }; }
    const leftArm=(ls&&le?dist(ls,le):0)+(le&&lw?dist(le,lw):0);
    const rightArm=(rs&&re?dist(rs,re):0)+(re&&rw?dist(re,rw):0);
    const leftLeg=(lh&&lk?dist(lh,lk):0)+(lk&&la?dist(lk,la):0);
    const rightLeg=(rh&&rk?dist(rh,rk):0)+(rk&&ra?dist(rk,ra):0);
    return { shoulderTilt, hipTilt, forwardHead, symmetry, limb:{ armDiff: Math.abs(leftArm-rightArm)/Math.max(leftArm,rightArm,1), legDiff: Math.abs(leftLeg-rightLeg)/Math.max(leftLeg,rightLeg,1) } };
  }

  function getScanInputs(){
    const sex=$('#bs-sex')?.value||'male';
    const age=Number($('#bs-age')?.value)||null;
    const metric=$('#bs-unit-metric')?.checked??true;
    let hM=null,wKg=null;
    if(metric){ const cm=Number($('#bs-height-cm')?.value)||0; const kg=Number($('#bs-weight-kg')?.value)||0; if(cm>0) hM=cm/100; if(kg>0) wKg=kg; }
    else { const ft=Number($('#bs-feet')?.value)||0; const inch=Number($('#bs-inches')?.value)||0; const lb=Number($('#bs-pounds')?.value)||0; if(ft||inch) hM=(((ft*12)+inch)*2.54)/100; if(lb>0) wKg=lb*0.45359237; }
    return {sex,age,heightM:hM,weightKg:wKg};
  }
  function computeBMI(hM,wKg){ if(!hM||!wKg) return null; return wKg/(hM*hM); }
  function bodyFatFromInputs(bmi,sex,age){ if(!Number.isFinite(bmi)) return null; const s=sex==='male'?1:0; const a=Number.isFinite(age)?age:30; const bf=1.2*bmi+0.23*a-10.8*s-5.4; return `${Math.max(3,Math.min(60,Math.round(bf)))}%`; }
  function fmtDeg(v){ return Number.isFinite(v)?`${v.toFixed(1)}°`:'—'; }
  function fmtPct(v){ return Number.isFinite(v)?`${(v*100).toFixed(0)}%`:'—'; }

  async function ensureMediaPipePose(){
    if (window.__fitlife_mp_pose_estimator) return window.__fitlife_mp_pose_estimator;
    try { await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/pose.js'); } catch {}
    if (!window.Pose) { try { await loadScript('https://unpkg.com/@mediapipe/pose@0.5.1675469404/pose.js'); } catch {} }
    if (!window.Pose) throw new Error('mediapipe pose unavailable');
    const pose = new window.Pose({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}` });
    pose.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    window.__fitlife_mp_pose_estimator = pose;
    return pose;
  }

  function toKeypointsFromMediaPipe(landmarks){
    if (!Array.isArray(landmarks)) return null;
    const map = (idx, name) => landmarks[idx] ? { name, x: landmarks[idx].x, y: landmarks[idx].y, score: landmarks[idx].visibility ?? 0.9 } : null;
    return [
      map(11,'left_shoulder'), map(12,'right_shoulder'), map(23,'left_hip'), map(24,'right_hip'),
      map(13,'left_elbow'), map(14,'right_elbow'), map(15,'left_wrist'), map(16,'right_wrist'),
      map(25,'left_knee'), map(26,'right_knee'), map(27,'left_ankle'), map(28,'right_ankle'),
      map(0,'nose'), map(7,'left_ear'), map(8,'right_ear'), map(2,'left_eye'), map(5,'right_eye')
    ].filter(Boolean);
  }

  async function estimateWithMediaPipe(imageLike){
    const pose = await ensureMediaPipePose();
    // Resample to a fixed width for deterministic results
    const srcW = imageLike.videoWidth || imageLike.naturalWidth || imageLike.width || 0;
    const srcH = imageLike.videoHeight || imageLike.naturalHeight || imageLike.height || 0;
    const targetW = Math.min(720, srcW || 720);
    const targetH = srcW ? Math.round((targetW / srcW) * (srcH || targetW)) : targetW;
    const c = document.createElement('canvas'); c.width = targetW; c.height = targetH;
    const cctx = c.getContext('2d');
    try { cctx.drawImage(imageLike, 0, 0, c.width, c.height); } catch {}
    const res = await new Promise((resolve, reject) => {
      pose.onResults((r) => resolve(r));
      try { pose.send({ image: c }); } catch (e) { reject(e); }
    });
    const kps = toKeypointsFromMediaPipe(res.poseLandmarks || []);
    if (targetW && targetH && Array.isArray(kps)) { kps.forEach(p => { p.x *= targetW; p.y *= targetH; }); }
    // Remember the input size so we can scale keypoints to the base image later
    window.__fitlife_last_input_size = { w: targetW, h: targetH };
    return kps;
  }

  function drawOverlayFallback(baseImage, keypoints, options){
    const canvas = document.getElementById('bs-annotated');
    const wrap = canvas?.parentElement; if (wrap) wrap.classList.remove('hidden');
    if (!canvas || !baseImage) return;
    const ctx = canvas.getContext('2d');
    const w = baseImage.videoWidth || baseImage.naturalWidth || baseImage.width;
    const h = baseImage.videoHeight || baseImage.naturalHeight || baseImage.height;
    if (!w || !h) return;
    canvas.width = w; canvas.height = h;
    ctx.clearRect(0,0,w,h);
    ctx.drawImage(baseImage, 0, 0, w, h);
    // Scale keypoints from last input size to base image size if needed
    let scaledKeypoints = keypoints;
    try {
      const inp = window.__fitlife_last_input_size;
      if (inp && inp.w && inp.h && (inp.w !== w || inp.h !== h)){
        const sx = w / inp.w; const sy = h / inp.h;
        scaledKeypoints = keypoints.map(k => ({ name: k.name||k.part, x: k.x * sx, y: k.y * sy, score: k.score }));
      }
    } catch {}
    const kpBy = new Map(); scaledKeypoints.forEach(k=> kpBy.set(k.name||k.part, k));
    const get = (n)=> kpBy.get(n);
    const mid = (a,b)=> (a&&b)?{x:(a.x+b.x)/2,y:(a.y+b.y)/2}:null;
    const drawCornerText = (x, y, text) => { ctx.font = `${Math.max(12, Math.floor(w/60))}px ui-sans-serif, system-ui`; ctx.lineWidth = Math.max(2, Math.floor(w/800)); ctx.strokeStyle = 'rgba(0,0,0,0.35)'; ctx.strokeText(text, x, y); ctx.fillStyle = '#ffffff'; ctx.fillText(text, x, y); };
    const drawBadge = (x,y,text)=>{ const size=Math.max(12,Math.floor(w/50)); ctx.font=`${size}px ui-sans-serif, system-ui`; const tw=ctx.measureText(text).width; const tx=Math.max(6,Math.min(w-tw-6,x)); const ty=Math.max(12,Math.min(h-6,y)); ctx.lineWidth=Math.max(2,Math.floor(w/600)); ctx.strokeStyle='rgba(0,0,0,0.35)'; ctx.strokeText(text,tx,ty); ctx.fillStyle='#ffffff'; ctx.fillText(text,tx,ty); };
    const pairs = [['left_shoulder','right_shoulder'],['left_hip','right_hip'],['left_shoulder','left_elbow'],['left_elbow','left_wrist'],['right_shoulder','right_elbow'],['right_elbow','right_wrist'],['left_hip','left_knee'],['left_knee','left_ankle'],['right_hip','right_knee'],['right_knee','right_ankle'],['left_shoulder','left_hip'],['right_shoulder','right_hip']];
    ctx.lineWidth = Math.max(2, w/400); ctx.strokeStyle = 'rgba(7,192,162,0.9)';
    pairs.forEach(([a,b])=>{ const p=get(a), q=get(b); if(p&&q){ ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke(); }});
    if (options && options.showLandmarks) {
      keypoints.forEach(k=>{ if(!k||(k.score!==undefined && k.score<0.3)) return; ctx.fillStyle='#07c0a2'; ctx.beginPath(); ctx.arc(k.x,k.y,Math.max(3,w/200),0,Math.PI*2); ctx.fill(); });
    }
    // Silhouette outline from convex hull of confident keypoints
    const pts = keypoints.filter(k=>!k||k.score===undefined||k.score>=0.4).map(k=>({x:k.x,y:k.y}));
    if (pts.length>=3){
      const cross=(o,a,b)=>((a.x-o.x)*(b.y-o.y))-((a.y-o.y)*(b.x-o.x));
      const sorted=[...pts].sort((a,b)=>a.x===b.x? a.y-b.y : a.x-b.x);
      const lower=[]; for(const p of sorted){ while(lower.length>=2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0) lower.pop(); lower.push(p);} 
      const upper=[]; for(let i=sorted.length-1;i>=0;i--){ const p=sorted[i]; while(upper.length>=2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0) upper.pop(); upper.push(p);} 
      const hull = lower.slice(0, lower.length-1).concat(upper.slice(0, upper.length-1));
      if (hull.length>=3){
        ctx.save();
        ctx.fillStyle='rgba(7,192,162,0.16)';
        ctx.strokeStyle='#07c0a2';
        ctx.lineWidth=Math.max(2,w/500);
        ctx.beginPath(); ctx.moveTo(hull[0].x, hull[0].y); for(let i=1;i<hull.length;i++){ ctx.lineTo(hull[i].x, hull[i].y);} ctx.closePath();
        ctx.fill(); ctx.stroke();
        ctx.restore();
      }
    }
    // Title
    ctx.fillStyle='rgba(0,0,0,0.55)'; ctx.fillRect(8,8,220,30); ctx.fillStyle='#fff'; ctx.font=`${Math.max(14, Math.floor(w/40))}px ui-sans-serif, system-ui`; ctx.fillText('FitLife Body Scan',16,30);
    // Ideal lines
    ctx.save(); ctx.setLineDash([6,4]); ctx.strokeStyle='rgba(255,255,255,0.6)'; ctx.lineWidth=Math.max(1,w/500);
    const ls=get('left_shoulder'), rs=get('right_shoulder'), lh=get('left_hip'), rh=get('right_hip'), nose=get('nose'); const ms=mid(ls,rs), mh=mid(lh,rh);
    if (ms) { ctx.beginPath(); ctx.moveTo(8, ms.y); ctx.lineTo(w-8, ms.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(ms.x, 8); ctx.lineTo(ms.x, h-8); ctx.stroke(); }
    if (mh) { ctx.beginPath(); ctx.moveTo(8, mh.y); ctx.lineTo(w-8, mh.y); ctx.stroke(); }
    ctx.restore();
    const m = window.__fitlife_last_metrics;
    const limb = window.__fitlife_last_limb;
    if (ls && rs && ms && m && Number.isFinite(m.shoulderTilt)) {
      const side = ls.y < rs.y ? 'Left higher' : rs.y < ls.y ? 'Right higher' : 'Level';
      ctx.strokeStyle='rgba(255,255,0,0.9)'; ctx.beginPath(); ctx.moveTo(ls.x,ls.y); ctx.lineTo(rs.x,rs.y); ctx.stroke();
      drawBadge(ms.x, ms.y-28, side==='Level' ? `Shoulders: level (≤ 2°)` : `Shoulders: ${side} by ${m.shoulderTilt.toFixed(1)}°`);
    }
    if (lh && rh && mh && m && Number.isFinite(m.hipTilt)) {
      const sideH = lh.y < rh.y ? 'Left higher' : rh.y < lh.y ? 'Right higher' : 'Level';
      ctx.strokeStyle='rgba(255,140,0,0.9)'; ctx.beginPath(); ctx.moveTo(lh.x,lh.y); ctx.lineTo(rh.x,rh.y); ctx.stroke();
      drawBadge(mh.x, mh.y+8, sideH==='Level' ? `Hips: level (≤ 2°)` : `Hips: ${sideH} by ${m.hipTilt.toFixed(1)}°`);
    }
    if (ms && nose && m && Number.isFinite(m.forwardHead)) { ctx.strokeStyle='rgba(173,216,230,0.9)'; ctx.beginPath(); ctx.moveTo(nose.x,nose.y); ctx.lineTo(ms.x,nose.y); ctx.stroke(); drawBadge(nose.x, Math.max(4, nose.y-24), `FHP: ${Math.round(m.forwardHead*100)}% (≤ 5%)`); }
    if (ls && lh && rs && rh && m && m.symmetry && Number.isFinite(m.symmetry.torsoDiffPct)) { const mt = mid(ms||ls,mh||lh)||{x:(w*0.05),y:(h*0.5)}; drawBadge(mt.x, mt.y, `Torso L/R: ${Math.round(m.symmetry.torsoDiffPct*100)}% (≤ 5%)`); }
    if (limb && Number.isFinite(limb.armDiff)) { drawBadge(w*0.04, h*0.08, `Arms diff: ${Math.round(limb.armDiff*100)}% (≤ 10%)`); }
    if (limb && Number.isFinite(limb.legDiff)) { drawBadge(w*0.04, h*0.08 + Math.max(28, h/20), `Legs diff: ${Math.round(limb.legDiff*100)}% (≤ 10%)`); }
    // Overlay scores and human-readable summary (top-right)
    const scores = options && options.scores || null;
    let y = 20;
    if (scores){ drawCornerText(w - 260, y, `Posture score: ${scores.posture}`); y += 18; drawCornerText(w - 260, y, `Symmetry score: ${scores.symmetry}`); y += 22; }
    const human=[];
    const mm = window.__fitlife_last_metrics || {};
    const sTilt = (Number.isFinite(mm.shoulderTilt)) ? (mm.shoulderTilt<=2?'Shoulders level':`Shoulders: ${mm.shoulderTilt.toFixed(1)}° tilt`) : null;
    const hTilt = (Number.isFinite(mm.hipTilt)) ? (mm.hipTilt<=2?'Hips level':`Hips: ${mm.hipTilt.toFixed(1)}° tilt`) : null;
    const fh = (Number.isFinite(mm.forwardHead)) ? ((mm.forwardHead*100)<=5?'Head neutral':`Head: ${Math.round(mm.forwardHead*100)}% forward`) : null;
    const torso = (mm.symmetry && Number.isFinite(mm.symmetry.torsoDiffPct)) ? ((mm.symmetry.torsoDiffPct*100)<=5?'Torso symmetric':`Torso: ${Math.round(mm.symmetry.torsoDiffPct*100)}% L/R diff`) : null;
    [sTilt,hTilt,fh,torso].filter(Boolean).forEach(s=>{ drawCornerText(w-260, y, s); y+=18; });

    try { canvas.scrollIntoView({ behavior: 'smooth', block: 'center' }); } catch {}
  }

  // Preview thumbnails with delete buttons in fallback
  function fileKey(f){ return `${f.name}_${f.size}_${f.lastModified||0}`; }
  const keypointCache = new Map();
  const skipImages = new Set();
  const skipVideos = new Set();
  function buildImagePreviews(){ const input = $('#bs-images'); const grid = $('#bs-previews'); if (!input || !grid) return; grid.innerHTML=''; Array.from(input.files||[]).forEach(f=>{ const key=fileKey(f); const url=URL.createObjectURL(f); const wrap=document.createElement('div'); wrap.className='thumb'; const img=new Image(); img.src=url; img.onload=()=>URL.revokeObjectURL(url); img.style.width='100%'; img.style.display='block'; img.style.borderRadius='10px'; const del=document.createElement('button'); del.className='del'; del.textContent='✕'; del.addEventListener('click', ()=>{ skipImages.add(key); wrap.remove(); }); wrap.appendChild(img); wrap.appendChild(del); grid.appendChild(wrap); }); }
  function buildVideoPreviews(){ const input = $('#bs-video'); const grid = $('#bs-video-previews'); if (!input || !grid) return; grid.innerHTML=''; Array.from(input.files||[]).forEach(f=>{ const key=fileKey(f); const url=URL.createObjectURL(f); const wrap=document.createElement('div'); wrap.className='thumb'; const vid=document.createElement('video'); vid.src=url; vid.muted=true; vid.playsInline=true; vid.controls=false; vid.style.width='100%'; vid.style.display='block'; vid.style.borderRadius='10px'; vid.onloadeddata=()=>{ vid.currentTime=Math.min(0.1, vid.duration||0); URL.revokeObjectURL(url); }; const del=document.createElement('button'); del.className='del'; del.textContent='✕'; del.addEventListener('click', ()=>{ skipVideos.add(key); wrap.remove(); }); wrap.appendChild(vid); wrap.appendChild(del); grid.appendChild(wrap); }); }
  $('#bs-images')?.addEventListener('change', buildImagePreviews);
  $('#bs-video')?.addEventListener('change', buildVideoPreviews);

  async function analyzeFallback(){
    try{
      const imgInput = $('#bs-images');
      const vidInput = $('#bs-video');
      const imgFiles = Array.from(imgInput?.files||[]).filter(f=>!skipImages.has(fileKey(f)));
      const vidFiles = Array.from(vidInput?.files||[]).filter(f=>!skipVideos.has(fileKey(f)));
      const hasImg = imgFiles.length>0;
      const hasVid = vidFiles.length>0;
      if(!hasImg && !hasVid){ setStatus('Please upload photos or a video first.'); return; }
      const timeout = new Promise((_,rej)=> setTimeout(()=> rej(new Error('Analysis timed out')), 15000));
      let det = null;
      try { det = await ensureFallbackDetector(); } catch (_) { det = null; }

      let kps = null; let baseImage = null;
      if(hasImg){
        for(const f of imgFiles){
          const cacheId = fileKey(f);
          const url=URL.createObjectURL(f);
          const img=await new Promise(res=>{ const i=new Image(); i.onload=()=>res(i); i.src=url; });
          let mpKps = keypointCache.get(cacheId);
          if (!mpKps){
            // best-of-N: run estimator multiple times and pick the set with most confident shoulder-hip landmarks
            const runs = [];
            for (let iRun=0;iRun<3;iRun++){
              try { const k = await Promise.race([estimateWithMediaPipe(img), timeout]); if (Array.isArray(k) && k.length) runs.push(k); } catch {}
            }
            const score = (k)=>{
              const sL = k.find(p=>p.name==='left_shoulder'); const sR = k.find(p=>p.name==='right_shoulder');
              const hL = k.find(p=>p.name==='left_hip'); const hR = k.find(p=>p.name==='right_hip');
              return (sL?.score||0)+(sR?.score||0)+(hL?.score||0)+(hR?.score||0);
            };
            runs.sort((a,b)=> score(b)-score(a));
            mpKps = runs[0] || [];
            keypointCache.set(cacheId, mpKps);
          }
          if (mpKps && mpKps.length) { kps = mpKps; baseImage = img; URL.revokeObjectURL(url); break; }
          URL.revokeObjectURL(url);
        }
      } else if (hasVid){
        for (const f of vidFiles){
          const url=URL.createObjectURL(f);
          const v=$('#bs-video-el');
          await new Promise(res=>{ v.onloadeddata=()=>res(); v.src=url; });
          try{ await v.play(); }catch{}; v.pause();
          // Deterministic frame capture
          try { v.currentTime = Math.min(0.1, v.duration||0); } catch {}
          if (v.videoWidth===0||v.videoHeight===0){ URL.revokeObjectURL(url); continue; }
          const runs = [];
          for (let iRun=0;iRun<2;iRun++){
            try { const k = await Promise.race([estimateWithMediaPipe(v), timeout]); if (Array.isArray(k) && k.length) runs.push(k); } catch {}
          }
          const score = (k)=>{
            const sL = k.find(p=>p.name==='left_shoulder'); const sR = k.find(p=>p.name==='right_shoulder');
            const hL = k.find(p=>p.name==='left_hip'); const hR = k.find(p=>p.name==='right_hip');
            return (sL?.score||0)+(sR?.score||0)+(hL?.score||0)+(hR?.score||0);
          };
          runs.sort((a,b)=> score(b)-score(a));
          const mpKps = runs[0] || [];
          if (mpKps && mpKps.length) { kps=mpKps; baseImage=v; URL.revokeObjectURL(url); break; }
          URL.revokeObjectURL(url);
        }
      }

      if(!kps){ setStatus('No person detected. Try a clearer, well-lit image.'); return; }

      const m = computeMetricsFromKeypoints(kps);
      const limb = (function(){ const l = {}; const ls=key(kps,'left_shoulder'), rs=key(kps,'right_shoulder'), le=key(kps,'left_elbow'), re=key(kps,'right_elbow'), lw=key(kps,'left_wrist'), rw=key(kps,'right_wrist'), lh=key(kps,'left_hip'), rh=key(kps,'right_hip'), lk=key(kps,'left_knee'), rk=key(kps,'right_knee'), la=key(kps,'left_ankle'), ra=key(kps,'right_ankle'); const leftArm=(ls&&le?dist(ls,le):0)+(le&&lw?dist(le,lw):0); const rightArm=(rs&&re?dist(rs,re):0)+(re&&rw?dist(re,rw):0); const leftLeg=(lh&&lk?dist(lh,lk):0)+(lk&&la?dist(lk,la):0); const rightLeg=(rh&&rk?dist(rh,rk):0)+(rk&&ra?dist(rk,ra):0); return { armDiff: Math.abs(leftArm-rightArm)/Math.max(leftArm,rightArm,1), legDiff: Math.abs(leftLeg-rightLeg)/Math.max(leftLeg,rightLeg,1) }; })();
      window.__fitlife_last_metrics = m;
      window.__fitlife_last_limb = limb;
      drawOverlayFallback(baseImage, kps);

      const {sex,age,heightM,weightKg} = getScanInputs();
      const bmi = computeBMI(heightM,weightKg);
      const bf = bodyFatFromInputs(bmi,sex,age);

      const results = $('#bs-results'); results.innerHTML='';
      const posture = document.createElement('div'); posture.className='metric'; posture.innerHTML = `
        <h4>Posture & Symmetry</h4>
        <div class="kv"><span>Shoulder tilt</span><span>${fmtDeg(m.shoulderTilt)}</span></div>
        <div class="kv"><span>Hip tilt</span><span>${fmtDeg(m.hipTilt)}</span></div>
        <div class="kv"><span>Forward head offset</span><span>${fmtPct(m.forwardHead)}</span></div>
        <div class="kv"><span>Torso L/R diff</span><span>${fmtPct(m.symmetry?.torsoDiffPct)}</span></div>
      `; results.appendChild(posture);
      const body = document.createElement('div'); body.className='metric'; body.innerHTML = `
        <h4>Body Data</h4>
        <div class="kv"><span>Sex</span><span>${sex}</span></div>
        <div class="kv"><span>Age</span><span>${Number.isFinite(age)?age:'—'}</span></div>
        <div class="kv"><span>BMI</span><span>${Number.isFinite(bmi)?bmi.toFixed(1):'—'}</span></div>
        <div class="kv"><span>Body fat (est.)</span><span>${bf??'—'}</span></div>
      `; results.appendChild(body);
      const limbCard = document.createElement('div'); limbCard.className='metric'; limbCard.innerHTML = `
        <h4>Limb Balance</h4>
        <div class="kv"><span>Arms L/R diff</span><span>${fmtPct(limb?.armDiff)}</span></div>
        <div class="kv"><span>Legs L/R diff</span><span>${fmtPct(limb?.legDiff)}</span></div>
      `; results.appendChild(limbCard);

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
          <li><em>Tip</em>: Visual estimates from one frame; for best accuracy use straight-on photos with good lighting.</li>
        </ul>
      `;
      results.appendChild(explain);
      try { results.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch {}

      const ls = key(kps,'left_shoulder'), rs = key(kps,'right_shoulder');
      const lh = key(kps,'left_hip'), rh = key(kps,'right_hip');
      const lines = [];
      if (Number.isFinite(m.shoulderTilt)){
        let t = 'Shoulders: level';
        if (ls && rs){
          if (ls.y < rs.y && m.shoulderTilt > 0.5) t = `Shoulders: left higher by ${m.shoulderTilt.toFixed(1)}°`;
          else if (rs.y < ls.y && m.shoulderTilt > 0.5) t = `Shoulders: right higher by ${m.shoulderTilt.toFixed(1)}°`;
        }
        lines.push(t);
      }
      if (Number.isFinite(m.hipTilt)){
        let t = 'Hips: level';
        if (lh && rh){
          if (lh.y < rh.y && m.hipTilt > 0.5) t = `Hips: left higher by ${m.hipTilt.toFixed(1)}°`;
          else if (rh.y < lh.y && m.hipTilt > 0.5) t = `Hips: right higher by ${m.hipTilt.toFixed(1)}°`;
        }
        lines.push(t);
      }
      if (Number.isFinite(m.forwardHead)) lines.push((m.forwardHead*100)<=5 ? 'Head: neutral' : `Head: ${Math.round(m.forwardHead*100)}% forward`);
      if (m.symmetry && Number.isFinite(m.symmetry.torsoDiffPct)) lines.push((m.symmetry.torsoDiffPct*100)<=5 ? 'Torso: symmetric' : `Torso: ${Math.round(m.symmetry.torsoDiffPct*100)}% L/R difference`);
      if (limb && Number.isFinite(limb.armDiff) && Number.isFinite(limb.legDiff)){
        const a = Math.round(limb.armDiff*100), g = Math.round(limb.legDiff*100);
        lines.push((a<=10 && g<=10) ? 'Arms/Legs: balanced' : `Arms: ${a}% diff · Legs: ${g}% diff`);
      }
      setStatus(lines.join(' · '));
    }catch(err){ console.error(err); setStatus('Analysis failed'); const msg=(err&&err.message)?String(err.message).slice(0,120):''; if(msg) setTimeout(()=>setStatus(`Analysis failed: ${msg}`),10); }
  }

  function fallbackClear(){ try{ const img=$('#bs-images'); if(img) img.value=''; const vid=$('#bs-video'); if(vid) vid.value=''; const p=$('#bs-previews'); if(p) p.innerHTML=''; const vp=$('#bs-video-previews'); if(vp) vp.innerHTML=''; skipImages.clear(); skipVideos.clear(); const res=$('#bs-results'); if(res) res.innerHTML=''; setStatus(''); const wrap=document.querySelector('.annotated-wrap'); wrap?.classList.add('hidden'); const cvs=document.getElementById('bs-annotated'); const ctx=cvs?.getContext && cvs.getContext('2d'); if(ctx&&cvs){ ctx.clearRect(0,0,cvs.width,cvs.height); } ['#bs-age','#bs-height-cm','#bs-weight-kg','#bs-feet','#bs-inches','#bs-pounds'].forEach(sel=>{ const el=$(sel); if(el) el.value=''; }); const sex=$('#bs-sex'); if(sex) sex.value='male'; const um=$('#bs-unit-metric'), ui=$('#bs-unit-imperial'); if(um&&ui){ um.checked=true; ui.checked=false; } }catch(_){}}

  // Guided capture (fallback)
  async function guidedAnalyzeFallback(){
    const q = document.getElementById('gc-quality'); const scoresEl = document.getElementById('gc-scores'); const summaryEl = document.getElementById('gc-summary');
    if (!q || !scoresEl || !summaryEl) return;
    q.textContent = 'Checking photos…'; scoresEl.innerHTML=''; summaryEl.innerHTML='';

    const fF = document.getElementById('gc-front')?.files?.[0];
    const fS = document.getElementById('gc-side')?.files?.[0];
    const fB = document.getElementById('gc-back')?.files?.[0];
    if (!fF && !fS && !fB){ q.textContent='Please add at least one photo.'; return; }

    let det=null; try { det = await ensureFallbackDetector(); } catch { det=null; }

    const checks=[]; const all=[];
    for (const [name, file] of [['front',fF],['side',fS],['back',fB]]){
      if (!file) continue;
      const img = await new Promise(res=>{ const url=URL.createObjectURL(file); const i=new Image(); i.onload=()=>{ URL.revokeObjectURL(url); res(i); }; i.src=url; });
      const c=document.createElement('canvas'); c.width=img.naturalWidth; c.height=img.naturalHeight; const ctx=c.getContext('2d'); ctx.drawImage(img,0,0);
      const luma=(()=>{ const d=ctx.getImageData(0,0,c.width,c.height).data; let s=0; for(let i=0;i<d.length;i+=4){ s+=0.2126*d[i]+0.7152*d[i+1]+0.0722*d[i+2]; } return s/(d.length/4)/255; })();
      if (luma < 0.25) checks.push(`${name}: too dark`);
      if (luma > 0.95) checks.push(`${name}: too bright`);
      let kps=null;
      if (det){ const poses=await det.estimatePoses(img,{maxPoses:1,flipHorizontal:false}); kps=poses[0]?.keypoints||[]; }
      if (!kps || !kps.length){ kps = await estimateWithMediaPipe(img); }
      if (!kps || !kps.length){ checks.push(`${name}: no person detected`); }
      const ls = kps?.find(k=>k.name==='left_shoulder'||k.part==='left_shoulder');
      const rs = kps?.find(k=>k.name==='right_shoulder'||k.part==='right_shoulder');
      if (ls && rs){ const cx=((ls.x+rs.x)/2)/img.naturalWidth; if (cx<0.3||cx>0.7) checks.push(`${name}: please center your body`); }
      all.push({ name, img, kps:kps||[] });
    }
    q.textContent = checks.length ? checks.join(' · ') : 'Looks good!';

    const metricsList = all.filter(v=>v.kps.length).map(v=>computeMetricsFromKeypoints(v.kps));
    if (!metricsList.length){ summaryEl.innerHTML='<p class="muted">No valid poses detected.</p>'; return; }
    const fused = metricsList.reduce((acc,m)=>({ shoulderTilt:(acc.shoulderTilt||0)+m.shoulderTilt, hipTilt:(acc.hipTilt||0)+m.hipTilt, forwardHead:(acc.forwardHead||0)+m.forwardHead, torso:(acc.torso||0)+((m.symmetry?.torsoDiffPct)||0)}),{});
    const n=metricsList.length; const fusedM={ shoulderTilt:fused.shoulderTilt/n, hipTilt:fused.hipTilt/n, forwardHead:fused.forwardHead/n, torsoDiffPct:fused.torso/n };

    // Scores
    const postureScore = Math.round(( Math.max(0,1-(Math.abs(fusedM.shoulderTilt)/2))*100 + Math.max(0,1-(Math.abs(fusedM.hipTilt)/2))*100 + Math.max(0,1-((fusedM.forwardHead*100)/5))*100 )/3);
    const symmetryScore = Math.round(Math.max(0,1-((fusedM.torsoDiffPct*100)/5))*100);
    function chip(label, s){ const cls = s>=85? 'good' : s>=65? 'warn':'bad'; return `<div class="score ${cls}"><span class="dot"></span><span>${label}: ${s}</span></div>`; }
    scoresEl.innerHTML = chip('Posture score', postureScore) + ' ' + chip('Symmetry score', symmetryScore);

    const shoulderStr = Math.abs(fusedM.shoulderTilt)<=2 ? 'Shoulders: level' : `Shoulders: ${fusedM.shoulderTilt.toFixed(1)}° high on one side`;
    const hipStr = Math.abs(fusedM.hipTilt)<=2 ? 'Hips: level' : `Hips: ${fusedM.hipTilt.toFixed(1)}° high on one side`;
    const headStr = (fusedM.forwardHead*100)<=5 ? 'Head: neutral' : `Head: ${Math.round(fusedM.forwardHead*100)}% forward`;
    const torsoStr = (fusedM.torsoDiffPct*100)<=5 ? 'Torso: symmetric' : `Torso: ${Math.round(fusedM.torsoDiffPct*100)}% L/R difference`;
    summaryEl.innerHTML = `<ul><li>${shoulderStr}</li><li>${hipStr}</li><li>${headStr}</li><li>${torsoStr}</li></ul>`;

    // Draw overlay on first valid view
    const bestView = all.find(v=>v.kps && v.kps.length) || all[0];
    if (bestView){
      // Ensure scaling matches the image we are drawing on
      window.__fitlife_last_input_size = { w: bestView.img.naturalWidth || bestView.img.width, h: bestView.img.naturalHeight || bestView.img.height };
      drawOverlayFallback(bestView.img, bestView.kps, { scores: { posture: postureScore, symmetry: symmetryScore }, summaries: [shoulderStr, hipStr, headStr, torsoStr], showLandmarks: false });
      try{ document.getElementById('bs-annotated')?.scrollIntoView({behavior:'smooth', block:'center'}); }catch{}
      try { const snap=document.getElementById('bs-annotated').toDataURL('image/png'); const img=new Image(); img.src=snap; img.style.width='100%'; img.style.borderRadius='12px'; img.style.border='1px solid var(--border)'; const holder=document.createElement('div'); holder.className='metric'; holder.innerHTML='<h4>Annotated view</h4>'; holder.appendChild(img); results.prepend(holder); } catch {}
    }

    setStatus('Done · ' + `Shoulder ${fmtDeg(fusedM.shoulderTilt)} · Hip ${fmtDeg(fusedM.hipTilt)} · FHP ${fmtPct(fusedM.forwardHead)} · Torso ${fmtPct(fusedM.torsoDiffPct)} · Arms ${fmtPct(limb?.armDiff)} · Legs ${fmtPct(limb?.legDiff)}`);
  }
  window.gcAnalyzeFallback = guidedAnalyzeFallback;

  async function guidedSetBaselineFallback(){
    const fF = document.getElementById('gc-front')?.files?.[0]; const fS = document.getElementById('gc-side')?.files?.[0]; const fB = document.getElementById('gc-back')?.files?.[0];
    if (!fF && !fS && !fB){ alert('Add at least one photo.'); return; }
    let det=null; try { det = await ensureFallbackDetector(); } catch { det=null; }
    const metricsList=[];
    for (const f of [fF,fS,fB].filter(Boolean)){
      const img=await new Promise(res=>{ const url=URL.createObjectURL(f); const i=new Image(); i.onload=()=>{ URL.revokeObjectURL(url); res(i); }; i.src=url; });
      let kps=null; if (det){ const poses=await det.estimatePoses(img,{maxPoses:1,flipHorizontal:false}); kps=poses[0]?.keypoints||[]; }
      if (!kps || !kps.length){ kps=await estimateWithMediaPipe(img); }
      if (kps && kps.length){ metricsList.push(computeMetricsFromKeypoints(kps)); }
    }
    if (!metricsList.length){ alert('No valid pose to set baseline.'); return; }
    const fused = metricsList.reduce((acc,m)=>({ shoulderTilt:(acc.shoulderTilt||0)+m.shoulderTilt, hipTilt:(acc.hipTilt||0)+m.hipTilt, forwardHead:(acc.forwardHead||0)+m.forwardHead, torso:(acc.torso||0)+((m.symmetry?.torsoDiffPct)||0)}),{});
    const n=metricsList.length; const fusedM={ shoulderTilt:fused.shoulderTilt/n, hipTilt:fused.hipTilt/n, forwardHead:fused.forwardHead/n, torsoDiffPct:fused.torso/n };
    const now=new Date().toISOString(); try { localStorage.setItem('fitlife_baseline', JSON.stringify({ date: now, metrics: fusedM })); alert('Baseline saved.'); } catch { alert('Failed to save baseline.'); }
  }
  window.gcSetBaselineFallback = guidedSetBaselineFallback;

  // Delegate guided capture CTAs if module not booted
  document.addEventListener('click', (e)=>{
    const t = e.target instanceof Element ? e.target : null; if (!t) return;
    if (!window.__fitlife_scan_booted && t.closest('#gc-analyze-btn')){ e.preventDefault(); guidedAnalyzeFallback(); }
    if (!window.__fitlife_scan_booted && t.closest('#gc-set-baseline')){ e.preventDefault(); guidedSetBaselineFallback(); }
  });

  document.addEventListener('click', (e)=>{
    const t = e.target instanceof Element ? e.target : null;
    if (!t) return;
    if (t.closest('#bs-analyze-btn')){
      if (!window.__fitlife_scan_booted) { e.preventDefault(); analyzeFallback(); }
    } else if (t.closest('#bs-download-report')){
      if (!window.__fitlife_scan_booted) {
        e.preventDefault();
        const rpt = window.__fitlife_last_report; if (rpt){ const blob=new Blob([JSON.stringify(rpt,null,2)],{type:'application/json'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='fitlife-body-scan.json'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);} else { setStatus('Run an analysis first to download a report.'); }
      }
    } else if (t.closest('#bs-clear-btn')){
      if (!window.__fitlife_scan_booted) { e.preventDefault(); fallbackClear(); }
    }
  });
})();