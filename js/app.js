// Camera Eye Tracker: front camera -> MediaPipe Face Landmarker -> eye features -> ridge regression -> screen point.
// Everything runs in the browser; no frame leaves the device.
import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs';
import { extractFeatures, GazeMapper, DriftFilter, OneEuro, OneEuro2D, EyeClosure, cleanSamples, errorStats, calibrationPoints,
  validationPoints, cameraPosition, FEATURE_NAMES } from './gaze.js?v=3';

const MEDIAPIPE_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const TEST = new URLSearchParams(location.search).has('test');   // short timings for tests/browser_smoke.mjs
// per dot: wait `settle` for the eyes to land, collect at least `collect`, keep going up to `maxCollect` until
// `want` frames are in (slow phones), accept the dot with `minSamples` (real phone 2026-10-21: 8 in 1 s failed at 6 fps)
const TIME = TEST ? { intro: 200, count: 100, settle: 150, collect: 350, maxCollect: 700, tap: 250, move: 150 }
                  : { intro: 1800, count: 500, settle: 700, collect: 1000, maxCollect: 2500, tap: 500, move: 1000 };
const WANT_SAMPLES = TEST ? 6 : 12;
const MIN_SAMPLES = TEST ? 3 : 4;
const MIN_DOTS = 5;                   // of 9 calibration dots
const closure = new EyeClosure();
const DWELL_MS = 1100;
const MAX_TAPS = 12;                  // tap corrections kept in the training set
const SMOOTHING = { low: [1.6, 0.8], medium: [0.7, 0.6], high: [0.3, 0.4] };   // One Euro filter [min cutoff Hz, beta]
// movement phase: eyes stay on the centre dot while the head turns, one cue per TIME.move
const MOVES = ['to the left', 'to the right', 'up', 'down', 'back to the middle'];
const ARROWS = ['←', '→', '↑', '↓', '•'];
const RHO_RANGE = [0.7, 1.45];        // distance now / at calibration that the distance correction is trusted for
const FAR_SD = 1.5, FAR_MS = 1000;     // beyond the calibrated pose range by this much, this long: ask for a re-centre tap

const $ = id => document.getElementById(id);
const video = $('video');
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const sleep = ms => new Promise(r => setTimeout(r, ms));

const S = {
  mode: 'intro', view: 'figs', frame: null, fps: 0, luma: null, delegate: null, landmarker: null,
  model: null, screen: null, val: null, taps: 0, cancel: false, log: [],
  collector: null, gaze: null, filter: new OneEuro2D(...SMOOTHING.medium), dwell: { i: -1, t0: 0 },
  drift: new DriftFilter(), far: { since: null, anchor: null },
};
// the distance changes slowly: the iris size is smoothed before it scales the gaze point
const irisFilter = new OneEuro(0.5, 0);
window.gazeApp = S;                   // read by the browser test

function setMode(mode) {
  S.mode = mode;
  document.body.dataset.mode = mode;
  for (const id of ['intro', 'setup', 'calib', 'track']) $(id).hidden = id !== mode;
}
function status(el, text, isError = false) { el.textContent = text; el.classList.toggle('error', isError); }

// ------------------------------------------------------------------ start: camera, then the face model

$('btnStart').addEventListener('click', async () => {
  const btn = $('btnStart'), msg = $('introStatus');
  btn.disabled = true;
  try {
    if (!window.isSecureContext) throw new Error('The camera needs a secure page (https). Open the GitHub Pages address.');
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser cannot use the camera.');
    status(msg, 'Asking for the camera…');
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      // 640 x 480 is plenty for the 256 px face crop MediaPipe uses, and much lighter for phones than 1280 x 720
      audio: false, video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
    });
    await video.play();
    status(msg, 'Loading the face model (about 4 MB, once)…');
    S.landmarker = await createLandmarker();
    status(msg, '');
    setMode('setup');
    nextFrame();
  } catch (e) {
    btn.disabled = false;
    status(msg, cameraError(e), true);
    console.error(e);
  }
});

function cameraError(e) {
  if (e?.name === 'NotAllowedError') return 'Camera access was blocked. Allow the camera for this site in the browser settings, then reload.';
  if (e?.name === 'NotFoundError' || e?.name === 'OverconstrainedError') return 'No front camera found.';
  if (e?.name === 'NotReadableError') return 'Another app is using the camera. Close it and try again.';
  return e?.message || String(e);
}

async function createLandmarker() {
  const files = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  let last;
  for (const delegate of ['GPU', 'CPU']) {
    try {
      const lm = await FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true,
      });
      S.delegate = delegate;
      return lm;
    } catch (e) { last = e; console.warn(`${delegate} delegate failed`, e); }
  }
  throw last;
}

// ------------------------------------------------------------------ frame loop

// requestVideoFrameCallback fires once per camera frame (Chrome, Safari 15.4+); requestAnimationFrame is the fallback
const HAS_RVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
const nextFrame = () => HAS_RVFC ? video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
let lastVideoTime = -1, lastStamp = 0, frames = 0, fpsStart = 0, lastUi = 0;
function tick(now) {
  nextFrame();
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  now = Math.max(now, lastStamp + 0.01);           // MediaPipe needs strictly increasing timestamps
  lastStamp = now;
  lastVideoTime = video.currentTime;
  let res;
  try { res = S.landmarker.detectForVideo(video, now); } catch (e) { console.error(e); return; }
  const lm = res.faceLandmarks?.[0];
  S.frame = lm ? extractFeatures(lm, res.faceBlendshapes?.[0]?.categories, video.videoWidth, video.videoHeight) : null;
  if (S.frame) { S.frame.closed = closure.closed(S.frame); S.frame.irisS = irisFilter.filter(S.frame.iris, now / 1000); }
  if (S.stats) { S.stats.frames++; if (!S.frame) S.stats.noFace++; else if (S.frame.closed) S.stats.closed++; }
  frames++;
  if (now - fpsStart >= 1000) { S.fps = frames * 1000 / (now - fpsStart); frames = 0; fpsStart = now; S.luma = measureLuma(); }
  if (S.collector) S.collector(S.frame, now);
  if (S.mode === 'setup') {
    drawEyes(S.frame);
    if (now - lastUi > 200) { lastUi = now; updateChecks(S.frame); }
  } else if (S.mode === 'track') {
    trackFrame(S.frame, now);
  }
}

const lumaCanvas = Object.assign(document.createElement('canvas'), { width: 32, height: 24 });
const lumaCtx = lumaCanvas.getContext('2d', { willReadFrequently: true });
function measureLuma() {
  try {
    lumaCtx.drawImage(video, 0, 0, 32, 24);
    const d = lumaCtx.getImageData(0, 0, 32, 24).data;
    let s = 0;
    for (let i = 0; i < d.length; i += 4) s += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    return s / (d.length / 4);
  } catch { return null; }
}

// The setup screen draws the two eyes from the landmarks only (no camera picture), mirrored like a mirror.
const eyesView = $('eyesView'), ectx = eyesView.getContext('2d');
let colours = null;
const readColours = () => {
  const cs = getComputedStyle(document.body);
  colours = { line: cs.getPropertyValue('--ink-2').trim(), iris: cs.getPropertyValue('--accent').trim(), muted: cs.getPropertyValue('--muted').trim() };
};
matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', readColours);
function drawEyes(f) {
  const r = eyesView.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  const cw = Math.round(r.width * dpr), ch = Math.round(r.height * dpr);
  if (!cw) return;
  if (eyesView.width !== cw || eyesView.height !== ch) { eyesView.width = cw; eyesView.height = ch; }
  if (!colours) readColours();
  ectx.clearRect(0, 0, cw, ch);
  if (!f) {
    ectx.fillStyle = colours.muted; ectx.font = `${14 * dpr}px system-ui, sans-serif`; ectx.textAlign = 'center';
    ectx.fillText('Looking for your eyes…', cw / 2, ch / 2);
    return;
  }
  // centre the pair, scale by the distance between the eyes, mirror left-right
  const [A, B] = f.eyes, cx = (A.c[0] + B.c[0]) / 2, cy = (A.c[1] + B.c[1]) / 2;
  const k = (cw * 0.55) / Math.max(f.iod, 1e-6);
  const P = ([x, y]) => [cw / 2 - (x - cx) * k, ch / 2 + (y - cy) * k];
  ectx.lineWidth = 2.5 * dpr; ectx.strokeStyle = colours.line; ectx.fillStyle = colours.iris;
  for (const e of f.eyes) {
    const [L, R, T, Bo, I] = [P(e.L), P(e.R), P(e.top), P(e.bottom), P(e.iris)];
    // almond through the corners; control points pushed out so the curve passes near the lid points
    ectx.beginPath();
    ectx.moveTo(...L);
    ectx.quadraticCurveTo(2 * T[0] - (L[0] + R[0]) / 2, 2 * T[1] - (L[1] + R[1]) / 2, ...R);
    ectx.quadraticCurveTo(2 * Bo[0] - (L[0] + R[0]) / 2, 2 * Bo[1] - (L[1] + R[1]) / 2, ...L);
    ectx.stroke();
    const rad = Math.hypot(R[0] - L[0], R[1] - L[1]) * 0.2;
    ectx.beginPath(); ectx.arc(I[0], I[1], rad, 0, 2 * Math.PI); ectx.fill();
  }
}

// ------------------------------------------------------------------ setup checks

function setCheck(key, state, text) {
  const li = document.querySelector(`[data-check="${key}"]`);
  li.dataset.state = state;
  li.querySelector('.txt').textContent = text;
}
function updateChecks(f) {
  if (!f) {
    setCheck('face', 'bad', 'No face found. Look at the screen with your whole face in the picture.');
    setCheck('eyes', 'idle', 'Eyes: waiting for a face.');
    setCheck('size', 'idle', 'Distance: waiting for a face.');
  } else {
    setCheck('face', 'ok', 'Face and both irises found.');
    const open = !f.closed;
    setCheck('eyes', open ? 'ok' : 'warn', open ? 'Both eyes open and visible.' : 'Eyes look closed or hidden (glare on glasses?).');
    const s = f.size;
    setCheck('size', s < 0.09 || s > 0.5 ? 'warn' : 'ok',
      s < 0.09 ? 'Your face is small in the picture: come closer to the camera.'
        : s > 0.5 ? 'Very close: move back a little.' : 'Distance is fine.');
  }
  const L = S.luma;
  setCheck('light', L === null ? 'idle' : L < 45 ? 'bad' : L < 75 ? 'warn' : 'ok',
    L === null ? 'Light: measuring…' : L < 45 ? 'Too dark. Light your face from the front.'
      : L < 75 ? 'A bit dark. More light on your face helps.' : 'Light is fine.');
  const fps = Math.round(S.fps);
  setCheck('speed', fps === 0 ? 'idle' : fps < 12 ? 'warn' : 'ok',
    fps === 0 ? 'Speed: measuring…' : `${fps} frames per second${fps < 12 ? ': slow, the dot will lag' : ''} (${S.delegate})`);
  $('btnCalibrate').disabled = !f;
}

// ------------------------------------------------------------------ calibration

$('btnCalibrate').addEventListener('click', calibrate);
$('btnCancel').addEventListener('click', () => { S.cancel = true; });
$('btnRecal').addEventListener('click', () => { status($('setupMsg'), ''); toSetup(); });

async function calibrate() {
  S.cancel = false;
  setMode('calib');
  keepAwake();
  const W = innerWidth, H = innerHeight, samples = [];
  S.stats = { frames: 0, noFace: 0, closed: 0, t0: performance.now() };
  await countdown();
  const pts = calibrationPoints();
  for (let k = 0; k < pts.length; k++) {
    const got = cleanSamples(await collectPoint(pts[k], `Dot ${k + 1} of ${pts.length}`));
    if (S.cancel) return backToSetup('Calibration cancelled.');
    if (got.length >= MIN_SAMPLES) got.forEach(s => samples.push({ ...s, target: pts[k], group: k }));
    else S.stats.missed = (S.stats.missed || 0) + 1;
  }
  const dots = new Set(samples.map(s => s.group)).size;
  if (dots < MIN_DOTS) return backToSetup(calibrationProblem(dots, pts.length));
  // eyes on one dot while the head turns: teaches the fit which feature changes do not move the gaze point
  const moved = cleanSamples(await collectMoves(), undefined, 5);
  if (S.cancel) return backToSetup('Calibration cancelled.');
  if (moved.length >= MIN_SAMPLES) moved.forEach(s => samples.push({ ...s, target: [0.5, 0.5], group: 50 }));
  const camera = cameraPosition(H >= W, screenAngle());
  let model = new GazeMapper(camera).fit(samples);
  const vpts = validationPoints(), val = [];
  for (let k = 0; k < vpts.length; k++) {
    const got = cleanSamples(await collectPoint(vpts[k], `Test dot ${k + 1} of ${vpts.length}`));
    if (S.cancel) return backToSetup('Calibration cancelled.');
    val.push({ target: vpts[k], preds: got.map(s => model.predict(s.x, s.iris)) });
    got.forEach(s => samples.push({ ...s, target: vpts[k], group: 100 + k }));
  }
  S.val = errorStats(val, W, H);            // measured before the test dots join the training set
  S.train = samples;
  S.model = new GazeMapper(camera).fit(samples);
  S.taps = 0;
  S.screen = { portrait: H >= W, w: W, h: H };
  // the offset starts as uncertain as the test dots were off; the clock and pose start now
  const p0 = Math.max(0.03, (S.val?.meanPx ?? 0.1 * Math.min(W, H)) / Math.min(W, H)) ** 2;
  S.drift.reset(p0, performance.now() / 1000, S.model.poseVector(samples[samples.length - 1].x));
  S.far = { since: null, anchor: null };
  S.log = samples.map(s => ({ phase: s.group === 50 ? 'move' : s.group >= 100 ? 'test' : 'dot', group: s.group, target: s.target, t: s.t, iris: s.iris, x: s.x }));
  S.filter.reset();
  setMode('track');
  layoutTrack();
  showResult(W, H);
}

function backToSetup(text) {
  S.collector = null;
  toSetup();
  status($('setupMsg'), text, true);
}
function toSetup() {
  $('anchor').hidden = true;
  $('saveLast').hidden = !S.log.length;
  setMode('setup');
}

// Numbers only (features, targets, times, screen size; no picture): lets a later version of the model be tried on
// a real calibration. Saved on this device; nothing is sent anywhere.
function saveData() {
  const data = {
    app: 'camera-eye-tracker', format: 1, saved: new Date().toISOString(), features: FEATURE_NAMES,
    screen: S.screen, camera: S.model?.camera, video: [video.videoWidth, video.videoHeight], fps: Math.round(S.fps),
    testError: S.val && { meanPx: S.val.meanPx, pctDiag: S.val.pctDiag }, samples: S.log,
  };
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' })),
    download: `eye-tracker-data-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.json`,
  });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// say why the dots got too few frames, so the next try can fix the right thing
function calibrationProblem(dots, total) {
  const st = S.stats, secs = (performance.now() - st.t0) / 1000, fps = st.frames / secs;
  const pct = n => Math.round(100 * n / Math.max(st.frames, 1));
  let tip = 'Keep your face in view and try again.';
  if (fps < 10) tip = 'The phone was slow. Close other apps and try again.';
  else if (pct(st.noFace) > 30) tip = 'Your face often left the picture. Hold the phone steady with your whole face in view.';
  else if (pct(st.closed) > 30) tip = 'Your eyes often looked closed. Hold the phone a little higher, so you look at it straight rather than down, and light your face.';
  return `Only ${dots} of ${total} dots got enough frames of your eyes (${Math.round(fps)} frames per second, ` +
    `${pct(st.noFace)} % without a face, ${pct(st.closed)} % with eyes closed). ${tip}`;
}

function screenAngle() {
  const a = screen.orientation?.angle ?? window.orientation ?? 0;
  return ((a % 360) + 360) % 360;
}

async function countdown() {
  const t = $('target'), msg = $('calibMsg');
  t.classList.remove('shrink');
  t.style.left = '50%'; t.style.top = '50%';
  msg.textContent = 'Follow the dot with your eyes. Keep your head still.';
  await sleep(TIME.intro);
  for (const n of ['3', '2', '1']) { msg.textContent = n; await sleep(TIME.count); }
}

// Show the dot at [x, y] (screen fractions), wait for the eyes to land, then keep the frames without a blink.
function collectPoint([x, y], label) {
  const target = $('target'), msg = $('calibMsg');
  target.style.left = `${x * 100}%`;
  target.style.top = `${y * 100}%`;
  return new Promise(resolve => {
    let tries = 0;
    const run = () => {
      msg.textContent = label;
      const got = [], t0 = performance.now();
      target.classList.remove('shrink');
      void target.offsetWidth;                        // restart the shrinking ring
      target.style.setProperty('--dur', `${TIME.settle + TIME.collect}ms`);
      target.classList.add('shrink');
      S.collector = (f, now) => {
        if (now - t0 >= TIME.settle && f && !f.closed) got.push(sampleOf(f, now));
      };
      const poll = setInterval(() => {
        const t = performance.now() - t0;
        if (!S.cancel && t < TIME.settle + TIME.collect) return;
        if (!S.cancel && got.length < WANT_SAMPLES && t < TIME.settle + TIME.maxCollect) return;
        clearInterval(poll);
        S.collector = null;
        if (got.length >= MIN_SAMPLES || ++tries > 1 || S.cancel) return resolve(got);
        msg.textContent = 'Lost your eyes for a moment. Look at the dot again.';
        setTimeout(run, 600);
      }, 50);
    };
    run();
  });
}

const sampleOf = (f, now) => ({ x: f.x, iris: f.irisS, t: Math.round(now) });

// Eyes on the centre dot while the head turns a little each way (or the phone tilts): one cue per TIME.move.
function collectMoves() {
  const target = $('target'), msg = $('calibMsg'), got = [], t0 = performance.now();
  target.style.left = '50%'; target.style.top = '50%';
  target.classList.remove('shrink');
  void target.offsetWidth;
  target.style.setProperty('--dur', `${TIME.settle + MOVES.length * TIME.move}ms`);
  target.classList.add('shrink');
  msg.textContent = 'Keep your eyes on the dot.';
  S.collector = (f, now) => { if (now - t0 >= TIME.settle && f && !f.closed) got.push(sampleOf(f, now)); };
  return new Promise(resolve => {
    const poll = setInterval(() => {
      const t = performance.now() - t0, k = Math.floor((t - TIME.settle) / TIME.move);
      if (k >= 0 && k < MOVES.length) msg.innerHTML = `Keep your eyes on the dot and slowly turn your head ${MOVES[k]} <span class="arrow">${ARROWS[k]}</span><br><small>or tilt the phone a little</small>`;
      if (!S.cancel && t < TIME.settle + MOVES.length * TIME.move) return;
      clearInterval(poll);
      S.collector = null;
      resolve(got);
    }, 50);
  });
}

// ------------------------------------------------------------------ tracking

function trackFrame(f, now) {
  if (!S.model) return;
  const dot = $('gaze');
  if (!f) { dot.classList.add('lost'); postureWarning(null, now); return; }
  watchRange(f, now);
  postureWarning(f, now);
  if (f.closed) return;                               // hold the last point through a blink
  const [rx, ry] = S.drift.apply(S.model.predict(f.x, f.irisS));
  const [x, y] = S.filter.filter(clamp(rx, -0.1, 1.1), clamp(ry, -0.1, 1.1), now / 1000);
  S.gaze = { x: clamp(x, 0, 1), y: clamp(y, 0, 1) };
  const px = S.gaze.x * innerWidth, py = S.gaze.y * innerHeight;
  dot.classList.remove('lost');
  dot.style.transform = `translate(${px}px, ${py}px)`;
  if (S.view === 'figs') updateFigs(px, py, now);
  else if (S.view === 'heat') paintHeat(px, py);
}

// Beyond the calibrated pose range for a second, and not yet re-centred there: show the centre dot and ask for a tap.
function watchRange(f, now) {
  const far = S.model.outside(f.x) > FAR_SD;
  const pose = S.model.poseVector(f.x), a = S.far.anchor;
  const recentred = a && Math.hypot(...pose.map((v, i) => v - a[i])) < FAR_SD;
  if (!far || recentred) S.far.since = null;
  else if (S.far.since === null) S.far.since = now;
  $('anchor').hidden = !(S.far.since !== null && now - S.far.since > FAR_MS);
}

let lastWarn = 0;
function postureWarning(f, now) {
  if (now - lastWarn < 400) return;
  lastWarn = now;
  let text = null;
  const rho = f ? S.model.rho(f.irisS) : 1;
  if (S.screen && S.screen.portrait !== (innerHeight >= innerWidth)) text = 'The screen turned. Turn it back or recalibrate.';
  else if (!f) text = 'No face in view.';
  else if (rho < RHO_RANGE[0]) text = 'Much closer than during calibration: move back a little.';
  else if (rho > RHO_RANGE[1]) text = 'Much farther than during calibration: come a little closer.';
  else if (!$('anchor').hidden) text = 'You moved a lot. Look at the dot in the middle and tap it, or recalibrate.';
  else if (S.luma !== null && S.luma < 45) text = 'It got dark. Light your face.';
  const toast = $('toast');
  toast.hidden = !text;
  if (text) toast.textContent = text;
}

// Tap where you look: half a second of frames measures the offset there, and the Kalman filter takes as much of it as
// it trusts (most after a move or a long time, less right after the last tap). The tap frames are kept for the export.
$('track').addEventListener('pointerdown', ev => {
  if (!S.model || ev.target.closest('.toolbar, .result, .clear')) return;
  let [cx, cy] = [ev.clientX, ev.clientY];
  const anchor = $('anchor');
  if (!anchor.hidden) {                               // a tap near the centre dot counts as looking at its centre
    const r = anchor.getBoundingClientRect(), ax = r.left + r.width / 2, ay = r.top + r.height / 2;
    if (Math.hypot(cx - ax, cy - ay) < 60) [cx, cy] = [ax, ay];
  }
  const x = cx / innerWidth, y = cy / innerHeight, t0 = performance.now(), got = [];
  ripple(cx, cy);
  const group = 1000 + (++S.taps);
  S.collector = (f, now) => { if (f && !f.closed && now - t0 > 80) got.push({ f, now }); };
  setTimeout(() => {
    S.collector = null;
    if (got.length < 3) return;
    const preds = got.map(({ f }) => S.model.predict(f.x, f.irisS));
    const mean = [0, 1].map(a => preds.reduce((s, p) => s + p[a], 0) / preds.length);
    const last = got[got.length - 1].f;
    const pose = S.model.poseVector(last.x);
    S.drift.correct([x - (mean[0] + S.drift.beta[0]), y - (mean[1] + S.drift.beta[1])], performance.now() / 1000, pose);
    S.far = { since: null, anchor: pose };
    anchor.hidden = true;
    got.forEach(({ f, now }) => S.log.push({ phase: 'tap', group, target: [x, y], ...sampleOf(f, now) }));
    $('hint').textContent = `Corrected (${S.taps} tap${S.taps > 1 ? 's' : ''}).`;
  }, TIME.tap);
});
function ripple(x, y) {
  const r = Object.assign(document.createElement('div'), { className: 'ripple' });
  r.style.left = `${x}px`; r.style.top = `${y}px`;
  $('track').append(r);
  setTimeout(() => r.remove(), 700);
}

// views
const HINTS = {
  figs: 'Look at one figure: only that one lights up. Hold your gaze to make it hop.',
  dot: 'Tap the spot you are looking at to correct the dot.',
  heat: 'Where you looked, the colour builds up. Tap where you look to correct.',
};
for (const b of document.querySelectorAll('.seg button')) b.addEventListener('click', () => setView(b.dataset.view));
for (const f of document.querySelectorAll('.fig')) f.append(Object.assign(document.createElement('span'), { className: 'bar' }));
function setView(view) {
  S.view = view;
  document.body.dataset.view = view;
  for (const b of document.querySelectorAll('.seg button')) b.setAttribute('aria-selected', String(b.dataset.view === view));
  $('btnClear').hidden = view !== 'heat';
  $('hint').textContent = HINTS[view];
  $('track').style.setProperty('--toolbar-h', `${$('toolbar').offsetHeight}px`);
  if (view === 'figs') measureFigs();
}
$('smooth').addEventListener('change', e => S.filter.set(...SMOOTHING[e.target.value]));
$('btnClear').addEventListener('click', clearHeat);
$('btnOk').addEventListener('click', () => { $('result').hidden = true; $('valmarks').innerHTML = ''; });
for (const id of ['btnSave', 'btnSave2']) $(id).addEventListener('click', saveData);
$('btnRecal2').addEventListener('click', () => { $('result').hidden = true; $('valmarks').innerHTML = ''; toSetup(); });

function layoutTrack() {
  $('track').style.setProperty('--toolbar-h', `${$('toolbar').offsetHeight}px`);
  sizeHeat();
  setView(S.view);
}
addEventListener('resize', () => { if (S.mode === 'track') layoutTrack(); });

// three figures: the one nearest the gaze point lights up; holding the gaze makes it hop
let figBox = null, figCentres = [];
function measureFigs() {
  requestAnimationFrame(() => {                      // after the view is laid out
    figBox = $('figs').getBoundingClientRect();
    figCentres = [...document.querySelectorAll('.fig')].map(f => { const r = f.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; });
    S.dwell = { i: -1, t0: 0 };
  });
}
function updateFigs(px, py, now) {
  if (!figBox) return;
  const figs = document.querySelectorAll('.fig');
  const inside = px >= figBox.left && px <= figBox.right && py >= figBox.top - 20 && py <= figBox.bottom;
  let hit = -1, best = Infinity;
  if (inside) figCentres.forEach(([cx, cy], i) => { const d = Math.hypot(px - cx, py - cy); if (d < best) { best = d; hit = i; } });
  if (hit !== S.dwell.i) {
    figs.forEach((f, i) => { f.classList.toggle('look', i === hit); f.style.setProperty('--p', 0); });
    S.dwell = { i: hit, t0: now };
    return;
  }
  if (hit < 0) return;
  const p = clamp((now - S.dwell.t0) / DWELL_MS, 0, 1), fig = figs[hit];
  fig.style.setProperty('--p', p);
  if (p >= 1) {
    fig.classList.remove('picked'); void fig.offsetWidth; fig.classList.add('picked');
    $('hint').textContent = `You looked at the ${fig.dataset.name}. Now try another one.`;
    S.dwell.t0 = now + 700;                          // short pause before it can hop again
  }
}

const heat = $('heat'), hctx = heat.getContext('2d');
function sizeHeat() {
  const dpr = devicePixelRatio || 1;
  heat.width = Math.round(innerWidth * dpr); heat.height = Math.round(innerHeight * dpr);
  hctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
function clearHeat() { hctx.clearRect(0, 0, innerWidth, innerHeight); }
function paintHeat(px, py) {
  const r = Math.max(innerWidth, innerHeight) * 0.06, g = hctx.createRadialGradient(px, py, 0, px, py, r);
  g.addColorStop(0, 'rgba(228, 87, 46, 0.07)');
  g.addColorStop(1, 'rgba(228, 87, 46, 0)');
  hctx.fillStyle = g;
  hctx.fillRect(px - r, py - r, 2 * r, 2 * r);
}

function showResult(W, H) {
  const v = S.val;
  $('acc').textContent = v ? `± ${Math.round(v.meanPx)} px` : '± ?';
  let text = 'The test dots could not be measured: your eyes were not visible.';
  $('btnRecal2').hidden = false;
  if (v) {
    // camera-only trackers land around 2-3 cm on a phone, roughly 15-20 % of its diagonal (see README)
    const verdict = v.pctDiag < 10 ? 'Good for camera-only tracking.'
      : v.pctDiag < 20 ? 'Typical for camera-only tracking: enough to tell the three figures apart.'
        : 'Poor. Recalibrate: more light on your face and a steady head help most.';
    $('btnRecal2').hidden = v.pctDiag < 20;
    text = `Average error on the 4 test dots: <b>${Math.round(v.meanPx)} px</b>, about ${Math.round(v.pctDiag)} % of the screen diagonal. ${verdict}`;
    const sx = x => x * W, sy = y => y * H;
    $('valmarks').innerHTML = v.perPoint.map(p => {
      const [tx, ty] = [sx(p.target[0]), sy(p.target[1])], [ex, ey] = [sx(clamp(p.mean[0], 0, 1)), sy(clamp(p.mean[1], 0, 1))];
      return `<line class="err" x1="${tx}" y1="${ty}" x2="${ex}" y2="${ey}"/>` +
        `<line class="cross" x1="${tx - 9}" y1="${ty}" x2="${tx + 9}" y2="${ty}"/><line class="cross" x1="${tx}" y1="${ty - 9}" x2="${tx}" y2="${ty + 9}"/>` +
        `<circle class="est" cx="${ex}" cy="${ey}" r="6"/>`;
    }).join('');
  }
  $('resultText').innerHTML = text;
  $('result').hidden = false;
}

// keep the screen on while calibrating and tracking
let wakeLock = null;
async function keepAwake() { try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* not offered: fine */ } }
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && (S.mode === 'calib' || S.mode === 'track')) keepAwake();
});
