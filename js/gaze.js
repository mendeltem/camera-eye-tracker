// Gaze math, free of browser APIs so it can be tested with `node --test`.
// Face landmarks (MediaPipe Face Landmarker, 478 points incl. iris) -> feature vector -> ridge regression -> screen point.

// MediaPipe face mesh indices. The two eyes are told apart by position in the image, not by name,
// so nothing depends on mirroring or on where the camera sits.
const EYES = [
  { corners: [33, 133], lids: [159, 145] },
  { corners: [362, 263], lids: [386, 374] },
];
const IRIS_CENTERS = [468, 473];
const NOSE_TIP = 1;
const CHIN = 152;
const BLEND = ['eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeLookDownLeft',
  'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight', 'eyeLookDownRight'];

export const FEATURE_NAMES = [
  'eyeA_h', 'eyeA_v', 'eyeA_open', 'eyeB_h', 'eyeB_v', 'eyeB_open',
  'blend_hL', 'blend_vL', 'blend_hR', 'blend_vR',
  'head_yaw', 'head_pitch', 'head_roll', 'face_x', 'face_y', 'face_size',
  'h_sq', 'v_sq', 'h_v',
];

const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

// One eye in its own frame: u along the corner line (image left -> right), v perpendicular (image down).
// h, v: iris offset from the eye centre / lid midpoint, in eye widths. open: lid gap in eye widths.
export function eyeFrame(pt, eye, irisPoint) {
  let L = pt(eye.corners[0]), R = pt(eye.corners[1]);
  if (L[0] > R[0]) [L, R] = [R, L];
  const w = Math.hypot(R[0] - L[0], R[1] - L[1]) || 1e-6;
  const u = [(R[0] - L[0]) / w, (R[1] - L[1]) / w];
  const v = [-u[1], u[0]];
  const c = mid(L, R);
  const top = pt(eye.lids[0]), bottom = pt(eye.lids[1]);
  const lidMid = mid(top, bottom);
  return {
    h: dot(sub(irisPoint, c), u) / w,
    v: dot(sub(irisPoint, lidMid), v) / w,
    open: dot(sub(bottom, top), v) / w,
    w, c, L, R, top, bottom, iris: irisPoint,
  };
}

// landmarks: [{x, y, z}] normalised to the frame; categories: MediaPipe blendshapes (optional).
export function extractFeatures(landmarks, categories, W, H) {
  if (!landmarks || landmarks.length < 478) return null;
  const pt = i => [landmarks[i].x * W, landmarks[i].y * H];
  // pair each iris with the nearer eye
  const centres = EYES.map(e => mid(pt(e.corners[0]), pt(e.corners[1])));
  const irises = IRIS_CENTERS.map(pt);
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const swap = d(irises[0], centres[0]) + d(irises[1], centres[1]) > d(irises[1], centres[0]) + d(irises[0], centres[1]);
  let eyes = EYES.map((e, k) => eyeFrame(pt, e, irises[swap ? 1 - k : k]));
  eyes.sort((a, b) => a.c[0] - b.c[0]);                 // eye A = left in the image
  const [A, B] = eyes;

  const iod = Math.hypot(B.c[0] - A.c[0], B.c[1] - A.c[1]) || 1e-6;
  const ex = [(B.c[0] - A.c[0]) / iod, (B.c[1] - A.c[1]) / iod];
  const ey = [-ex[1], ex[0]];
  const m = mid(A.c, B.c);
  const nose = sub(pt(NOSE_TIP), m), chin = sub(pt(CHIN), m);
  const yaw = dot(nose, ex) / iod;
  const pitch = dot(nose, ey) / Math.max(dot(chin, ey), 1e-6);
  const roll = Math.atan2(ex[1], ex[0]);

  const score = {};
  for (const c of categories || []) score[c.categoryName] = c.score;
  const g = n => score[n] ?? 0;
  const blend = [g(BLEND[1]) - g(BLEND[0]), g(BLEND[2]) - g(BLEND[3]), g(BLEND[5]) - g(BLEND[4]), g(BLEND[6]) - g(BLEND[7])];
  const blink = Math.max(g('eyeBlinkLeft'), g('eyeBlinkRight'));

  const hm = (A.h + B.h) / 2, vm = (A.v + B.v) / 2;
  const size = iod / Math.min(W, H);
  const x = [A.h, A.v, A.open, B.h, B.v, B.open, ...blend, yaw, pitch, roll, m[0] / W, m[1] / H, size, hm * hm, vm * vm, hm * vm];
  return { x, blink, size, yaw, pitch, roll, faceX: m[0] / W, faceY: m[1] / H, minOpen: Math.min(A.open, B.open), eyes, iod };
}

// ---------------------------------------------------------------- ridge regression

function cholSolve(A, B) {            // A: d x d symmetric positive definite, B: d x m
  const d = A.length, L = A.map(() => new Float64Array(d));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
    }
  }
  return B[0].map((_, col) => {
    const y = new Float64Array(d), x = new Float64Array(d);
    for (let i = 0; i < d; i++) { let s = B[i][col]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
    for (let i = d - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < d; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
    return x;
  });
}

function fitRidge(Z, Y, idx, lambda) {
  const d = Z[0].length, n = idx.length;
  const zm = new Float64Array(d), ym = [0, 0];
  for (const i of idx) { for (let j = 0; j < d; j++) zm[j] += Z[i][j] / n; ym[0] += Y[i][0] / n; ym[1] += Y[i][1] / n; }
  const A = Array.from({ length: d }, () => new Float64Array(d)), B = Array.from({ length: d }, () => [0, 0]);
  for (const i of idx) {
    const z = Z[i].map((v, j) => v - zm[j]);
    for (let a = 0; a < d; a++) {
      for (let b = 0; b <= a; b++) A[a][b] += z[a] * z[b] / n;
      B[a][0] += z[a] * (Y[i][0] - ym[0]) / n;
      B[a][1] += z[a] * (Y[i][1] - ym[1]) / n;
    }
  }
  for (let a = 0; a < d; a++) { for (let b = 0; b < a; b++) A[b][a] = A[a][b]; A[a][a] += lambda; }
  const [wx, wy] = cholSolve(A, B);
  return { zm, ym, wx, wy };
}

function predictRidge(f, z) {
  let x = f.ym[0], y = f.ym[1];
  for (let j = 0; j < z.length; j++) { const c = z[j] - f.zm[j]; x += c * f.wx[j]; y += c * f.wy[j]; }
  return [x, y];
}

export const LAMBDAS = [1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1];

// Features are standardised; lambda is picked by leave-one-calibration-point-out cross-validation.
export class GazeModel {
  fit(X, Y, groups, lambdas = LAMBDAS) {
    const n = X.length, d = X[0].length;
    this.mu = new Float64Array(d); this.sd = new Float64Array(d);
    for (const x of X) for (let j = 0; j < d; j++) this.mu[j] += x[j] / n;
    for (const x of X) for (let j = 0; j < d; j++) this.sd[j] += (x[j] - this.mu[j]) ** 2 / n;
    for (let j = 0; j < d; j++) this.sd[j] = Math.sqrt(this.sd[j]) > 1e-9 ? Math.sqrt(this.sd[j]) : 1;
    const Z = X.map(x => this.standardise(x));
    const all = [...Array(n).keys()];
    const uniq = [...new Set(groups)];
    this.lambda = 0.01; this.cvError = null;
    if (uniq.length >= 4) {
      let best = Infinity;
      for (const lam of lambdas) {
        let err = 0;
        for (const g of uniq) {
          const train = all.filter(i => groups[i] !== g), test = all.filter(i => groups[i] === g);
          const f = fitRidge(Z, Y, train, lam);
          for (const i of test) { const p = predictRidge(f, Z[i]); err += Math.hypot(p[0] - Y[i][0], p[1] - Y[i][1]) / n; }
        }
        if (err < best) { best = err; this.lambda = lam; }
      }
      this.cvError = best;
    }
    this.f = fitRidge(Z, Y, all, this.lambda);
    return this;
  }
  standardise(x) { return x.map((v, j) => (v - this.mu[j]) / this.sd[j]); }
  predict(x) { return predictRidge(this.f, this.standardise(x)); }
}

// Drop samples whose iris features sit far from the point's median (saccades, half blinks).
export function cleanSamples(samples, keys = [0, 1, 3, 4], cut = 3.5) {
  if (samples.length < 4) return samples;
  const med = a => { const s = [...a].sort((p, q) => p - q); const k = s.length >> 1; return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2; };
  const stats = keys.map(k => {
    const col = samples.map(s => s.x[k]), m = med(col);
    return { k, m, mad: 1.4826 * med(col.map(v => Math.abs(v - m))) };
  });
  const kept = samples.filter(s => stats.every(({ k, m, mad }) => mad === 0 || Math.abs(s.x[k] - m) <= cut * mad));
  return kept.length >= samples.length / 2 ? kept : samples;
}

// ---------------------------------------------------------------- smoothing (One Euro filter, Casiez et al. 2012)

class LowPass {
  constructor() { this.y = null; }
  filter(x, a) { this.y = this.y === null ? x : a * x + (1 - a) * this.y; return this.y; }
}
export class OneEuro {
  constructor(minCutoff = 1, beta = 0.5, dCutoff = 1) { Object.assign(this, { minCutoff, beta, dCutoff }); this.reset(); }
  reset() { this.x = new LowPass(); this.dx = new LowPass(); this.t = null; }
  static alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  filter(v, tSec) {
    if (this.t === null) { this.t = tSec; this.dx.filter(0, 1); return this.x.filter(v, 1); }
    const dt = Math.max(1e-3, tSec - this.t);
    this.t = tSec;
    const dv = this.dx.filter((v - this.x.y) / dt, OneEuro.alpha(this.dCutoff, dt));
    return this.x.filter(v, OneEuro.alpha(this.minCutoff + this.beta * Math.abs(dv), dt));
  }
}
export class OneEuro2D {
  constructor(minCutoff, beta) { this.fx = new OneEuro(minCutoff, beta); this.fy = new OneEuro(minCutoff, beta); }
  set(minCutoff, beta) { for (const f of [this.fx, this.fy]) { f.minCutoff = minCutoff; f.beta = beta; } }
  reset() { this.fx.reset(); this.fy.reset(); }
  filter(x, y, tSec) { return [this.fx.filter(x, tSec), this.fy.filter(y, tSec)]; }
}

// ---------------------------------------------------------------- calibration layout and error

export function calibrationPoints() {        // 3 x 3, snake order so the eyes never jump across the screen
  const xs = [0.1, 0.5, 0.9], ys = [0.1, 0.5, 0.9];
  return ys.flatMap((y, r) => (r % 2 ? [...xs].reverse() : xs).map(x => [x, y]));
}
export function validationPoints() { return [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]; }

// points: [{target: [x, y], preds: [[x, y], ...]}] in screen fractions; W, H: screen size in px.
export function errorStats(points, W, H) {
  const per = points.filter(p => p.preds.length).map(p => {
    const mx = p.preds.reduce((s, q) => s + q[0], 0) / p.preds.length;
    const my = p.preds.reduce((s, q) => s + q[1], 0) / p.preds.length;
    return { target: p.target, mean: [mx, my], errPx: Math.hypot((mx - p.target[0]) * W, (my - p.target[1]) * H) };
  });
  if (!per.length) return null;
  const meanPx = per.reduce((s, p) => s + p.errPx, 0) / per.length;
  return { meanPx, pctDiag: 100 * meanPx / Math.hypot(W, H), perPoint: per };
}
