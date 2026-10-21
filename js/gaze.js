// Gaze math, free of browser APIs so it can be tested with `node --test`.
// Face landmarks (MediaPipe Face Landmarker, 478 points incl. iris) -> feature vector -> ridge regression -> screen point.

// MediaPipe face mesh indices. The two eyes are told apart by position in the image, not by name,
// so nothing depends on mirroring or on where the camera sits.
const EYES = [
  { corners: [33, 133], lids: [159, 145] },
  { corners: [362, 263], lids: [386, 374] },
];
const IRIS_CENTERS = [468, 473];
const IRIS_RINGS = [[469, 470, 471, 472], [474, 475, 476, 477]];   // four points on each iris edge
const NOSE_TIP = 1;
const CHIN = 152;
const BLEND = ['eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeLookDownLeft',
  'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight', 'eyeLookDownRight'];

export const FEATURE_NAMES = [
  'eyeA_h', 'eyeA_v', 'eyeA_vlid', 'eyeA_open', 'eyeB_h', 'eyeB_v', 'eyeB_vlid', 'eyeB_open',
  'blend_hL', 'blend_vL', 'blend_hR', 'blend_vR',
  'head_yaw', 'head_pitch', 'head_roll', 'face_x', 'face_y',
  'h_sq', 'v_sq', 'h_v',
];
export const IRIS_KEYS = [0, 1, 4, 5];            // iris offsets of both eyes (outlier check)
export const POSE_KEYS = [12, 13, 14, 15, 16];    // head pose and face position (range check, drift)
// smallest spread assumed per pose feature, so a calibration without movement does not make every later wobble "far"
const POSE_FLOOR = [0.03, 0.03, 0.05, 0.02, 0.02];

const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

// One eye. h, v: iris offset from the corner midpoint along the image axes, in eye widths. The iris sits in front of
// the eyeball centre, so this offset measures the angle between the gaze and the line from the eye to the camera:
// it stays the same when the head moves or turns while the eyes keep looking at the same screen point (to first
// order), and it rotates with the screen when the head rolls, as the gaze point does. vlid: iris below the lid
// midpoint (image axis), open: lid gap, both in eye widths.
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
    h: (irisPoint[0] - c[0]) / w,
    v: (irisPoint[1] - c[1]) / w,
    vlid: (irisPoint[1] - lidMid[1]) / w,
    open: dot(sub(bottom, top), v) / w,
    w, c, L, R, top, bottom, iris: irisPoint,
  };
}

// Iris diameter in pixels: the larger of the two edge-to-edge spans (a sideways look shortens one of them).
// The visible iris is about 11.7 mm wide in nearly everyone, so its size in the image gives the distance to the camera.
export function irisDiameter(pt, ring) {
  const d = (a, b) => Math.hypot(pt(a)[0] - pt(b)[0], pt(a)[1] - pt(b)[1]);
  return Math.max(d(ring[0], ring[2]), d(ring[1], ring[3]));
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
  const iris = (irisDiameter(pt, IRIS_RINGS[0]) + irisDiameter(pt, IRIS_RINGS[1])) / 2 / W;
  const x = [A.h, A.v, A.vlid, A.open, B.h, B.v, B.vlid, B.open, ...blend, yaw, pitch, roll, m[0] / W, m[1] / H,
    hm * hm, vm * vm, hm * vm];
  return { x, blink, size, iris, yaw, pitch, roll, faceX: m[0] / W, faceY: m[1] / H, minOpen: Math.min(A.open, B.open), eyes, iod };
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
    this.zlo = Z[0].map((_, j) => Math.min(...Z.map(z => z[j])) - 1);
    this.zhi = Z[0].map((_, j) => Math.max(...Z.map(z => z[j])) + 1);
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
  // features are held within the training range plus one standard deviation: a value the fit never saw (a pose far
  // from calibration) would otherwise be multiplied by a weight that was only ever tested on a tiny spread
  predict(x) { return predictRidge(this.f, this.standardise(x).map((z, j) => Math.min(this.zhi[j], Math.max(this.zlo[j], z)))); }
}

// Drop samples whose iris features sit far from the point's median (saccades, half blinks).
export function cleanSamples(samples, keys = IRIS_KEYS, cut = 3.5) {
  if (samples.length < 4) return samples;
  const med = a => { const s = [...a].sort((p, q) => p - q); const k = s.length >> 1; return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2; };
  const stats = keys.map(k => {
    const col = samples.map(s => s.x[k]), m = med(col);
    return { k, m, mad: 1.4826 * med(col.map(v => Math.abs(v - m))) };
  });
  const kept = samples.filter(s => stats.every(({ k, m, mad }) => mad === 0 || Math.abs(s.x[k] - m) <= cut * mad));
  return kept.length >= samples.length / 2 ? kept : samples;
}

// ---------------------------------------------------------------- from features to the screen, robust to moving

const median = a => { const s = [...a].sort((p, q) => p - q), k = s.length >> 1; return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2; };

// Where the front camera sits, in screen fractions. Only the distance correction uses it, so a rough guess is enough:
// phones have the camera at the top edge in portrait and at a side in landscape, laptops at the top.
// angle: screen.orientation.angle, 90 = the top of the phone turned to the left.
export function cameraPosition(portrait, angle = 0) {
  if (portrait) return angle === 180 ? [0.5, 1.02] : [0.5, -0.02];
  if (angle === 90) return [-0.02, 0.5];
  if (angle === 270 || angle === -90) return [1.02, 0.5];
  return [0.5, -0.02];
}

// The ridge model plus geometry, so the head or the phone may move after calibration:
//  - distance: the gaze point's offset from the camera grows with the distance (offset = distance x tan(angle)), so
//    targets are brought to the calibration distance before the fit and predictions are scaled back by
//    rho = distance now / distance at calibration = iris size at calibration / iris size now;
//  - pose: the calibration includes a phase with the eyes on one dot while the head moves, so the fit learns which
//    feature changes do not move the gaze point; outside() says how far the pose has gone beyond that range.
export class GazeMapper {
  constructor(camera = [0.5, -0.02]) { this.camera = camera; }
  rho(iris) { return iris > 0 && this.irisRef > 0 ? this.irisRef / iris : 1; }
  fit(samples) {                       // samples: [{x, iris, target: [x, y], group}]
    this.irisRef = median(samples.map(s => s.iris || 0));
    const c = this.camera, toRef = s => { const r = this.rho(s.iris); return [0, 1].map(a => c[a] + (s.target[a] - c[a]) / r); };
    this.model = new GazeModel().fit(samples.map(s => s.x), samples.map(toRef), samples.map(s => s.group));
    this.poseLo = POSE_KEYS.map(k => Math.min(...samples.map(s => s.x[k])));
    this.poseHi = POSE_KEYS.map(k => Math.max(...samples.map(s => s.x[k])));
    this.poseScale = POSE_KEYS.map((k, i) => Math.max(this.model.sd[k], POSE_FLOOR[i]));
    return this;
  }
  get lambda() { return this.model.lambda; }
  get cvError() { return this.model.cvError; }
  predict(x, iris) {
    const g = this.model.predict(x), r = this.rho(iris), c = this.camera;
    return [c[0] + r * (g[0] - c[0]), c[1] + r * (g[1] - c[1])];
  }
  // how far the head pose is beyond the range seen in calibration, in standard deviations (0 inside)
  outside(x) {
    return Math.max(0, ...POSE_KEYS.map((k, i) => Math.max(this.poseLo[i] - x[k], x[k] - this.poseHi[i]) / this.poseScale[i]));
  }
  poseVector(x) { return POSE_KEYS.map((k, i) => x[k] / this.poseScale[i]); }
}

// ---------------------------------------------------------------- drift (Kalman filter)

// The slow offset of the gaze point (drift over minutes, a new grip on the phone) is the state of a Kalman filter
// (Kalman 1960), modelled as a random walk: its variance P grows with the time since the last correction and with how
// far the pose has moved since then. A tap where the user looks measures the offset, and the gain K = P / (P + R)
// decides how much of it to take: nearly all of it after a big move, a share when the last correction was recent.
// Jitter is left to the One Euro filter; this one only carries the offset.
export class DriftFilter {
  constructor({ p0 = 0.05 ** 2, qTime = 0.003 ** 2, qPose = 0.03 ** 2, r = 0.03 ** 2 } = {}) {
    Object.assign(this, { qTime, qPose, r });
    this.reset(p0);
  }
  reset(p0 = this.p0, tSec = null, pose = null) {
    Object.assign(this, { p0, P: p0, beta: [0, 0], t: tSec, pose, n: 0 });
  }
  variance(tSec, pose) {               // tSec in seconds, pose from GazeMapper.poseVector
    let P = this.P;
    if (this.t !== null) P += this.qTime * Math.max(0, tSec - this.t);
    if (this.pose && pose) P += this.qPose * this.pose.reduce((s, v, i) => s + (pose[i] - v) ** 2, 0);
    return P;
  }
  correct(err, tSec, pose, r = this.r) {   // err: tapped point - (prediction + offset), screen fractions
    const P = this.variance(tSec, pose), K = P / (P + r);
    this.beta = [this.beta[0] + K * err[0], this.beta[1] + K * err[1]];
    Object.assign(this, { P: (1 - K) * P, t: tSec, pose, n: this.n + 1 });
    return K;
  }
  apply([x, y]) { return [x + this.beta[0], y + this.beta[1]]; }
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

// ---------------------------------------------------------------- closed eyes

// A frame counts as "eyes closed" when the lid gap falls below half of this person's usual gap (75th percentile of
// the last ~3 s), or the blink blendshape is very high. A fixed blendshape cut-off (0.45) threw away most frames on a
// phone held low: looking down narrows the lids and raises eyeBlink although the eyes are open (real phone, 2026-10-21).
export class EyeClosure {
  constructor(window = 90, ratio = 0.5, blendCut = 0.85) { Object.assign(this, { window, ratio, blendCut }); this.hist = []; }
  closed(f) {
    if (!f) return true;
    const ref = this.hist.length >= 15 ? [...this.hist].sort((a, b) => a - b)[Math.floor(this.hist.length * 0.75)] : null;
    const shut = f.blink > this.blendCut || (ref !== null && f.minOpen < this.ratio * ref);
    this.hist.push(f.minOpen);
    if (this.hist.length > this.window) this.hist.shift();
    return shut;
  }
}
