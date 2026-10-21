// node --test tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import { extractFeatures, GazeModel, OneEuro, cleanSamples, errorStats, calibrationPoints, FEATURE_NAMES } from '../js/gaze.js';

// A crude face in a 640 x 480 frame: two eyes, irises, nose, chin. Coordinates in pixels.
function face({ irisDx = 0, irisDy = 0, rotate = 0, shift = [0, 0] } = {}) {
  const pts = Array.from({ length: 478 }, () => [320, 240]);
  const set = (i, x, y) => { pts[i] = [x, y]; };
  set(33, 250, 200); set(133, 290, 200); set(159, 270, 192); set(145, 270, 208);   // eye 1
  set(362, 350, 200); set(263, 390, 200); set(386, 370, 192); set(374, 370, 208);  // eye 2
  set(468, 270 + irisDx, 200 + irisDy); set(473, 370 + irisDx, 200 + irisDy);
  set(1, 320, 250); set(152, 320, 320);
  const c = Math.cos(rotate), s = Math.sin(rotate);
  return pts.map(([x, y]) => {
    const dx = x - 320, dy = y - 240;
    return { x: (320 + c * dx - s * dy + shift[0]) / 640, y: (240 + s * dx + c * dy + shift[1]) / 480, z: 0 };
  });
}
const feats = o => extractFeatures(face(o), [], 640, 480);

test('feature vector has one value per name', () => {
  assert.equal(feats().x.length, FEATURE_NAMES.length);
});

test('iris moving right raises h, moving down raises v', () => {
  const a = feats(), r = feats({ irisDx: 5 }), d = feats({ irisDy: 3 });
  assert.ok(r.x[0] > a.x[0] && r.x[3] > a.x[3]);
  assert.ok(d.x[1] > a.x[1] && d.x[4] > a.x[4]);
});

test('eye features do not change when the head rolls or moves in the frame', () => {
  const a = feats({ irisDx: 4, irisDy: 2 }), b = feats({ irisDx: 4, irisDy: 2, rotate: 0.3, shift: [40, -25] });
  for (const k of [0, 1, 2, 3, 4, 5]) assert.ok(Math.abs(a.x[k] - b.x[k]) < 1e-9, FEATURE_NAMES[k]);
  assert.ok(Math.abs(b.roll - 0.3) < 1e-9);
});

test('swapped iris indices are paired with the right eye', () => {
  const lm = face({ irisDx: 3 });
  [lm[468], lm[473]] = [lm[473], lm[468]];
  const a = extractFeatures(lm, [], 640, 480), b = feats({ irisDx: 3 });
  assert.deepEqual(a.x, b.x);
});

test('too few landmarks (no iris model) gives null', () => {
  assert.equal(extractFeatures(face().slice(0, 468), [], 640, 480), null);
});

// synthetic calibration: the screen point is a smooth function of two hidden gaze angles plus noise
function synth(n, seed = 1) {
  let s = seed; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const pts = calibrationPoints(), X = [], Y = [], groups = [];
  pts.forEach(([tx, ty], g) => {
    for (let k = 0; k < n; k++) {
      const h = (tx - 0.5) * 0.3 + (rnd() - 0.5) * 0.01, v = (ty - 0.5) * 0.2 + (rnd() - 0.5) * 0.01;
      const x = [h, v, 0.3, h, v, 0.3, h * 2, -v * 2, h * 2, -v * 2, 0, 0.5, 0, 0.5, 0.4, 0.2, h * h, v * v, h * v]
        .map(val => val + (rnd() - 0.5) * 0.002);
      X.push(x); Y.push([tx, ty]); groups.push(g);
    }
  });
  return { X, Y, groups };
}

test('ridge model recovers the screen point on clean synthetic data', () => {
  const { X, Y, groups } = synth(20);
  const m = new GazeModel().fit(X, Y, groups);
  const err = X.reduce((s, x, i) => { const p = m.predict(x); return s + Math.hypot(p[0] - Y[i][0], p[1] - Y[i][1]); }, 0) / X.length;
  assert.ok(err < 0.05, `mean error ${err}`);
  assert.ok(m.cvError !== null && m.cvError < 0.1, `cv error ${m.cvError}`);
});

test('constant features (a still photo) give finite predictions, not NaN', () => {
  const X = Array.from({ length: 30 }, () => feats().x), Y = X.map((_, i) => [(i % 3) / 2, 0.5]);
  const m = new GazeModel().fit(X, Y, X.map((_, i) => i % 5));
  assert.ok(m.predict(X[0]).every(Number.isFinite));
});

test('cleanSamples drops a saccade sample and keeps the rest', () => {
  const s = Array.from({ length: 10 }, (_, i) => ({ x: [0.1 + i * 1e-4, 0.2, 0, 0.1, 0.2] }));
  s.push({ x: [0.6, 0.9, 0, 0.6, 0.9] });
  assert.equal(cleanSamples(s).length, 10);
});

test('One Euro filter calms jitter and still follows a jump', () => {
  const f = new OneEuro(1, 0.5);
  let s = 3, out = [];
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 0.1;
  for (let k = 0; k < 120; k++) out.push(f.filter(0.5 + rnd(), k / 30));
  const sd = a => Math.sqrt(a.reduce((p, v) => p + (v - 0.5) ** 2, 0) / a.length);
  assert.ok(sd(out.slice(30)) < 0.5 * 0.1 / Math.sqrt(12), 'jitter reduced by half or more');
  let last;
  for (let k = 120; k < 150; k++) last = f.filter(0.9, k / 30);
  assert.ok(Math.abs(last - 0.9) < 0.02, `after 1 s at the new place: ${last}`);
});

test('errorStats reports pixels and percent of the screen diagonal', () => {
  const st = errorStats([{ target: [0.5, 0.5], preds: [[0.6, 0.5], [0.6, 0.5]] }], 1000, 1000);
  assert.ok(Math.abs(st.meanPx - 100) < 1e-9);
  assert.ok(Math.abs(st.pctDiag - 100 * 100 / Math.hypot(1000, 1000)) < 1e-9);
});

test('EyeClosure: looking down is not a blink, a real blink is', async () => {
  const { EyeClosure } = await import('../js/gaze.js');
  const gate = new EyeClosure();
  for (let k = 0; k < 30; k++) gate.closed({ minOpen: 0.30, blink: 0.1 });
  assert.equal(gate.closed({ minOpen: 0.20, blink: 0.6 }), false, 'looking down: lids narrower, blendshape 0.6');
  assert.equal(gate.closed({ minOpen: 0.05, blink: 0.7 }), true, 'lids almost shut');
  assert.equal(gate.closed({ minOpen: 0.28, blink: 0.9 }), true, 'blendshape very high');
  assert.equal(new EyeClosure().closed({ minOpen: 0.3, blink: 0.5 }), false, 'no history yet: only the blendshape decides');
});
