// node --test tests/*.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { extractFeatures, GazeModel, GazeMapper, DriftFilter, OneEuro, cleanSamples, errorStats, calibrationPoints, cameraPosition,
  FEATURE_NAMES, POSE_KEYS } from '../js/gaze.js';

// A crude face in a 640 x 480 frame: two eyes, irises with their edge points, nose, chin. Coordinates in pixels.
// zoom scales the face about the frame centre (a face closer to the camera).
function face({ irisDx = 0, irisDy = 0, rotate = 0, shift = [0, 0], zoom = 1 } = {}) {
  const pts = Array.from({ length: 478 }, () => [320, 240]);
  const set = (i, x, y) => { pts[i] = [x, y]; };
  set(33, 250, 200); set(133, 290, 200); set(159, 270, 192); set(145, 270, 208);   // eye 1
  set(362, 350, 200); set(263, 390, 200); set(386, 370, 192); set(374, 370, 208);  // eye 2
  [[468, 270], [473, 370]].forEach(([c, x0], e) => {
    const x = x0 + irisDx, y = 200 + irisDy;
    set(c, x, y);
    [[6, 0], [0, -6], [-6, 0], [0, 6]].forEach(([dx, dy], k) => set(469 + 5 * e + k, x + dx, y + dy));   // radius 6 px
  });
  set(1, 320, 250); set(152, 320, 320);
  const c = Math.cos(rotate), s = Math.sin(rotate);
  return pts.map(([x, y]) => {
    const dx = (x - 320) * zoom, dy = (y - 240) * zoom;
    return { x: (320 + c * dx - s * dy + shift[0]) / 640, y: (240 + s * dx + c * dy + shift[1]) / 480, z: 0 };
  });
}
const feats = o => extractFeatures(face(o), [], 640, 480);

test('feature vector has one value per name', () => {
  assert.equal(feats().x.length, FEATURE_NAMES.length);
});

test('iris moving right raises h, moving down raises v', () => {
  const a = feats(), r = feats({ irisDx: 5 }), d = feats({ irisDy: 3 });
  assert.ok(r.x[0] > a.x[0] && r.x[4] > a.x[4]);
  assert.ok(d.x[1] > a.x[1] && d.x[5] > a.x[5]);
});

test('eye features do not change when the face moves in the frame', () => {
  const a = feats({ irisDx: 4, irisDy: 2 }), b = feats({ irisDx: 4, irisDy: 2, shift: [40, -25] });
  for (const k of [0, 1, 2, 3, 4, 5, 6, 7]) assert.ok(Math.abs(a.x[k] - b.x[k]) < 1e-9, FEATURE_NAMES[k]);
});

test('when the head rolls, the iris offset turns with it, as the gaze point on the screen does', () => {
  const t = 0.3, a = feats({ irisDx: 4, irisDy: 2 }), b = feats({ irisDx: 4, irisDy: 2, rotate: t });
  const [h, v] = [a.x[0], a.x[1]];
  assert.ok(Math.abs(b.x[0] - (Math.cos(t) * h - Math.sin(t) * v)) < 1e-9);
  assert.ok(Math.abs(b.x[1] - (Math.sin(t) * h + Math.cos(t) * v)) < 1e-9);
  assert.ok(Math.abs(b.roll - t) < 1e-9);
  assert.ok(Math.abs(b.x[3] - a.x[3]) < 1e-9, 'lid gap is a length and does not turn');
});

test('iris size follows the distance; eye features do not', () => {
  const a = feats({ irisDx: 3 }), b = feats({ irisDx: 3, zoom: 1.25 });
  assert.ok(Math.abs(b.iris / a.iris - 1.25) < 1e-9, `${b.iris / a.iris}`);
  for (const k of [0, 1, 2, 3]) assert.ok(Math.abs(a.x[k] - b.x[k]) < 1e-9, FEATURE_NAMES[k]);
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
      const x = [h, v, v, 0.3, h, v, v, 0.3, h * 2, -v * 2, h * 2, -v * 2, 0, 0.5, 0, 0.5, 0.4, h * h, v * v, h * v]
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
  const s = Array.from({ length: 10 }, (_, i) => ({ x: [0.1 + i * 1e-4, 0.2, 0, 0, 0.1, 0.2] }));
  s.push({ x: [0.6, 0.9, 0, 0, 0.6, 0.9] });
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

test('features far beyond the training range are held at its edge', () => {
  const { X, Y, groups } = synth(20);
  const m = new GazeModel().fit(X, Y, groups);
  const far = [...X[0]], farther = [...X[0]];
  far[12] = 50; farther[12] = 5000;                     // head yaw, nearly constant in calibration
  assert.deepEqual(m.predict(far), m.predict(farther));
  assert.ok(m.predict(farther).every(Number.isFinite));
});

test('GazeMapper: farther from the camera, the same eye features point farther from it', () => {
  const { X, Y, groups } = synth(20);
  const cam = cameraPosition(true);
  const mapper = new GazeMapper(cam).fit(X.map((x, i) => ({ x, iris: 0.03, target: Y[i], group: groups[i] })));
  const near = mapper.predict(X[0], 0.03), far = mapper.predict(X[0], 0.025);   // 0.03 / 0.025 = 1.2 times as far
  for (const a of [0, 1]) assert.ok(Math.abs(far[a] - (cam[a] + 1.2 * (near[a] - cam[a]))) < 1e-9);
  assert.equal(mapper.outside(X[0]), 0);
  const moved = [...X[0]]; moved[POSE_KEYS[3]] += 0.2;  // face moved a fifth of the frame
  assert.ok(mapper.outside(moved) > 5);
});

// Toy physics behind the movement phase. The gaze angle to the camera (theta) follows the target; the head pose psi
// is free. Iris offsets measure theta (noisy), lid and blendshape cues measure the eye in the head, theta - psi (less
// noisy), pose features measure psi. Without movement in calibration the fit cannot tell psi apart and errs once the
// head turns; with the movement phase it learns that psi plus the eye in the head is what counts.
function toy(seed, moves) {
  let s = seed; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5);
  const feat = ([tx, ty], [px, py]) => {
    const th = [(tx - 0.5) * 0.4, (ty - 0.5) * 0.3];
    return [th[0] + rnd() * 0.06, th[1] + rnd() * 0.06, th[0] - px + rnd() * 0.01, th[1] - py + rnd() * 0.01,
      px + rnd() * 0.004, py + rnd() * 0.004];
  };
  const X = [], Y = [], G = [];
  calibrationPoints().forEach((t, g) => { for (let k = 0; k < 15; k++) { X.push(feat(t, [rnd() * 0.01, rnd() * 0.01])); Y.push(t); G.push(g); } });
  if (moves) for (let k = 0; k < 60; k++) { X.push(feat([0.5, 0.5], [rnd() * 0.3, rnd() * 0.3])); Y.push([0.5, 0.5]); G.push(50); }
  const m = new GazeModel().fit(X, Y, G);
  const err = psi => {
    let e = 0, n = 0;
    for (const t of [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]) for (let k = 0; k < 20; k++) {
      const p = m.predict(feat(t, psi)); e += Math.hypot(p[0] - t[0], p[1] - t[1]); n++;
    }
    return e / n;
  };
  return { still: err([0, 0]), turned: err([0.1, 0.06]) };
}

test('movement phase: after the head turns, the error stays near the still error', () => {
  const without = toy(7, false), withMoves = toy(7, true);
  if (process.env.VERBOSE) console.log({ without, withMoves });
  assert.ok(withMoves.turned < 0.5 * without.turned, `turned: ${withMoves.turned.toFixed(3)} vs ${without.turned.toFixed(3)} without`);
  assert.ok(withMoves.turned < 2 * withMoves.still + 0.02, `with: still ${withMoves.still.toFixed(3)}, turned ${withMoves.turned.toFixed(3)}`);
});

test('DriftFilter (Kalman): a first tap moves the offset most of the way, a quick second one less', () => {
  const d = new DriftFilter({ p0: 0.1 ** 2, r: 0.03 ** 2 });
  assert.deepEqual(d.apply([0.5, 0.5]), [0.5, 0.5]);
  const k1 = d.correct([0.1, 0], 0, [0, 0]);
  assert.ok(k1 > 0.9 && Math.abs(d.beta[0] - 0.1 * k1) < 1e-12, `first gain ${k1}`);
  const k2 = d.correct([0, 0], 1, [0, 0]);
  assert.ok(k2 < 0.6 * k1, `second gain ${k2}`);
});

test('DriftFilter (Kalman): the offset is trusted less after time passes or the pose moves', () => {
  const d = new DriftFilter();
  d.correct([0, 0], 0, [0, 0]); d.correct([0, 0], 1, [0, 0]);
  const soon = d.variance(2, [0, 0]), moved = d.variance(2, [3, 0]), later = d.variance(600, [0, 0]);
  assert.ok(moved > 5 * soon && later > 5 * soon, `variance soon ${soon}, after a move ${moved}, 10 min later ${later}`);
});
