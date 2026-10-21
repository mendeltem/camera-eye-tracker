# Camera Eye Tracker

Eye tracking in the browser with nothing but the normal front camera. Three figures sit on the screen; only the one you
look at lights up, and a dot shows where your eyes point.

**Open it:** https://mendeltem.github.io/camera-eye-tracker/ (phone or laptop; allow the camera when asked)

- The camera picture is never shown and never leaves the device. Face landmarks are computed in the browser
  (MediaPipe Face Landmarker, WebAssembly); nothing is uploaded, nothing is stored.
- The camera can sit anywhere: above, below or beside the screen, or an external webcam. Calibration learns the
  geometry. Afterwards head and phone may move a little and the distance may change; after a large move the app
  shows a dot to look at and tap once.
- No build step, no server code: three static files and one JavaScript module, served by GitHub Pages.

## How to use it

1. **Set up.** The app draws your two eyes as it sees them (from landmarks, not the camera picture) and checks face,
   open eyes, distance, light and frame rate.
2. **Calibrate.** Follow 9 dots with your eyes. Then keep looking at one dot while you turn your head slowly left,
   right, up and down (or tilt the phone). Then 4 test dots measure the error, which the app shows in pixels and as
   a share of the screen diagonal. About 30 seconds in all.
3. **Look.** *Figures*: the figure you look at lights up; hold your gaze about a second and it hops.
   *Dot*: the gaze point alone. *Heat*: where you looked builds up colour.
   Tap the spot you are looking at to correct the tracker: half a second of frames measures the offset there.

Good results need light on your face (a bright screen helps). Glasses with reflections, a dark room or a window behind
you make it worse. Turning the phone between portrait and landscape means calibrating again.

After calibration the result card offers **Save the data**: a JSON file with the features, targets and timings of the
calibration (numbers only, no picture), saved on your device. It lets a later version of the model be tried on a
real calibration.

## Accuracy you can expect

Camera-only gaze on a phone lands around **2 to 3 cm** of error (best case about 1.5 cm); that tells three large
figures apart, not single words. The app measures your own error after every calibration instead of promising a
number. Published figures, for comparison:

| Study | Setup | Reported error |
|---|---|---|
| Krafka et al. 2016 | phone, trained CNN (iTracker), with calibration | 1.34 cm |
| Valliappan et al. 2020 | phone on a stand, CNN fine-tuned per person | 0.46 cm (hand-held 0.50 cm) |
| Papoutsaki et al. 2016 | laptop browser, WebGazer, in the lab | 4.17° (about 1.6 inch) |
| Semmelmann & Weigelt 2018 | desktop browser, WebGazer, online | 207 px, about 18 % of the screen |
| Davalos et al. 2025 | browser, few-shot model (WebEyeTrack) | 2.32 cm on GazeCapture |

This app uses hand-built eye features and a per-person regression, closer to WebGazer than to the trained networks.

## How it works

```
front camera ─► MediaPipe Face Landmarker (478 points incl. iris, blendshapes)
             ─► 20 features: per eye the iris offset from the eye-corner midpoint along the image axes and below the
                lid midpoint (in eye widths) and the lid gap; blendshape gaze (look in/out/up/down); head yaw, pitch,
                roll and face position from the landmarks; three quadratic terms. Iris diameter for the distance.
             ─► ridge regression to the screen point, features standardised and held within the calibrated range,
                lambda chosen by leave-one-calibration-dot-out cross-validation
             ─► distance correction (iris size), offset carried by a Kalman filter that taps update
             ─► One Euro filter (Casiez et al. 2012), closed eyes skipped
             ─► figures / dot / heat view
```

- Calibration frames are taken only after the eyes have settled on each dot, outliers per dot are dropped (median
  absolute deviation), and the 4 test dots are scored before they join the training set.
- `js/gaze.js` holds all the math and has no browser code; `js/app.js` is camera, screens and drawing.

### Moving after calibration

A regression fitted while head and phone stayed still has never seen the pose change. Its weights for head pose and
face position were fitted on a tiny spread, and a later move multiplies them by values far outside it, so the point
jumps. Four parts address this:

1. **Features that move with the gaze, not with the head.** The iris sits in front of the eyeball centre, so its
   offset in the image measures the angle between the gaze and the line from the eye to the camera. To first order
   it stays the same when the head moves or turns while the eyes keep looking at the same screen point, and it
   rotates with the screen when the head rolls, as the gaze point does. The offset is therefore measured along the
   image axes, not in the eye's own tilted frame.
2. **A movement phase in the calibration.** The eyes stay on one dot while the head turns. These frames teach the
   fit which feature changes do not move the gaze point (for example lids and blendshapes, which follow the eye in
   the head). Features are held within the calibrated range plus one standard deviation, so a pose beyond it cannot
   throw the point across the screen.
3. **Distance from the iris size.** The visible iris is about 11.7 mm wide in nearly everyone (Vakunov & Lagun
   2020), so its size in the image gives the distance relative to calibration. The gaze point's offset from the
   camera grows with the distance, and predictions are scaled by that ratio. The camera position this needs is a
   guess (top edge in portrait, side in landscape); only the distance correction uses it.
4. **The slow offset as a Kalman state.** Drift and a new grip shift the whole point. The offset is the state of a
   Kalman filter (Kalman 1960), modelled as a random walk whose variance grows with time and with how far the pose
   moved since the last correction. A tap where you look measures it, and the gain decides how much to take: nearly
   all after a large move, a share right after the previous tap. When the pose leaves the calibrated range for a
   second, a dot appears in the middle to look at and tap.

The phone's gyroscope is not used. It measures how the phone turns in the room, while the gaze point depends on
the phone relative to the head, which the camera already sees. Walking with the phone held still in front of the face
would read as movement, and iOS would ask for one more permission.

The toy simulation in `tests/gaze.test.mjs` shows the mechanism: after a head turn the error with the movement
phase stays at the still error, without it the error grows to about a quarter of the screen. That simulation
follows the assumptions above; the gain on a real phone has not been measured yet.

## Run it yourself

```bash
python3 -m http.server 8000      # then open http://localhost:8000 (the camera needs https or localhost)
node --test tests/*.test.mjs     # unit tests of the gaze math
```

End-to-end check in headless Chromium with a fake camera playing a face clip (`.y4m`). It checks that the whole
chain runs without errors; a still photo cannot follow the dots, so it says nothing about accuracy.

```bash
npm i playwright && npx playwright install chromium
FAKE_VIDEO=face.y4m node tests/browser_smoke.mjs
```

## Known limits and next steps

- Tested in headless Chromium with a fake camera; check it on your own phone (Chrome on Android, Safari on iOS 15.4+).
- `detectForVideo` runs on the main thread; on slow phones a Web Worker would keep the page smoother.
- Accuracy drifts over minutes (head and lighting change). Tap corrections help; recalibrate when the dot is off.
- The offset is one value for the whole screen and every pose: after a re-centre far from the calibrated pose,
  going back may need another tap.
- Possible gains: kernel ridge regression, a smooth-pursuit calibration with more frames, a small eye-image patch.

## Sources

Each was opened and checked on 2026-10-21.

- Papoutsaki A, Sangkloy P, Laskey J, Daskalova N, Huang J, Hays J (2016). WebGazer: Scalable webcam eye tracking using
  user interactions. IJCAI-16, 3839-3845. https://www.ijcai.org/Proceedings/16/Papers/540.pdf
  Ridge regression from small eye patches to screen pixels, trained at each click; 4.17° in the lab.
- Semmelmann K, Weigelt S (2018). Online webcam-based eye tracking in cognitive science: A first look. Behav Res
  Methods 50(2):451-465. doi:10.3758/s13428-017-0913-7. WebGazer offsets of 15 % (lab) and 18 % (online) of the screen.
- Krafka K, Khosla A, Kellnhofer P, Kannan H, Bhandarkar S, Matusik W, Torralba A (2016). Eye tracking for everyone.
  CVPR 2016. arXiv:1606.05814. GazeCapture data set (1,450+ people) and iTracker; 1.34 cm on phones with calibration.
- Valliappan N et al. (2020). Accelerating eye movement research via accurate and affordable smartphone eye tracking.
  Nat Commun 11:4553. doi:10.1038/s41467-020-18360-5. 0.46 cm after under 30 s of personal calibration; fails with
  extreme head pose, looking down, glasses, dark rooms.
- Heck M, Becker C, Deutscher V (2023). Webcam eye tracking for desktop and mobile devices: A systematic review.
  HICSS-56, 6820-6829. https://hdl.handle.net/10125/103459. Beyond about 4 calibration points gains are small; head
  motion, poor light and glasses cost the most; phones do worse relative to screen size.
- Kaduk T, Goeke C, Finger H, König P (2023). Webcam eye tracking close to laboratory standards. Behav Res Methods.
  doi:10.3758/s13428-023-02237-8. 1.4° on desktop, but only with a long calibration and a head-movement guard.
- Slim MS, Kandel M, Yacovone A, Snedeker J (2024). Webcams as windows to the mind? Open Mind 8:1369-1424.
  doi:10.1162/opmi_a_00171. WebGazer offset about 24 % of the screen in smooth pursuit vs 13 % for an infrared tracker.
- Davalos E et al. (2025). WebEyeTrack: Scalable eye-tracking for the browser via on-device few-shot
  personalization. arXiv:2508.19544. MediaPipe landmarks plus an eye-patch model adapted with up to 9 calibration
  samples; 2.32 cm on GazeCapture, error drifts over 20-minute sessions.
- Vakunov A, Lagun D (2020). MediaPipe Iris: Real-time iris tracking and depth estimation. Google Research blog,
  2020-08-06. https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/
  Horizontal iris diameter 11.7 ± 0.5 mm across a wide population; distance from iris size with a mean relative
  error of 4.3 % against a depth sensor.
- Kalman RE (1960). A new approach to linear filtering and prediction problems. J Basic Eng 82(1):35-45.
  doi:10.1115/1.3662552. The recursive estimator used here for the offset.
- Casiez G, Roussel N, Vogel D (2012). 1 € filter: A simple speed-based low-pass filter for noisy input in
  interactive systems. CHI 2012. https://gery.casiez.net/1euro/. The smoothing filter used here, re-implemented.
- MediaPipe Face Landmarker for Web. https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/web_js
  Package `@mediapipe/tasks-vision` (Apache-2.0), loaded from jsDelivr; the model file is loaded from Google's
  model storage, as in the official guide.

No code from WebGazer (GPL-3.0) or other trackers is included; the ideas are re-implemented from the papers.

## License

MIT, see [LICENSE](LICENSE).
