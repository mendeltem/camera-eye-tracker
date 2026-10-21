# Camera Eye Tracker

Eye tracking in the browser with nothing but the normal front camera. Three figures sit on the screen; only the one you
look at lights up, and a dot shows where your eyes point.

**Open it:** https://mendeltem.github.io/camera-eye-tracker/ (phone or laptop; allow the camera when asked)

- The camera picture is never shown and never leaves the device. Face landmarks are computed in the browser
  (MediaPipe Face Landmarker, WebAssembly); nothing is uploaded, nothing is stored.
- The camera can sit anywhere: above, below or beside the screen, or an external webcam. Calibration learns the
  geometry. Keep camera and screen in the same place afterwards.
- No build step, no server code: three static files and one JavaScript module, served by GitHub Pages.

## How to use it

1. **Set up.** The app draws your two eyes as it sees them (from landmarks, not the camera picture) and checks face,
   open eyes, distance, light and frame rate.
2. **Calibrate.** Follow 9 dots with your eyes, about 20 seconds, head still. Then 4 test dots measure the error,
   which the app shows in pixels and as a share of the screen diagonal.
3. **Look.** *Figures*: the figure you look at lights up; hold your gaze about a second and it hops.
   *Dot*: the gaze point alone. *Heat*: where you looked builds up colour.
   Tap the spot you are looking at to correct the tracker; each tap adds half a second of training frames.

Good results need light on your face (a bright screen helps), a steady head and the device at the distance you
calibrated at. Glasses with reflections, a dark room or a window behind you make it worse. Turning the phone means
calibrating again.

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
             ─► 19 features: per eye the iris offset in the eye's own frame (along / across the corner line, in eye
                widths) and the lid gap; blendshape gaze (look in/out/up/down); head yaw, pitch, roll from the
                landmarks; face position and size in the frame; three quadratic terms
             ─► ridge regression to the screen point, features standardised, lambda chosen by
                leave-one-calibration-dot-out cross-validation
             ─► One Euro filter (Casiez et al. 2012), blinks skipped (eyeBlink blendshape)
             ─► figures / dot / heat view
```

- Eye features are measured in the eye's own frame, so a tilted head or a camera off to the side does not shift them;
  head pose and face position are separate features, so the regression learns where the camera is.
- Calibration frames are taken only after the eyes have settled on each dot (700 ms), outliers per dot are dropped
  (median absolute deviation), and the 4 test dots are scored before they join the training set.
- `js/gaze.js` holds all the math and has no browser code; `js/app.js` is camera, screens and drawing.

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
- Casiez G, Roussel N, Vogel D (2012). 1 € filter: A simple speed-based low-pass filter for noisy input in
  interactive systems. CHI 2012. https://gery.casiez.net/1euro/. The smoothing filter used here, re-implemented.
- MediaPipe Face Landmarker for Web. https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/web_js
  Package `@mediapipe/tasks-vision` (Apache-2.0), loaded from jsDelivr; the model file is loaded from Google's
  model storage, as in the official guide.

No code from WebGazer (GPL-3.0) or other trackers is included; the ideas are re-implemented from the papers.

## License

MIT, see [LICENSE](LICENSE).
