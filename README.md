# SignEase Bridge 🤟

**A browser extension for two-way sign language ↔ speech translation on video calls.**

SignEase Bridge lets a deaf/mute person and a hearing person understand each
other on a live video call (Google Meet, Zoom Web, Teams Web), in both
directions:

1. **Sign → Speech**: the deaf/mute person signs into their webcam. An
   on-device hand-tracking pipeline recognizes ASL letters or short words and
   the extension speaks the result out loud (and shows it as a caption).
2. **Speech → Sign**: the hearing person's voice from the call is
   transcribed and matched against a library of recorded sign sequences,
   replayed to the deaf/mute person as an animated hand skeleton.

This is a ground-up rebuild of an earlier hackathon project — the original
(a standalone React web app with a TensorFlow model and fabricated benchmark
numbers) has been replaced with real trained PyTorch models, a single clean
FastAPI backend, and a Manifest V3 browser extension as the actual delivery
target.

## Honest numbers

No number below is hardcoded or invented — both are the model's accuracy on
a held-out test split it never saw during training (`backend/checkpoints/*_report.json`).

| Model | Classes | Train / Val / Test samples | Test accuracy |
|---|---|---|---|
| Letter classifier (ResMLP) | 28 (A–Z, minus "nothing") | 11,851 / 2,540 / 2,540 | **98.66%** |
| Word classifier (Conv1D + Transformer) | 45 curated words | 9,450 / 2,025 / 2,025 | **80.94%** |

**Known limitation:** the word model was trained on hand *and* upper-body
pose landmarks (MediaPipe Holistic), but the extension currently only
captures hands (MediaPipe HandLandmarker) — there's no pose tracking in the
browser yet. Feeding zeroed-out pose data at inference costs real accuracy on
pose-dependent signs. Measured offline: forcing pose to all-zero drops word
accuracy from 80.94% to ~52%. This is a documented gap, not a silent bug —
closing it means adding MediaPipe's PoseLandmarker to the client capture
pipeline, tracked as a follow-up.

Recognized word vocabulary (45 words, not general ASL — sourced from a
baby/family sign-language subset of the Kaggle `asl-signs` dataset):
`hello, bye, please, thankyou, yes, no, water, drink, food, hungry, thirsty,
sick, happy, sad, hot, sleepy, open, close, wait, go, look, listen, home,
potty, tomorrow, later, now, time, where, why, who, finish, clean, dirty,
quiet, loud, fine, bad, mom, dad, sleep, shower, milk, up, down`.

## Architecture

```
backend/                       FastAPI service — the only backend, replaces every
                                duplicate Flask server from the old repo.
  training/
    common.py                  Landmark normalization/feature engineering shared
                                by both pipelines.
    prepare_letters.py         Kaggle asl-alphabet -> per-frame landmark features.
    prepare_words.py           Kaggle asl-signs -> landmark sequences (45 curated
                                words), bulk download from the extracted archive.
    train_letters.py           ResMLP training, reports real val/test accuracy.
    train_words.py             Conv1D + TransformerEncoder + masked attention
                                pooling, same honest-accuracy reporting.
    export_sign_references.py  Picks one representative landmark sample/sequence
                                per class (nearest-to-centroid) for the
                                speech->sign skeleton replay — generated from
                                training data, no licensed video/assets.
  models/
    letter_classifier.py       ResMLP architecture.
    word_classifier.py         Conv1D+Transformer architecture.
  inference/engine.py          Loads both checkpoints once, serves predictions.
  services/
    tts.py                     Sarvam TTS (bulbul:v2) primary, local Piper
                                fallback.
    stt.py                     Sarvam ASR (saaras:v3) primary, local
                                faster-whisper fallback.
  api/routes.py                /api/predict/letter, /api/predict/word,
                                /api/speak, /api/transcribe, /api/sign-references,
                                /api/health.
  app.py                       FastAPI app + CORS (localhost dev origins +
                                chrome-extension://* via regex).

extension/                     Manifest V3 browser extension — the primary product.
  src/background/
    service-worker.ts          Routes messages between content script, offscreen
                                document, and the backend (backend calls run here
                                and in the offscreen doc, both outside the host
                                page's CSP).
  src/offscreen/
    offscreen.ts                Owns the persistent offscreen document (survives
                                service-worker suspension); lazy-imports handTracking
                                to dodge a message-listener registration race.
    handTracking.ts             Camera capture + MediaPipe hand tracking + prediction
                                loop. Runs on a setInterval, not requestAnimationFrame
                                — offscreen documents are never painted, so rAF is
                                throttled/stalled there.
  src/lib/mediapipeHands.ts     HandLandmarker wrapper (MediaPipe Tasks API, CPU
                                delegate). Uses MediaPipe's raw Left/Right handedness
                                labels directly — matching how the training data
                                was labeled.
  src/content-scripts/          Injected into the call page: overlay panel (mode
                                toggles, captions, debug log), speech-to-sign relay.
  src/permission/               Dedicated persistent tab for granting camera
                                permission (offscreen documents can't prompt for it,
                                and the popup auto-closes on blur mid-prompt).
  src/popup/                    Settings (backend URL).

camera-test/                   Standalone test harness — plain HTML/JS page, no
                                extension involved. Used to isolate and verify the
                                camera + MediaPipe + prediction pipeline independent
                                of any Chrome-extension-specific bugs. The MediaPipe
                                wasm/model binaries it loads aren't tracked in git
                                (they're just copies of extension/ assets) — copy
                                them in before running:
                                `cp -r extension/dist/mediapipe-wasm extension/dist/hand_landmarker.task extension/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs camera-test/`
                                then `python3 -m http.server 5500` from this directory.

signease-frontend/             Kept as a secondary, installable-extension-free
                                practice-mode site (not the primary product anymore).
```

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env`:
```env
SARVAM_API_KEY=your_sarvam_key   # optional — falls back to local Piper/faster-whisper if unset
```

Run it:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8001
```

Training your own models needs Kaggle API credentials (`~/.kaggle/kaggle.json`)
and accepting the `asl-signs` competition rules on kaggle.com once, then:
```bash
python -m backend.training.prepare_letters
python -m backend.training.train_letters
python -m backend.training.prepare_words
python -m backend.training.train_words
python -m backend.training.export_sign_references
```

### Extension

```bash
cd extension
npm install
npm run build
```

Load it in Chrome: `chrome://extensions` → enable Developer mode → **Load
unpacked** → select `extension/dist`. Set the backend URL in the extension
popup if it isn't running on the default `http://localhost:8001`. Open a
Google Meet call, click the SignEase Bridge panel, grant camera permission
via the permission tab it opens, and hit Start.

## Verification

- Backend accuracy numbers above are computed on a real held-out test split
  by `train_letters.py` / `train_words.py` — rerun them to reproduce.
- `camera-test/` was used to verify the camera → MediaPipe → prediction
  pipeline in isolation before debugging any extension-specific behavior
  (offscreen document lifecycle, message races, permission prompting).
- Live-tested end-to-end in a real Google Meet call for both letter
  spelling and word signing.

## Roadmap

- Add MediaPipe PoseLandmarker to the client capture pipeline to close the
  pose-accuracy gap on the word model.
- In-call virtual camera/mic injection (`replaceTrack()` on the live
  `RTCPeerConnection`) so the *other* call participant also hears/sees the
  translation automatically, without looking at the extension's own panel.
- Dedicated hardware / native mobile app — future phases, not started.
