import { HandLandmarkTracker } from "../lib/mediapipeHands";
import { predictLetter, predictWord, speak } from "../lib/api";
import type { ExtensionMessage, HandTrackingMode, Landmark, SignLanguage, WordFrame } from "../types";

// Runs entirely inside the offscreen document — see the long comment on
// START_HAND_TRACKING in types.ts for why this can't live in the content
// script. Offscreen documents are also not subject to the host page's CSP,
// so this calls the backend directly (lib/api.ts) rather than through the
// background-relay lib/apiRelay.ts that content-script code needs.

const LETTER_PREDICT_INTERVAL_MS = 700;
const MIN_CONFIDENCE = 0.55;
const HAND_TRACKING_FPS = 12;
const HAND_TRACKING_INTERVAL_MS = 1000 / HAND_TRACKING_FPS;
const DIAGNOSTIC_INTERVAL_MS = 2000;

// TTS engines mumble/skip bare single characters ("V" often comes out as
// near-silence or a stray consonant sound) — speaking the letter's name
// instead ("Vee") is what fingerspelling apps do and is what's actually
// intelligible out loud. Only affects what's spoken, not the on-screen
// caption, which still shows the plain letter.
const LETTER_NAMES: Record<string, string> = {
  A: "Ay", B: "Bee", C: "See", D: "Dee", E: "Ee", F: "Eff", G: "Jee",
  H: "Aitch", I: "Eye", J: "Jay", K: "Kay", L: "El", M: "Em", N: "En",
  O: "Oh", P: "Pee", Q: "Cue", R: "Ar", S: "Ess", T: "Tee", U: "You",
  V: "Vee", W: "Double-you", X: "Ex", Y: "Why", Z: "Zee",
};

let video: HTMLVideoElement | null = null;
let tracker: HandLandmarkTracker | null = null;
let stream: MediaStream | null = null;
let intervalId: ReturnType<typeof setInterval> | null = null;
let mode: HandTrackingMode = "letter";
let language: SignLanguage = "en-IN";
let lastLetterPredictAt = 0;
let lastDiagnosticAt = 0;
let wordBuffer: WordFrame[] = [];
let wordCapturing = false;
let audioEl: HTMLAudioElement | null = null;
let lastSpokenLetter: string | null = null;

function report(message: ExtensionMessage) {
  chrome.runtime.sendMessage(message);
}

function debug(label: string, ok: boolean, detail: string, durationMs: number) {
  report({ type: "HAND_TRACKING_DEBUG", entry: { label, ok, detail, durationMs } });
}

export async function startHandTracking(newMode: HandTrackingMode, newLanguage: SignLanguage) {
  stopHandTracking();
  mode = newMode;
  language = newLanguage;

  video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  audioEl = document.createElement("audio");
  document.body.appendChild(audioEl);

  tracker = new HandLandmarkTracker();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  } catch (err) {
    report({ type: "HAND_TRACKING_ERROR", message: err instanceof Error ? err.message : String(err) });
    return;
  }
  video.srcObject = stream;
  await video.play();
  // requestAnimationFrame does NOT reliably fire in an offscreen document —
  // those pages are never painted/composited (they're never visible), and
  // Chrome throttles/stalls rAF callbacks for anything that isn't actually
  // rendering to screen. That was the root cause of the live extension's
  // hand-tracking loop freezing after a single frame (framesAttempted stuck
  // at 1 forever). setInterval has no such dependency on painting.
  intervalId = setInterval(loop, HAND_TRACKING_INTERVAL_MS);
}

export function stopHandTracking() {
  if (intervalId !== null) clearInterval(intervalId);
  intervalId = null;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  tracker?.close();
  tracker = null;
  video = null;
  audioEl?.remove();
  audioEl = null;
  wordBuffer = [];
  wordCapturing = false;
  lastSpokenLetter = null;
}

export function setHandTrackingMode(newMode: HandTrackingMode) {
  mode = newMode;
  wordBuffer = [];
  lastSpokenLetter = null;
}

export function setWordCapturing(capturing: boolean) {
  wordCapturing = capturing;
  if (capturing) {
    wordBuffer = [];
  } else {
    void finishWordCapture();
  }
}

function loop() {
  if (!video || !tracker) return;
  const now = performance.now();

  tracker
    .processFrame(video)
    .then(() => {
      // Only push once a fresh detection actually landed — pushing on
      // every tick instead of once per detection would flood the buffer
      // with duplicate frames carrying no new temporal information.
      if (!wordCapturing || !tracker) return;
      const { left, right } = tracker.getLatest();
      if (left || right) wordBuffer.push({ leftHand: left, rightHand: right, pose: null });
    })
    .catch((err) => report({ type: "HAND_TRACKING_ERROR", message: err instanceof Error ? err.message : String(err) }));

  if (mode === "letter" && now - lastLetterPredictAt > LETTER_PREDICT_INTERVAL_MS) {
    lastLetterPredictAt = now;
    void predictCurrentLetter();
  }

  if (now - lastDiagnosticAt > DIAGNOSTIC_INTERVAL_MS) {
    lastDiagnosticAt = now;
    const stats = tracker.getStats();
    debug(
      "MEDIAPIPE_STATS",
      stats.handsDetectedCount > 0,
      `video ${video.videoWidth}x${video.videoHeight} readyState=${video.readyState} | attempted=${stats.framesAttempted} notReady=${stats.framesNotReady} sent=${stats.framesSent} results=${stats.resultsReceived} handsSeen=${stats.handsDetectedCount} sendErr=${stats.lastSendError ?? "none"}`,
      0
    );
  }
}

async function predictCurrentLetter() {
  if (!tracker) return;
  const { left, right } = tracker.getLatest();
  const hand: Landmark[] | null = right ?? left;
  if (!hand) {
    // Hand dropped out of frame — next time any letter is shown (even the
    // same one as before), treat it as a fresh sign worth announcing again.
    lastSpokenLetter = null;
    return;
  }
  const t0 = performance.now();
  try {
    const result = await predictLetter(hand);
    debug("PREDICT_LETTER", true, `${result.label} ${(result.confidence * 100).toFixed(0)}%`, performance.now() - t0);
    // Only announce on change — otherwise holding one letter steady re-fires
    // the caption + TTS every ~700ms (the "vvvvvvv" repeat) since the model
    // keeps re-predicting the same still-held sign.
    if (result.confidence >= MIN_CONFIDENCE && result.label !== lastSpokenLetter) {
      lastSpokenLetter = result.label;
      report({ type: "HAND_TRACKING_CAPTION", text: result.label, confidence: result.confidence });
      await playSpeech(LETTER_NAMES[result.label.toUpperCase()] ?? result.label);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    debug("PREDICT_LETTER", false, msg, performance.now() - t0);
    report({ type: "HAND_TRACKING_ERROR", message: msg });
  }
}

async function finishWordCapture() {
  if (wordBuffer.length < 3) return;
  const frames = wordBuffer;
  wordBuffer = [];
  const t0 = performance.now();
  try {
    const result = await predictWord(frames);
    debug("PREDICT_WORD", true, `${result.label} ${(result.confidence * 100).toFixed(0)}%`, performance.now() - t0);
    if (result.confidence >= MIN_CONFIDENCE) {
      report({ type: "HAND_TRACKING_CAPTION", text: result.label, confidence: result.confidence });
      await playSpeech(result.label);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    debug("PREDICT_WORD", false, msg, performance.now() - t0);
    report({ type: "HAND_TRACKING_ERROR", message: msg });
  }
}

async function playSpeech(text: string) {
  if (!audioEl) return;
  try {
    const buf = await speak(text, language);
    const blob = new Blob([buf], { type: "audio/wav" });
    audioEl.src = URL.createObjectURL(blob);
    await audioEl.play();
  } catch {
    // Sarvam/Piper both unreachable — caption still shown; audio is a bonus.
  }
}
