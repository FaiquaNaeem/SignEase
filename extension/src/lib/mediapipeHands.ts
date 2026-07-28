import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import type { Landmark } from "../types";

/**
 * Hand-landmark tracking via MediaPipe's Tasks API (HandLandmarker), not the
 * legacy @mediapipe/hands package. The legacy package's callback-based
 * Hands.send() broke after the first frame in this bundled extension
 * context (a minified "X is not a function" from its internal global-
 * namespace init code) — a known deprecation issue; the Tasks API is
 * Google's maintained replacement and exposes a synchronous
 * detectForVideo() instead, which sidesteps that whole class of bug.
 */
export class HandLandmarkTracker {
  private landmarker: HandLandmarker | null = null;
  private latest: { left: Landmark[] | null; right: Landmark[] | null } = {
    left: null,
    right: null,
  };

  // Diagnostics: lets callers tell "model never finished loading" apart from
  // "frames are sent but no hand is ever detected" apart from "detectForVideo
  // throws" — different bugs that all look identical from the outside
  // (getLatest() staying {left:null,right:null} forever).
  framesAttempted = 0;
  framesNotReady = 0;
  framesSent = 0;
  resultsReceived = 0;
  handsDetectedCount = 0;
  lastSendError: string | null = null;
  initError: string | null = null;
  initDone = false;
  private closed = false;

  constructor() {
    void this.init();
  }

  private async init() {
    try {
      // GPU delegate needs a WebGL context; that init hung indefinitely in
      // the old content-script location rather than failing loudly. CPU
      // delegate has no such dependency — slightly slower per frame, but
      // reliable, and plenty fast at 640x480.
      //
      // useModule defaults to false (classic-script loader,
      // vision_wasm_internal.js) deliberately: this code now runs inside
      // the offscreen document, a normal single-JS-world page, where the
      // loader's <script>-tag-injection + global-variable technique works
      // as designed. useModule=true was a wrong diagnosis from when this
      // ran in a content script's isolated world — and it's also just
      // broken in this package version regardless of world: the loader
      // always injects a classic (non type="module") <script> tag even
      // when told to use the ES-module variant, which throws "Cannot use
      // 'import.meta' outside a module" immediately.
      const fileset = await FilesetResolver.forVisionTasks(chrome.runtime.getURL("mediapipe-wasm"));
      const landmarker = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: chrome.runtime.getURL("hand_landmarker.task"),
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.5,
      });
      // close() can be called (Stop clicked) while this is still in flight —
      // model loading takes several seconds. Finishing init after that point
      // would otherwise leak a landmarker instance nothing ever closes.
      if (this.closed) {
        landmarker.close();
      } else {
        this.landmarker = landmarker;
      }
      this.initDone = true;
    } catch (err) {
      this.initError = err instanceof Error ? err.message : String(err);
      this.initDone = true;
    }
  }

  subscribe(_callback: (result: { left: Landmark[] | null; right: Landmark[] | null }) => void) {
    // Kept for API compatibility with the previous callback-based tracker;
    // detectForVideo is synchronous so callers just read getLatest() instead.
  }

  async processFrame(video: HTMLVideoElement) {
    this.framesAttempted++;
    if (video.readyState < 2) {
      this.framesNotReady++;
      return;
    }
    if (!this.initDone) {
      // Don't block forever if init() itself hangs — surface it via stats
      // instead of processFrame silently never progressing (what happened
      // with the GPU delegate before this fix).
      this.lastSendError = "model still initializing";
      return;
    }
    if (this.initError) {
      this.lastSendError = `init failed: ${this.initError}`;
      return;
    }
    if (!this.landmarker) return;

    this.framesSent++;
    try {
      const result = this.landmarker.detectForVideo(video, performance.now());
      this.resultsReceived++;

      let left: Landmark[] | null = null;
      let right: Landmark[] | null = null;
      result.landmarks.forEach((landmarks, i) => {
        const label = result.handedness[i]?.[0]?.categoryName; // "Left" | "Right"
        const converted = landmarks.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z }));
        // No swap: use MediaPipe's raw Left/Right labels directly — the
        // training data's left_hand/right_hand columns came from
        // MediaPipe's own output too, so matching it directly is what
        // agrees with what the model was trained on. Swapping here (the
        // previous code, based on a wrong assumption about correcting for
        // a mirror convention) was confirmed to crash the word model's
        // real held-out test accuracy from 80.94% to 5.83% when
        // reproduced offline — this is what was causing the "always
        // predicts nearly the same wrong word" behavior live.
        if (label === "Left") left = converted;
        else if (label === "Right") right = converted;
      });
      if (left || right) this.handsDetectedCount++;
      this.latest = { left, right };
    } catch (err) {
      this.lastSendError = err instanceof Error ? err.message : String(err);
    }
  }

  getStats() {
    return {
      framesAttempted: this.framesAttempted,
      framesNotReady: this.framesNotReady,
      framesSkippedBusy: 0,
      framesSent: this.framesSent,
      resultsReceived: this.resultsReceived,
      handsDetectedCount: this.handsDetectedCount,
      lastSendError: this.lastSendError,
    };
  }

  getLatest() {
    return this.latest;
  }

  close() {
    this.closed = true;
    this.landmarker?.close();
  }
}
