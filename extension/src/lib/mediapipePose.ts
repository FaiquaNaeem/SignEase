import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { Landmark } from "../types";

/**
 * Upper-body pose tracking via MediaPipe's PoseLandmarker (Tasks API) —
 * companion to HandLandmarkTracker. The word model was trained on hand
 * *and* pose landmarks (backend/training/prepare_words.py), but the
 * extension originally only captured hands, silently sending zeroed-out
 * pose at inference. Measured offline: that mismatch alone drops the
 * 250-class word model's real accuracy from ~72% to ~41% (and specific
 * signs much further — "hello" 69.5% -> 22%). This closes that gap.
 *
 * Only extracts the 8 points backend/training/prepare_words.py's
 * POSE_KEEP_IDX keeps — left/right shoulder, elbow, wrist, hip (BlazePose
 * topology indices 11,12,13,14,15,16,23,24) — in that exact order, since
 * the backend's normalize_pose_frame assumes index 0/1 are the shoulders.
 *
 * NOT currently wired into handTracking.ts's live loop — running this
 * alongside HandLandmarkTracker on every tick, even sampled at a reduced
 * rate, wasn't stable in a real video call: the offscreen document kept
 * getting killed under memory/CPU pressure and silently recreated,
 * dropping tracking mid-session. This class itself works and was verified
 * end-to-end against the live backend before that was found. To revisit:
 * re-fetch the model file (deleted to avoid bloating the built extension
 * with an unused 5.7MB asset) —
 *   curl -sL -o extension/public/pose_landmarker.task \
 *     "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
 * — add it back to manifest.config.ts's web_accessible_resources, and
 * likely needs either a lower capture resolution or moving one of the two
 * models to a Web Worker before it's safe to run continuously.
 */
const POSE_KEEP_IDX = [11, 12, 13, 14, 15, 16, 23, 24];

export class PoseLandmarkTracker {
  private landmarker: PoseLandmarker | null = null;
  private latest: Landmark[] | null = null;

  framesAttempted = 0;
  framesSent = 0;
  resultsReceived = 0;
  poseDetectedCount = 0;
  lastSendError: string | null = null;
  initError: string | null = null;
  initDone = false;
  private closed = false;

  constructor() {
    void this.init();
  }

  private async init() {
    try {
      const fileset = await FilesetResolver.forVisionTasks(chrome.runtime.getURL("mediapipe-wasm"));
      const landmarker = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: chrome.runtime.getURL("pose_landmarker.task"),
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
      });
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

  async processFrame(video: HTMLVideoElement) {
    this.framesAttempted++;
    if (video.readyState < 2) return;
    if (!this.initDone) {
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

      const pose = result.landmarks[0];
      if (pose && pose.length >= 33) {
        this.latest = POSE_KEEP_IDX.map((i) => ({ x: pose[i].x, y: pose[i].y, z: pose[i].z }));
        this.poseDetectedCount++;
      } else {
        this.latest = null;
      }
      this.lastSendError = null;
    } catch (err) {
      this.lastSendError = err instanceof Error ? err.message : String(err);
    }
  }

  getStats() {
    return {
      framesAttempted: this.framesAttempted,
      framesSent: this.framesSent,
      resultsReceived: this.resultsReceived,
      poseDetectedCount: this.poseDetectedCount,
      lastSendError: this.lastSendError,
    };
  }

  getLatest(): Landmark[] | null {
    return this.latest;
  }

  close() {
    this.closed = true;
    this.landmarker?.close();
  }
}
