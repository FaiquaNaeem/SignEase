export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface Alternative {
  label: string;
  confidence: number;
}

export interface PredictResponse {
  label: string;
  confidence: number;
  alternatives: Alternative[];
  inference_ms: number;
}

export interface WordFrame {
  leftHand: Landmark[] | null;
  rightHand: Landmark[] | null;
  pose: Landmark[] | null;
}

export type SignLanguage = "en-IN" | "hi-IN";

export interface SignReferenceStatic {
  type: "static";
  hand: [number, number, number][];
}

export interface SignReferenceSequence {
  type: "sequence";
  frames: {
    leftHand: [number, number, number][];
    rightHand: [number, number, number][];
    pose: [number, number, number][];
  }[];
}

export interface SignReferences {
  letters: Record<string, SignReferenceStatic>;
  words: Record<string, SignReferenceSequence>;
}

export type HandTrackingMode = "letter" | "word";

// Runtime messages passed between content script / background / offscreen doc.
export type ExtensionMessage =
  | { type: "START_TAB_TRANSCRIPTION"; tabId: number }
  | { type: "STOP_TAB_TRANSCRIPTION" }
  | { type: "TAB_STREAM_ID"; streamId: string }
  | { type: "TRANSCRIPT_RESULT"; transcript: string; language_code?: string }
  | { type: "TRANSCRIPTION_ERROR"; message: string }
  // Sign->speech hand tracking must run inside the offscreen document, not
  // the content script: MediaPipe's WASM loader injects a <script> tag and
  // reads back a global it sets, but a content script's DOM-injected script
  // tags execute in the host page's MAIN world while the content script
  // itself runs in the ISOLATED world — two separate JS global scopes that
  // share a DOM but not variables, so the global never becomes visible to
  // our code ("ModuleFactory not set", reproduced every time regardless of
  // loader variant). An offscreen document is a normal single-world page,
  // so the same technique just works there.
  | { type: "START_HAND_TRACKING"; mode: HandTrackingMode; language: SignLanguage }
  | { type: "STOP_HAND_TRACKING" }
  | { type: "SET_HAND_TRACKING_MODE"; mode: HandTrackingMode }
  | { type: "HAND_TRACKING_CAPTION"; text: string; confidence: number }
  | { type: "HAND_TRACKING_ERROR"; message: string }
  | { type: "HAND_TRACKING_DEBUG"; entry: { label: string; ok: boolean; detail: string; durationMs: number } };

// Content scripts run inside the host page's (e.g. meet.google.com) CSP,
// which blocks fetch() to arbitrary hosts like our localhost backend
// (Chrome enforces page CSP on content-script network requests). The
// background service worker isn't subject to that CSP, so content-script
// code relays backend calls through it via these messages instead of
// fetching directly. See lib/apiRelay.ts (sender) and
// background/service-worker.ts (handler, using lib/api.ts's direct fetches).
export type BackendRequest =
  | { type: "BACKEND_HEALTH" }
  | { type: "BACKEND_PREDICT_LETTER"; landmarks: Landmark[] }
  | { type: "BACKEND_PREDICT_WORD"; frames: WordFrame[] }
  | { type: "BACKEND_SPEAK"; text: string; language: SignLanguage }
  | { type: "BACKEND_TRANSCRIBE"; audioBase64: string; languageCode: string };

export type BackendResponse<T = unknown> = { ok: true; data: T } | { ok: false; error: string };
