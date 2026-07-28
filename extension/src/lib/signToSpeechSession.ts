import { logDebug } from "./debugLog";
import type { ExtensionMessage, HandTrackingMode, SignLanguage } from "../types";

export type { HandTrackingMode as SignToSpeechMode };

export interface SignToSpeechCallbacks {
  onCaption: (text: string, confidence: number) => void;
  onError: (message: string) => void;
}

/**
 * Content-script-side handle for hand tracking. The actual camera capture
 * and MediaPipe processing run in the offscreen document (see
 * offscreen/handTracking.ts) — a content script can't host MediaPipe's
 * WASM loader reliably (it injects a <script> tag that runs in the host
 * page's MAIN world, while the content script itself runs in the ISOLATED
 * world; the global it sets is never visible back to isolated-world code).
 * This class just sends control messages and listens for results.
 */
export class SignToSpeechSession {
  private listener = (message: ExtensionMessage) => {
    if (message.type === "HAND_TRACKING_CAPTION") {
      this.callbacks.onCaption(message.text, message.confidence);
    } else if (message.type === "HAND_TRACKING_ERROR") {
      this.callbacks.onError(message.message);
    } else if (message.type === "HAND_TRACKING_DEBUG") {
      logDebug(message.entry);
    }
  };

  constructor(private mode: HandTrackingMode, private callbacks: SignToSpeechCallbacks, private language: SignLanguage) {}

  async start() {
    chrome.runtime.onMessage.addListener(this.listener);
    chrome.runtime.sendMessage({
      type: "START_HAND_TRACKING",
      mode: this.mode,
      language: this.language,
    } satisfies ExtensionMessage);
  }

  stop() {
    chrome.runtime.sendMessage({ type: "STOP_HAND_TRACKING" } satisfies ExtensionMessage);
    chrome.runtime.onMessage.removeListener(this.listener);
  }
}
