import * as backend from "../lib/api";
import { bytesToBase64, base64ToBytes } from "../lib/base64";
import type { BackendRequest, BackendResponse, ExtensionMessage } from "../types";

const OFFSCREEN_PATH = "src/offscreen/offscreen.html";
let activeTranscriptionTabId: number | null = null;
let activeHandTrackingTabId: number | null = null;

// A held Port keeps this service worker alive for as long as the content
// script wants (SignToSpeechSession opens one for the duration of a live
// session) — the standard MV3 workaround for the 30s idle-suspend timer
// that otherwise causes intermittent "Failed to fetch" mid-session.
chrome.runtime.onConnect.addListener((port) => {
  port.onDisconnect.addListener(() => {
    /* no-op: connecting/disconnecting is the whole mechanism */
  });
});

async function ensureOffscreenDocument() {
  const existing = await chrome.runtime.getContexts?.({
    contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
  });
  if (existing && existing.length > 0) return;

  await chrome.offscreen.createDocument({
    url: OFFSCREEN_PATH,
    reasons: [chrome.offscreen.Reason.USER_MEDIA],
    justification:
      "Capture the video-call tab's audio for speech-to-sign transcription, and run camera-based hand tracking for sign-to-speech (must run in a normal single-world page — see START_HAND_TRACKING in types.ts).",
  });
  // createDocument() resolving doesn't guarantee the page's own script has
  // started running yet, and a message broadcast right after can still (in
  // principle) beat its onMessage listener registration — a small, one-time
  // grace period on first creation is cheap insurance against that residual
  // race, on top of the lazy-import fix in offscreen.ts that removes the
  // much bigger delay (the ~150KB MediaPipe bundle loading before anything
  // else in that file could run).
  await new Promise((resolve) => setTimeout(resolve, 150));
}

chrome.runtime.onMessage.addListener((message: ExtensionMessage | BackendRequest, sender, sendResponse) => {
  if (message.type.startsWith("BACKEND_")) {
    handleBackendRequest(message as BackendRequest).then(sendResponse);
    return true;
  }
  handleMessage(message as ExtensionMessage, sender).then(sendResponse);
  return true; // keep the message channel open for the async response
});

/** Runs the actual backend fetch (lib/api.ts) here, in the background
 * service worker, which — unlike a content script injected into
 * meet.google.com — isn't subject to the host page's CSP. See the
 * BackendRequest doc comment in types.ts. */
async function handleBackendRequest(message: BackendRequest): Promise<BackendResponse> {
  try {
    switch (message.type) {
      case "BACKEND_HEALTH":
        return { ok: true, data: await backend.checkHealth() };
      case "BACKEND_PREDICT_LETTER":
        return { ok: true, data: await backend.predictLetter(message.landmarks) };
      case "BACKEND_PREDICT_WORD":
        return { ok: true, data: await backend.predictWord(message.frames) };
      case "BACKEND_SPEAK": {
        const buf = await backend.speak(message.text, message.language);
        return { ok: true, data: bytesToBase64(new Uint8Array(buf)) };
      }
      case "BACKEND_TRANSCRIBE": {
        const bytes = base64ToBytes(message.audioBase64);
        const blob = new Blob([bytes as unknown as BlobPart], { type: "audio/webm" });
        return { ok: true, data: await backend.transcribe(blob, message.languageCode) };
      }
    }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

async function handleMessage(message: ExtensionMessage, sender: chrome.runtime.MessageSender) {
  switch (message.type) {
    case "START_TAB_TRANSCRIPTION": {
      // Content scripts don't know their own tabId; sender.tab.id (set by
      // Chrome for messages from a content script) is the reliable source,
      // with message.tabId only as a fallback for non-content-script callers.
      const tabId = sender.tab?.id ?? message.tabId;
      activeTranscriptionTabId = tabId;
      await ensureOffscreenDocument();
      // getMediaStreamId must be called from the background/service worker;
      // the resulting streamId is then consumed inside the offscreen
      // document via getUserMedia's chromeMediaSource constraint (regular
      // page/content-script contexts cannot use that constraint directly).
      const streamId = await new Promise<string>((resolve, reject) => {
        chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, (id) => {
          if (chrome.runtime.lastError || !id) {
            reject(new Error(chrome.runtime.lastError?.message ?? "getMediaStreamId failed"));
          } else {
            resolve(id);
          }
        });
      });
      chrome.runtime.sendMessage({ type: "TAB_STREAM_ID", streamId } satisfies ExtensionMessage);
      return { ok: true };
    }
    case "STOP_TAB_TRANSCRIPTION": {
      activeTranscriptionTabId = null;
      chrome.runtime.sendMessage(message);
      return { ok: true };
    }
    case "TRANSCRIPT_RESULT":
    case "TRANSCRIPTION_ERROR": {
      // Relayed from the offscreen document; forward to the content script
      // in the tab that started transcription (offscreen docs can't reach
      // content scripts directly).
      if (activeTranscriptionTabId !== null) {
        chrome.tabs.sendMessage(activeTranscriptionTabId, message);
      }
      return { ok: true };
    }
    case "START_HAND_TRACKING": {
      activeHandTrackingTabId = sender.tab?.id ?? null;
      await ensureOffscreenDocument();
      chrome.runtime.sendMessage(message);
      return { ok: true };
    }
    case "STOP_HAND_TRACKING":
    case "SET_HAND_TRACKING_MODE": {
      // Forwarded straight through to the offscreen document, which is the
      // only thing listening for these besides the sender itself.
      chrome.runtime.sendMessage(message);
      return { ok: true };
    }
    case "HAND_TRACKING_CAPTION":
    case "HAND_TRACKING_ERROR":
    case "HAND_TRACKING_DEBUG": {
      // Relayed from the offscreen document; forward to the content script
      // that started hand tracking.
      if (activeHandTrackingTabId !== null) {
        chrome.tabs.sendMessage(activeHandTrackingTabId, message);
      }
      return { ok: true };
    }
    default:
      return { ok: false, error: `Unhandled message from ${sender.id}` };
  }
}
