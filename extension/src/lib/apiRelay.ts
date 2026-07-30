import type {
  BackendRequest,
  BackendResponse,
  Landmark,
  PredictResponse,
  SignLanguage,
  SignReferences,
  WordFrame,
} from "../types";
import { base64ToBytes, bytesToBase64 } from "./base64";
import { logDebug } from "./debugLog";

/** Backend-call client for code running in a content script (injected into
 * meet.google.com/zoom.us/teams.microsoft.com). Routes every call through
 * the background service worker instead of fetching directly — see the
 * BackendRequest doc comment in types.ts for why. */

function sendOnce<T>(message: BackendRequest): Promise<T> {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response: BackendResponse<T> | undefined) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      if (!response || !response.ok) {
        reject(new Error(response?.error ?? "Background relay returned no response"));
        return;
      }
      resolve(response.data);
    });
  });
}

/** MV3 background service workers can be torn down after ~30s idle and get
 * woken back up per-message, but there's a real-world flakiness window
 * right at wake-up where an in-flight fetch can fail with the browser's
 * generic "Failed to fetch" — observed during sustained live use (repeated
 * predict calls every ~700ms) even though the exact same request always
 * succeeds standalone. One transparent retry papers over that transient
 * case without hiding a genuinely dead backend (which fails the retry too). */
async function send<T>(message: BackendRequest): Promise<T> {
  const t0 = performance.now();
  const hasBackgroundPage = typeof chrome !== "undefined" && !!chrome.runtime?.id;
  try {
    const data = await sendOnce<T>(message);
    logDebug({ label: message.type, ok: true, detail: `runtime.id=${hasBackgroundPage}`, durationMs: performance.now() - t0 });
    return data;
  } catch (err) {
    const firstError = err instanceof Error ? err.message : String(err);
    if (err instanceof Error && /failed to fetch/i.test(err.message)) {
      await new Promise((r) => setTimeout(r, 250));
      try {
        const data = await sendOnce<T>(message);
        logDebug({
          label: message.type,
          ok: true,
          detail: `retry succeeded after: ${firstError}`,
          durationMs: performance.now() - t0,
        });
        return data;
      } catch (retryErr) {
        const retryMsg = retryErr instanceof Error ? retryErr.message : String(retryErr);
        logDebug({
          label: message.type,
          ok: false,
          detail: `1st: ${firstError} | retry: ${retryMsg} | runtime.id=${hasBackgroundPage}`,
          durationMs: performance.now() - t0,
        });
        throw retryErr;
      }
    }
    logDebug({ label: message.type, ok: false, detail: `${firstError} | runtime.id=${hasBackgroundPage}`, durationMs: performance.now() - t0 });
    throw err;
  }
}

export async function checkHealth(): Promise<{ reachable: boolean; url: string; detail: string }> {
  return send({ type: "BACKEND_HEALTH" });
}

export async function predictLetter(landmarks: Landmark[]): Promise<PredictResponse> {
  return send({ type: "BACKEND_PREDICT_LETTER", landmarks });
}

export async function predictWord(frames: WordFrame[]): Promise<PredictResponse> {
  return send({ type: "BACKEND_PREDICT_WORD", frames });
}

/** Returns an object URL for the synthesized speech; caller must revoke it. */
export async function speak(text: string, language: SignLanguage): Promise<string> {
  const base64 = await send<string>({ type: "BACKEND_SPEAK", text, language });
  const bytes = base64ToBytes(base64);
  // TS's DOM lib types Uint8Array's .buffer as ArrayBufferLike (which
  // includes SharedArrayBuffer) rather than BlobPart's stricter ArrayBuffer
  // — a known lib.dom typing quirk, not a real runtime concern here.
  const blob = new Blob([bytes as unknown as BlobPart], { type: "audio/wav" });
  return URL.createObjectURL(blob);
}

export async function transcribe(
  audioBlob: Blob,
  languageCode = "unknown"
): Promise<{ transcript: string; language_code?: string; provider: string }> {
  const buf = await audioBlob.arrayBuffer();
  const audioBase64 = bytesToBase64(new Uint8Array(buf));
  return send({ type: "BACKEND_TRANSCRIBE", audioBase64, languageCode });
}

/** Sign references are static data generated once at training time — bundled
 * directly into the extension package (web_accessible_resources) instead of
 * fetched from the backend, so this works even if the backend is down and
 * never touches the message relay (that 1MB+ payload was too slow/large to
 * push through chrome.runtime.sendMessage reliably). */
export async function fetchSignReferences(): Promise<SignReferences> {
  const url = chrome.runtime.getURL("sign_references.json");
  const t0 = performance.now();
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    logDebug({ label: "SIGN_REFERENCES", ok: true, detail: url, durationMs: performance.now() - t0 });
    return data;
  } catch (err) {
    const detail = `${url} — ${err instanceof Error ? err.message : String(err)}`;
    logDebug({ label: "SIGN_REFERENCES", ok: false, detail, durationMs: performance.now() - t0 });
    throw new Error(detail);
  }
}
