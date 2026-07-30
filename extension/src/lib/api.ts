import type { Landmark, PredictResponse, SignLanguage, SignReferences, WordFrame } from "../types";

const DEFAULT_BASE_URL = "http://localhost:8001";

async function getBaseUrl(): Promise<string> {
  // chrome.storage is unreliable from inside an offscreen document (this
  // module runs there for hand-tracking predictions, not just in the
  // background service worker) — reads have been observed to throw
  // "Cannot read properties of undefined (reading 'sync')" there even
  // though the "storage" permission is granted and the same call works
  // fine from the service worker. Fall back to the default rather than
  // letting every prediction call crash on it.
  try {
    const stored = await chrome.storage.sync.get("backendBaseUrl");
    return (stored.backendBaseUrl as string | undefined) || DEFAULT_BASE_URL;
  } catch {
    return DEFAULT_BASE_URL;
  }
}

/** Diagnostic used by the panel's connection-status check. A plain fetch()
 * failure (TypeError) only ever surfaces the generic message "Failed to
 * fetch" to JS — by browser design, to avoid leaking network details to
 * scripts — so this reports the exact URL it tried, which is the one piece
 * of context that actually helps distinguish "wrong port" from "server
 * down" from "blocked by something else" without opening DevTools. */
export async function checkHealth(): Promise<{ reachable: boolean; url: string; detail: string }> {
  const base = await getBaseUrl();
  const url = `${base}/api/health`;
  try {
    const res = await fetch(url);
    if (!res.ok) return { reachable: false, url, detail: `HTTP ${res.status}` };
    const data = await res.json();
    return { reachable: true, url, detail: JSON.stringify(data) };
  } catch (err) {
    const detail = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    return { reachable: false, url, detail };
  }
}

export async function predictLetter(landmarks: Landmark[]): Promise<PredictResponse> {
  const base = await getBaseUrl();
  const res = await fetch(`${base}/api/predict/letter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ landmarks }),
  });
  if (!res.ok) throw new Error(`predictLetter failed: ${res.status}`);
  return res.json();
}

export async function predictWord(frames: WordFrame[]): Promise<PredictResponse> {
  const base = await getBaseUrl();
  const res = await fetch(`${base}/api/predict/word`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames }),
  });
  if (!res.ok) throw new Error(`predictWord failed: ${res.status}`);
  return res.json();
}

/** Returns raw WAV bytes. Callers in a page/content-script context should go
 * through apiRelay.ts instead — object URLs created in the background
 * service worker aren't resolvable from a content script's document. */
export async function speak(text: string, language: SignLanguage): Promise<ArrayBuffer> {
  const base = await getBaseUrl();
  const res = await fetch(`${base}/api/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language }),
  });
  if (!res.ok) throw new Error(`speak failed: ${res.status}`);
  return res.arrayBuffer();
}

export async function transcribe(
  audioBlob: Blob,
  languageCode = "unknown"
): Promise<{ transcript: string; language_code?: string; provider: string }> {
  const base = await getBaseUrl();
  const form = new FormData();
  form.append("file", audioBlob, "chunk.webm");
  const res = await fetch(`${base}/api/transcribe?language_code=${languageCode}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`transcribe failed: ${res.status}`);
  return res.json();
}

export async function fetchSignReferences(): Promise<SignReferences> {
  const base = await getBaseUrl();
  const res = await fetch(`${base}/api/sign-references`);
  if (!res.ok) throw new Error(`sign-references failed: ${res.status}`);
  return res.json();
}
