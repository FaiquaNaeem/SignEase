/** In-panel diagnostic log so failures can be read straight off a screenshot
 * of the extension panel, without needing DevTools navigation. */

export interface DebugEntry {
  time: string;
  label: string;
  ok: boolean;
  detail: string;
  durationMs: number;
}

const entries: DebugEntry[] = [];
const listeners = new Set<(entries: DebugEntry[]) => void>();
const MAX_ENTRIES = 15;

export function logDebug(entry: Omit<DebugEntry, "time">) {
  entries.unshift({ ...entry, time: new Date().toLocaleTimeString() });
  if (entries.length > MAX_ENTRIES) entries.length = MAX_ENTRIES;
  listeners.forEach((l) => l([...entries]));
}

export function subscribeDebugLog(listener: (entries: DebugEntry[]) => void): () => void {
  listeners.add(listener);
  listener([...entries]);
  return () => listeners.delete(listener);
}
