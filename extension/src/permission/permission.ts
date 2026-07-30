// Requesting getUserMedia directly in the extension's action popup is
// unreliable: showing the native permission prompt can make the popup lose
// focus, and Chrome auto-closes popups on blur — which cancels the pending
// request before the user can respond. A normal, persistent tab (opened via
// chrome.tabs.create) doesn't have that problem, and the resulting grant is
// tied to the extension's origin, so it's reused later by getUserMedia calls
// from the offscreen document too (see popup/Popup.tsx and
// background/service-worker.ts for why hand tracking runs there).

const statusEl = document.getElementById("status")!;
const buttonEl = document.getElementById("grant")!;

buttonEl.addEventListener("click", async () => {
  statusEl.textContent = "Requesting…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    stream.getTracks().forEach((t) => t.stop());
    statusEl.textContent = "Granted — you can close this tab.";
    statusEl.style.color = "#16a34a";
  } catch (err) {
    statusEl.textContent = `Denied: ${err instanceof Error ? err.message : String(err)}`;
    statusEl.style.color = "#dc2626";
  }
});
