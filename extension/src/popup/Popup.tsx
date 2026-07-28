import { useEffect, useState } from "react";

export function Popup() {
  const [backendUrl, setBackendUrl] = useState("http://localhost:8001");
  const [status, setStatus] = useState<"unknown" | "ok" | "error">("unknown");

  useEffect(() => {
    chrome.storage.sync.get("backendBaseUrl").then((stored) => {
      if (stored.backendBaseUrl) setBackendUrl(stored.backendBaseUrl as string);
    });
  }, []);

  const save = async () => {
    await chrome.storage.sync.set({ backendBaseUrl: backendUrl });
    checkHealth();
  };

  const checkHealth = async () => {
    try {
      const res = await fetch(`${backendUrl}/api/health`);
      setStatus(res.ok ? "ok" : "error");
    } catch {
      setStatus("error");
    }
  };

  const openCameraPermissionTab = () => {
    // Requesting getUserMedia directly here (in the action popup) is
    // unreliable — showing the native prompt can blur the popup, and Chrome
    // auto-closes popups on blur, canceling the pending request. A real tab
    // doesn't have that problem, and the resulting grant is reused by the
    // offscreen document's later getUserMedia calls (see
    // START_HAND_TRACKING in types.ts for why hand tracking runs there).
    chrome.tabs.create({ url: chrome.runtime.getURL("src/permission/permission.html") });
  };

  return (
    <div style={{ padding: 14 }}>
      <strong>SignEase Bridge</strong>
      <p style={{ fontSize: 12, opacity: 0.8 }}>
        Open a Google Meet / Zoom / Teams call to see the live translation panel.
      </p>
      <label style={{ fontSize: 11, display: "block", marginTop: 8 }}>Backend URL</label>
      <input
        style={{ width: "100%", boxSizing: "border-box" }}
        value={backendUrl}
        onChange={(e) => setBackendUrl(e.target.value)}
      />
      <button style={{ marginTop: 8, width: "100%" }} onClick={save}>
        Save & test connection
      </button>
      {status !== "unknown" && (
        <div style={{ marginTop: 6, fontSize: 12, color: status === "ok" ? "#4ade80" : "#f87171" }}>
          {status === "ok" ? "Connected" : "Could not reach backend"}
        </div>
      )}

      <label style={{ fontSize: 11, display: "block", marginTop: 12 }}>Camera access (one-time)</label>
      <button style={{ marginTop: 4, width: "100%" }} onClick={openCameraPermissionTab}>
        Grant camera access
      </button>
      <p style={{ fontSize: 11, opacity: 0.7, marginTop: 4 }}>Opens a tab — grant there, then you can close it.</p>
    </div>
  );
}
