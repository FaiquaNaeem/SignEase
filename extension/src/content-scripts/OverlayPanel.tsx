import { useEffect, useRef, useState } from "react";
import { SignToSpeechSession, type SignToSpeechMode } from "../lib/signToSpeechSession";
import { SignPlaybackPlayer } from "../lib/signPlayback";
import { checkHealth, fetchSignReferences } from "../lib/apiRelay";
import { subscribeDebugLog, type DebugEntry } from "../lib/debugLog";
import type { ExtensionMessage, SignLanguage } from "../types";

type Direction = "signToSpeech" | "speechToSign";

export function OverlayPanel() {
  const [direction, setDirection] = useState<Direction>("signToSpeech");
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState<SignToSpeechMode>("letter");
  const [language, setLanguage] = useState<SignLanguage>("en-IN");
  const [caption, setCaption] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<{ reachable: boolean; url: string; detail: string } | null>(
    null
  );
  const [debugEntries, setDebugEntries] = useState<DebugEntry[]>([]);
  const [showDebug, setShowDebug] = useState(false);

  const sessionRef = useRef<SignToSpeechSession | null>(null);
  const playerRef = useRef(new SignPlaybackPlayer());
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    fetchSignReferences()
      .then((refs) => playerRef.current.setReferences(refs))
      .catch((err) => setError(`sign-references: ${err.message}`));
    checkHealth().then(setBackendStatus).catch((err) =>
      setBackendStatus({ reachable: false, url: "?", detail: err.message })
    );
    return subscribeDebugLog(setDebugEntries);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    playerRef.current.setCanvas(canvas ? canvas.getContext("2d") : null);
  }, [direction]);

  useEffect(() => {
    if (direction !== "speechToSign") return;
    const listener = (message: ExtensionMessage) => {
      if (message.type === "TRANSCRIPT_RESULT") {
        setCaption(message.transcript);
        setError(null);
        playerRef.current.enqueueText(message.transcript);
      } else if (message.type === "TRANSCRIPTION_ERROR") {
        setError(message.message);
      }
    };
    chrome.runtime.onMessage.addListener(listener);
    return () => chrome.runtime.onMessage.removeListener(listener);
  }, [direction]);

  const start = async () => {
    setError(null);
    if (direction === "signToSpeech") {
      const session = new SignToSpeechSession(
        mode,
        {
          onCaption: (text, conf) => {
            setCaption(text);
            setConfidence(conf);
            setError(null);
          },
          onError: (message) => setError(message),
        },
        language
      );
      sessionRef.current = session;
      try {
        await session.start();
        setRunning(true);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } else {
      chrome.runtime.sendMessage({ type: "START_TAB_TRANSCRIPTION", tabId: -1 } satisfies ExtensionMessage);
      setRunning(true);
    }
  };

  const stop = () => {
    if (direction === "signToSpeech") {
      sessionRef.current?.stop();
      sessionRef.current = null;
    } else {
      chrome.runtime.sendMessage({ type: "STOP_TAB_TRANSCRIPTION" } satisfies ExtensionMessage);
      playerRef.current.stop();
    }
    setRunning(false);
  };

  const toggleWordHold = (holding: boolean) => sessionRef.current?.setWordCapturing(holding);

  return (
    <div style={panelStyle}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <strong style={{ fontSize: 13 }}>SignEase Bridge</strong>
        <span style={{ fontSize: 11, opacity: 0.7 }}>{running ? "live" : "idle"}</span>
      </div>

      <div
        style={{
          marginTop: 6,
          fontSize: 10,
          display: "flex",
          alignItems: "center",
          gap: 6,
          color: backendStatus?.reachable ? "#4ade80" : "#f87171",
        }}
      >
        <span>{backendStatus === null ? "checking backend…" : backendStatus.reachable ? "backend connected" : "backend unreachable"}</span>
        <button
          style={{ fontSize: 9, padding: "1px 5px", background: "transparent", color: "inherit", border: "1px solid currentColor", borderRadius: 4, cursor: "pointer" }}
          onClick={() =>
            checkHealth()
              .then(setBackendStatus)
              .catch((err) => setBackendStatus({ reachable: false, url: "?", detail: err.message }))
          }
        >
          retest
        </button>
      </div>
      {backendStatus && !backendStatus.reachable && (
        <div style={{ fontSize: 10, opacity: 0.8, marginTop: 2 }}>
          tried {backendStatus.url} — {backendStatus.detail}
        </div>
      )}

      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button
          style={tabStyle(direction === "signToSpeech")}
          onClick={() => !running && setDirection("signToSpeech")}
        >
          Sign → Speech
        </button>
        <button
          style={tabStyle(direction === "speechToSign")}
          onClick={() => !running && setDirection("speechToSign")}
        >
          Speech → Sign
        </button>
      </div>

      {direction === "signToSpeech" && (
        <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
          <button style={tabStyle(mode === "letter")} onClick={() => !running && setMode("letter")}>
            Spell letters
          </button>
          <button style={tabStyle(mode === "word")} onClick={() => !running && setMode("word")}>
            Sign a word
          </button>
        </div>
      )}

      <div style={{ marginTop: 6 }}>
        <select value={language} onChange={(e) => setLanguage(e.target.value as SignLanguage)} disabled={running}>
          <option value="en-IN">English</option>
          <option value="hi-IN">Hindi</option>
        </select>
      </div>

      <button style={{ ...tabStyle(running), marginTop: 8, width: "100%" }} onClick={running ? stop : start}>
        {running ? "Stop" : "Start"}
      </button>

      {direction === "signToSpeech" && mode === "word" && running && (
        <button
          style={{ marginTop: 6, width: "100%" }}
          onMouseDown={() => toggleWordHold(true)}
          onMouseUp={() => toggleWordHold(false)}
          onMouseLeave={() => toggleWordHold(false)}
        >
          Hold to sign a word
        </button>
      )}

      <div style={{ marginTop: 10, fontSize: 20, fontWeight: 600, minHeight: 28 }}>{caption || "…"}</div>
      {confidence > 0 && <div style={{ fontSize: 11, opacity: 0.7 }}>confidence {(confidence * 100).toFixed(0)}%</div>}

      {direction === "speechToSign" && (
        <canvas ref={canvasRef} width={220} height={160} style={{ marginTop: 8, background: "#111", borderRadius: 6 }} />
      )}

      {error && <div style={{ marginTop: 6, fontSize: 11, color: "#f87171" }}>{error}</div>}

      <button
        style={{ marginTop: 8, width: "100%", fontSize: 10, background: "transparent", color: "#a1a1aa", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 6, padding: "3px 6px", cursor: "pointer" }}
        onClick={() => setShowDebug((v) => !v)}
      >
        {showDebug ? "Hide" : "Show"} debug log ({debugEntries.length})
      </button>
      {showDebug && (
        <div style={{ marginTop: 6, maxHeight: 140, overflowY: "auto", fontSize: 9, fontFamily: "monospace", background: "#0a0a0a", borderRadius: 6, padding: 6 }}>
          {debugEntries.length === 0 && <div style={{ opacity: 0.6 }}>no calls yet</div>}
          {debugEntries.map((e, i) => (
            <div key={i} style={{ marginBottom: 4, color: e.ok ? "#4ade80" : "#f87171" }}>
              [{e.time}] {e.label} {e.ok ? "OK" : "FAIL"} ({e.durationMs.toFixed(0)}ms)
              <div style={{ color: "#a1a1aa", wordBreak: "break-all" }}>{e.detail}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const panelStyle: React.CSSProperties = {
  position: "fixed",
  bottom: 16,
  right: 16,
  zIndex: 2147483647,
  width: 260,
  padding: 12,
  borderRadius: 10,
  background: "rgba(20, 20, 24, 0.92)",
  color: "#f5f5f5",
  fontFamily: "system-ui, sans-serif",
  boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
};

function tabStyle(active: boolean): React.CSSProperties {
  return {
    flex: 1,
    fontSize: 11,
    padding: "4px 6px",
    borderRadius: 6,
    border: "1px solid rgba(255,255,255,0.15)",
    background: active ? "#4ade80" : "transparent",
    color: active ? "#0a0a0a" : "#f5f5f5",
    cursor: "pointer",
  };
}
