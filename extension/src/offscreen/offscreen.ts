import { transcribe } from "../lib/api";
import type { ExtensionMessage } from "../types";

// MV3 service workers get suspended when idle, which would kill a live
// MediaStream. This offscreen document exists purely to hold the
// tabCapture stream and run MediaRecorder for as long as transcription is
// active, per Chrome's documented workaround for this limitation.

const CHUNK_INTERVAL_MS = 4000;

let recorder: MediaRecorder | null = null;
let stream: MediaStream | null = null;

chrome.runtime.onMessage.addListener((message: ExtensionMessage) => {
  if (message.type === "TAB_STREAM_ID") {
    startCapture(message.streamId).catch((err) => {
      chrome.runtime.sendMessage({
        type: "TRANSCRIPTION_ERROR",
        message: err instanceof Error ? err.message : String(err),
      } satisfies ExtensionMessage);
    });
  } else if (message.type === "STOP_TAB_TRANSCRIPTION") {
    stopCapture();
  }
});

async function startCapture(streamId: string) {
  stopCapture();

  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      // @ts-expect-error chromeMediaSource is a non-standard Chrome-only constraint
      mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId },
    },
  });

  // Keep the tab's own audio audible locally (otherwise capturing it mutes
  // the call for the user); loop it back through a silent-volume path isn't
  // needed since tabCapture by default still lets audio play through.
  recorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
  recorder.ondataavailable = async (event) => {
    if (event.data.size === 0) return;
    try {
      const result = await transcribe(event.data, "unknown");
      if (result.transcript.trim()) {
        chrome.runtime.sendMessage({
          type: "TRANSCRIPT_RESULT",
          transcript: result.transcript,
          language_code: result.language_code,
        } satisfies ExtensionMessage);
      }
    } catch (err) {
      chrome.runtime.sendMessage({
        type: "TRANSCRIPTION_ERROR",
        message: err instanceof Error ? err.message : String(err),
      } satisfies ExtensionMessage);
    }
  };
  recorder.start(CHUNK_INTERVAL_MS);
}

function stopCapture() {
  recorder?.stop();
  recorder = null;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
}
