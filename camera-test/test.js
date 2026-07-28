import { FilesetResolver, HandLandmarker } from "./vision_bundle.mjs";

const BACKEND_URL = "http://localhost:8001";
const PREDICT_INTERVAL_MS = 700;
const WORD_CAPTURE_HZ = 15;

const statusEl = document.getElementById("status");
const captionEl = document.getElementById("caption");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const modeLetterBtn = document.getElementById("mode-letter");
const modeWordBtn = document.getElementById("mode-word");
const wordHoldBtn = document.getElementById("word-hold");

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

function log(msg, cls = "") {
  console.log(msg);
  statusEl.innerHTML += `\n<span class="${cls}">${msg}</span>`;
}

let framesSent = 0;
let handsSeen = 0;
let mode = "letter"; // "letter" | "word"
let wordCapturing = false;
let wordBuffer = [];
let wordCaptureTimer = null;

function setMode(newMode) {
  mode = newMode;
  modeLetterBtn.disabled = mode === "letter";
  modeWordBtn.disabled = mode === "word";
  wordHoldBtn.disabled = mode !== "word";
}
modeLetterBtn.addEventListener("click", () => setMode("letter"));
modeWordBtn.addEventListener("click", () => setMode("word"));
setMode("letter");

// MediaPipe's handedness assumes a mirrored/selfie input image; our raw
// (unmirrored) camera feed means its "Left"/"Right" labels are swapped
// relative to true anatomy — same convention used in the extension and in
// how the training data (Kaggle asl-signs, extracted via MediaPipe) is
// labeled, so this keeps client and model consistent.
function splitHands(result) {
  let left = null;
  let right = null;
  result.landmarks.forEach((landmarks, i) => {
    const label = result.handedness[i]?.[0]?.categoryName;
    const converted = landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z }));
    // No swap: use MediaPipe's raw Left/Right labels directly. The training
    // data's left_hand/right_hand columns came from MediaPipe's own output
    // too, whatever its internal mirror-convention is — it's applied the
    // same way both times, so matching it directly (not "correcting" it) is
    // what actually agrees with what the model was trained on. Swapping
    // here was confirmed to crash offline test accuracy from 80.94% to
    // 5.83% when reproduced against the real held-out test set.
    if (label === "Left") left = converted;
    else if (label === "Right") right = converted;
  });
  return { left, right };
}

async function main() {
  log("Requesting camera…");
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  video.srcObject = stream;
  await video.play();
  log("Camera OK: " + video.videoWidth + "x" + video.videoHeight, "ok");

  log("Loading MediaPipe WASM fileset (local files, no CDN)…");
  const fileset = await FilesetResolver.forVisionTasks("./mediapipe-wasm");
  log("Fileset loaded.", "ok");

  log("Creating HandLandmarker (CPU delegate, local .task model)…");
  const landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "./hand_landmarker.task", delegate: "CPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.5,
  });
  log("HandLandmarker ready! Hold your hand up to the camera.", "ok");

  let latestResult = null;
  let lastPredictAt = 0;
  let predictInFlight = false;

  async function predictLetter(landmarks) {
    predictInFlight = true;
    try {
      const res = await fetch(`${BACKEND_URL}/api/predict/letter`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ landmarks: landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z })) }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      captionEl.textContent = `${data.label}  (${(data.confidence * 100).toFixed(0)}%)`;
      captionEl.className = data.confidence >= 0.55 ? "ok" : "";
      log(`letter: ${data.label} ${(data.confidence * 100).toFixed(0)}% (${data.inference_ms.toFixed(1)}ms)`);
    } catch (err) {
      captionEl.textContent = "predict failed";
      captionEl.className = "bad";
      log("predict ERROR: " + (err && err.message ? err.message : String(err)), "bad");
    } finally {
      predictInFlight = false;
    }
  }

  async function predictWord(frames) {
    captionEl.textContent = "…";
    captionEl.className = "";
    try {
      const res = await fetch(`${BACKEND_URL}/api/predict/word`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      captionEl.textContent = `${data.label}  (${(data.confidence * 100).toFixed(0)}%)`;
      captionEl.className = data.confidence >= 0.55 ? "ok" : "";
      log(`word: ${data.label} ${(data.confidence * 100).toFixed(0)}% from ${frames.length} frames (${data.inference_ms.toFixed(1)}ms)`);
    } catch (err) {
      captionEl.textContent = "predict failed";
      captionEl.className = "bad";
      log("predict ERROR: " + (err && err.message ? err.message : String(err)), "bad");
    }
  }

  wordHoldBtn.addEventListener("mousedown", () => {
    if (mode !== "word") return;
    wordCapturing = true;
    wordBuffer = [];
    wordHoldBtn.textContent = "Recording… release when done";
    wordCaptureTimer = setInterval(() => {
      if (!latestResult) return;
      const { left, right } = splitHands(latestResult);
      if (left || right) wordBuffer.push({ leftHand: left, rightHand: right, pose: null });
    }, 1000 / WORD_CAPTURE_HZ);
  });
  const stopWordCapture = () => {
    if (!wordCapturing) return;
    wordCapturing = false;
    wordHoldBtn.textContent = "Hold to sign a word";
    clearInterval(wordCaptureTimer);
    log(`captured ${wordBuffer.length} frames for word prediction`);
    if (wordBuffer.length >= 3) void predictWord(wordBuffer);
    wordBuffer = [];
  };
  wordHoldBtn.addEventListener("mouseup", stopWordCapture);
  wordHoldBtn.addEventListener("mouseleave", stopWordCapture);

  function loop() {
    framesSent++;
    const now = performance.now();
    const result = landmarker.detectForVideo(video, now);
    latestResult = result;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (result.landmarks.length > 0) {
      handsSeen++;
      ctx.strokeStyle = "#4ade80";
      ctx.fillStyle = "#4ade80";
      ctx.lineWidth = 3;
      for (const hand of result.landmarks) {
        for (const [a, b] of HAND_CONNECTIONS) {
          ctx.beginPath();
          ctx.moveTo(hand[a].x * canvas.width, hand[a].y * canvas.height);
          ctx.lineTo(hand[b].x * canvas.width, hand[b].y * canvas.height);
          ctx.stroke();
        }
        for (const pt of hand) {
          ctx.beginPath();
          ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      if (mode === "letter" && !predictInFlight && now - lastPredictAt > PREDICT_INTERVAL_MS) {
        lastPredictAt = now;
        void predictLetter(result.landmarks[0]);
      }
    }

    if (framesSent % 30 === 0) {
      statusEl.lastChild.textContent = `frames sent: ${framesSent} | hands detected in: ${handsSeen} frames | currently: ${result.landmarks.length} hand(s)`;
    }
    requestAnimationFrame(loop);
  }
  const summaryLine = document.createElement("div");
  summaryLine.className = "ok";
  statusEl.appendChild(summaryLine);
  loop();
}

main().catch((err) => {
  log("ERROR: " + (err && err.message ? err.message : String(err)), "bad");
  console.error(err);
});
