import { defineManifest } from "@crxjs/vite-plugin";
import pkg from "./package.json";

export default defineManifest({
  manifest_version: 3,
  name: "SignEase Bridge",
  description:
    "Real-time two-way sign language <-> speech translation for video calls.",
  version: pkg.version,
  icons: {
    16: "icons/icon16.png",
    48: "icons/icon48.png",
    128: "icons/icon128.png",
  },
  action: {
    default_popup: "src/popup/index.html",
  },
  background: {
    service_worker: "src/background/service-worker.ts",
    type: "module",
  },
  permissions: ["tabCapture", "offscreen", "storage", "activeTab", "scripting"],
  // MV3's default extension-page CSP (script-src 'self'; object-src 'self')
  // doesn't permit WebAssembly compilation at all — 'wasm-unsafe-eval' has
  // to be added explicitly, or MediaPipe's WebAssembly.instantiate() in the
  // offscreen document fails with a CSP violation error every time.
  content_security_policy: {
    extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
  },
  host_permissions: [
    "https://meet.google.com/*",
    "https://*.zoom.us/*",
    "https://teams.microsoft.com/*",
  ],
  content_scripts: [
    {
      matches: [
        "https://meet.google.com/*",
        "https://*.zoom.us/*",
        "https://teams.microsoft.com/*",
      ],
      js: ["src/content-scripts/overlay-root.tsx"],
      run_at: "document_idle",
    },
  ],
  web_accessible_resources: [
    {
      resources: ["mediapipe-wasm/*", "hand_landmarker.task", "pose_landmarker.task", "sign_references.json"],
      matches: [
        "https://meet.google.com/*",
        "https://*.zoom.us/*",
        "https://teams.microsoft.com/*",
      ],
    },
  ],
});
