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
      resources: ["mediapipe/*", "sign_references.json"],
      matches: [
        "https://meet.google.com/*",
        "https://*.zoom.us/*",
        "https://teams.microsoft.com/*",
      ],
    },
  ],
});
