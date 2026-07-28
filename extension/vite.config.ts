import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { crx } from "@crxjs/vite-plugin";
import { viteStaticCopy } from "vite-plugin-static-copy";
import manifest from "./manifest.config";

export default defineConfig({
  plugins: [
    react(),
    crx({ manifest }),
    // The legacy @mediapipe/hands package (callback-based Hands.send()) is
    // deprecated by Google and broke unpredictably after the first frame in
    // this bundled context ("X is not a function" from its minified global-
    // namespace init code) — the same underlying deprecation that forced a
    // Python-side migration to the Tasks API for training data extraction.
    // @mediapipe/tasks-vision's HandLandmarker is the maintained replacement:
    // synchronous detectForVideo(), no global script-injection tricks. It
    // still needs its wasm assets served locally (MV3 CSP blocks CDN fetch).
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/@mediapipe/tasks-vision/wasm/*",
          dest: "mediapipe-wasm",
        },
      ],
    }),
  ],
  // offscreen.html is never referenced by any manifest field (its URL is
  // only a runtime string passed to chrome.offscreen.createDocument), so
  // CRXJS's manifest-driven entry-point scan can't discover it on its own —
  // it needs to be listed explicitly or the file silently never gets built
  // at all, and chrome.offscreen.createDocument fails at runtime with no
  // build-time signal that anything was wrong.
  build: {
    rollupOptions: {
      input: {
        offscreen: "src/offscreen/offscreen.html",
        permission: "src/permission/permission.html",
      },
    },
  },
  server: {
    port: 5175,
    strictPort: true,
    hmr: { port: 5175 },
  },
});
