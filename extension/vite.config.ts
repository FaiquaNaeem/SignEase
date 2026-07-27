import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { crx } from "@crxjs/vite-plugin";
import { viteStaticCopy } from "vite-plugin-static-copy";
import manifest from "./manifest.config";

export default defineConfig({
  plugins: [
    react(),
    crx({ manifest }),
    // MediaPipe Hands needs its .wasm/.data/.binarypb assets served locally —
    // an MV3 extension's default CSP blocks fetching them from a CDN at
    // runtime, unlike the old repo's web app which loaded them from
    // cdn.jsdelivr.net. Copy the package's dist assets in at build time.
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/@mediapipe/hands/*.{wasm,data,binarypb,js}",
          dest: "mediapipe",
        },
      ],
    }),
  ],
  server: {
    port: 5175,
    strictPort: true,
    hmr: { port: 5175 },
  },
});
