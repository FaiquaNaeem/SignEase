import { createRoot } from "react-dom/client";
import { OverlayPanel } from "./OverlayPanel";

// Shadow DOM keeps the host page's CSS (Meet/Zoom/Teams) from bleeding into
// our panel and vice versa.
const host = document.createElement("div");
host.id = "signease-bridge-root";
document.documentElement.appendChild(host);
const shadow = host.attachShadow({ mode: "open" });
const mountPoint = document.createElement("div");
shadow.appendChild(mountPoint);

createRoot(mountPoint).render(<OverlayPanel />);
