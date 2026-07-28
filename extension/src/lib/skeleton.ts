// Standard 21-point MediaPipe hand landmark bone connections.
export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [5, 9], [9, 10], [10, 11], [11, 12], // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17],
];

/** Draws a single normalized (wrist-relative, scale-normalized) hand pose
 * centered in the canvas. Used to replay speech->sign references, which are
 * stored in the same normalized space the classifier trains on
 * (see backend/training/common.py:normalize_hand_landmarks). */
export function drawHand(
  ctx: CanvasRenderingContext2D,
  points: [number, number, number][],
  color = "#4ade80"
) {
  const { width, height } = ctx.canvas;
  const scale = Math.min(width, height) * 0.35;
  const cx = width / 2;
  const cy = height / 2;

  const project = (p: [number, number, number]): [number, number] => [
    cx + p[0] * scale,
    cy + p[1] * scale,
  ];

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  for (const [a, b] of HAND_CONNECTIONS) {
    const [ax, ay] = project(points[a]);
    const [bx, by] = project(points[b]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  ctx.fillStyle = color;
  for (const p of points) {
    const [x, y] = project(p);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

const ZERO_HAND: [number, number, number][] = Array.from({ length: 21 }, () => [0, 0, 0]);

function isZeroHand(points: [number, number, number][]): boolean {
  return points.every((p) => p[0] === 0 && p[1] === 0 && p[2] === 0);
}

export { ZERO_HAND, isZeroHand };
