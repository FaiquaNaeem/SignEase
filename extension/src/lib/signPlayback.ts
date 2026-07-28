import { drawHand } from "./skeleton";
import type { SignReferences } from "../types";

const FRAME_MS = 60; // ~16fps playback of recorded sequences
const LETTER_HOLD_MS = 500;

/** Tokenizes ASR transcript text into playable units: known words play as
 * their recorded motion sequence; anything else gets fingerspelled letter
 * by letter using the static letter references. */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .split(/\s+/)
    .filter(Boolean);
}

export class SignPlaybackPlayer {
  private queue: string[] = [];
  private playing = false;
  private ctx: CanvasRenderingContext2D | null = null;
  private references: SignReferences | null = null;
  private stopped = false;

  setCanvas(ctx: CanvasRenderingContext2D | null) {
    this.ctx = ctx;
  }

  setReferences(references: SignReferences) {
    this.references = references;
  }

  enqueueText(text: string) {
    this.queue.push(...tokenize(text));
    if (!this.playing) void this.drain();
  }

  stop() {
    this.stopped = true;
    this.queue = [];
  }

  private async drain() {
    this.playing = true;
    this.stopped = false;
    while (this.queue.length > 0 && !this.stopped) {
      const word = this.queue.shift()!;
      await this.playWord(word);
    }
    this.playing = false;
  }

  private async playWord(word: string) {
    if (!this.references) return;
    const wordRef = this.references.words[word];
    if (wordRef) {
      for (const frame of wordRef.frames) {
        if (this.stopped) return;
        this.render(frame.leftHand, frame.rightHand);
        await sleep(FRAME_MS);
      }
      return;
    }
    // Not in the curated word vocabulary: fingerspell it.
    for (const char of word) {
      if (this.stopped) return;
      const letterRef = this.references.letters[char.toUpperCase()];
      if (!letterRef) continue;
      this.render(letterRef.hand, null);
      await sleep(LETTER_HOLD_MS);
    }
  }

  private render(
    left: [number, number, number][] | null,
    right: [number, number, number][] | null
  ) {
    if (!this.ctx) return;
    this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
    if (left) drawHand(this.ctx, left, "#60a5fa");
    if (right) drawHand(this.ctx, right, "#4ade80");
  }
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
