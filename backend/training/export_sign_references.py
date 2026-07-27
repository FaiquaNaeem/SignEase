"""Build the asset that drives speech->sign playback: for every letter and
word class, pick the real training example closest to that class's centroid
and save its raw landmark geometry (not the classifier's engineered feature
vector). The extension replays these as an animated hand skeleton — no
licensed video/images, generated entirely from data we already collected.

    python -m backend.training.export_sign_references
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LETTERS_NPZ = Path("backend/data/letters/processed/letters.npz")
WORDS_NPZ = Path("backend/data/words/processed/words.npz")
OUT_PATH = Path("backend/checkpoints/sign_references.json")


def nearest_to_centroid(features: np.ndarray) -> int:
    """Index of the sample closest (L2) to the feature-space centroid."""
    centroid = features.reshape(features.shape[0], -1).mean(axis=0)
    flat = features.reshape(features.shape[0], -1)
    dists = np.linalg.norm(flat - centroid, axis=1)
    return int(np.argmin(dists))


def export_letters() -> dict:
    if not LETTERS_NPZ.exists():
        print(f"Skipping letters: {LETTERS_NPZ} not found (run prepare_letters.py + train first).")
        return {}
    data = np.load(LETTERS_NPZ, allow_pickle=True)
    class_names = [str(c) for c in data["class_names"]]
    x_train, y_train = data["x_train"], data["y_train"]

    out = {}
    for idx, name in enumerate(class_names):
        class_features = x_train[y_train == idx]
        if len(class_features) == 0:
            continue
        best = nearest_to_centroid(class_features)
        # First 63 values of the 83-dim feature vector are the normalized
        # (x, y, z) hand landmarks — see backend/training/common.py:letter_feature_vector.
        landmarks = class_features[best][:63].reshape(21, 3).tolist()
        out[name] = {"type": "static", "hand": landmarks}
    return out


def export_words() -> dict:
    if not WORDS_NPZ.exists():
        print(f"Skipping words: {WORDS_NPZ} not found (run prepare_words.py + train first).")
        return {}
    data = np.load(WORDS_NPZ, allow_pickle=True)
    class_names = [str(c) for c in data["class_names"]]
    x_train, y_train = data["x_train"], data["y_train"]

    out = {}
    for idx, name in enumerate(class_names):
        class_seqs = x_train[y_train == idx]  # (n, T, 150)
        if len(class_seqs) == 0:
            continue
        best = nearest_to_centroid(class_seqs)
        seq = class_seqs[best][::2]  # (T, 150) = left(63) + right(63) + pose(24); every 2nd
        # frame is plenty for a skeleton replay and halves the asset size.
        frames = []
        for frame in seq:
            left = frame[:63].reshape(21, 3).tolist()
            right = frame[63:126].reshape(21, 3).tolist()
            pose = frame[126:].reshape(8, 3).tolist()
            frames.append({"leftHand": left, "rightHand": right, "pose": pose})
        out[name] = {"type": "sequence", "frames": frames}
    return out


def round_floats(obj, digits=4):
    """Full float64 precision is meaningless for a skeleton animation and
    roughly triples JSON size for no visual benefit — this asset gets
    bundled directly into the extension package, so size matters."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, list):
        return [round_floats(v, digits) for v in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    return obj


def main():
    references = round_floats({"letters": export_letters(), "words": export_words()})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(references, f, separators=(",", ":"))
    n_letters = len(references["letters"])
    n_words = len(references["words"])
    print(f"Wrote {n_letters} letter references and {n_words} word references to {OUT_PATH}")


if __name__ == "__main__":
    main()
