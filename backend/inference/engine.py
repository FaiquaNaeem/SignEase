"""Loads both trained checkpoints once and runs inference. Single source of
truth for normalization (via backend.training.common) so the server can
never drift from how the models were trained — the old repo's #1 problem.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from backend.models.letter_classifier import LetterClassifier
from backend.models.word_classifier import WordClassifier
from backend.training.common import (
    hand_geometry_features,
    normalize_hand_landmarks,
    normalize_pose_frame,
    resample_sequence,
)

CHECKPOINT_DIR = Path("backend/checkpoints")
WORD_MAX_FRAMES = 64
WORD_POSE_POINTS = 8


def _landmarks_to_array(landmarks: list[dict]) -> np.ndarray:
    return np.array([[p["x"], p["y"], p["z"]] for p in landmarks], dtype=np.float32)


class InferenceEngine:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.letter_model, self.letter_classes = self._load_letter_model()
        self.word_model, self.word_classes = self._load_word_model()
        self.started_at = time.time()

    def _load_letter_model(self):
        ckpt_path = CHECKPOINT_DIR / "letter_classifier.pt"
        if not ckpt_path.exists():
            return None, None
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model = LetterClassifier(input_dim=ckpt["input_dim"], num_classes=len(ckpt["class_names"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        return model, ckpt["class_names"]

    def _load_word_model(self):
        ckpt_path = CHECKPOINT_DIR / "word_classifier.pt"
        if not ckpt_path.exists():
            return None, None
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model = WordClassifier(input_dim=ckpt["input_dim"], num_classes=len(ckpt["class_names"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        return model, ckpt["class_names"]

    def predict_letter(self, landmarks: list[dict], top_k: int = 3) -> dict:
        if self.letter_model is None:
            raise RuntimeError("Letter model not loaded — run training/train_letters.py first.")
        raw = _landmarks_to_array(landmarks)
        normalized = normalize_hand_landmarks(raw)
        geometry = hand_geometry_features(normalized)
        feature = np.concatenate([normalized.reshape(-1), geometry]).astype(np.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            x = torch.from_numpy(feature).unsqueeze(0).to(self.device)
            logits = self.letter_model(x)
            probs = torch.softmax(logits, dim=1)[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        return self._format_result(probs, self.letter_classes, top_k, inference_ms)

    def predict_word(self, frames: list[dict], top_k: int = 3) -> dict:
        if self.word_model is None:
            raise RuntimeError("Word model not loaded — run training/train_words.py first.")

        seq = []
        for frame in frames:
            left = frame.get("leftHand")
            right = frame.get("rightHand")
            pose = frame.get("pose")

            left_norm = (
                normalize_hand_landmarks(_landmarks_to_array(left)) if left else np.zeros((21, 3), np.float32)
            )
            right_norm = (
                normalize_hand_landmarks(_landmarks_to_array(right)) if right else np.zeros((21, 3), np.float32)
            )
            pose_arr = _landmarks_to_array(pose) if pose else np.zeros((WORD_POSE_POINTS, 3), np.float32)
            pose_norm = normalize_pose_frame(pose_arr) if pose else pose_arr

            seq.append(np.concatenate([left_norm.reshape(-1), right_norm.reshape(-1), pose_norm.reshape(-1)]))

        sequence = resample_sequence(np.stack(seq).astype(np.float32), WORD_MAX_FRAMES)

        t0 = time.perf_counter()
        with torch.no_grad():
            x = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            mask = torch.ones(1, sequence.shape[0], dtype=torch.bool, device=self.device)
            logits = self.word_model(x, mask)
            probs = torch.softmax(logits, dim=1)[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        return self._format_result(probs, self.word_classes, top_k, inference_ms)

    @staticmethod
    def _format_result(probs: torch.Tensor, class_names: list[str], top_k: int, inference_ms: float) -> dict:
        top_probs, top_idx = torch.topk(probs, k=min(top_k, probs.shape[0]))
        alternatives = [
            {"label": class_names[i], "confidence": float(p)} for p, i in zip(top_probs.tolist(), top_idx.tolist())
        ]
        return {
            "label": alternatives[0]["label"],
            "confidence": alternatives[0]["confidence"],
            "alternatives": alternatives,
            "inference_ms": inference_ms,
        }
