"""Shared landmark normalization, augmentation, and dataset utilities.

Used by both the static-letter pipeline (single-frame hand landmarks) and the
dynamic-word pipeline (sequences of hand+pose landmarks). Keeping this in one
place is what the old repo never had: four different, mutually-incompatible
normalization schemes scattered across files.
"""
from __future__ import annotations

import numpy as np

# MediaPipe Hands landmark indices we rely on for normalization/features.
WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
MIDDLE_MCP = 9
FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [2, 5, 9, 13, 17]
FINGER_PIPS = [3, 6, 10, 14, 18]


def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Translate to wrist-relative, scale by wrist->middle-MCP distance.

    landmarks: (21, 3) array of raw (x, y, z) MediaPipe hand landmarks.
    Returns a (21, 3) array, invariant to hand position and overall size
    (but not to in-plane rotation — the classifier learns that instead of
    us throwing away useful orientation information).
    """
    origin = landmarks[WRIST]
    centered = landmarks - origin
    scale = np.linalg.norm(centered[MIDDLE_MCP])
    scale = max(scale, 1e-6)
    return centered / scale


def hand_geometry_features(normalized: np.ndarray) -> np.ndarray:
    """Engineered features on top of normalized landmarks: fingertip
    distances from wrist and per-finger curl angles. These make the small
    residual MLP far more sample-efficient than raw coordinates alone.
    """
    feats = []
    # Fingertip distance from wrist (already-origin, so just the norm).
    for tip in FINGER_TIPS:
        feats.append(float(np.linalg.norm(normalized[tip])))
    # Per-finger curl angle at the PIP joint (MCP -> PIP -> TIP).
    for mcp, pip, tip in zip(FINGER_MCPS, FINGER_PIPS, FINGER_TIPS):
        v1 = normalized[mcp] - normalized[pip]
        v2 = normalized[tip] - normalized[pip]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        feats.append(float(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    # Pairwise fingertip-to-fingertip distances (captures pinch/spread shapes).
    for i in range(len(FINGER_TIPS)):
        for j in range(i + 1, len(FINGER_TIPS)):
            feats.append(
                float(np.linalg.norm(normalized[FINGER_TIPS[i]] - normalized[FINGER_TIPS[j]]))
            )
    return np.array(feats, dtype=np.float32)


def letter_feature_vector(landmarks: np.ndarray) -> np.ndarray:
    """Full feature vector for the static-letter model: normalized raw
    coordinates (63) concatenated with engineered geometry (20).
    """
    normalized = normalize_hand_landmarks(landmarks)
    geometry = hand_geometry_features(normalized)
    return np.concatenate([normalized.reshape(-1), geometry]).astype(np.float32)


LETTER_FEATURE_DIM = 21 * 3 + 5 + 5 + 10  # 63 raw + 5 tip-dist + 5 curl + 10 pairwise = 83


def augment_landmarks(landmarks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small random rotation (about z), isotropic scale jitter, and additive
    noise applied to *raw* landmarks before normalization. Cheap but
    effective augmentation for a small landmark dataset.
    """
    angle = rng.uniform(-0.26, 0.26)  # ~+-15 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scale = rng.uniform(0.9, 1.1)
    noise = rng.normal(0.0, 0.01, size=landmarks.shape).astype(np.float32)
    return (landmarks @ rot.T) * scale + noise


def normalize_pose_frame(pose: np.ndarray) -> np.ndarray:
    """Normalize a single frame of upper-body pose landmarks: origin at
    shoulder midpoint, scale by shoulder width. pose is (N, 3) with at
    least indices for left/right shoulder at positions 0 and 1.
    """
    left_shoulder, right_shoulder = pose[0], pose[1]
    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(right_shoulder - left_shoulder)
    scale = max(scale, 1e-6)
    return (pose - center) / scale


def resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a (T, D) sequence to (target_len, D) along time.
    Standard trick for variable-length landmark sequences (used by winning
    solutions to the Kaggle isolated-sign-language competitions).
    """
    t_orig = sequence.shape[0]
    if t_orig == target_len:
        return sequence
    if t_orig == 1:
        return np.repeat(sequence, target_len, axis=0)
    x_old = np.linspace(0.0, 1.0, t_orig)
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, sequence.shape[1]), dtype=sequence.dtype)
    for d in range(sequence.shape[1]):
        out[:, d] = np.interp(x_new, x_old, sequence[:, d])
    return out
