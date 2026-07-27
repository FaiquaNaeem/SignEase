"""Download the Kaggle ASL-alphabet image dataset and turn it into normalized
hand-landmark features via *real* MediaPipe Hands detection.

This replaces every previous "data prep" script in the repo, none of which
actually ran end-to-end (see plan doc for the audit). Run with:

    python -m backend.training.prepare_letters --data-dir backend/data/letters
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.training.common import letter_feature_vector  # noqa: E402

KAGGLE_DATASET = "grassknoted/asl-alphabet"


def download_dataset(data_dir: Path) -> Path:
    zip_path = data_dir / "asl-alphabet.zip"
    extract_dir = data_dir / "raw"
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Dataset already extracted at {extract_dir}, skipping download.")
        return extract_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading {KAGGLE_DATASET} via Kaggle API ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(data_dir)],
            check=True,
        )
    print(f"Extracting {zip_path} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def find_class_dirs(raw_dir: Path) -> dict[str, Path]:
    # The kaggle archive nests as asl_alphabet_train/asl_alphabet_train/<CLASS>/*.jpg
    candidates = list(raw_dir.rglob("asl_alphabet_train"))
    train_root = candidates[-1] if candidates else raw_dir
    class_dirs = {p.name: p for p in train_root.iterdir() if p.is_dir()}
    if not class_dirs:
        raise RuntimeError(f"No class directories found under {train_root}")
    return class_dirs


MEDIAPIPE_MODEL_PATH = Path(__file__).resolve().parents[1] / "mediapipe_models" / "hand_landmarker.task"


def extract_features(class_dirs: dict[str, Path], images_per_class: int | None, seed: int):
    # mediapipe>=0.10.something dropped the legacy mp.solutions.hands API in
    # favor of the Tasks API, which needs a downloaded .task model bundle
    # (see MEDIAPIPE_MODEL_PATH) instead of being usable out of the box.
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not MEDIAPIPE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MEDIAPIPE_MODEL_PATH}. Download it with:\n"
            "  curl -sL -o backend/mediapipe_models/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )

    detector = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
        )
    )
    rng = np.random.default_rng(seed)

    features, labels, class_names = [], [], sorted(class_dirs.keys())
    no_hand_count = 0

    for class_name in class_names:
        image_paths = sorted(class_dirs[class_name].glob("*.jpg")) + sorted(
            class_dirs[class_name].glob("*.png")
        )
        if images_per_class is not None:
            rng.shuffle(image_paths)
            image_paths = image_paths[:images_per_class]

        for path in tqdm(image_paths, desc=class_name, leave=False):
            image = cv2.imread(str(path))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = detector.detect(mp_image)
            if not result.hand_landmarks:
                no_hand_count += 1
                continue
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
                dtype=np.float32,
            )
            features.append(letter_feature_vector(landmarks))
            labels.append(class_name)

    detector.close()
    print(f"Extracted {len(features)} samples, {no_hand_count} images had no detected hand.")
    return np.stack(features), np.array(labels), class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("backend/data/letters"))
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=800,
        help="Cap per class to keep MediaPipe extraction time reasonable (None = all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = download_dataset(args.data_dir)
    class_dirs = find_class_dirs(raw_dir)
    # "nothing" is meaningless for a landmark-based classifier: by definition
    # MediaPipe finds no hand in those images, so there's no landmark vector
    # to classify. That's exactly the "no hand detected" case the inference
    # layer already handles before ever calling the model — keeping the
    # class here just starves it of real samples (799/800 correctly produce
    # no detection) and broke the stratified split entirely.
    class_dirs.pop("nothing", None)
    print(f"Found {len(class_dirs)} classes: {sorted(class_dirs.keys())}")

    features, labels, class_names = extract_features(class_dirs, args.images_per_class, args.seed)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    label_idx = np.array([class_to_idx[l] for l in labels], dtype=np.int64)

    x_train, x_temp, y_train, y_temp = train_test_split(
        features, label_idx, test_size=0.3, random_state=args.seed, stratify=label_idx
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )

    out_dir = args.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "letters.npz",
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
        class_names=np.array(class_names),
    )
    print(f"Saved processed dataset to {out_dir / 'letters.npz'}")
    print(f"Splits: train={len(y_train)} val={len(y_val)} test={len(y_test)}")


if __name__ == "__main__":
    main()
