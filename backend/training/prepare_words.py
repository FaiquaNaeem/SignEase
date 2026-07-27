"""Download a curated subset of Kaggle's `asl-signs` competition dataset
(Google - Isolated Sign Language Recognition) and turn the already-MediaPipe-
extracted landmark parquet files into fixed-length hand+pose sequences.

The competition's bulk `kaggle competitions download` is tens of GB (~94,477
sequences across 250 signs) — far more than we need for a ~30-word curated
vocabulary. Instead this fetches only the specific per-sequence parquet files
for our curated words, in parallel via the Kaggle API client directly
(avoids ~3.7s/file subprocess overhead of shelling out to the CLI per file).

NOTE: downloading requires the user to have accepted the competition rules
at https://www.kaggle.com/competitions/asl-signs/rules at least once (a
one-time manual step Kaggle enforces for competition data).

    python -m backend.training.prepare_words --data-dir backend/data/words
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.training.common import normalize_hand_landmarks, normalize_pose_frame, resample_sequence  # noqa: E402

COMPETITION = "asl-signs"

# Curated set of common, everyday words to keep dataset size/training time
# reasonable. This dataset's vocabulary comes from PopSign (a baby/family
# sign-language app), not generic adult words — verified against the real
# `sign` column of train.csv rather than guessed, since about half of an
# initial guess (sorry/help/love/more/family/...) turned out not to exist.
CURATED_WORDS = [
    "hello", "bye", "please", "thankyou", "yes", "no", "water", "drink", "food",
    "hungry", "thirsty", "sick", "happy", "sad", "hot", "sleepy", "open", "close",
    "wait", "go", "look", "listen", "home", "potty", "tomorrow", "later", "now",
    "time", "where", "why", "who", "finish", "clean", "dirty", "quiet", "loud",
    "fine", "bad", "mom", "dad", "sleep", "shower", "milk", "up", "down",
]

# MediaPipe Holistic pose landmark indices we keep (upper body only).
POSE_KEEP_IDX = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, hips
NUM_HAND_LANDMARKS = 21
MAX_FRAMES = 64
DOWNLOAD_WORKERS = 24


def download_file_if_missing(api: KaggleApi, remote_path: str, root: Path) -> Path:
    local_path = root / remote_path
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    api.competition_download_file(COMPETITION, remote_path, path=str(local_path.parent), quiet=True, force=True)

    if local_path.exists():
        return local_path
    # Kaggle's API zip-wraps some files (observed for .csv) but not others
    # (observed for .parquet) — handle both rather than assume one.
    zip_path = local_path.with_name(local_path.name + ".zip")
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(local_path.parent)
        zip_path.unlink()
    if not local_path.exists():
        raise FileNotFoundError(f"Expected {local_path} after download but it's missing (zip or direct).")
    return local_path


def resolve_file(remote_path: str, local_archive_dir: Path | None, api: KaggleApi | None, root: Path) -> Path:
    """Prefer an already-extracted local copy of the full competition
    archive (reliable — no per-file API flakiness) over the per-file
    download API (observed ~88% 404 rate on this closed competition,
    apparently due to individual-file storage gaps that the bulk archive
    doesn't have)."""
    if local_archive_dir is not None:
        local_copy = local_archive_dir / remote_path
        if local_copy.exists():
            return local_copy
    if api is None:
        raise FileNotFoundError(f"{remote_path} not found in local archive and no API fallback configured.")
    return download_file_if_missing(api, remote_path, root)


def load_frame_features(parquet_path: Path) -> np.ndarray | None:
    """Read one sequence's landmark parquet file, return (T, D) normalized
    hand+pose features, or None if neither hand is ever visible."""
    df = pd.read_parquet(parquet_path)
    frames = sorted(df["frame"].unique())
    seq = []
    any_hand_seen = False

    for frame in frames:
        fdf = df[df["frame"] == frame]

        def get_landmarks(row_type: str, n: int) -> np.ndarray:
            sub = fdf[fdf["type"] == row_type].sort_values("landmark_index")
            if len(sub) < n:
                return np.full((n, 3), np.nan, dtype=np.float32)
            return sub[["x", "y", "z"]].to_numpy(dtype=np.float32)[:n]

        left_hand = get_landmarks("left_hand", NUM_HAND_LANDMARKS)
        right_hand = get_landmarks("right_hand", NUM_HAND_LANDMARKS)
        pose_full = get_landmarks("pose", 33)
        pose = pose_full[POSE_KEEP_IDX]

        left_valid = not np.isnan(left_hand).all()
        right_valid = not np.isnan(right_hand).all()
        if left_valid or right_valid:
            any_hand_seen = True

        left_norm = normalize_hand_landmarks(np.nan_to_num(left_hand)) if left_valid else np.zeros((21, 3), np.float32)
        right_norm = normalize_hand_landmarks(np.nan_to_num(right_hand)) if right_valid else np.zeros((21, 3), np.float32)
        pose_clean = np.nan_to_num(pose)
        pose_norm = normalize_pose_frame(pose_clean) if not np.isnan(pose).all() else np.zeros_like(pose_clean)

        frame_vec = np.concatenate([left_norm.reshape(-1), right_norm.reshape(-1), pose_norm.reshape(-1)])
        seq.append(frame_vec)

    if not any_hand_seen:
        return None
    return resample_sequence(np.stack(seq).astype(np.float32), MAX_FRAMES)


def _extract_one(item: tuple[Path, int]) -> tuple[np.ndarray, int] | None:
    """Module-level (picklable) worker for ProcessPoolExecutor."""
    path, label = item
    try:
        seq = load_frame_features(path)
    except Exception:  # noqa: BLE001 - skip unreadable files, don't kill the whole run
        return None
    if seq is None:
        return None
    return seq, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("backend/data/words"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class", type=int, default=300, help="Cap sequences per word (keeps download/extraction time bounded)."
    )
    parser.add_argument(
        "--local-archive-dir",
        type=Path,
        default=None,
        help="Path to an already-extracted full `kaggle competitions download -c asl-signs` archive "
        "(e.g. backend/data/words_full/raw). Used instead of the flaky per-file download API when present.",
    )
    args = parser.parse_args()

    root = args.data_dir / "raw"
    root.mkdir(parents=True, exist_ok=True)

    api = None
    if args.local_archive_dir is None or not (args.local_archive_dir / "train.csv").exists():
        api = KaggleApi()
        api.authenticate()

    train_csv = resolve_file("train.csv", args.local_archive_dir, api, root)
    train_df = pd.read_csv(train_csv)

    available_words = [w for w in CURATED_WORDS if w in set(train_df["sign"])]
    missing = sorted(set(CURATED_WORDS) - set(available_words))
    if missing:
        print(f"Warning: these curated words aren't in the dataset's sign vocabulary: {missing}")
    print(f"Using {len(available_words)} classes: {available_words}")

    # Plain per-word sampling instead of groupby(...).apply(...): the latter
    # is version-sensitive about whether the grouping column ("sign") ends
    # up as a column or gets absorbed into the index, which broke row["sign"]
    # lookups below on this pandas version.
    sampled_parts = []
    for word in available_words:
        word_rows = train_df[train_df["sign"] == word]
        n = min(len(word_rows), args.max_per_class)
        sampled_parts.append(word_rows.sample(n=n, random_state=args.seed))
    subset = pd.concat(sampled_parts, ignore_index=True)
    print(f"Resolving {len(subset)} landmark files...")

    def fetch(path: str) -> tuple[str, Path | None]:
        try:
            return path, resolve_file(path, args.local_archive_dir, api, root)
        except Exception as exc:  # noqa: BLE001 - report and skip, don't kill the whole run
            print(f"  failed to resolve {path}: {exc}")
            return path, None

    paths = subset["path"].tolist()
    downloaded: dict[str, Path] = {}
    if args.local_archive_dir is not None:
        # Local filesystem lookups are fast enough to just do serially.
        for path in tqdm(paths, desc="Resolving"):
            _, local = fetch(path)
            if local is not None:
                downloaded[path] = local
    else:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futures = [pool.submit(fetch, p) for p in paths]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                path, local = future.result()
                if local is not None:
                    downloaded[path] = local

    class_names = sorted(available_words)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    items = [
        (downloaded[row["path"]], class_to_idx[row["sign"]])
        for _, row in subset.iterrows()
        if row["path"] in downloaded
    ]
    sequences, labels = [], []
    # Parquet parsing + normalization is pure CPU work, independent per
    # file — parallelize across cores instead of a slow single-process loop.
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for result in tqdm(pool.map(_extract_one, items, chunksize=8), total=len(items), desc="Extracting sequences"):
            if result is not None:
                seq, label = result
                sequences.append(seq)
                labels.append(label)

    x = np.stack(sequences).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    print(f"Extracted {len(y)} sequences across {len(class_names)} classes.")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )

    out_dir = args.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "words.npz",
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
        class_names=np.array(class_names),
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"max_frames": MAX_FRAMES, "feature_dim": int(x.shape[-1])}, f, indent=2)
    print(f"Saved processed dataset to {out_dir / 'words.npz'}")
    print(f"Splits: train={len(y_train)} val={len(y_val)} test={len(y_test)}")


if __name__ == "__main__":
    main()
