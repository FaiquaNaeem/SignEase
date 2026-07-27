"""Train the dynamic-word Conv1D+Transformer classifier on the processed
landmark-sequence dataset produced by prepare_words.py. Same honest
train/val/test discipline as train_letters.py: real accuracy, no shortcuts.

    python -m backend.training.train_words
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.models.word_classifier import WordClassifier  # noqa: E402

PAD_NOISE_STD = 0.01


class SequenceDataset(Dataset):
    """x: (N, T, D) already resampled to a fixed T by prepare_words.py. All
    frames are "valid" here since resampling removed variable length, but we
    still carry a mask so the model/pooling code paths exercise real masking
    logic (useful once we support live variable-length capture too).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.augment:
            rng = np.random.default_rng()
            x = x + rng.normal(0.0, PAD_NOISE_STD, size=x.shape).astype(np.float32)
            if rng.random() < 0.5:
                # Random temporal crop+pad-by-repeat-edge: mild robustness to
                # where in the sign the "start" is detected live.
                t = x.shape[0]
                shift = int(rng.integers(-3, 4))
                x = np.roll(x, shift, axis=0)
        mask = np.ones(x.shape[0], dtype=bool)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.tensor(self.y[idx])


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return float((preds == labels).mean()), preds, labels


def main():
    data_path = Path("backend/data/words/processed/words.npz")
    data = np.load(data_path, allow_pickle=True)
    class_names = [str(c) for c in data["class_names"]]
    num_classes = len(class_names)
    feature_dim = data["x_train"].shape[-1]

    train_ds = SequenceDataset(data["x_train"], data["y_train"], augment=True)
    val_ds = SequenceDataset(data["x_val"], data["y_val"])
    test_ds = SequenceDataset(data["x_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WordClassifier(input_dim=feature_dim, num_classes=num_classes).to(device)
    print(model)
    print(f"Device: {device}, params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    epochs = 80
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    patience, patience_counter = 12, 0
    ckpt_dir = Path("backend/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "word_classifier.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        scheduler.step()

        val_acc, _, _ = evaluate(model, val_loader, device)
        train_loss = total_loss / len(train_ds)
        print(
            f"epoch {epoch:3d}/{epochs} loss={train_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} time={time.time()-t0:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "input_dim": feature_dim,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break

    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"\n=== FINAL TEST ACCURACY: {test_acc:.4f} ===\n")
    print(
        classification_report(
            labels, preds, labels=list(range(num_classes)), target_names=class_names, digits=3, zero_division=0
        )
    )

    report = {
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "num_classes": num_classes,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
    }
    with open(ckpt_dir / "word_classifier_report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.save(ckpt_dir / "word_confusion_matrix.npy", confusion_matrix(labels, preds))
    print(f"Saved checkpoint to {ckpt_path} and report to {ckpt_dir / 'word_classifier_report.json'}")


if __name__ == "__main__":
    main()
