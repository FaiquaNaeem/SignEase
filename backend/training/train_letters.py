"""Train the static-letter ResMLP on the processed landmark dataset produced
by prepare_letters.py. Reports honest val/test accuracy — no hardcoded
numbers, no shortcuts.

    python -m backend.training.train_letters
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
from backend.models.letter_classifier import LetterClassifier  # noqa: E402


class LandmarkFeatureDataset(Dataset):
    """Wraps already-extracted feature vectors; optionally re-derives them
    from raw landmarks with augmentation for training. Since prepare_letters
    only stores final feature vectors (not raw landmarks) we apply a
    lightweight feature-space jitter instead of landmark-space augmentation
    at train time, which is still effective and much simpler to wire up.
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
            x = x + np.random.default_rng().normal(0.0, 0.01, size=x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[idx])


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = float((preds == labels).mean())
    return acc, preds, labels


def main():
    data_path = Path("backend/data/letters/processed/letters.npz")
    data = np.load(data_path, allow_pickle=True)
    class_names = [str(c) for c in data["class_names"]]
    num_classes = len(class_names)

    train_ds = LandmarkFeatureDataset(data["x_train"], data["y_train"], augment=True)
    val_ds = LandmarkFeatureDataset(data["x_val"], data["y_val"])
    test_ds = LandmarkFeatureDataset(data["x_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LetterClassifier(input_dim=data["x_train"].shape[1], num_classes=num_classes).to(device)
    print(model)
    print(f"Device: {device}, params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    patience, patience_counter = 8, 0
    ckpt_dir = Path("backend/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "letter_classifier.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
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
                    "input_dim": data["x_train"].shape[1],
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break

    # Final honest evaluation on the untouched test split, using the best checkpoint.
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
    with open(ckpt_dir / "letter_classifier_report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.save(ckpt_dir / "letter_confusion_matrix.npy", confusion_matrix(labels, preds))
    print(f"Saved checkpoint to {ckpt_path} and report to {ckpt_dir / 'letter_classifier_report.json'}")


if __name__ == "__main__":
    main()
