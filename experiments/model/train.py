import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

from model.cnn import build_model
from model.baseline import run_baselines
from datalayer.build_dataset import INDEX_TO_CHAR


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _prepare_tensors(X, y, device):
    """Convert numpy arrays to torch tensors with proper shape/dtype."""
    X_t = torch.from_numpy(X).float().unsqueeze(1) / 255.0   # (N, 1, 28, 28)
    y_t = torch.from_numpy(y).long()
    return X_t, y_t


def _make_loader(X_t, y_t, batch_size, shuffle):
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ──────────────────────────────────────────────
# 5.3  Training
# ──────────────────────────────────────────────

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=1e-3,
):
    """
    Train the CNN with CrossEntropyLoss + Adam.
    Returns training history (loss and accuracy per epoch).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += X_batch.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- validate ----
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:>3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    return history


# ──────────────────────────────────────────────
# 5.4  Validation / Evaluation
# ──────────────────────────────────────────────

def evaluate(model, loader, device, criterion=None):
    """Return (loss, accuracy) on the given loader."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += X_batch.size(0)

    return running_loss / total, correct / total


def full_evaluation(model, loader, device, num_classes):
    """
    Run a detailed evaluation: per-class classification report
    with character-name labels.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    present = sorted(set(all_labels) | set(all_preds))
    target_names = [INDEX_TO_CHAR[i] for i in present]

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        labels=present,
        target_names=target_names,
        zero_division=0,
    )

    print(f"\n{'='*50}")
    print(f"CNN Test Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print(report)

    return acc, report


# ──────────────────────────────────────────────
# 5.5  Model Export
# ──────────────────────────────────────────────

def save_model(model, path):
    """Save full model state dict."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Model Export] Saved model → {path}")


def load_model(path, num_classes=36, device=None):
    """Load a previously saved model."""
    model, device = build_model(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    print(f"[Model Export] Loaded model ← {path}")
    return model, device


# ──────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────

def run_training_pipeline(
    data_dir="dataset/ready/1k_pbm",
    output_dir="model/saved",
    epochs=20,
    batch_size=64,
    lr=1e-3,
    run_baseline=True,
):
    """
    End-to-end: load data → baselines → CNN train → evaluate → export.
    """
    # ---- Load data ----
    train_data = np.load(os.path.join(data_dir, "train.npz"))
    test_data = np.load(os.path.join(data_dir, "test.npz"))
    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    num_classes = len(set(y_train) | set(y_test))
    print(f"[Training Pipeline] Train: {X_train.shape}, Test: {X_test.shape}, "
          f"Classes: {num_classes}")

    # 5.1  Baselines
    if run_baseline:
        print("\n" + "="*60)
        print("  STEP 5.1 — Baseline Models (KNN + SVM)")
        print("="*60)
        run_baselines(X_train, y_train, X_test, y_test)

    # 5.2 + 5.3  Build & train CNN
    print("\n" + "="*60)
    print("  STEP 5.2 + 5.3 — CNN Training")
    print("="*60)
    model, device = build_model(num_classes=num_classes)
    print(f"Device: {device}")
    print(model)

    X_train_t, y_train_t = _prepare_tensors(X_train, y_train, device)
    X_test_t, y_test_t = _prepare_tensors(X_test, y_test, device)

    train_loader = _make_loader(X_train_t, y_train_t, batch_size, shuffle=True)
    test_loader = _make_loader(X_test_t, y_test_t, batch_size, shuffle=False)

    t0 = time.time()
    history = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=lr)
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.1f}s")

    # 5.4  Detailed evaluation
    print("\n" + "="*60)
    print("  STEP 5.4 — Validation")
    print("="*60)
    acc, report = full_evaluation(model, test_loader, device, num_classes)

    # 5.5  Export
    print("\n" + "="*60)
    print("  STEP 5.5 — Model Export")
    print("="*60)
    model_path = os.path.join(output_dir, "captcha_cnn.pt")
    save_model(model, model_path)

    return model, history, acc


if __name__ == "__main__":
    run_training_pipeline()
