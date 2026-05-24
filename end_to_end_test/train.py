"""
train.py
────────
Defines a CRNN (Convolutional Recurrent Neural Network) model and trains it
with CTC loss on preprocessed CAPTCHA data.

Usage:
    python end_to_end_test/train.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE    = os.path.join(SCRIPT_DIR, 'preprocessed_data.npz')
VOCAB_FILE   = os.path.join(SCRIPT_DIR, 'vocab.json')
MODEL_FILE   = os.path.join(SCRIPT_DIR, 'crnn_model.pth')


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class CaptchaDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed NumPy arrays."""

    def __init__(self, images, encoded_labels, label_lengths):
        """
        Parameters
        ----------
        images : np.ndarray, shape (N, H, W), float32 in [0, 1]
        encoded_labels : np.ndarray, shape (N, max_len), int32
        label_lengths : np.ndarray, shape (N,), int32
        """
        self.images = images
        self.encoded_labels = encoded_labels
        self.label_lengths = label_lengths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # (1, H, W) – single-channel grayscale
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)
        lbl = torch.from_numpy(self.encoded_labels[idx].astype(np.int32))
        lbl_len = torch.tensor(self.label_lengths[idx], dtype=torch.int32)
        return img, lbl, lbl_len


# ──────────────────────────────────────────────
# CRNN Model
# ──────────────────────────────────────────────
class CRNN(nn.Module):
    """
    CRNN for sequence recognition (whole-image, no segmentation).

    Architecture
    ────────────
    CNN  :  5 conv layers (+ BN + ReLU + pool) to extract spatial features.
            Input  (B, 1, 32, 128) → Output (B, 256, 1, 32)
    Map  :  Squeeze height dim → (B, 256, 32) → permute → (32, B, 256)
    RNN  :  2-layer bidirectional LSTM → (32, B, 256)
    FC   :  Linear(256, num_classes)  → (32, B, num_classes)

    The sequence length T = 32 corresponds to the width dimension after
    all pooling layers.  This is suitable for CTC decoding.
    """

    def __init__(self, num_classes: int = 37):
        super().__init__()

        # ─── CNN feature extractor ───
        self.cnn = nn.Sequential(
            # Block 1: (1, 32, 128) → (64, 16, 64)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (64, 16, 64) → (128, 8, 32)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (128, 8, 32) → (256, 4, 32)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # pool height only

            # Block 4: (256, 4, 32) → (256, 2, 32)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # pool height only

            # Block 5: (256, 2, 32) → (256, 1, 32)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # pool height only
        )

        # ─── RNN sequence modelling ───
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.3,
        )

        # ─── Classifier ───
        self.fc = nn.Linear(256, num_classes)  # 128*2 (bidirectional)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, 1, 32, 128)

        Returns
        -------
        log_probs : (T, B, num_classes)   – log-softmax output for CTC
        """
        # CNN
        conv = self.cnn(x)                     # (B, 256, 1, 32)
        conv = conv.squeeze(2)                 # (B, 256, 32)
        conv = conv.permute(2, 0, 1)           # (T=32, B, 256)

        # RNN
        rnn_out, _ = self.rnn(conv)            # (T, B, 256)

        # Classifier
        output = self.fc(rnn_out)              # (T, B, num_classes)
        log_probs = torch.nn.functional.log_softmax(output, dim=2)
        return log_probs


# ──────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for imgs, labels, label_lens in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        label_lens = label_lens.to(device)

        log_probs = model(imgs)                          # (T, B, C)
        T, B, _ = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.int32).to(device)

        # Flatten targets for CTC
        targets = []
        for i in range(B):
            targets.append(labels[i, :label_lens[i]])
        targets = torch.cat(targets).to(device)

        loss = criterion(log_probs, targets, input_lengths, label_lens)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to stabilise training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for imgs, labels, label_lens in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)

            log_probs = model(imgs)
            T, B, _ = log_probs.shape
            input_lengths = torch.full((B,), T, dtype=torch.int32).to(device)

            targets = []
            for i in range(B):
                targets.append(labels[i, :label_lens[i]])
            targets = torch.cat(targets).to(device)

            loss = criterion(log_probs, targets, input_lengths, label_lens)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────
def run_training(epochs=30, batch_size=64, lr=1e-3, val_split=0.2):
    """
    Full training pipeline: load data → split → build model → train → save.
    """
    # ── Load preprocessed data ──
    print(f"[Train] Loading data from {DATA_FILE} ...")
    data = np.load(DATA_FILE, allow_pickle=True)
    images         = data['images']           # (N, 32, 128)
    encoded_labels = data['encoded_labels']   # (N, max_len)
    label_lengths  = data['label_lengths']    # (N,)

    # ── Load vocabulary ──
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    num_classes = vocab['num_classes']

    N = len(images)
    print(f"[Train] Total samples: {N}, Num classes: {num_classes}")

    # ── Train / Validation split ──
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(N * (1 - val_split))
    train_idx = indices[:split]
    val_idx   = indices[split:]

    train_ds = CaptchaDataset(images[train_idx], encoded_labels[train_idx], label_lengths[train_idx])
    val_ds   = CaptchaDataset(images[val_idx],   encoded_labels[val_idx],   label_lengths[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Train] Train set: {len(train_ds)}, Val set: {len(val_ds)}")

    # ── Save split indices for later use by Predicted.py ──
    np.savez(os.path.join(SCRIPT_DIR, 'split_indices.npz'),
             train_idx=train_idx, val_idx=val_idx)
    print(f"[Train] Saved split indices → split_indices.npz")

    # ── Build model ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(num_classes=num_classes).to(device)
    print(f"[Train] Device: {device}")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Total parameters: {total_params:,}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ── Training loop ──
    best_val_loss = float('inf')
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  Starting training — {epochs} epochs")
    print(f"{'='*70}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch:>3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"lr={current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'epoch': epoch,
                'val_loss': val_loss,
            }, MODEL_FILE)
            print(f"  ★ Saved best model (val_loss={val_loss:.4f}) → {MODEL_FILE}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Training finished in {elapsed:.1f}s")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")

    return model


if __name__ == '__main__':
    # Default to 5 epochs for CPU training to keep the wait time minimal
    # but allow specifying via command line: python train.py <epochs>
    epochs = 5
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            pass
    run_training(epochs=epochs)
