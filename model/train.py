import os
import time
import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

from model.cnn import build_model
from model.baseline import run_baselines
from datalayer.build_dataset import INDEX_TO_CHAR
from model.multi_label import CaptchaMultiLabelCNN
from model.crnn import CaptchaCRNN


# ──────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────

class CaptchaDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed NumPy arrays for sequence/whole-image models."""
    def __init__(self, images, encoded_labels, label_lengths):
        """
        Parameters
        ----------
        images : np.ndarray, shape (N, H, W) or (N, 1, H, W)
        encoded_labels : np.ndarray, shape (N, max_len)
        label_lengths : np.ndarray, shape (N,)
        """
        self.images = images
        self.encoded_labels = encoded_labels
        self.label_lengths = label_lengths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 2:
            img_t = torch.from_numpy(img).float().unsqueeze(0)
        else:
            img_t = torch.from_numpy(img).float()
            
        # Ensure values are normalized if they are in 0-255 range
        if img_t.max() > 1.0:
            img_t = img_t / 255.0
            
        lbl = torch.from_numpy(self.encoded_labels[idx]).long()
        lbl_len = torch.tensor(self.label_lengths[idx], dtype=torch.int32)
        return img_t, lbl, lbl_len


# ──────────────────────────────────────────────
# Mappings & Loaders for Whole Images
# ──────────────────────────────────────────────

CHARSET = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
CRNN_CHAR_TO_INDEX = {ch: idx + 1 for idx, ch in enumerate(CHARSET)}  # 1-indexed (0 = blank)
CRNN_INDEX_TO_CHAR = {idx + 1: ch for idx, ch in enumerate(CHARSET)}
CRNN_INDEX_TO_CHAR[0] = ''

ML_CHAR_TO_INDEX = {ch: idx for idx, ch in enumerate(CHARSET)}  # 0-indexed
ML_INDEX_TO_CHAR = {idx: ch for idx, ch in enumerate(CHARSET)}


def extract_label_from_filename(filename):
    """Extract label from filename stem (e.g. 0824_orig.png -> 0824)."""
    stem = os.path.splitext(filename)[0]
    if stem.startswith('gan'):
        return None
    # Split by underscore
    parts = stem.split('_')
    if len(parts) > 0:
        label = parts[0]
        if len(label) >= 4:
            return label.upper()
    return stem.upper()


def load_captcha_dataset_from_folder(img_dir, csv_path, char_to_index, preprocess_fn=None):
    """
    Load CAPTCHA images and labels directly from a directory and matching CSV split file.
    
    Parameters
    ----------
    img_dir : str
        Directory containing the preprocessed or raw images (e.g., "dataset/processed/1k_pbm").
    csv_path : str
        Path to the train.csv or test.csv file containing the list of filenames.
    char_to_index : dict
        Vocabulary mapping.
    preprocess_fn : callable, optional
        Preprocessing function to apply if loading raw images.
    """
    images = []
    labels = []
    lengths = []
    
    if not os.path.exists(csv_path):
        print(f"[Warning] CSV file {csv_path} not found.")
        return None
        
    with open(csv_path, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]
        
    for fname in filenames:
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            base, ext = os.path.splitext(fname)
            for alt_ext in ['.png', '.pbm', '.jpg', '.jpeg']:
                alt_path = os.path.join(img_dir, base + alt_ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
                    
        if not os.path.exists(img_path):
            continue
            
        label_text = extract_label_from_filename(fname)
        if label_text is None:
            continue
            
        valid = True
        encoded = []
        for ch in label_text:
            if ch not in char_to_index:
                valid = False
                break
            encoded.append(char_to_index[ch])
            
        if not valid:
            continue
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        if preprocess_fn is not None:
            img = preprocess_fn(img)
            
        images.append(img)
        labels.append(encoded)
        lengths.append(len(encoded))
        
    if len(images) == 0:
        return None
        
    max_len = max(lengths)
    padded_labels = np.zeros((len(labels), max_len), dtype=np.int32)
    for i, lbl in enumerate(labels):
        padded_labels[i, :len(lbl)] = lbl
        
    images_arr = np.array(images, dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.int32)
    
    return CaptchaDataset(images_arr, padded_labels, lengths_arr)


def get_loaders_for_whole_images(
    processed_img_dir="dataset/processed/1k_pbm",
    metadata_dir="dataset/meta/1k_pbm",
    model_type="crnn",
    batch_size=64,
    preprocess_fn=None,
):
    """
    Build train and validation DataLoaders using processed whole images in the dataset folder.
    """
    char_to_index = CRNN_CHAR_TO_INDEX if model_type.lower() == "crnn" else ML_CHAR_TO_INDEX
    
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "test.csv")
    
    train_ds = load_captcha_dataset_from_folder(
        processed_img_dir, train_csv, char_to_index, preprocess_fn
    )
    val_ds = load_captcha_dataset_from_folder(
        processed_img_dir, val_csv, char_to_index, preprocess_fn
    )
    
    if train_ds is None or val_ds is None:
        raise ValueError("Failed to load train or validation dataset. Please verify folder paths and metadata CSV files.")
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader





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


# ──────────────────────────────────────────────
# Multi-Label Training
# ──────────────────────────────────────────────

def train_multi_label(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=1e-3,
):
    """
    Train a multi-label CNN (whole image, multi-head output).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_char_acc": [], "train_word_acc": [], 
               "val_loss": [], "val_char_acc": [], "val_word_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_chars = 0
        correct_captchas = 0
        total_chars = 0
        total_captchas = 0

        for X_batch, y_batch, _ in train_loader:
            # X_batch: (B, 1, H, W)
            # y_batch: (B, num_chars)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)  # (B, num_chars, num_classes)
            
            # Compute cross entropy over each character position
            loss = criterion(outputs.view(-1, model.num_classes), y_batch.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            
            # Accuracy computation
            preds = outputs.argmax(dim=2)  # (B, num_chars)
            char_matches = (preds == y_batch)
            correct_chars += char_matches.sum().item()
            total_chars += y_batch.numel()

            correct_captchas += char_matches.all(dim=1).sum().item()
            total_captchas += X_batch.size(0)

        train_loss = running_loss / total_captchas
        train_char_acc = correct_chars / total_chars
        train_word_acc = correct_captchas / total_captchas

        # validate
        val_loss, val_char_acc, val_word_acc = evaluate_multi_label(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_char_acc"].append(train_char_acc)
        history["train_word_acc"].append(train_word_acc)
        history["val_loss"].append(val_loss)
        history["val_char_acc"].append(val_char_acc)
        history["val_word_acc"].append(val_word_acc)

        print(f"Epoch {epoch:>3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_char_acc={train_char_acc:.4f}  train_word_acc={train_word_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_char_acc={val_char_acc:.4f}  val_word_acc={val_word_acc:.4f}")

    return history


def evaluate_multi_label(model, loader, device, criterion=None):
    """Return (loss, char_accuracy, word_accuracy) for a multi-label model."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    running_loss = 0.0
    correct_chars = 0
    correct_captchas = 0
    total_chars = 0
    total_captchas = 0

    with torch.no_grad():
        for X_batch, y_batch, _ in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1, model.num_classes), y_batch.view(-1))

            running_loss += loss.item() * X_batch.size(0)
            
            preds = outputs.argmax(dim=2)
            char_matches = (preds == y_batch)
            correct_chars += char_matches.sum().item()
            total_chars += y_batch.numel()

            correct_captchas += char_matches.all(dim=1).sum().item()
            total_captchas += X_batch.size(0)

    return running_loss / total_captchas, correct_chars / total_chars, correct_captchas / total_captchas



# ──────────────────────────────────────────────
# CRNN Training
# ──────────────────────────────────────────────

def train_crnn(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=1e-3,
    index_to_char=None,
):
    """
    Train the CRNN model with CTCLoss.
    """
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    history = {"train_loss": [], "val_loss": [], "val_word_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for imgs, labels, label_lens in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)
            B = imgs.size(0)

            optimizer.zero_grad()
            log_probs = model(imgs)  # (T, B, C)
            T = log_probs.size(0)
            input_lengths = torch.full((B,), T, dtype=torch.int32, device=device)

            # Flatten targets for CTC Loss
            targets = []
            for i in range(B):
                targets.append(labels[i, :label_lens[i]])
            targets = torch.cat(targets).to(device)

            loss = criterion(log_probs, targets, input_lengths, label_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * B
            total_samples += B

        train_loss = running_loss / total_samples

        # Validation
        val_loss, val_word_acc = evaluate_crnn(model, val_loader, device, criterion, index_to_char)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_word_acc"].append(val_word_acc)

        print(f"Epoch {epoch:>3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_word_acc={val_word_acc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

    return history


def evaluate_crnn(model, loader, device, criterion=None, index_to_char=None):
    """Return (loss, word_accuracy) on the given loader for CRNN."""
    if criterion is None:
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct_captchas = 0

    with torch.no_grad():
        for imgs, labels, label_lens in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)
            B = imgs.size(0)

            log_probs = model(imgs)  # (T, B, C)
            T = log_probs.size(0)
            input_lengths = torch.full((B,), T, dtype=torch.int32, device=device)

            # Flatten targets for CTC Loss
            targets = []
            for i in range(B):
                targets.append(labels[i, :label_lens[i]])
            targets = torch.cat(targets).to(device)

            loss = criterion(log_probs, targets, input_lengths, label_lens)
            running_loss += loss.item() * B
            total_samples += B

            # Greedy decoding for accuracy check
            if index_to_char is not None:
                log_probs_np = log_probs.cpu().numpy()
                for b in range(B):
                    pred_str = _greedy_decode_single(log_probs_np[:, b, :], index_to_char, blank=0)
                    
                    target_indices = labels[b, :label_lens[b]].cpu().numpy()
                    target_str = "".join([index_to_char.get(int(idx), "") for idx in target_indices])
                    
                    if pred_str == target_str:
                        correct_captchas += 1

    val_loss = running_loss / total_samples
    val_word_acc = correct_captchas / total_samples if index_to_char is not None else 0.0
    return val_loss, val_word_acc


def _greedy_decode_single(log_probs, index_to_char, blank=0):
    """Greedy CTC decoding for a single sample."""
    best_path = np.argmax(log_probs, axis=1)
    decoded_chars = []
    prev = -1
    for idx in best_path:
        if idx != prev:
            if idx != blank:
                decoded_chars.append(index_to_char.get(int(idx), ''))
        prev = idx
    return "".join(decoded_chars)





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
    print(f"[Model Export] Saved model -> {path}")


def load_model(path, num_classes=36, device=None):
    """Load a previously saved model."""
    model, device = build_model(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    print(f"[Model Export] Loaded model <- {path}")
    return model, device


def save_multi_label_model(model, path, epoch=None, val_loss=None):
    """Save CaptchaMultiLabelCNN state dict and architecture parameters."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'num_classes': model.num_classes,
        'num_chars': model.num_chars,
        'epoch': epoch,
        'val_loss': val_loss,
    }
    torch.save(state, path)
    print(f"[Model Export] Saved Multi-Label model -> {path}")


def load_multi_label_model(path, device=None):
    """Load CaptchaMultiLabelCNN model from saved checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    num_classes = checkpoint.get('num_classes', 36)
    num_chars = checkpoint.get('num_chars', 5)
    
    model = CaptchaMultiLabelCNN(num_classes=num_classes, num_chars=num_chars).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[Model Export] Loaded Multi-Label model <- {path} (epoch: {checkpoint.get('epoch')}, val_loss: {checkpoint.get('val_loss')})")
    return model, device


def save_crnn_model(model, path, epoch=None, val_loss=None):
    """Save CaptchaCRNN state dict and architecture parameters."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'num_classes': model.fc.out_features,
        'epoch': epoch,
        'val_loss': val_loss,
    }
    torch.save(state, path)
    print(f"[Model Export] Saved CRNN model -> {path}")


def load_crnn_model(path, device=None):
    """Load CaptchaCRNN model from saved checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    num_classes = checkpoint.get('num_classes', 37)
    
    model = CaptchaCRNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[Model Export] Loaded CRNN model <- {path} (epoch: {checkpoint.get('epoch')}, val_loss: {checkpoint.get('val_loss')})")
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
    model_name="captcha_cnn.pt",
):
    """
    End-to-end: load data → baselines → CNN train → evaluate → export.
    """
    # ---- Load data ----
    train_data = np.load(os.path.join(data_dir, "train.npz"))
    test_data = np.load(os.path.join(data_dir, "test.npz"))
    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    num_classes = len(INDEX_TO_CHAR)
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
    model_path = os.path.join(output_dir, model_name)
    save_model(model, model_path)

    return model, history, acc


def run_multi_label_training_pipeline(
    processed_img_dir="dataset/processed/1k_pbm",
    metadata_dir="dataset/meta/1k_pbm",
    output_dir="model/saved",
    epochs=20,
    batch_size=64,
    lr=1e-3,
    model_name="captcha_multi_label.pt",
):
    """
    End-to-end multi-label training pipeline.
    """
    print("\n" + "="*60)
    print("  MULTI-LABEL CNN TRAINING PIPELINE")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    img_dir = processed_img_dir
    if not os.path.exists(img_dir):
        raw_dir = processed_img_dir.replace("processed", "raw")
        if os.path.exists(raw_dir):
            img_dir = raw_dir
            print(f"[Multi-Label] Processed dir not found. Using raw dir: {img_dir}")
            
    train_loader, val_loader = get_loaders_for_whole_images(
        processed_img_dir=img_dir,
        metadata_dir=metadata_dir,
        model_type="multi_label",
        batch_size=batch_size,
        preprocess_fn=lambda x: cv2.resize(x, (128, 64))
    )
    
    num_classes = 36
    num_chars = train_loader.dataset.encoded_labels.shape[1]
    print(f"Num classes: {num_classes}, Num chars (sequence length): {num_chars}")
    
    model = CaptchaMultiLabelCNN(num_classes=num_classes, num_chars=num_chars).to(device)
    print(model)
    
    t0 = time.time()
    history = train_multi_label(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.1f}s")
    
    model_path = os.path.join(output_dir, model_name)
    val_loss = history["val_loss"][-1]
    save_multi_label_model(model, model_path, epoch=epochs, val_loss=val_loss)
    
    return model, history


def run_crnn_training_pipeline(
    processed_img_dir="dataset/processed/1k_pbm",
    metadata_dir="dataset/meta/1k_pbm",
    output_dir="model/saved",
    epochs=20,
    batch_size=64,
    lr=1e-3,
    model_name="captcha_crnn.pt",
):
    """
    End-to-end CRNN training pipeline.
    """
    print("\n" + "="*60)
    print("  CRNN TRAINING PIPELINE")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    img_dir = processed_img_dir
    if not os.path.exists(img_dir):
        raw_dir = processed_img_dir.replace("processed", "raw")
        if os.path.exists(raw_dir):
            img_dir = raw_dir
            print(f"[CRNN] Processed dir not found. Using raw dir: {img_dir}")
            
    train_loader, val_loader = get_loaders_for_whole_images(
        processed_img_dir=img_dir,
        metadata_dir=metadata_dir,
        model_type="crnn",
        batch_size=batch_size,
        preprocess_fn=lambda x: cv2.resize(x, (128, 32))
    )
    
    num_classes = 37
    
    model = CaptchaCRNN(num_classes=num_classes).to(device)
    print(model)
    
    t0 = time.time()
    history = train_crnn(
        model, 
        train_loader, 
        val_loader, 
        device, 
        epochs=epochs, 
        lr=lr, 
        index_to_char=CRNN_INDEX_TO_CHAR
    )
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.1f}s")
    
    model_path = os.path.join(output_dir, model_name)
    val_loss = history["val_loss"][-1]
    save_crnn_model(model, model_path, epoch=epochs, val_loss=val_loss)
    
    return model, history


if __name__ == "__main__":
    run_training_pipeline()
