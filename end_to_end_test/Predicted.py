"""
Predicted.py
────────────
Loads the trained CRNN model and runs inference on the validation set.
Uses greedy CTC decoding to produce predicted CAPTCHA strings.

Usage:
    python end_to_end_test/Predicted.py          (standalone demo)
    from Predicted import run_prediction         (imported by evaluate.py)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import numpy as np

import torch
import torch.nn as nn

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE    = os.path.join(SCRIPT_DIR, 'preprocessed_data.npz')
VOCAB_FILE   = os.path.join(SCRIPT_DIR, 'vocab.json')
MODEL_FILE   = os.path.join(SCRIPT_DIR, 'crnn_model.pth')
SPLIT_FILE   = os.path.join(SCRIPT_DIR, 'split_indices.npz')

# Import CRNN architecture from train.py
sys.path.insert(0, SCRIPT_DIR)
from train import CRNN


# ──────────────────────────────────────────────
# Greedy CTC decoder
# ──────────────────────────────────────────────
def greedy_decode(log_probs, index_to_char, blank=0):
    """
    Greedy CTC decoding for a single sample.

    Parameters
    ----------
    log_probs : np.ndarray, shape (T, num_classes)
        Log probabilities from the model.
    index_to_char : dict
        Mapping from index → character.
    blank : int
        Index of the CTC blank token.

    Returns
    -------
    decoded : str
        The decoded string after collapsing repeated characters
        and removing blank tokens.
    """
    # Argmax at each time step
    best_path = np.argmax(log_probs, axis=1)  # (T,)

    # Collapse consecutive duplicates and remove blanks
    decoded_chars = []
    prev = -1
    for idx in best_path:
        if idx != prev:
            if idx != blank:
                ch = index_to_char.get(int(idx), '?')
                decoded_chars.append(ch)
        prev = idx

    return ''.join(decoded_chars)


def batch_greedy_decode(log_probs_batch, index_to_char, blank=0):
    """
    Greedy CTC decoding for a batch.

    Parameters
    ----------
    log_probs_batch : np.ndarray, shape (T, B, C) or torch.Tensor

    Returns
    -------
    decoded_list : list[str]
    """
    if isinstance(log_probs_batch, torch.Tensor):
        log_probs_batch = log_probs_batch.cpu().numpy()

    T, B, C = log_probs_batch.shape
    results = []
    for b in range(B):
        decoded = greedy_decode(log_probs_batch[:, b, :], index_to_char, blank)
        results.append(decoded)
    return results


# ──────────────────────────────────────────────
# Prediction pipeline
# ──────────────────────────────────────────────
def run_prediction(batch_size=128):
    """
    Load the trained model and run prediction on the validation set.

    Returns
    -------
    results : list[dict]
        Each dict has keys: 'filename', 'source', 'true_label', 'predicted_label'
    """
    # ── Load vocabulary ──
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    num_classes = vocab['num_classes']
    charset = vocab['charset']
    index_to_char = {i + 1: ch for i, ch in enumerate(charset)}
    index_to_char[0] = ''  # blank

    # ── Load data ──
    data = np.load(DATA_FILE, allow_pickle=True)
    images      = data['images']
    raw_labels  = data['raw_labels']
    filenames   = data['filenames']
    sources     = data['sources']

    # ── Load split indices ──
    split = np.load(SPLIT_FILE)
    val_idx = split['val_idx']

    val_images    = images[val_idx]
    val_labels    = raw_labels[val_idx]
    val_filenames = filenames[val_idx]
    val_sources   = sources[val_idx]

    print(f"[Predicted] Validation set size: {len(val_idx)}")

    # ── Load model ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(num_classes=num_classes).to(device)

    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[Predicted] Loaded model from {MODEL_FILE} (epoch {checkpoint['epoch']})")

    # ── Run inference in batches ──
    all_predictions = []

    with torch.no_grad():
        for start in range(0, len(val_images), batch_size):
            end = min(start + batch_size, len(val_images))
            batch_imgs = val_images[start:end]

            # (B, 1, H, W)
            batch_tensor = torch.from_numpy(batch_imgs).unsqueeze(1).to(device)
            log_probs = model(batch_tensor)  # (T, B, C)

            predictions = batch_greedy_decode(log_probs, index_to_char, blank=0)
            all_predictions.extend(predictions)

    # ── Build results ──
    results = []
    for i in range(len(val_idx)):
        results.append({
            'filename':        str(val_filenames[i]),
            'source':          str(val_sources[i]),
            'true_label':      str(val_labels[i]),
            'predicted_label': all_predictions[i],
        })

    # ── Print some examples ──
    print(f"\n{'='*60}")
    print(f"  Sample Predictions (first 20)")
    print(f"{'='*60}")
    print(f"  {'Filename':<30s} {'True':<10s} {'Predicted':<10s} {'Match'}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*5}")
    for r in results[:20]:
        match = '✓' if r['true_label'] == r['predicted_label'] else '✗'
        print(f"  {r['filename']:<30s} {r['true_label']:<10s} {r['predicted_label']:<10s} {match}")

    correct = sum(1 for r in results if r['true_label'] == r['predicted_label'])
    print(f"\n  Quick accuracy: {correct}/{len(results)} = {correct/len(results)*100:.2f}%\n")

    return results


if __name__ == '__main__':
    results = run_prediction()
