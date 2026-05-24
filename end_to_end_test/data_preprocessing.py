"""
data_preprocessing.py
─────────────────────
Reads CAPTCHA images from three directories:
    • dataset_v2/kaggle_captcha
    • dataset_v2/raw
    • dataset_v2/processed

For every image the script:
    1. Converts it to grayscale.
    2. Resizes it to a standard size (IMG_HEIGHT × IMG_WIDTH).
    3. Normalises pixel values to [0, 1].
    4. Extracts the ground-truth label from the filename.

All data are gathered into NumPy arrays and persisted as
``preprocessed_data.npz`` in the same directory for downstream training.
"""

import os
import sys
import cv2
import json
import numpy as np

# Resolve project root (parent of end_to_end_test)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Insert PROJECT_ROOT to sys.path to allow importing from preprocess module
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocess.strategies import Dataset2PreprocessingStrategy

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
IMG_HEIGHT = 32
IMG_WIDTH  = 128

# Character set: 0-9 and A-Z  (index 0 is reserved for CTC blank)
CHARSET = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
CHAR_TO_INDEX = {ch: idx + 1 for idx, ch in enumerate(CHARSET)}  # 1-indexed (0 = blank)
INDEX_TO_CHAR = {idx + 1: ch for idx, ch in enumerate(CHARSET)}
INDEX_TO_CHAR[0] = ''  # CTC blank token
NUM_CLASSES  = len(CHARSET) + 1   # 36 chars + 1 blank = 37

DATA_DIRS = {
    'kaggle_captcha': os.path.join(PROJECT_ROOT, 'dataset_v2', 'kaggle_captcha'),
    'raw':            os.path.join(PROJECT_ROOT, 'dataset_v2', 'raw'),
    'processed':      os.path.join(PROJECT_ROOT, 'dataset_v2', 'processed'),
}

OUTPUT_FILE  = os.path.join(SCRIPT_DIR, 'preprocessed_data.npz')
VOCAB_FILE   = os.path.join(SCRIPT_DIR, 'vocab.json')


# ──────────────────────────────────────────────
# Label extraction
# ──────────────────────────────────────────────
def extract_label(filename: str, source: str) -> str | None:
    """
    Extract the CAPTCHA text from a filename.

    Rules
    -----
    * ``raw`` / ``processed``: first 4 characters of stem.
      e.g. ``0008_noise_0.png`` → ``0008``
    * ``kaggle_captcha``: first token when split by ``_``.
      If there is no ``_`` then the entire stem is used.
      e.g. ``2b827.png`` → ``2B827``   (5 chars)
           ``0008_noise_0.png`` → ``0008`` (4 chars)

    All labels are uppercased.
    """
    stem = os.path.splitext(filename)[0]

    # Skip auxiliary files (e.g. gan_0.png)
    if stem.startswith('gan'):
        return None

    if source in ('raw', 'processed'):
        label = stem[:4]
    else:  # kaggle_captcha
        label = stem.split('_')[0]

    label = label.upper()

    # Validate: every character must be in CHARSET
    for ch in label:
        if ch not in CHAR_TO_INDEX:
            return None

    return label


# ──────────────────────────────────────────────
# Image loading & preprocessing
# ──────────────────────────────────────────────
# Global preprocessing strategy imported from preprocess module
PREPROCESS_STRATEGY = Dataset2PreprocessingStrategy(
    size=(IMG_WIDTH, IMG_HEIGHT),  # (128, 32)
    denoise_method="median",
    threshold_method="adaptive",
    line_removal_kernel_size=3,
    denoising_kernel_size=3,
    maintain_aspect_ratio=False,
    normalize=True
)


def load_and_preprocess(path: str) -> np.ndarray | None:
    """
    Read an image, convert to grayscale, resize to
    (IMG_HEIGHT, IMG_WIDTH) and normalise to [0, 1] using preprocess module strategy.

    Uses np.fromfile + cv2.imdecode to support Unicode paths on Windows.
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    
    # Process using the imported strategy (grayscale, resize, adaptive threshold, line removal, normalize)
    img = PREPROCESS_STRATEGY.preprocess(img)
    return img


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def run_preprocessing():
    images = []
    labels = []
    filenames = []
    sources = []

    for source_name, dir_path in DATA_DIRS.items():
        if not os.path.isdir(dir_path):
            print(f"[WARNING] Directory not found, skipping: {dir_path}")
            continue

        png_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith('.png')])
        loaded = 0

        for fname in png_files:
            label = extract_label(fname, source_name)
            if label is None:
                continue

            img = load_and_preprocess(os.path.join(dir_path, fname))
            if img is None:
                continue

            images.append(img)
            labels.append(label)
            filenames.append(fname)
            sources.append(source_name)
            loaded += 1

        print(f"[Data Preprocessing] {source_name}: loaded {loaded} images from {dir_path}")

    # Convert labels to encoded sequences (variable-length, padded)
    max_label_len = max(len(lbl) for lbl in labels)
    encoded_labels = np.zeros((len(labels), max_label_len), dtype=np.int32)
    label_lengths  = np.array([len(lbl) for lbl in labels], dtype=np.int32)

    for i, lbl in enumerate(labels):
        for j, ch in enumerate(lbl):
            encoded_labels[i, j] = CHAR_TO_INDEX[ch]

    images_arr = np.array(images, dtype=np.float32)  # (N, H, W)

    print(f"\n{'='*60}")
    print(f"  Data Preprocessing Summary")
    print(f"{'='*60}")
    print(f"  Total images : {len(images_arr)}")
    print(f"  Image shape  : {images_arr.shape[1:]}")
    print(f"  Max label len: {max_label_len}")
    print(f"  Vocabulary   : {len(CHARSET)} characters")
    print(f"  Num classes   : {NUM_CLASSES} (including CTC blank)")
    print(f"{'='*60}\n")

    # Save everything
    np.savez_compressed(
        OUTPUT_FILE,
        images=images_arr,
        encoded_labels=encoded_labels,
        label_lengths=label_lengths,
        filenames=np.array(filenames, dtype=object),
        sources=np.array(sources, dtype=object),
        raw_labels=np.array(labels, dtype=object),
    )
    print(f"[Data Preprocessing] Saved preprocessed data → {OUTPUT_FILE}")

    # Save vocabulary info for other scripts
    vocab_info = {
        'charset': CHARSET,
        'char_to_index': CHAR_TO_INDEX,
        'num_classes': NUM_CLASSES,
        'img_height': IMG_HEIGHT,
        'img_width': IMG_WIDTH,
        'max_label_len': max_label_len,
    }
    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocab_info, f, indent=2)
    print(f"[Data Preprocessing] Saved vocabulary info → {VOCAB_FILE}")

    return images_arr, encoded_labels, label_lengths, filenames


if __name__ == '__main__':
    run_preprocessing()
