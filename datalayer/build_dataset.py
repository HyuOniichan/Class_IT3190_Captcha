import os
import csv
import cv2
import numpy as np


CHARSET = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
CHAR_TO_INDEX = {ch: idx for idx, ch in enumerate(CHARSET)}
INDEX_TO_CHAR = {idx: ch for idx, ch in enumerate(CHARSET)}


# ──────────────────────────────────────────────
# 4.1  Label Parsing
# ──────────────────────────────────────────────

def parse_labels(metadata_path="dataset/meta/1k_pbm"):
    """
    Read train.csv / test.csv (each line is a filename like '0824.pbm'),
    extract the CAPTCHA string from the filename stem, and split it into
    individual characters.

    Returns
    -------
    train_labels : list[tuple[str, list[str]]]
        [(filename, [char0, char1, ...]), ...]
    test_labels  : same structure for test set
    """
    train_labels = _parse_split(os.path.join(metadata_path, "train.csv"))
    test_labels = _parse_split(os.path.join(metadata_path, "test.csv"))

    labels_csv_path = os.path.join(metadata_path, "labels.csv")
    _write_labels_csv(labels_csv_path, train_labels + test_labels)
    print(f"[Label Parsing] Wrote {labels_csv_path}  "
          f"(train={len(train_labels)}, test={len(test_labels)})")

    return train_labels, test_labels


def _parse_split(csv_path):
    labels = []
    with open(csv_path, "r") as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue
            stem = os.path.splitext(filename)[0]
            chars = list(stem.upper())
            labels.append((filename, chars))
    return labels


def _write_labels_csv(path, all_labels):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "char_0", "char_1", "char_2", "char_3"])
        for filename, chars in all_labels:
            stem = os.path.splitext(filename)[0]
            row = [filename, stem] + chars[:4]
            writer.writerow(row)


# ──────────────────────────────────────────────
# 4.2  Dataset Mapping
# ──────────────────────────────────────────────

def build_dataset(labels, segmented_dir="dataset/segmented/1k_pbm"):
    """
    Pair each segmented character image with its label.

    Parameters
    ----------
    labels : list[tuple[str, list[str]]]
        Output of parse_labels (one split).
    segmented_dir : str
        Directory containing files like '0824_char_0.png'.

    Returns
    -------
    X : np.ndarray, shape (N, 28, 28), dtype uint8
    y : list[str]   – character labels ('0'-'9', 'A'-'Z')
    """
    X_list = []
    y_list = []
    skipped = 0

    for filename, chars in labels:
        stem = os.path.splitext(filename)[0]
        for i, ch in enumerate(chars):
            img_path = os.path.join(segmented_dir, f"{stem}_char_{i}.png")
            if not os.path.exists(img_path):
                skipped += 1
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue
            X_list.append(img)
            y_list.append(ch)

    X = np.array(X_list, dtype=np.uint8)
    print(f"[Dataset Mapping] Loaded {len(y_list)} samples, skipped {skipped}")
    return X, y_list


# ──────────────────────────────────────────────
# 4.3  Label Encoding
# ──────────────────────────────────────────────

def encode_labels(y_chars):
    """
    Map character labels to integer indices.
    '0'→0 … '9'→9, 'A'→10 … 'Z'→35.

    Parameters
    ----------
    y_chars : list[str]

    Returns
    -------
    y_encoded : np.ndarray, shape (N,), dtype int32
    """
    y_encoded = np.array([CHAR_TO_INDEX[ch] for ch in y_chars], dtype=np.int32)

    unique = sorted(set(y_chars))
    print(f"[Label Encoding] {len(y_encoded)} labels encoded, "
          f"{len(unique)} unique classes: {unique}")
    return y_encoded


def decode_labels(y_encoded):
    """Reverse of encode_labels."""
    return [INDEX_TO_CHAR[int(idx)] for idx in y_encoded]


# ──────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────

def run_dataset_pipeline(
    metadata_path="dataset/meta/1k_pbm",
    segmented_dir="dataset/segmented/1k_pbm",
    output_dir="dataset/ready/1k_pbm",
):
    """
    Execute steps 4.1 → 4.2 → 4.3 and persist the result as .npz files
    that can be loaded directly for model training.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 4.1 Label Parsing
    train_labels, test_labels = parse_labels(metadata_path)

    # 4.2 Dataset Mapping
    X_train, y_train_chars = build_dataset(train_labels, segmented_dir)
    X_test, y_test_chars = build_dataset(test_labels, segmented_dir)

    # 4.3 Label Encoding
    y_train = encode_labels(y_train_chars)
    y_test = encode_labels(y_test_chars)

    # Persist
    train_path = os.path.join(output_dir, "train.npz")
    test_path = os.path.join(output_dir, "test.npz")
    np.savez(train_path, X=X_train, y=y_train)
    np.savez(test_path, X=X_test, y=y_test)

    print(f"[Dataset Pipeline] Saved {train_path}  "
          f"(X={X_train.shape}, y={y_train.shape})")
    print(f"[Dataset Pipeline] Saved {test_path}  "
          f"(X={X_test.shape}, y={y_test.shape})")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    run_dataset_pipeline()
