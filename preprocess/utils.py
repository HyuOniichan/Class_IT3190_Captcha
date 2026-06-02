import os
import random
import cv2
import csv
import numpy as np





# ------------------------------
# Simple processing
# ------------------------------

# Resize & Normalize
def resize_image(img, size=(200, 72)):
    """
    Standardize image size
    Input: raw image
    Output: resized image
    """
    resized = cv2.resize(img, size)
    return resized


# Grayscale Conversion
def to_grayscale(img):
    """
    Convert image to grayscale
    Input: resized image
    Output: grayscale image (1 channel)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray


# Noise Reduction
def denoise_image(img, method="median"):
    """
    Remove noise from image
    Input: grayscale image
    Output: denoised image
    """
    if method == "median":
        return cv2.medianBlur(img, 3)
    elif method == "gaussian":
        return cv2.GaussianBlur(img, (3, 3), 0)
    else:
        return img


# Thresholding
def threshold_image(img, method="otsu"):
    """
    Convert image to binary form
    Input: denoised image
    Output: binary image
    """
    if method == "otsu":
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        th = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
    else:
        _, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    return th


# Morphological Processing
def morph_process(img, kernel_size=2):
    """
    Clean binary image using morphological operations
    Input: binary image
    Output: cleaned image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # remove small noise
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # fill gaps in characters
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed

# Full Pipeline
def simple_preprocess_pipeline(img):
    """
    Full simple preprocessing pipeline
    """
    img = resize_image(img)
    img = to_grayscale(img)
    img = denoise_image(img)
    img = threshold_image(img)
    img = morph_process(img)
    return img





# ------------------------------
# Segmentation
# ------------------------------

# Ensure binary image
def prepare_binary_image(processed_img):
    """
    Ensure image is clean binary and correct foreground (white chars on black background)
    Fix common issues with PBM images.
    """
    # 1. Convert to uint8
    if processed_img.dtype != np.uint8:
        processed_img = processed_img.astype(np.uint8)

    # 2. Ensure binary
    unique_vals = np.unique(processed_img)
    if len(unique_vals) > 2:
        _, img = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY)
    else:
        img = processed_img.copy()

    # 3. Ensure foreground (text) is white (255)
    # Check if white pixel is more -> invert
    white_pixels = np.sum(img == 255)
    black_pixels = np.sum(img == 0)

    if white_pixels > black_pixels:
        img = cv2.bitwise_not(img)

    # 4. Optional: remove border
    if img.shape[0] > 4 and img.shape[1] > 4:
        img = img[2:-2, 2:-2]

    return img

# Contour Detection
def find_contours(processed_img):
    """
    Detect character regions using OpenCV contours
    Input: processed (binary) image
    Output: contours list
    """
    contours, _ = cv2.findContours(
        processed_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


# Character Extraction
def extract_characters(processed_img, contours, min_area=50):
    """
    Crop character regions from image
    Input: contours + processed image
    Output: list of character images + bounding boxes
    """
    characters = []
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # filter noise
        if w * h >= min_area:
            char_img = processed_img[y:y+h, x:x+w]
            characters.append(char_img)
            boxes.append((x, y, w, h))

    return characters, boxes


# Character Sorting
def sort_characters(characters, boxes):
    """
    Sort characters from left to right
    Input: character images + bounding boxes
    Output: ordered character images
    """
    # sort by x coordinate
    sorted_data = sorted(zip(characters, boxes), key=lambda b: b[1][0])

    sorted_chars = [item[0] for item in sorted_data]
    sorted_boxes = [item[1] for item in sorted_data]

    return sorted_chars, sorted_boxes


# Character Resizing
def resize_characters(characters, size=(28, 28)):
    """
    Normalize character size
    Input: list of character images
    Output: resized character images
    """
    resized_chars = []

    for char in characters:
        resized = cv2.resize(char, size)
        resized_chars.append(resized)

    return resized_chars


# Full Pipeline
def segmentation_pipeline(processed_img):
    """
    Full segmentation pipeline:
    1. Ensure binary image
    2. Detect contours
    3. Extract characters
    4. Sort characters
    5. Resize characters
    """
    processed_img = prepare_binary_image(processed_img)
    contours = find_contours(processed_img)
    chars, boxes = extract_characters(processed_img, contours)
    chars, boxes = sort_characters(chars, boxes)
    chars = resize_characters(chars)
    return chars

def run_segmentation_pipeline(
    input_dir="dataset/processed/1k_pbm", 
    output_dir="dataset/segmented/1k_pbm"
):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # The segmentation_pipeline returns a list of character images
        segmented_chars = segmentation_pipeline(img)

        # Save each segmented character as a separate image
        base_name = os.path.splitext(img_name)[0]
        for i, char_img in enumerate(segmented_chars):
            char_file_path = os.path.join(output_dir, f"{base_name}_char_{i}.png")
            cv2.imwrite(char_file_path, char_img)





# ------------------------------
# Dataset builder
# ------------------------------

SPLIT_RATIO = 0.8 # 80/20
CHARSET = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
CHAR_TO_INDEX = {ch: idx for idx, ch in enumerate(CHARSET)}
INDEX_TO_CHAR = {idx: ch for idx, ch in enumerate(CHARSET)}

# Train test split
def split_train_test(
    dataset_path="dataset/lv1_1k_pbm", 
    metadata_path="output/dataset/lv1_1k_pbm/meta", 
    split_ratio=0.8
):
    os.makedirs(metadata_path, exist_ok=True)

    # Train test split
    files = [f for f in os.listdir(dataset_path) if f.endswith(".pbm")]
    random.shuffle(files)

    split_idx = int(split_ratio * len(files))
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    with open(os.path.join(metadata_path, "train.csv"), "w") as f:
        for file in train_files:
            f.write(file + "\n")

    with open(os.path.join(metadata_path, "test.csv"), "w") as f:
        for file in test_files:
            f.write(file + "\n")


# Label Parsing
def parse_labels(metadata_path="output/dataset/lv1_1k_pbm/meta"):
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


# Dataset Mapping
def build_dataset(labels, segmented_dir="output/dataset/lv1_1k_pbm/segmented"):
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
    y : list[str] - character labels ('0'-'9', 'A'-'Z')
    """
    X_list = []
    y_list = []
    skipped = 0

    for filename, chars in labels:
        stem = os.path.splitext(filename)[0]

        # Temp arrays, only use those chars if no chars are lost
        stem_X, stem_y = [], []
        isLost = False
        
        for i, ch in enumerate(chars):
            img_path = os.path.join(segmented_dir, f"{stem}_char_{i}.png")
            if not os.path.exists(img_path):
                skipped += 1
                isLost = True
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                isLost = True
                continue
            stem_X.append(img)
            stem_y.append(ch)
        
        # All chars are reserved
        if not isLost:
            X_list.extend(stem_X)
            y_list.extend(stem_y)

    X = np.array(X_list, dtype=np.uint8)
    print(f"[Dataset Mapping] Loaded {len(y_list)} samples, skipped {skipped}")
    return X, y_list


# Label Encoding
def encode_labels(y_chars):
    """
    Map character labels to integer indices.
    '0'->0 ... '9'->9, 'A'->10 ... 'Z'->35.

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


# Full pipeline
def build_dataset_pipeline(
    raw_dir="dataset/lv1_1k_pbm",
    metadata_path="output/dataset/lv1_1k_pbm/meta",
    segmented_dir="output/dataset/lv1_1k_pbm/segmented",
    output_dir="output/dataset/lv1_1k_pbm/build",
):
    """
    Execute steps 4.1 → 4.2 → 4.3 and persist the result as .npz files
    that can be loaded directly for model training.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Train Test Split
    split_train_test(raw_dir, metadata_path)

    # Label Parsing
    train_labels, test_labels = parse_labels(metadata_path)

    # Dataset Mapping
    X_train, y_train_chars = build_dataset(train_labels, segmented_dir)
    X_test, y_test_chars = build_dataset(test_labels, segmented_dir)

    # Label Encoding
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

    # return X_train, y_train, X_test, y_test


