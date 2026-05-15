import cv2
import numpy as np
import os
from typing import List, Tuple

from segment.strategies import SegmentationStrategy, Dataset1SegmentationStrategy

# NOTE: This file is kept for backward compatibility and to illustrate
# how the old procedural functions map to the new strategy pattern.
# For new development, prefer using the strategy classes directly from `strategies.py`.

def prepare_binary_image(processed_img: np.ndarray) -> np.ndarray:
    """
    Ensure image is clean binary and correct foreground (white chars on black background)
    Fix common issues with PBM images.
    """
    if processed_img.dtype != np.uint8:
        processed_img = processed_img.astype(np.uint8)

    unique_vals = np.unique(processed_img)
    if len(unique_vals) > 2:
        _, img = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY)
    else:
        img = processed_img.copy()

    white_pixels = np.sum(img == 255)
    black_pixels = np.sum(img == 0)

    if white_pixels > black_pixels:
        img = cv2.bitwise_not(img)

    if img.shape[0] > 4 and img.shape[1] > 4:
        img = img[2:-2, 2:-2]

    return img

def find_contours(processed_img: np.ndarray) -> List[np.ndarray]:
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

def extract_characters(processed_img: np.ndarray, contours: List[np.ndarray], min_area: int = 50) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Crop character regions from image
    Input: contours + processed image
    Output: list of character images + bounding boxes
    """
    characters = []
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w * h >= min_area:
            char_img = processed_img[y:y+h, x:x+w]
            characters.append(char_img)
            boxes.append((x, y, w, h))

    return characters, boxes

def sort_characters(characters: List[np.ndarray], boxes: List[Tuple[int, int, int, int]]) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Sort characters from left to right
    Input: character images + bounding boxes
    Output: ordered character images + bounding boxes
    """
    sorted_data = sorted(zip(characters, boxes), key=lambda b: b[1][0])

    sorted_chars = [item[0] for item in sorted_data]
    sorted_boxes = [item[1] for item in sorted_data]

    return sorted_chars, sorted_boxes

def resize_characters(characters: List[np.ndarray], size: Tuple[int, int] = (28, 28)) -> List[np.ndarray]:
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

def segmentation_pipeline(processed_img: np.ndarray) -> List[np.ndarray]:
    """
    Full segmentation pipeline using Dataset1SegmentationStrategy for backward compatibility.
    """
    strategy = Dataset1SegmentationStrategy()
    return strategy.segment(processed_img)

def run_segmentation_pipeline(
    input_dir: str = "dataset/processed/1k_pbm", 
    output_dir: str = "dataset/segmented/1k_pbm",
    strategy: SegmentationStrategy = Dataset1SegmentationStrategy()
):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        segmented_chars = strategy.segment(img)

        base_name = os.path.splitext(img_name)[0]
        for i, char_img in enumerate(segmented_chars):
            char_file_path = os.path.join(output_dir, f"{base_name}_char_{i}.png")
            cv2.imwrite(char_file_path, char_img)

if __name__ == "__main__":
    run_segmentation_pipeline()