import cv2
import numpy as np
import os
import os

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

if __name__ == "__main__":
    run_segmentation_pipeline()