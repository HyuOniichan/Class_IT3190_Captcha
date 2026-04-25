import cv2
import numpy as np
import os

# Resize & Normalize
def resize_image(img, size=(128, 64)):
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
def preprocess_pipeline(img):
    """
    Full preprocessing pipeline
    """
    img = resize_image(img)
    img = to_grayscale(img)
    img = denoise_image(img)
    img = threshold_image(img)
    img = morph_process(img)
    return img

def run_preprocessing_pipeline(
    input_dir="dataset/raw/1k_pbm", 
    output_dir="dataset/processed/1k_pbm"
):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        processed_img = preprocess_pipeline(img)
        cv2.imwrite(os.path.join(output_dir, img_name), processed_img)

if __name__ == "__main__":
    import os
    run_preprocessing_pipeline()