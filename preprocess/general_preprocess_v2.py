import cv2
import os
from typing import Literal

from preprocess.strategies import PreprocessingStrategy, Dataset2PreprocessingStrategy

def run_preprocessing_pipeline_v2(
    input_dir: str = "dataset_v2/raw",
    output_dir: str = "dataset/processed/dataset_v2_processed",
    strategy: PreprocessingStrategy = Dataset2PreprocessingStrategy()
):
    """
    Runs the preprocessing pipeline for images in a given directory using a specified strategy.

    Args:
        input_dir (str): Directory containing raw images.
        output_dir (str): Directory to save processed images.
        strategy (PreprocessingStrategy): The preprocessing strategy to use.
    """
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path) # Read as is, strategy handles grayscale
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        processed_img = strategy.preprocess(img)
        cv2.imwrite(os.path.join(output_dir, img_name), processed_img)

if __name__ == "__main__":
    # Example usage for Dataset 2 preprocessing
    # You can customize the strategy parameters here
    dataset2_strategy = Dataset2PreprocessingStrategy(
        denoise_method="gaussian", # "median" || "gaussian"
        threshold_method="otsu", # "otsu" || "adaptive"
        line_removal_kernel_size=3,
        denoising_kernel_size=3
    )
    run_preprocessing_pipeline_v2(
        input_dir="dataset_v2/raw",
        output_dir="dataset/processed/dataset_v2_processed",
        strategy=dataset2_strategy
    )
