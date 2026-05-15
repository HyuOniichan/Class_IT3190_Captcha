import cv2
import os
from typing import List

from segment.strategies import SegmentationStrategy, Dataset2SegmentationStrategy

def run_segmentation_pipeline_v2(
    input_dir: str = "dataset/processed/dataset_v2_processed",
    output_dir: str = "dataset/segmented/dataset_v2_segmented",
    strategy: SegmentationStrategy = Dataset2SegmentationStrategy()
):
    """
    Runs the segmentation pipeline for images in a given directory using a specified strategy.

    Args:
        input_dir (str): Directory containing preprocessed binary images.
        output_dir (str): Directory to save segmented character images.
        strategy (SegmentationStrategy): The segmentation strategy to use.
    """
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Read as grayscale for segmentation
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        segmented_chars = strategy.segment(img)

        base_name = os.path.splitext(img_name)[0]
        for i, char_img in enumerate(segmented_chars):
            char_file_path = os.path.join(output_dir, f"{base_name}_char_{i}.png")
            cv2.imwrite(char_file_path, char_img)

if __name__ == "__main__":
    # Example usage for Dataset 2 segmentation
    dataset2_segment_strategy = Dataset2SegmentationStrategy(
        min_area=30,  # Lower threshold for noisy images
        char_size=(28, 28),
        min_aspect_ratio=0.1,  # More lenient aspect ratio
        max_aspect_ratio=3.0,
        min_height=5  # Lower minimum height
    )
    run_segmentation_pipeline_v2(
        input_dir="dataset/processed/dataset_v2_processed",
        output_dir="dataset/segmented/dataset_v2_segmented",
        strategy=dataset2_segment_strategy
    )
