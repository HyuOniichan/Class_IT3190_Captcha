import os
import cv2
import numpy as np

from .utils import simple_preprocess_pipeline, segmentation_pipeline, lv1_build_dataset_pipeline

def preprocess_1k_pbm(
    input_dir="dataset/lv1_1k_pbm",
    output_dir="output/dataset/lv1_1k_pbm"
):
    """
    Preprocessing for dataset "lv1_1k_pbm"
    """
    
    # Paths
    preprocessed_dir = os.path.join(output_dir, 'preprocessed')
    segmented_dir = os.path.join(output_dir, 'segmented')
    meta_dir = os.path.join(output_dir, 'meta')
    build_dir = os.path.join(output_dir, 'build')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(segmented_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(build_dir, exist_ok=True)
    
    # Simple preprocess
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        processed_img = simple_preprocess_pipeline(img)
        cv2.imwrite(os.path.join(preprocessed_dir, img_name), processed_img)
    
    # Segmentation
    for img_name in os.listdir(preprocessed_dir):
        img_path = os.path.join(preprocessed_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # The segmentation_pipeline returns a list of character images
        segmented_chars = segmentation_pipeline(img)

        # Save each segmented character as a separate image
        base_name = os.path.splitext(img_name)[0]
        for i, char_img in enumerate(segmented_chars):
            char_file_path = os.path.join(segmented_dir, f"{base_name}_char_{i}.png")
            cv2.imwrite(char_file_path, char_img)
    
    # Build dataset
    lv1_build_dataset_pipeline(
        raw_dir=input_dir,
        metadata_path=meta_dir,
        segmented_dir=segmented_dir,
        output_dir=build_dir
    )
    
