import os
import cv2
import numpy as np

from .utils import lv0_build_dataset_pipeline

def preprocess_emnist(
    input_dir="dataset/lv0_emnist",
    output_dir="output/dataset/lv0_emnist"
):
    """
    Preprocessing for dataset "lv0_emnist"
    """
    
    # Paths
    build_dir = os.path.join(output_dir, 'build')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(build_dir, exist_ok=True)
    
    # Build dataset
    lv0_build_dataset_pipeline(
        raw_dir=input_dir,
        output_dir=build_dir
    )
    
