import os
import random
# from PIL import Image
# import numpy as np
# import cv2

random.seed(36)

def process_1k_pbm_dataset(
    dataset_path="dataset/raw/1k_pbm", 
    metadata_path="dataset/meta/1k_pbm", 
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

if __name__ == '__main__':
    process_1k_pbm_dataset()
