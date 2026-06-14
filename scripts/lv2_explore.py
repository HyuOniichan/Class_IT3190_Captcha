# TODO: Thong ke phan bo du lieu cua dataset lv2
import os
import numpy as np


def count_char_distribution():
    """
    Phan phoi cac ky tu
    """
    dataset_dir = 'dataset/lv2_1k_5digits'
    mapping = '0123456789abcdefghijklmnopqrstuvwxyz'
    label_to_idx = { ch: idx for idx, ch in enumerate(mapping) }
    idx_to_label = { idx: ch for idx, ch in enumerate(mapping) }
    
    char_counts = [0 for _ in range(len(mapping))]
    
    for filename in os.listdir(dataset_dir):
        # filename: 2b827.png
        # name: 2b827
        name = filename.split(".")[0]
        
        for c in name:
            idx = label_to_idx[c]
            char_counts[idx] += 1

    n = len(os.listdir(dataset_dir)) * 5
    probs = [(count * 100 / n) for count in char_counts]
    
    print(f"Total: {len(os.listdir(dataset_dir))}")
    
    for label in mapping:
        print(f"Number of char {label}: {char_counts[label_to_idx[label]]} ({probs[label_to_idx[label]]:.2f}%)")
    
    print(f"Mean: {np.mean(probs):.2f}%")
    print(f"Variance: {np.var(probs):.2f}%")
    print(f"Standard Deviation: {np.std(probs):.2f}%")
    
    
    

import cv2
image = cv2.imread('dataset/lv2_1k_5digits/2b827.png')
print(image.shape)

# count_char_distribution()


