# TODO: Thong ke phan bo du lieu cua dataset lv1
import os
import numpy as np


def check_oversegmented():
    """
    Kiem tra cac anh co phan doan thanh bao nhieu ky tu
    """
    dataset_dir = 'output/dataset/lv1_1k_pbm/segmented'
    char_idx_counts = [0 for _ in range(10)]

    for filename in os.listdir(dataset_dir):
        # filename format: 0008_char_0.png
        # name: 0008_char_0
        name = filename.split(".")[0]
        # char index: 0
        char_idx = int(name[-1])
        char_idx_counts[char_idx] += 1
        if char_idx > 3:
            print(f"Over-segmented: {name}")

    print(f"Total: {len(os.listdir(dataset_dir))}")
    for i in range(len(char_idx_counts)):
        print(f"Number of char {i}: {char_idx_counts[i]}")



def count_char_distribution():
    """
    Phan phoi cac ky tu
    """
    dataset_dir = 'dataset/lv1_1k_pbm'
    char_counts = [0 for _ in range(10)]
    
    for filename in os.listdir(dataset_dir):
        # filename: 0008.pbm
        # name: 0008
        name = filename.split(".")[0]
        
        for c in name:
            idx = int(c)
            char_counts[idx] += 1

    n = len(os.listdir(dataset_dir)) * 4
    probs = [(char_counts[i] * 100 / n) for i in range(10)]

    for i in range(10):
        print(f"Number of char {i}: {char_counts[i]} ({probs[i]:.2f}%)")
    
    print(f"Mean: {np.mean(probs):.2f}%")
    print(f"Variance: {np.var(probs):.2f}%")
    print(f"Standard Deviation: {np.std(probs):.2f}%")




# check_oversegmented()
count_char_distribution()
