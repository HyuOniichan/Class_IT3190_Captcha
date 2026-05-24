"""
VP segmentation hyperparameter benchmark.

This script evaluates 15 evenly distributed parameter combinations from the
search space on two datasets:
- original dataset: dataset/raw/1k_pbm
- dataset_v2: dataset_v2/processed

Each run prints:
- dataset name
- hyperparameters
- total images processed
- number of images with exactly expected segments
- correct ratio = exact_count / total
- distribution of segment counts

Reading the output:
- higher correct ratio is better.
- raw dataset expects 4 segments.
- kaggle dataset expects 5 segments.
- best settings are those with highest correct ratio and lowest variability.
"""
import os
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from segment.strategies import VerticalProjectionSegmentationStrategy

DATASETS = [
    ("raw", "dataset_v2/raw", 4),
    ("kaggle", "dataset_v2/kaggle_captcha", 5),
]

GAP_RATIOS = [0.08, 0.10, 0.12, 0.15, 0.20]
MIN_WIDTHS = [4, 6, 8]
SMOOTH_KERNELS = [5, 9, 11, 21]


def choose_combinations():
    return list(product(GAP_RATIOS, MIN_WIDTHS, SMOOTH_KERNELS))


def list_images(directory):
    if not os.path.isdir(directory):
        return []
    valid_ext = {".png", ".jpg", ".jpeg", ".pbm"}
    return sorted(
        [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in valid_ext]
    )


def _binarize_segment(segment):
    if segment is None:
        return segment
    if segment.dtype != np.uint8:
        segment = segment.astype(np.uint8)
    _, binary = cv2.threshold(segment, 127, 255, cv2.THRESH_BINARY)
    return binary


def _is_too_bright(segment, threshold=0.93):
    if segment is None or segment.size == 0:
        return True
    bright = np.count_nonzero(segment == 255)
    return (bright / float(segment.size)) > threshold


def evaluate_dataset(directory, expected_segments, strategy):
    image_files = list_images(directory)
    if not image_files:
        return None

    count_ok = 0
    segment_counts = []
    unreadable = 0
    for fname in image_files:
        path = os.path.join(directory, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            unreadable += 1
            continue

        valid_segments = []
        for segment in strategy.segment(img):
            segment = _binarize_segment(segment)
            if _is_too_bright(segment, threshold=0.93):
                continue
            valid_segments.append(segment)

        count = len(valid_segments)
        segment_counts.append(count)
        if count == expected_segments:
            count_ok += 1

    total = len(segment_counts)
    if total == 0:
        return None

    distribution = Counter(segment_counts)
    mean = float(np.mean(segment_counts))
    std = float(np.std(segment_counts))
    correct_ratio = count_ok / total
    return {
        "total_images": total,
        "correct_count": count_ok,
        "correct_ratio": correct_ratio,
        "mean_segments": mean,
        "std_segments": std,
        "distribution": dict(sorted(distribution.items())),
        "unreadable": unreadable,
    }


def print_result(dataset_name, directory, params, result):
    gap_ratio, min_width, smooth_kernel = params
    print("=" * 80)
    print(f"PARAMS: gap_ratio={gap_ratio}, min_width={min_width}, smooth_kernel={smooth_kernel}")
    print(f"DATASET: {dataset_name}")
    print(f"DIR: {directory}")
    if result is None:
        print("Result: no readable images found.")
        print("=" * 80)
        print()
        return

    correct_percent = result["correct_ratio"] * 100.0
    print(f"Images processed: {result['total_images']}")
    print(f"Correct splits: {result['correct_count']} / {result['total_images']} ({correct_percent:.2f}%)")
    print(f"Mean segments: {result['mean_segments']:.3f}")
    print(f"Std dev segments: {result['std_segments']:.3f}")
    print(f"Unreadable files skipped: {result['unreadable']}")
    print("Segment count distribution:")
    for count, freq in result["distribution"].items():
        print(f"  {count} segments: {freq}")
    print("=" * 80)
    print()


def print_top_params(dataset_name, results, top_n=5):
    print("#" * 80)
    print(f"TOP {top_n} PARAMS for {dataset_name}")
    print("#" * 80)
    for rank, item in enumerate(results[:top_n], start=1):
        params = item["params"]
        result = item["result"]
        gap_ratio, min_width, smooth_kernel = params
        print(f"{rank}. gap_ratio={gap_ratio}, min_width={min_width}, smooth_kernel={smooth_kernel}")
        if result is None:
            print("   No readable images.")
            continue
        correct_percent = result["correct_ratio"] * 100.0
        print(f"   Correct splits: {result['correct_count']} / {result['total_images']} ({correct_percent:.2f}%)")
        print(f"   Mean segments: {result['mean_segments']:.3f}, Std dev: {result['std_segments']:.3f}")
        print()


def main():
    combos = choose_combinations()
    print("Running 120 benchmark evaluations: 60 parameter sets x 2 datasets\n")
    print("Hyperparameter combinations selected:")
    for i, (gap_ratio, min_width, smooth_kernel) in enumerate(combos, start=1):
        print(f"  {i:02d}. gap_ratio={gap_ratio}, min_width={min_width}, smooth_kernel={smooth_kernel}")
    print()

    all_results = {"raw": [], "kaggle": []}

    for dataset_name, directory, expected_segments in DATASETS:
        for params in combos:
            gap_ratio, min_width, smooth_kernel = params
            strategy = VerticalProjectionSegmentationStrategy(
                char_size=(28, 28),
                min_width=min_width,
                gap_ratio=gap_ratio,
                smooth_kernel=smooth_kernel,
            )
            result = evaluate_dataset(directory, expected_segments, strategy)
            print_result(dataset_name, directory, params, result)
            all_results[dataset_name].append({
                "params": params,
                "result": result,
            })

    for dataset_name in ["raw", "kaggle"]:
        results = all_results[dataset_name]
        sorted_results = sorted(
            results,
            key=lambda item: (
                item["result"]["correct_ratio"] if item["result"] is not None else 0.0,
                -item["result"]["std_segments"] if item["result"] is not None else 0.0,
            ),
            reverse=True,
        )
        print_top_params(dataset_name, sorted_results, top_n=5)

    print("Benchmark finished.")
    print("Interpretation guide:")
    print(" - Higher correct ratio means more images were segmented into exactly the expected number of parts.")
    print(" - Raw dataset expects 4 segments; Kaggle dataset expects 5 segments.")
    print(" - Best parameters are those with the highest correct ratio, then lower variability.")


if __name__ == "__main__":
    main()
