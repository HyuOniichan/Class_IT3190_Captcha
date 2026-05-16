import argparse
import csv
import os
import random
import time
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from segment.strategies import Dataset1SegmentationStrategy, Dataset2SegmentationStrategy

IMAGE_EXTS = {".pbm", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class DatasetConfig:
    name: str
    input_dir: str
    expected_chars: int
    threshold_default: str
    resize_mode: str
    min_area: int
    min_aspect_ratio: Optional[float]
    max_aspect_ratio: Optional[float]
    min_height: Optional[int]


@dataclass
class PreprocessVariant:
    name: str
    denoise: str = "none"
    denoise_kernel: int = 3
    bilateral_params: Tuple[int, int, int] = (5, 50, 50)
    threshold: str = "auto"
    morph_sequence: Tuple[str, ...] = ()
    morph_kernel: int = 2
    line_removal: str = "none"
    line_kernel: int = 3
    hough_params: Optional[Dict[str, int]] = None


def list_images(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []
    paths = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            paths.append(path)
    return sorted(paths)


def resize_for_dataset(img: np.ndarray, config: DatasetConfig) -> np.ndarray:
    if config.resize_mode == "fixed":
        return cv2.resize(img, (128, 64))
    target_height = 64
    h, w = img.shape[:2]
    if h == 0:
        return img
    target_width = max(1, int(target_height * (w / h)))
    return cv2.resize(img, (target_width, target_height))


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def denoise_image(img: np.ndarray, variant: PreprocessVariant) -> np.ndarray:
    if variant.denoise == "median":
        return cv2.medianBlur(img, variant.denoise_kernel)
    if variant.denoise == "gaussian":
        return cv2.GaussianBlur(img, (variant.denoise_kernel, variant.denoise_kernel), 0)
    if variant.denoise == "bilateral":
        d, sigma_color, sigma_space = variant.bilateral_params
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    return img


def remove_lines_morph(img: np.ndarray, kernel_size: int) -> np.ndarray:
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size * 5, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size * 5))
    temp = cv2.morphologyEx(img, cv2.MORPH_OPEN, horiz_kernel)
    img = cv2.subtract(img, temp)
    temp = cv2.morphologyEx(img, cv2.MORPH_OPEN, vert_kernel)
    return cv2.subtract(img, temp)


def remove_lines_hough(img: np.ndarray, params: Dict[str, int]) -> np.ndarray:
    edges = cv2.Canny(img, params["canny1"], params["canny2"])
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=params["hough_thresh"],
        minLineLength=params["min_line_length"],
        maxLineGap=params["max_line_gap"],
    )
    if lines is None:
        return img
    mask = np.zeros_like(img)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    return cv2.inpaint(img, mask, params["inpaint_radius"], cv2.INPAINT_TELEA)


def threshold_image(img: np.ndarray, method: str) -> np.ndarray:
    if method == "otsu":
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    if method == "otsu_inv":
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return th
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 10
        )
    if method == "adaptive_inv":
        return cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 10
        )
    if method == "simple_inv":
        _, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        return th
    _, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return th


def apply_morph_sequence(binary: np.ndarray, variant: PreprocessVariant) -> np.ndarray:
    if not variant.morph_sequence:
        return binary
    kernel = np.ones((variant.morph_kernel, variant.morph_kernel), np.uint8)
    img = binary
    for step in variant.morph_sequence:
        if step == "open":
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif step == "close":
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def preprocess_image(img: np.ndarray, variant: PreprocessVariant, config: DatasetConfig) -> np.ndarray:
    resized = resize_for_dataset(img, config)
    gray = to_grayscale(resized)
    gray = denoise_image(gray, variant)

    if variant.line_removal == "morph":
        gray = remove_lines_morph(gray, variant.line_kernel)
    elif variant.line_removal == "hough" and variant.hough_params:
        gray = remove_lines_hough(gray, variant.hough_params)

    threshold_method = variant.threshold
    if threshold_method == "auto":
        threshold_method = config.threshold_default

    binary = threshold_image(gray, threshold_method)
    binary = apply_morph_sequence(binary, variant)
    return binary


def foreground_ratio(binary: np.ndarray) -> float:
    white = float(np.sum(binary == 255))
    black = float(np.sum(binary == 0))
    total = max(1.0, white + black)
    return min(white, black) / total


def filtered_contours(binary: np.ndarray, config: DatasetConfig) -> List[np.ndarray]:
    white = np.sum(binary == 255)
    black = np.sum(binary == 0)
    img = binary if white <= black else cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < config.min_area:
            continue
        if config.min_height is not None and h < config.min_height:
            continue
        if config.min_aspect_ratio is not None and config.max_aspect_ratio is not None:
            aspect = w / float(h) if h > 0 else 0
            if not (config.min_aspect_ratio < aspect < config.max_aspect_ratio):
                continue
        filtered.append(cnt)
    return filtered


def draw_boxes(binary: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
    white = np.sum(binary == 255)
    black = np.sum(binary == 0)
    img = binary if white <= black else cv2.bitwise_not(binary)
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return canvas


def save_sample_images(
    out_dir: str,
    base_name: str,
    raw: np.ndarray,
    binary: np.ndarray,
    overlay: np.ndarray
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_raw.png"), raw)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_binary.png"), binary)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_boxes.png"), overlay)


def evaluate_variant(
    image_paths: List[str],
    variant: PreprocessVariant,
    config: DatasetConfig,
    seg_strategy,
    max_samples: Optional[int],
    sample_dir: Optional[str],
    samples_per_variant: int,
) -> Dict[str, float]:
    seg_success = 0
    seg_count_total = 0
    contour_count_total = 0
    contour_area_total = 0.0
    contour_area_samples = 0
    fg_ratio_total = 0.0

    sample_saved = 0
    for idx, path in enumerate(image_paths):
        if max_samples is not None and idx >= max_samples:
            break
        img = cv2.imread(path)
        if img is None:
            continue

        binary = preprocess_image(img, variant, config)
        seg_chars = seg_strategy.segment(binary)
        seg_count = len(seg_chars)

        seg_count_total += seg_count
        if seg_count == config.expected_chars:
            seg_success += 1

        fg_ratio_total += foreground_ratio(binary)

        contours = filtered_contours(binary, config)
        contour_count_total += len(contours)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            contour_area_total += float(np.mean(areas))
            contour_area_samples += 1

        if sample_dir and sample_saved < samples_per_variant:
            base_name = os.path.splitext(os.path.basename(path))[0]
            resized = resize_for_dataset(img, config)
            overlay = draw_boxes(binary, contours)
            save_sample_images(sample_dir, base_name, resized, binary, overlay)
            sample_saved += 1

    count = min(len(image_paths), max_samples) if max_samples else len(image_paths)
    count = max(1, count)
    return {
        "num_images": float(count),
        "seg_success_rate": seg_success / count,
        "seg_count_mean": seg_count_total / count,
        "contour_count_mean": contour_count_total / count,
        "contour_area_mean": (contour_area_total / contour_area_samples) if contour_area_samples else 0.0,
        "foreground_ratio_mean": fg_ratio_total / count,
    }


def build_variants(stage: str) -> List[PreprocessVariant]:
    variants: List[PreprocessVariant] = []

    if stage in {"filters", "all"}:
        variants.extend([
            PreprocessVariant(name="gaussian_k3", denoise="gaussian", denoise_kernel=3),
            PreprocessVariant(name="gaussian_k5", denoise="gaussian", denoise_kernel=5),
            PreprocessVariant(name="median_k3", denoise="median", denoise_kernel=3),
            PreprocessVariant(name="median_k5", denoise="median", denoise_kernel=5),
            PreprocessVariant(name="bilateral_d5", denoise="bilateral", bilateral_params=(5, 50, 50)),
            PreprocessVariant(name="bilateral_d7", denoise="bilateral", bilateral_params=(7, 75, 75)),
        ])

    if stage in {"morph", "all"}:
        variants.extend([
            PreprocessVariant(name="open_k2", morph_sequence=("open",), morph_kernel=2),
            PreprocessVariant(name="open_k3", morph_sequence=("open",), morph_kernel=3),
            PreprocessVariant(name="close_k2", morph_sequence=("close",), morph_kernel=2),
            PreprocessVariant(name="close_k3", morph_sequence=("close",), morph_kernel=3),
            PreprocessVariant(name="open_close_k2", morph_sequence=("open", "close"), morph_kernel=2),
            PreprocessVariant(name="close_open_k2", morph_sequence=("close", "open"), morph_kernel=2),
        ])

    if stage in {"lines", "all"}:
        variants.extend([
            PreprocessVariant(name="line_morph_k3", line_removal="morph", line_kernel=3),
            PreprocessVariant(name="line_morph_k5", line_removal="morph", line_kernel=5),
            PreprocessVariant(
                name="line_hough_s1",
                line_removal="hough",
                hough_params={
                    "canny1": 50,
                    "canny2": 150,
                    "hough_thresh": 80,
                    "min_line_length": 20,
                    "max_line_gap": 5,
                    "inpaint_radius": 3,
                },
            ),
            PreprocessVariant(
                name="line_hough_s2",
                line_removal="hough",
                hough_params={
                    "canny1": 100,
                    "canny2": 200,
                    "hough_thresh": 120,
                    "min_line_length": 30,
                    "max_line_gap": 8,
                    "inpaint_radius": 3,
                },
            ),
        ])

    if stage in {"combo", "all"}:
        variants.extend([
            PreprocessVariant(
                name="combo_median_k3_open_k2",
                denoise="median",
                denoise_kernel=3,
                morph_sequence=("open",),
                morph_kernel=2,
            ),
            PreprocessVariant(
                name="combo_median_k3_open_k2_hough_s1",
                denoise="median",
                denoise_kernel=3,
                morph_sequence=("open",),
                morph_kernel=2,
                line_removal="hough",
                hough_params={
                    "canny1": 50,
                    "canny2": 150,
                    "hough_thresh": 80,
                    "min_line_length": 20,
                    "max_line_gap": 5,
                    "inpaint_radius": 3,
                },
            ),
            PreprocessVariant(
                name="combo_gaussian_k3_open_k2",
                denoise="gaussian",
                denoise_kernel=3,
                morph_sequence=("open",),
                morph_kernel=2,
            ),
        ])

    return variants


def run_benchmark(
    datasets: Iterable[DatasetConfig],
    stage: str,
    max_images: Optional[int],
    out_dir: str,
    save_samples: bool,
    samples_per_variant: int,
) -> None:
    variants = build_variants(stage)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"preprocess_benchmark_{stage}.csv")

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dataset",
                "variant",
                "num_images",
                "seg_success_rate",
                "seg_count_mean",
                "contour_count_mean",
                "contour_area_mean",
                "foreground_ratio_mean",
                "time_sec",
            ],
        )
        writer.writeheader()

        for config in datasets:
            image_paths = list_images(config.input_dir)
            if not image_paths:
                print(f"[Skip] No images in {config.input_dir}")
                continue

            seg_strategy = (
                Dataset1SegmentationStrategy()
                if config.name == "dataset1"
                else Dataset2SegmentationStrategy()
            )

            for variant in variants:
                start = time.time()
                sample_dir = None
                if save_samples:
                    sample_dir = os.path.join(out_dir, "samples", config.name, variant.name)

                metrics = evaluate_variant(
                    image_paths,
                    variant,
                    config,
                    seg_strategy,
                    max_images,
                    sample_dir,
                    samples_per_variant,
                )
                elapsed = time.time() - start

                row = {
                    "dataset": config.name,
                    "variant": variant.name,
                    **metrics,
                    "time_sec": elapsed,
                }
                writer.writerow(row)
                print(
                    f"[{config.name}] {variant.name}: "
                    f"seg_success={metrics['seg_success_rate']:.3f}, "
                    f"seg_count={metrics['seg_count_mean']:.2f}, "
                    f"contours={metrics['contour_count_mean']:.2f}"
                )

    print(f"Results saved to {csv_path}")


def build_dataset_configs(dataset2_dir: Optional[str]) -> List[DatasetConfig]:
    configs = [
        DatasetConfig(
            name="dataset1",
            input_dir="dataset/raw/1k_pbm",
            expected_chars=4,
            threshold_default="otsu",
            resize_mode="fixed",
            min_area=50,
            min_aspect_ratio=None,
            max_aspect_ratio=None,
            min_height=None,
        ),
        DatasetConfig(
            name="dataset2",
            input_dir=dataset2_dir or "dataset_v2/raw",
            expected_chars=4,
            threshold_default="adaptive_inv",
            resize_mode="height",
            min_area=30,
            min_aspect_ratio=0.1,
            max_aspect_ratio=3.0,
            min_height=5,
        ),
    ]
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess benchmarking (experimental)")
    parser.add_argument(
        "--stage",
        choices=["filters", "morph", "lines", "combo", "all"],
        default="filters",
    )
    parser.add_argument("--dataset", choices=["1", "2", "all"], default="all")
    parser.add_argument("--dataset2-dir", default=None, help="Override dataset 2 input dir")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--out-dir", default="experiments/output")
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--samples-per-variant", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    configs = build_dataset_configs(args.dataset2_dir)
    if args.dataset == "1":
        configs = [configs[0]]
    elif args.dataset == "2":
        configs = [configs[1]]

    run_benchmark(
        datasets=configs,
        stage=args.stage,
        max_images=args.max_images,
        out_dir=args.out_dir,
        save_samples=args.save_samples,
        samples_per_variant=args.samples_per_variant,
    )


if __name__ == "__main__":
    main()
