
from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import Tuple, List, Literal

class PreprocessingStrategy(ABC):
    """Abstract base class for preprocessing strategies."""

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a series of preprocessing steps to an image.

        Args:
            image (np.ndarray): The input image (raw or partially processed).

        Returns:
            np.ndarray: The fully preprocessed image.
        """
        pass

class Dataset1PreprocessingStrategy(PreprocessingStrategy):
    """
    Preprocessing strategy for Dataset 1 (simple, clean images).
    This strategy includes resizing, grayscale conversion, basic denoising,
    Otsu's thresholding, and morphological operations.
    """
    def __init__(self, size: Tuple[int, int] = (128, 64),
                 denoise_method: Literal["median", "gaussian", "none"] = "median",
                 threshold_method: Literal["otsu", "adaptive", "simple"] = "otsu",
                 morph_kernel_size: int = 2):
        """
        Initializes the Dataset1PreprocessingStrategy.

        Args:
            size (Tuple[int, int]): Target size for resizing images. Defaults to (128, 64).
            denoise_method (Literal["median", "gaussian", "none"]): Denoising method to use. Defaults to "median".
            threshold_method (Literal["otsu", "adaptive", "simple"]): Thresholding method. Defaults to "otsu".
            morph_kernel_size (int): Kernel size for morphological operations. Defaults to 2.
        """
        self.size = size
        self.denoise_method = denoise_method
        self.threshold_method = threshold_method
        self.morph_kernel_size = morph_kernel_size

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Standardize image size."""
        return cv2.resize(img, self.size)

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _denoise_image(self, img: np.ndarray) -> np.ndarray:
        """Remove noise from image using the specified method."""
        if self.denoise_method == "median":
            return cv2.medianBlur(img, 3)
        elif self.denoise_method == "gaussian":
            return cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def _threshold_image(self, img: np.ndarray) -> np.ndarray:
        """Convert image to binary form using the specified method."""
        if self.threshold_method == "otsu":
            _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.threshold_method == "adaptive":
            th = cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        else:
            _, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return th

    def _morph_process(self, img: np.ndarray) -> np.ndarray:
        """Clean binary image using morphological operations."""
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the Dataset 1 specific preprocessing pipeline.
        """
        img = self._resize_image(image)
        img = self._to_grayscale(img)
        img = self._denoise_image(img)
        img = self._threshold_image(img)
        img = self._morph_process(img)
        return img

class Dataset2PreprocessingStrategy(PreprocessingStrategy):
    """
    Preprocessing strategy for Dataset 2 (lightly noisy images with lines and dots).
    This strategy includes advanced noise reduction, adaptive thresholding,
    and line removal techniques.
    """
    def __init__(
        self, size: Tuple[int, int] = (128, 64),
        denoise_method: Literal["median", "gaussian"] = "median",
        threshold_method: Literal["otsu", "adaptive"] = "adaptive",
        line_removal_kernel_size: int = 3,
        denoising_kernel_size: int = 3
    ):
        """
        Initializes the Dataset2PreprocessingStrategy.

        Args:
            size (Tuple[int, int]): Target size for resizing images. Defaults to (128, 64).
            denoise_method (Literal["median", "gaussian"]): Denoising method to use. Defaults to "median".
            threshold_method (Literal["otsu", "adaptive"]): Thresholding method. Defaults to "adaptive".
            line_removal_kernel_size (int): Kernel size for morphological operations for line removal. Defaults to 3.
            denoising_kernel_size (int): Kernel size for denoising. Defaults to 3.
        """
        self.size = size
        self.denoise_method = denoise_method
        self.threshold_method = threshold_method
        self.line_removal_kernel_size = line_removal_kernel_size
        self.denoising_kernel_size = denoising_kernel_size

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Standardize image size while maintaining aspect ratio."""
        target_height = self.size[1]  # Use the height from size tuple
        h, w = img.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(img, (target_width, target_height))

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _denoise_image_advanced(self, img: np.ndarray) -> np.ndarray:
        """
        Advanced noise reduction using Gaussian Blur or Median Filter.
        """
        if self.denoise_method == "median":
            return cv2.medianBlur(img, self.denoising_kernel_size)
        elif self.denoise_method == "gaussian":
            return cv2.GaussianBlur(img, (self.denoising_kernel_size, self.denoising_kernel_size), 0)
        return img

    def _line_removal(self, img: np.ndarray) -> np.ndarray:
        """
        Remove lines from the image using morphological operations.
        This focuses on horizontal and vertical lines.
        """
        # Create a rectangular kernel for horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.line_removal_kernel_size * 5, 1))
        # Use opening to remove horizontal lines
        temp_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel)
        img = cv2.subtract(img, temp_img)

        # Create a rectangular kernel for vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.line_removal_kernel_size * 5))
        # Use opening to remove vertical lines
        temp_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel)
        img = cv2.subtract(img, temp_img)
        
        return img


    def _threshold_image_adaptive(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive thresholding for better binarization in noisy images.
        """
        if self.threshold_method == "adaptive":
            return cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, # Invert to get white text on black background
                21, 10
            )
        elif self.threshold_method == "otsu":
            _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return th
        
        _, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV) # Default to simple binary inverse
        return th


    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the Dataset 2 specific preprocessing pipeline.
        """
        img = self._resize_image(image)
        img = self._to_grayscale(img)
        img = self._denoise_image_advanced(img)
        img = self._line_removal(img)
        img = self._threshold_image_adaptive(img)
        
        # Apply a final morphological closing to connect broken characters if any
        kernel = np.ones((2,2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return img
