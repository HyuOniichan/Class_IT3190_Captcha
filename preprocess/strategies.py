
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
        denoising_kernel_size: int = 3,
        maintain_aspect_ratio: bool = True,
        normalize: bool = False
    ):
        """
        Initializes the Dataset2PreprocessingStrategy.

        Args:
            size (Tuple[int, int]): Target size for resizing images. Defaults to (128, 64).
            denoise_method (Literal["median", "gaussian"]): Denoising method to use. Defaults to "median".
            threshold_method (Literal["otsu", "adaptive"]): Thresholding method. Defaults to "adaptive".
            line_removal_kernel_size (int): Kernel size for morphological operations for line removal. Defaults to 3.
            denoising_kernel_size (int): Kernel size for denoising. Defaults to 3.
            maintain_aspect_ratio (bool): If True, resizes keeping aspect ratio. If False, resizes to size directly. Defaults to True.
            normalize (bool): If True, converts image to float32 and scales to [0, 1]. Defaults to False.
        """
        self.size = size
        self.denoise_method = denoise_method
        self.threshold_method = threshold_method
        self.line_removal_kernel_size = line_removal_kernel_size
        self.denoising_kernel_size = denoising_kernel_size
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.normalize = normalize

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Standardize image size."""
        if not self.maintain_aspect_ratio:
            return cv2.resize(img, self.size)
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
        
        if self.normalize:
            img = img.astype(np.float32) / 255.0
            
        return img

class VerticalProjectionSegmentationStrategy:
    """
    Segmentation by vertical projection profile.
    Params:
      - char_size: output character size (w,h) e.g. (28,28)
      - min_width: minimum column width to accept a segment
      - gap_ratio: fraction of max projection used to detect gaps
      - smooth_kernel: odd kernel size for 1-D smoothing (must be odd)
      - max_white_ratio: reject segments with too many white pixels after resizing
    """
    def __init__(self, char_size=(28, 28), min_width=8, gap_ratio=0.2, smooth_kernel=11, max_white_ratio=0.93):
        self.char_size = char_size
        self.min_width = max(3, min_width)
        self.gap_ratio = float(gap_ratio)
        self.smooth_kernel = int(smooth_kernel) if int(smooth_kernel) % 2 == 1 else int(smooth_kernel) + 1
        self.max_white_ratio = float(max_white_ratio)

    def _prepare_binary(self, img):
        import numpy as np, cv2
        if img is None:
            return img
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        # ensure binary
        unique = np.unique(img)
        if len(unique) > 2:
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # foreground should be black for projection (we count black pixels)
        white = int((img == 255).sum())
        black = int((img == 0).sum())
        if black > white:
            img = cv2.bitwise_not(img)
        # optional small border crop
        if img.shape[0] > 6 and img.shape[1] > 6:
            img = img[2:-2, 2:-2]
        return img

    def _smooth_projection(self, proj):
        import numpy as np, cv2
        # proj is 1D numpy float array; make it 2D for cv2.GaussianBlur
        arr = proj.reshape(1, -1).astype('float32')
        k = max(3, self.smooth_kernel)
        # kernel must be odd; use (k,1)
        if k % 2 == 0:
            k += 1
        sm = cv2.GaussianBlur(arr, (k, 1), 0).reshape(-1)
        return sm

    def _find_cuts(self, binary):
        import numpy as np
        # projection: count black pixels per column
        proj = np.sum(binary == 0, axis=0).astype(float)
        if proj.size == 0:
            return [0, binary.shape[1]]
        sm = self._smooth_projection(proj)
        thresh = sm.max() * self.gap_ratio
        gaps = sm < thresh
        # turn gaps boolean -> continuous gap intervals
        cuts = [0]
        in_gap = False
        gap_start = None
        for i, g in enumerate(gaps):
            if g and not in_gap:
                in_gap = True
                gap_start = i
            elif (not g) and in_gap:
                in_gap = False
                gap_end = i
                mid = (gap_start + gap_end) // 2
                cuts.append(mid)
        if in_gap:
            cuts.append((gap_start + len(gaps)) // 2)
        cuts.append(binary.shape[1])
        # ensure sorted unique and at least two boundaries
        cuts = sorted(list(dict.fromkeys(cuts)))
        return cuts

    def _is_too_bright(self, segment):
        import numpy as np
        if segment is None or segment.size == 0:
            return True
        white_pixels = int((segment == 255).sum())
        white_ratio = white_pixels / float(segment.size)
        return white_ratio > self.max_white_ratio

    def segment(self, image):
        import cv2, numpy as np
        img = self._prepare_binary(image)
        if img is None:
            return []
        cuts = self._find_cuts(img)
        segments = []
        for i in range(len(cuts) - 1):
            l, r = cuts[i], cuts[i + 1]
            width = r - l
            if width < self.min_width:
                continue
            crop = img[:, l:r]
            try:
                resized = cv2.resize(crop, self.char_size, interpolation=cv2.INTER_LINEAR)
            except Exception:
                # fallback: pad/reshape
                h = img.shape[0]
                import numpy as np
                canvas = np.ones((h, max(self.min_width, width)), dtype=img.dtype) * 255
                canvas[:, :width] = crop
                resized = cv2.resize(canvas, self.char_size, interpolation=cv2.INTER_LINEAR)
            if resized.ndim == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
            if self._is_too_bright(resized):
                continue
            segments.append(resized)
        return segments