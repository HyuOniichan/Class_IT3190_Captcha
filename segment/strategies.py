from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import List, Tuple

class SegmentationStrategy(ABC):
    """Abstract base class for segmentation strategies."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Applies character segmentation to a preprocessed image.

        Args:
            image (np.ndarray): The preprocessed binary image.

        Returns:
            List[np.ndarray]: A list of segmented character images.
        """
        pass

class Dataset1SegmentationStrategy(SegmentationStrategy):
    """
    Segmentation strategy for Dataset 1 (simple, clean images).
    This strategy uses basic contour detection and filtering.
    """
    def __init__(self, min_area: int = 50, char_size: Tuple[int, int] = (28, 28)):
        """
        Initializes the Dataset1SegmentationStrategy.

        Args:
            min_area (int): Minimum contour area to consider as a character. Defaults to 50.
            char_size (Tuple[int, int]): Target size for segmented characters. Defaults to (28, 28).
        """
        self.min_area = min_area
        self.char_size = char_size

    def _prepare_binary_image(self, processed_img: np.ndarray) -> np.ndarray:
        """
        Ensure image is clean binary and correct foreground (white chars on black background)
        Fix common issues with PBM images.
        """
        # 1. Convert to uint8
        if processed_img.dtype != np.uint8:
            processed_img = processed_img.astype(np.uint8)

        # 2. Ensure binary
        unique_vals = np.unique(processed_img)
        if len(unique_vals) > 2:
            _, img = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY)
        else:
            img = processed_img.copy()

        # 3. Ensure foreground (text) is white (255)
        # Check if white pixel is more -> invert
        white_pixels = np.sum(img == 255)
        black_pixels = np.sum(img == 0)

        if white_pixels > black_pixels:
            img = cv2.bitwise_not(img)

        # 4. Optional: remove border
        if img.shape[0] > 4 and img.shape[1] > 4:
            img = img[2:-2, 2:-2]

        return img

    def _find_contours(self, processed_img: np.ndarray) -> List[np.ndarray]:
        """
        Detect character regions using OpenCV contours
        Input: processed (binary) image
        Output: contours list
        """
        contours, _ = cv2.findContours(
            processed_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _extract_characters(self, processed_img: np.ndarray, contours: List[np.ndarray]) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Crop character regions from image
        Input: contours + processed image
        Output: list of character images + bounding boxes
        """
        characters = []
        boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # filter noise
            if w * h >= self.min_area:
                char_img = processed_img[y:y+h, x:x+w]
                characters.append(char_img)
                boxes.append((x, y, w, h))

        return characters, boxes

    def _sort_characters(self, characters: List[np.ndarray], boxes: List[Tuple[int, int, int, int]]) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Sort characters from left to right
        Input: character images + bounding boxes
        Output: ordered character images + bounding boxes
        """
        # sort by x coordinate
        sorted_data = sorted(zip(characters, boxes), key=lambda b: b[1][0])

        sorted_chars = [item[0] for item in sorted_data]
        sorted_boxes = [item[1] for item in sorted_data]

        return sorted_chars, sorted_boxes

    def _resize_characters(self, characters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalize character size
        Input: list of character images
        Output: resized character images
        """
        resized_chars = []

        for char in characters:
            resized = cv2.resize(char, self.char_size)
            resized_chars.append(resized)

        return resized_chars

    def segment(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Full segmentation pipeline for Dataset 1:
        1. Ensure binary image
        2. Detect contours
        3. Extract characters
        4. Sort characters
        5. Resize characters
        """
        processed_img = self._prepare_binary_image(image)
        
        contours = self._find_contours(processed_img)

        chars, boxes = self._extract_characters(processed_img, contours)

        chars, boxes = self._sort_characters(chars, boxes)

        chars = self._resize_characters(chars)

        return chars


class Dataset2SegmentationStrategy(SegmentationStrategy):
    """
    Segmentation strategy for Dataset 2 (noisy images).
    This strategy focuses on improved contour analysis and filtering to handle noise.
    """
    def __init__(self, min_area: int = 30, char_size: Tuple[int, int] = (28, 28),
                 min_aspect_ratio: float = 0.1, max_aspect_ratio: float = 3.0,
                 min_height: int = 5):
        """
        Initializes the Dataset2SegmentationStrategy.

        Args:
            min_area (int): Minimum contour area to consider as a character. Defaults to 30 (lower for noisy images).
            char_size (Tuple[int, int]): Target size for segmented characters. Defaults to (28, 28).
            min_aspect_ratio (float): Minimum aspect ratio (width/height) for contours. Defaults to 0.1.
            max_aspect_ratio (float): Maximum aspect ratio (width/height) for contours. Defaults to 3.0.
            min_height (int): Minimum height for contours. Defaults to 5 (lower for noisy images).
        """
        self.min_area = min_area
        self.char_size = char_size
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_height = min_height

    def _prepare_binary_image(self, processed_img: np.ndarray) -> np.ndarray:
        """
        Ensure image is clean binary with white foreground on black background.
        """
        if processed_img.dtype != np.uint8:
            processed_img = processed_img.astype(np.uint8)

        unique_vals = np.unique(processed_img)
        if len(unique_vals) > 2:
            _, img = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY)
        else:
            img = processed_img.copy()

        white_pixels = np.sum(img == 255)
        black_pixels = np.sum(img == 0)

        if white_pixels > black_pixels:
            img = cv2.bitwise_not(img)

        # Optional: remove border (be cautious with noisy images, as borders might touch characters)
        # For Dataset 2, it might be safer to avoid aggressive border removal if characters are close to edges.
        # If needed, can add a parameter to control this.
        # if img.shape[0] > 4 and img.shape[1] > 4:
        #     img = img[2:-2, 2:-2]

        return img

    def _find_and_filter_contours(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Detects contours and filters them based on area, aspect ratio, and height to remove noise.
        """
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area, aspect ratio, and height
        filtered_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0
            area = cv2.contourArea(cnt)

            # Heuristic filtering: adjust these values based on your dataset characteristics
            if (area > self.min_area and 
                self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio and 
                h > self.min_height):
                filtered_contours.append(cnt)

        return filtered_contours

    def _extract_and_sort_characters(self, img: np.ndarray, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extracts characters based on contours and sorts them from left to right.
        """
        characters = []
        boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            char_img = img[y:y+h, x:x+w]
            characters.append(char_img)
            boxes.append((x, y, w, h))

        # Sort characters from left to right
        sorted_data = sorted(zip(characters, boxes), key=lambda b: b[1][0])
        sorted_chars = [item[0] for item in sorted_data]
        
        return sorted_chars

    def _resize_characters(self, characters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalizes character sizes to a target size.
        """
        resized_chars = []
        for char in characters:
            resized = cv2.resize(char, self.char_size)
            resized_chars.append(resized)
        return resized_chars

    def segment(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Applies the Dataset 2 specific segmentation pipeline.
        """
        img = self._prepare_binary_image(image)
        contours = self._find_and_filter_contours(img)
        chars = self._extract_and_sort_characters(img, contours)
        chars = self._resize_characters(chars)
        return chars
