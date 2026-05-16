import cv2
import torch
from abc import ABC, abstractmethod
from datalayer.build_dataset import INDEX_TO_CHAR
from preprocess.general_preprocess import preprocess_pipeline
from segment.general_segmentation import segmentation_pipeline


class BasePredictor(ABC):
    @abstractmethod
    def predict(self, image):
        pass
