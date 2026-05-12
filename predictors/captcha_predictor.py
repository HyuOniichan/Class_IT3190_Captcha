import numpy as np
import cv2
from preprocess.general_preprocess import preprocess_pipeline
from segment.general_segmentation import segmentation_pipeline

class CaptchaPredictor:
    def __init__(
        self, char_predictor=None,
        preprocess_fn=preprocess_pipeline,
        segment_fn=segmentation_pipeline
    ):

        self.char_predictor = char_predictor
        self.preprocess_fn = preprocess_fn
        self.segment_fn = segment_fn

    def predict_captcha(self, img):
        """
        Full captcha prediction pipeline.
        Input: raw image
        Output: predicted text
        """
        # 0. convert to numpy
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # 1. preprocess
        processed = self.preprocess_fn(img)

        # 2. segment
        chars = self.segment_fn(processed)

        # 3. predict each character
        predictions = []
        
        for char_img in chars:
            pred_char = self.char_predictor.predict_char(char_img)
            predictions.append(pred_char)

        return "".join(predictions)

    def predict_from_path(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        return self.predict_captcha(img)

