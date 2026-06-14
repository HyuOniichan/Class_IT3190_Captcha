import os
import numpy as np
import cv2
import joblib

from preprocess.utils import simple_preprocess_pipeline, segmentation_pipeline

# # [lv0] KNN + PCA
# MODEL_PATH = "weights/test/knn-pca-68_dims.joblib"
# TRANSFORMER_PATH = "weights/test/knn-pca_transformer.joblib"
# CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

# [lv1] KNN
MODEL_PATH = "weights/lv1_1k_pbm/knn.joblib"
TRANSFORMER_PATH = None
CHARSET = "0123456789"



class Predictor():
    def __init__(
        self, 
        model_path: str = MODEL_PATH, 
        transformer_path: str = TRANSFORMER_PATH, 
        charset: str = CHARSET
    ):
        self.model = None
        self.transformer = None
        self.charset = charset
        
        self.setup(model_path, transformer_path)

    
    def setup(self, model_path, transformer_path):
        if model_path:
            self.model = joblib.load(model_path)
        if transformer_path:
            self.transformer = joblib.load(transformer_path)
    
    
    def predict(self, raw_image):
        """
        Prediction pipeline.  
        (Currently apply lv1 dataset preprocessing pipeline)
        """
        
        if raw_image is None:
            print("[Predictor] Input image is invalid")
            return ""
        
        img_np = np.array(raw_image) 
        
        # Convert to BGR (OpenCV)
        if len(img_np.shape) == 3:
            if img_np.shape[2] == 4:  # RGBA
                img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_cv2 = img_np  # Already binary image
        
        # Simple preprocess
        processed_img = simple_preprocess_pipeline(img_cv2)

        # Segmentation
        segmented_chars = segmentation_pipeline(processed_img)

        if not segmented_chars:
            print("[Predictor] Segmentation failed")
            return ""

        # Shape: (num_chars, 28, 28)
        X_chars = np.array(segmented_chars, dtype=np.uint8)
        
        # Flatten
        num_samples = X_chars.shape[0]
        input_tensor = X_chars.reshape(num_samples, -1)

        # Dimension reduction
        if self.transformer:
            input_tensor = self.transformer.transform(input_tensor)
        
        # Prediction (indexes)
        predicted_labels, _ = self.model.predict(input_tensor)
        predicted_ids = predicted_labels.flatten().astype(int)
        
        # Final prediction
        predicted_text = "".join([self.charset[int(idx)] for idx in predicted_ids])
        
        return predicted_text


