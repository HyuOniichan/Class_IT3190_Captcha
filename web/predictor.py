import os
import numpy as np
import cv2
import joblib
from tensorflow import keras
from preprocess.utils import simple_preprocess_pipeline, segmentation_pipeline


NAME = "lv2_sis"


if NAME == "lv0":
    # KNN + PCA
    MODEL_PATH = "weights/test/knn-pca-68_dims.joblib"
    TRANSFORMER_PATH = "weights/test/knn-pca_transformer.joblib"
    CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    
elif NAME == "lv1":
    # KNN
    MODEL_PATH = "weights/lv1_1k_pbm/knn.joblib"
    TRANSFORMER_PATH = None
    CHARSET = "0123456789"
    
elif NAME == "lv2_sis":
    # CNN
    MODEL_PATH = "weights/lv2_ctt_sis/result_model.keras"
    TRANSFORMER_PATH = None
    CHARSET = "023456789"
    
elif NAME == "lv2_kaggle":
    # CNN
    MODEL_PATH = "weights/lv2_1k_5digits/result_model.h5"
    TRANSFORMER_PATH = None
    CHARSET = "2345678bcdefgmnpwxy"



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
            self.model = keras.models.load_model(model_path)
        if transformer_path:
            self.transformer = joblib.load(transformer_path)
    
    
    def predict(self, raw_image):
        """
        Prediction pipeline.  
        """
        
        if raw_image is None:
            print("[Predictor] Input image is invalid")
            return ""
        
        img_np = np.array(raw_image) 
        
        if len(img_np.shape) == 3:
            if img_np.shape[2] == 4:  # RGBA (ảnh web thường có kênh Alpha)
                img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
            else:
                img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_cv2 = img_np

        thresh_img1 = cv2.adaptiveThreshold(img_cv2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
        thresh_img1 = ~thresh_img1
        close_img1 = cv2.morphologyEx(thresh_img1, cv2.MORPH_CLOSE, np.ones((5,2), np.uint8))
        dilate_img1 = cv2.dilate(close_img1, np.ones((2,2), np.uint8), iterations = 1)
        gauss_img1 = cv2.GaussianBlur(dilate_img1, (1,1), 0)
        gauss_img1 = cv2.resize(gauss_img1,(200,50),interpolation=cv2.INTER_LINEAR)

        if NAME == "lv2_sis":
            segmented_chars = [
                gauss_img1[0:50, 40:65],
                gauss_img1[0:50, 65:90],
                gauss_img1[0:50, 90:115],
                gauss_img1[0:50, 115:140],
                gauss_img1[0:50, 140:165]
            ]
        elif NAME == "lv2_kaggle":
            segmented_chars = [
                gauss_img1[10:50, 30:50],
                gauss_img1[10:50, 50:70],
                gauss_img1[10:50, 70:90],
                gauss_img1[10:50, 90:110],
                gauss_img1[10:50, 110:130]
            ]

        X_chars = np.array(segmented_chars, dtype=np.float32)
        
        X_chars = np.expand_dims(X_chars, axis=-1)
        
        X_chars /= 255.0
        
        ydemo = self.model.predict(X_chars)
        
        predicted_ids = np.argmax(ydemo, axis=1)

        predicted_text = "".join([self.charset[int(idx)] for idx in predicted_ids])
        
        return predicted_text



