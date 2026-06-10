import cv2
import joblib

from preprocess.utils import simple_preprocess_pipeline
from models.utils import flatten

# Utils
def load(filepath):
    """Load a model from a file."""
    model = joblib.load(filepath)
    return model

# Load models
knn_model = load(filepath="weights/lv1_knn_model.joblib")
dt_model = load(filepath="weights/lv1_dt_model.joblib")
rf_model = load(filepath="weights/lv1_rf_model.joblib")
svm_model = load(filepath="weights/lv1_svm_model.joblib")
cnn_model = load(filepath="weights/lv1_cnn_model.joblib")


# Predict
input_path = "output/dataset/lv1_1k_pbm/segmented/0008_char_3.png"
input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
input_tensor = input_image.reshape(1, -1)

print(f"Raw: {input_image.shape}")
print(f"Flatten: {input_tensor.shape}")

knn_pred, knn_probs = knn_model.predict(input_tensor)
dt_pred, dt_probs = dt_model.predict(input_tensor)
rf_pred, rf_probs = rf_model.predict(input_tensor)
svm_pred, svm_probs = svm_model.predict(input_tensor)
# cnn_pred = cnn_model.predict(input_tensor)

print(f"knn pred: {knn_pred} | {knn_probs}")
print(f"dt pred: {dt_pred} | {dt_probs}")
print(f"rf pred: {rf_pred} | {rf_probs}")
print(f"svm pred: {svm_pred} | {svm_probs}")
# print(f"cnn pred: {cnn_pred} | {cnn_probs}")

