import cv2
import joblib

from preprocess.utils import simple_preprocess_pipeline
from models.utils import flatten

# Utils
def load(filepath):
    """Load a model from a file."""
    model = joblib.load(filepath)
    return model

def run_prediction(model_path, input):
    model = load(filepath=model_path)
    pred, probs = model.predict(input)
    print(f"Prediction: {pred} | {probs}")


# Load models
# knn_model = load(filepath="weights/lv1_knn_model.joblib")
# dt_model = load(filepath="weights/lv1_dt_model.joblib")
# rf_model = load(filepath="weights/lv1_rf_model.joblib")
# svm_model = load(filepath="weights/lv1_svm_model.joblib")
# cnn_model = load(filepath="weights/lv1_cnn_model.joblib")


# Predict
# input_path = "output/dataset/lv1_1k_pbm/segmented/0008_char_3.png"
input_path = "output/dataset/lv2_1k_5digits/over_segmented/2b827_2_7.png"

input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (28, 28))
input_tensor = input_image.reshape(1, -1)

print(f"Raw: {input_image.shape}")
print(f"Flatten: {input_tensor.shape}")

run_prediction(
    model_path="weights/lv1_svm_model.joblib",
    input=input_tensor
)

# TODO: Train lai model tren handwritten dataset
# Yeu cau dataset: Co chu (a-z, A-Z) va so (0-9) + tu tao them nhieu

