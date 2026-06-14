import cv2
import numpy as np
import torch
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

    # For EMNIST dataset
    EMNIST_MAPPING = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    print(EMNIST_MAPPING[pred[0]])


def run_lv0_cnn_prediction(input_image):
    from models.cnn import SimpleCNN

    device = 'cpu'
    weights_path = 'weights/lv0_emnist/cnn.pth'
    
    model = SimpleCNN(num_classes=47).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        else:
            state_dict = state_dict
    else:
        raise RuntimeError(f"Fail to load weights file")

    model.load_state_dict(state_dict)
    model.eval()

    input_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        preds_idx = outputs.argmax(1).cpu().numpy()
        probs = probs.cpu().numpy()

    print(f"Prediction: {preds_idx} | {probs}")
    
    EMNIST_MAPPING = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    pred_char = EMNIST_MAPPING[preds_idx[0]]
    
    print(f"Character: '{pred_char}'")
    print(f"Confidence: {np.max(probs) * 100:.2f}%")



# Predict

# Output expected: 8
input_path = "output/dataset/lv1_1k_pbm/segmented/0008_char_3.png"

# # Output expected: 8
# input_path = "output/dataset/lv2_1k_5digits/over_segmented/2b827_2_7.png"

input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

# # Switch color if needed
# input_image = cv2.bitwise_not(input_image)

input_image = cv2.resize(input_image, (28, 28))
input_tensor = input_image.reshape(1, -1)

print(f"Raw: {input_image.shape}")
print(f"Flatten: {input_tensor.shape}")

# For CNN (no flatten)
input_tensor_cnn = input_image.reshape(1, 28, 28)


run_prediction(
    model_path="weights/lv0_emnist/random_forest.joblib",
    input=input_tensor_cnn
)

# run_lv0_cnn_prediction(input_image)

