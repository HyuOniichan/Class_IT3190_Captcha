import os
import cv2
import torch
import numpy as np
from preprocess.general_preprocess import preprocess_pipeline
from segment.general_segmentation import segmentation_pipeline
from datalayer.build_dataset import INDEX_TO_CHAR
from predictors.loader import load_inference_model

def predict_captcha(img_path, model, device):
    """
    End-to-end inference pipeline.
    Takes a raw image path, returns the predicted CAPTCHA sequence and segmented characters.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Error] Could not read image: {img_path}")
        return "", []

    # 1. Preprocessing
    processed_img = preprocess_pipeline(img)

    # 2. Segmentation
    chars = segmentation_pipeline(processed_img)

    # 3. Character Prediction & Sequence Reconstruction
    predictions = []
    with torch.no_grad():
        for char_img in chars:
            # Prepare tensor (N, 1, 28, 28)
            char_tensor = torch.from_numpy(char_img).float().unsqueeze(0).unsqueeze(0) / 255.0
            char_tensor = char_tensor.to(device)

            # Predict
            outputs = model(char_tensor)
            pred_idx = outputs.argmax(1).item()
            predictions.append(INDEX_TO_CHAR[pred_idx])

    return "".join(predictions), chars

def run_inference_demo(image_dir="dataset/raw/1k_pbm", num_samples=5):
    """Run a quick demonstration of the inference pipeline on a few samples."""
    model, device = load_inference_model()
    
    if not os.path.exists(image_dir):
        print(f"[Error] Image directory {image_dir} not found.")
        return
        
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".pbm") or f.endswith(".png")]
    # take a few random samples
    np.random.shuffle(image_files)
    sample_files = image_files[:num_samples]
    
    print("\n" + "="*50)
    print("  STEP 6 — Inference Pipeline Demo")
    print("="*50)
    
    for filename in sample_files:
        img_path = os.path.join(image_dir, filename)
        pred_text, chars = predict_captcha(img_path, model, device)
        print(f"File: {filename} | Predicted: {pred_text} | Segments: {len(chars)}")

if __name__ == "__main__":
    run_inference_demo()
