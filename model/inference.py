import os
import cv2
import torch
import numpy as np
from preprocess.general_preprocess import preprocess_pipeline
from segment.general_segmentation import segmentation_pipeline
from datalayer.build_dataset import INDEX_TO_CHAR
from predictors.loader import load_inference_model
from preprocess.strategies import Dataset2PreprocessingStrategy

def predict_captcha(img_path, model, device, dataset="1"):
    """
    End-to-end inference pipeline.
    Takes a raw image path, returns the predicted CAPTCHA sequence and segmented characters.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Error] Could not read image: {img_path}")
        return "", []

    # 1. Preprocessing & Segmentation based on dataset choice
    if dataset == "1":
        from preprocess.general_preprocess import preprocess_pipeline
        from segment.general_segmentation import segmentation_pipeline
        processed_img = preprocess_pipeline(img)
        chars = segmentation_pipeline(processed_img)
    else:
        from preprocess.strategies import Dataset2PreprocessingStrategy
        from segment.strategies import Dataset2SegmentationStrategy
        prep = Dataset2PreprocessingStrategy(
            denoise_method="median",
            threshold_method="adaptive",
            line_removal_kernel_size=3,
            denoising_kernel_size=3
        )
        seg = Dataset2SegmentationStrategy(
            min_area=30,
            char_size=(28, 28),
            min_aspect_ratio=0.1,
            max_aspect_ratio=3.0,
            min_height=5
        )
        processed_img = prep.preprocess(img)
        chars = seg.segment(processed_img)

    # 3. Character Prediction & Sequence Reconstruction
    predictions = []
    with torch.no_grad():
        for char_img in chars:
            # Prepare tensor (N, 1, 28, 28)
            if char_img.max() > 1.0:
                char_img = char_img.astype(np.float32) / 255.0
            char_tensor = torch.from_numpy(char_img).float().unsqueeze(0).unsqueeze(0)
            char_tensor = char_tensor.to(device)

            # Predict
            outputs = model(char_tensor)
            pred_idx = outputs.argmax(1).item()
            predictions.append(INDEX_TO_CHAR[pred_idx])

    return "".join(predictions), chars


# Preprocessing configurations for sequence recognition
PREPROCESS_ML = Dataset2PreprocessingStrategy(
    size=(128, 64),
    denoise_method="median",
    threshold_method="adaptive",
    line_removal_kernel_size=3,
    denoising_kernel_size=3,
    maintain_aspect_ratio=False,
    normalize=True
)

PREPROCESS_CRNN = Dataset2PreprocessingStrategy(
    size=(128, 32),
    denoise_method="median",
    threshold_method="adaptive",
    line_removal_kernel_size=3,
    denoising_kernel_size=3,
    maintain_aspect_ratio=False,
    normalize=True
)


def predict_captcha_multi_label(img_path, model, device, dataset="1"):
    """
    Inference for Multi-Label model.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Error] Could not read image: {img_path}")
        return ""
    
    # Choose preprocessing strategy based on dataset
    if dataset == "1":
        from preprocess.strategies import Dataset1PreprocessingStrategy
        strategy = Dataset1PreprocessingStrategy(size=(128, 64))
    else:
        from preprocess.strategies import Dataset2PreprocessingStrategy
        strategy = Dataset2PreprocessingStrategy(
            size=(128, 64),
            denoise_method="median",
            threshold_method="adaptive",
            line_removal_kernel_size=3,
            denoising_kernel_size=3,
            maintain_aspect_ratio=False,
            normalize=True
        )
        
    img_processed = strategy.preprocess(img)
    if img_processed.max() > 1.0:
        img_processed = img_processed.astype(np.float32) / 255.0
        
    img_tensor = torch.from_numpy(img_processed).float().unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)  # (1, num_chars, num_classes)
        preds = outputs.argmax(dim=2).squeeze(0).cpu().numpy()
        
    # Decode
    from datalayer.build_dataset import INDEX_TO_CHAR
    decoded = "".join([INDEX_TO_CHAR[int(idx)] for idx in preds])
    return decoded


def predict_captcha_crnn(img_path, model, device, dataset="1"):
    """
    Inference for CRNN model.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Error] Could not read image: {img_path}")
        return ""
    
    # Choose preprocessing strategy based on dataset
    if dataset == "1":
        from preprocess.strategies import Dataset1PreprocessingStrategy
        strategy = Dataset1PreprocessingStrategy(size=(128, 32))
    else:
        from preprocess.strategies import Dataset2PreprocessingStrategy
        strategy = Dataset2PreprocessingStrategy(
            size=(128, 32),
            denoise_method="median",
            threshold_method="adaptive",
            line_removal_kernel_size=3,
            denoising_kernel_size=3,
            maintain_aspect_ratio=False,
            normalize=True
        )
        
    img_processed = strategy.preprocess(img)
    if img_processed.max() > 1.0:
        img_processed = img_processed.astype(np.float32) / 255.0
        
    img_tensor = torch.from_numpy(img_processed).float().unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        log_probs = model(img_tensor)  # (T, 1, num_classes)
        
    # Greedy CTC decode
    log_probs_np = log_probs.squeeze(1).cpu().numpy()  # (T, num_classes)
    best_path = np.argmax(log_probs_np, axis=1)
    
    from model.train import CRNN_INDEX_TO_CHAR
    decoded_chars = []
    prev = -1
    for idx in best_path:
        if idx != prev:
            if idx != 0:  # 0 is blank
                decoded_chars.append(CRNN_INDEX_TO_CHAR.get(int(idx), ''))
        prev = idx
        
    return "".join(decoded_chars)


def run_inference_demo(image_dir="dataset/raw/1k_pbm", num_samples=5, model_path=None, model_type="cnn", dataset="1"):
    """Run a quick demonstration of the inference pipeline on a few samples."""
    if model_type == "cnn":
        path = model_path if model_path else "model/saved/captcha_cnn.pt"
        model, device = load_inference_model(model_path=path)
    elif model_type == "multi_label":
        path = model_path if model_path else "model/saved/captcha_multi_label.pt"
        from model.train import load_multi_label_model
        model, device = load_multi_label_model(path)
    elif model_type == "crnn":
        path = model_path if model_path else "model/saved/captcha_crnn.pt"
        from model.train import load_crnn_model
        model, device = load_crnn_model(path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    if not os.path.exists(image_dir):
        print(f"[Error] Image directory {image_dir} not found.")
        return
        
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".pbm") or f.endswith(".png") or f.endswith(".jpg")]
    if not image_files:
        print(f"[Warning] No image files found in {image_dir}")
        return
        
    # take a few random samples
    np.random.shuffle(image_files)
    sample_files = image_files[:num_samples]
    
    print("\n" + "="*50)
    print(f"  STEP 6 — Inference Pipeline Demo ({model_type.upper()})")
    print("="*50)
    
    for filename in sample_files:
        img_path = os.path.join(image_dir, filename)
        if model_type == "cnn":
            pred_text, chars = predict_captcha(img_path, model, device, dataset=dataset)
            print(f"File: {filename} | Predicted: {pred_text} | Segments: {len(chars)}")
        elif model_type == "multi_label":
            pred_text = predict_captcha_multi_label(img_path, model, device, dataset=dataset)
            print(f"File: {filename} | Predicted: {pred_text}")
        elif model_type == "crnn":
            pred_text = predict_captcha_crnn(img_path, model, device, dataset=dataset)
            print(f"File: {filename} | Predicted: {pred_text}")

if __name__ == "__main__":
    run_inference_demo()
