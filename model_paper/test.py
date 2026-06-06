import os
import glob
import torch

try:
    from .model import PaperCRNN
    from .train import decode_predictions, extract_label_from_filename, CRNN_CHAR_TO_INDEX
    from .preprocess import preprocess_image
except ImportError:
    from model import PaperCRNN
    from train import decode_predictions, extract_label_from_filename, CRNN_CHAR_TO_INDEX
    from preprocess import preprocess_image

def test(dataset_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "best_model.pt")
        
    if not os.path.exists(model_path):
        print(f"Error: Model file not found. Please train the model first.")
        return
        
    model = PaperCRNN(num_classes=37).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # 2. Get images from dataset
    project_dir = os.path.dirname(base_dir)
    if dataset_dir is None:
        dataset_dir = os.path.join(project_dir, "0_dataset", "raw", "ctt_sis")
    
    image_paths = []
    if os.path.exists(dataset_dir):
        image_paths.extend(glob.glob(os.path.join(dataset_dir, "*.png")))
        image_paths.extend(glob.glob(os.path.join(dataset_dir, "*.jpg")))
    
    # Filter valid labeled files
    valid_paths = []
    valid_labels = []
    for f in image_paths:
        lbl = extract_label_from_filename(f)
        if lbl and all(ch in CRNN_CHAR_TO_INDEX for ch in lbl):
            valid_paths.append(f)
            valid_labels.append(lbl)
            
    if len(valid_paths) == 0:
        print(f"No labeled images found in {dataset_dir}")
        return
        
    print(f"Found {len(valid_paths)} labeled images to test.")
    
    # 3. Predict and save results
    results = []
    correct_count = 0
    
    with torch.no_grad():
        for i, img_path in enumerate(valid_paths):
            true_label = valid_labels[i]
            # Preprocess
            try:
                # preprocess_image returns shape (1, 1, 80, 200)
                tensor = preprocess_image(img_path) 
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
                continue
                
            tensor = tensor.to(device)
            
            # Predict
            log_probs = model(tensor) # (T, 1, C)
            
            # Decode
            pred_text = decode_predictions(log_probs)[0]
            
            is_correct = pred_text == true_label
            if is_correct:
                correct_count += 1
            
            filename = os.path.basename(img_path)
            results.append((filename, true_label, pred_text, is_correct))
            print(f"Image: {filename} -> True: {true_label} | Pred: {pred_text} | {'[MATCH]' if is_correct else '[WRONG]'}")
            
    # Calculate and print Accuracy
    acc = correct_count / len(valid_paths)
    print(f"\n--- Results ---")
    print(f"Total Tested: {len(valid_paths)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            
    # Save to file
    out_csv = os.path.join(base_dir, "0_dataset_predictions.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("filename,true_label,prediction,is_correct\n")
        for filename, true_label, pred, is_correct in results:
            f.write(f"{filename},{true_label},{pred},{is_correct}\n")
            
    print(f"\nAll predictions saved to {out_csv}")

if __name__ == "__main__":
    import sys
    custom_path = None
    if len(sys.argv) > 1:
        custom_path = sys.argv[1]
    test(custom_path)
