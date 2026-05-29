import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from model.inference import load_inference_model, predict_captcha
from datalayer.build_dataset import INDEX_TO_CHAR, CHAR_TO_INDEX

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Generate and save confusion matrix."""
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Character Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix to {output_path}")

def plot_sample_errors(errors, output_path, max_samples=16):
    """
    Generate and save a grid of error samples.
    errors is a list of dicts: {'image_path': path, 'true': text, 'pred': text}
    """
    if not errors:
        return
        
    n_samples = min(len(errors), max_samples)
    cols = 4
    rows = int(np.ceil(n_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i, err in enumerate(errors[:n_samples]):
        img = cv2.imread(err['image_path'], cv2.IMREAD_GRAYSCALE)
        ax = axes[i]
        if img is not None:
            ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {err['true']}\nPred: {err['pred']}", color='red')
        ax.axis('off')
        
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Evaluation] Saved sample errors to {output_path}")

def evaluate_system(test_csv="dataset/meta/1k_pbm/test.csv", raw_dir="dataset/raw/1k_pbm", output_dir="evaluation", model_path=None, model_type="cnn", dataset="1"):
    """
    Run full evaluation on the test set.
    Calculates Character Accuracy, CAPTCHA Accuracy, and performs Error Analysis.
    """
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

    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print(f"  STEP 7 — System Evaluation ({model_type.upper()})")
    print("="*50)
    
    if not os.path.exists(test_csv):
        print(f"[Error] Test file {test_csv} not found.")
        return
        
    total_samples = 0
    correct_captchas = 0
    total_chars = 0
    correct_chars = 0
    
    seg_errors = 0
    
    y_true_chars = []
    y_pred_chars = []
    
    error_samples = []
    
    with open(test_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            filename = row[0].strip()
            # Split off the stem and remove suffixes like _orig to get target label
            stem = os.path.splitext(filename)[0]
            if '_' in stem:
                true_text = stem.split('_')[0].upper()
            else:
                true_text = stem.upper()
            
            img_path = os.path.join(raw_dir, filename)
            if not os.path.exists(img_path):
                # Try finding alternative extensions if file not found
                base, ext = os.path.splitext(filename)
                found = False
                for alt_ext in ['.png', '.pbm', '.jpg', '.jpeg']:
                    alt_path = os.path.join(raw_dir, base + alt_ext)
                    if os.path.exists(alt_path):
                        img_path = alt_path
                        found = True
                        break
                if not found:
                    continue
                
            if model_type == "cnn":
                pred_text, chars = predict_captcha(img_path, model, device, dataset=dataset)
            elif model_type == "multi_label":
                from model.inference import predict_captcha_multi_label
                pred_text = predict_captcha_multi_label(img_path, model, device, dataset=dataset)
            elif model_type == "crnn":
                from model.inference import predict_captcha_crnn
                pred_text = predict_captcha_crnn(img_path, model, device, dataset=dataset)
            
            total_samples += 1
            total_chars += len(true_text)
            
            # CAPTCHA Accuracy
            if pred_text == true_text:
                correct_captchas += 1
            else:
                error_samples.append({'image_path': img_path, 'true': true_text, 'pred': pred_text})
                
            # Segmentation Error Analysis
            if len(pred_text) != len(true_text):
                seg_errors += 1
                # If length differs, character accuracy is hard to align perfectly.
                # We do a simple positional match up to the min length
                min_len = min(len(pred_text), len(true_text))
                for i in range(min_len):
                    y_true_chars.append(true_text[i])
                    y_pred_chars.append(pred_text[i])
                    if true_text[i] == pred_text[i]:
                        correct_chars += 1
            else:
                # Exact length match
                for t, p in zip(true_text, pred_text):
                    y_true_chars.append(t)
                    y_pred_chars.append(p)
                    if t == p:
                        correct_chars += 1
                        
    captcha_acc = correct_captchas / total_samples if total_samples > 0 else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    
    print(f"\n[Results] Total Samples Tested: {total_samples}")
    print(f"[Results] CAPTCHA Accuracy: {captcha_acc:.4f} ({correct_captchas}/{total_samples})")
    print(f"[Results] Character Accuracy: {char_acc:.4f} ({correct_chars}/{total_chars})")
    if model_type == "cnn":
        print(f"[Results] Segmentation Errors (Length Mismatch): {seg_errors}/{total_samples}")
    
    # Visualizations
    if y_true_chars and y_pred_chars:
        plot_confusion_matrix(y_true_chars, y_pred_chars, os.path.join(output_dir, f"confusion_matrix_{model_type}.png"))
        
    if error_samples:
        plot_sample_errors(error_samples, os.path.join(output_dir, f"sample_errors_{model_type}.png"))

if __name__ == "__main__":
    evaluate_system()
