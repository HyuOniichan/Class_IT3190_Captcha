import os
import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from PIL import Image
import numpy as np

try:
    from .model import PaperCRNN
    from .preprocess import preprocess_image
except ImportError:
    from model import PaperCRNN
    from preprocess import preprocess_image

# Character Set Mapping (10 digits + 26 alphabet)
CHARSET = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
CRNN_CHAR_TO_INDEX = {ch: idx + 1 for idx, ch in enumerate(CHARSET)}
CRNN_INDEX_TO_CHAR = {idx + 1: ch for idx, ch in enumerate(CHARSET)}
CRNN_INDEX_TO_CHAR[0] = '' # Blank symbol for CTC

def extract_label_from_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    ignore_prefixes = ('gan', 'captcha', 'image', 'ocr', 'sample', 'test', 'validation')
    if stem.lower().startswith(ignore_prefixes):
        return None
    parts = stem.split('_')
    if len(parts) > 0:
        label = parts[0]
        if len(label) >= 4:
            return label.upper()
    return stem.upper()

class PaperCaptchaDataset(Dataset):
    def __init__(self, image_paths, labels, char_to_index):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_index = char_to_index
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_text = self.labels[idx]
        
        # Preprocess returns (1, 1, 80, 200). We remove the batch dim to get (1, 80, 200)
        tensor = preprocess_image(img_path).squeeze(0)
        
        # Encode label
        encoded = [self.char_to_index.get(ch, 0) for ch in label_text]
        encoded_tensor = torch.tensor(encoded, dtype=torch.long)
        length_tensor = torch.tensor(len(encoded), dtype=torch.int32)
        
        return tensor, encoded_tensor, length_tensor

def collate_fn(batch):
    # Padding sequences for CTC
    tensors, labels, lengths = zip(*batch)
    tensors = torch.stack(tensors)
    
    max_len = max([l.item() for l in lengths])
    padded_labels = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, lbl in enumerate(labels):
        padded_labels[i, :len(lbl)] = lbl
        
    lengths = torch.stack(lengths)
    return tensors, padded_labels, lengths

def decode_predictions(log_probs):
    # log_probs shape: (T, B, C)
    best_paths = log_probs.argmax(dim=2).permute(1, 0) # (B, T)
    decoded = []
    for path in best_paths:
        path = path.cpu().numpy()
        chars = []
        prev = -1
        for p in path:
            if p != prev:
                if p != 0:
                    chars.append(CRNN_INDEX_TO_CHAR.get(p, ''))
            prev = p
        decoded.append("".join(chars))
    return decoded

def train(resume=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Collect Data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dirs = [
        #os.path.join(project_dir, "dataset", "raw", "1k_pbm", "*.pbm"),
        #os.path.join(project_dir, "dataset", "raw", "1k_pbm", "*.png"),
        #os.path.join(project_dir, "dataset_v2", "raw", "*.png"),
        #os.path.join(project_dir, "dataset_v2", "kaggle_captcha", "*.png"),
        os.path.join(project_dir, "0_dataset", "raw", "ctt_sis", "*.png")
    ]
    
    import random
    all_files = []
    
    for search_path in data_dirs:
        files = glob.glob(search_path)
        # Limit dataset_v3 to 4000 images randomly picked
        if "dataset_v3" in search_path:
            random.seed(42)
            random.shuffle(files)
            files = files[:8000]
        all_files.extend(files)
    
    valid_files = []
    valid_labels = []
    for f in all_files:
        lbl = extract_label_from_filename(f)
        if lbl and all(ch in CRNN_CHAR_TO_INDEX for ch in lbl):
            valid_files.append(f)
            valid_labels.append(lbl)
            
    print(f"Found {len(valid_files)} valid images across dataset and dataset_v2.")
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        valid_files, valid_labels, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}, Validation size: {len(X_test)}")
    
    # 3. Datasets and Loaders
    batch_size = 32
    train_ds = PaperCaptchaDataset(X_train, y_train, CRNN_CHAR_TO_INDEX)
    val_ds = PaperCaptchaDataset(X_test, y_test, CRNN_CHAR_TO_INDEX)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 4. Model, Loss, Optimizer
    model = PaperCRNN(num_classes=37).to(device)
    
    save_path = os.path.join(base_dir, "best_model_1.pt")
    if resume and os.path.exists(save_path):
        print(f"\n[INFO] Loading existing model from {save_path} to resume training...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        lr = 5e-4 # Reduce learning rate when resuming to prevent destroying learned weights
    else:
        print("\n[INFO] Starting training from scratch...")
        lr = 5e-4

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 50
    patience = 10
    patience_counter = 0
    
    best_val_loss = float('inf')
    

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        t0 = time.time()
        for X, y, lengths in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            B = X.size(0)
            
            optimizer.zero_grad()
            log_probs = model(X) # (T, B, C)
            T = log_probs.size(0)
            input_lengths = torch.full((B,), T, dtype=torch.int32, device=device)
            
            loss = criterion(log_probs, y, input_lengths, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_loss += loss.item() * B
            total_samples += B
            
        train_loss = running_loss / total_samples
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        
        all_true_chars = []
        all_pred_chars = []
        
        with torch.no_grad():
            for X, y, lengths in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                X, y, lengths = X.to(device), y.to(device), lengths.to(device)
                B = X.size(0)
                
                log_probs = model(X)
                T = log_probs.size(0)
                input_lengths = torch.full((B,), T, dtype=torch.int32, device=device)
                
                loss = criterion(log_probs, y, input_lengths, lengths)
                val_loss_sum += loss.item() * B
                val_samples += B
                
                # Accuracy & Metrics
                decoded_preds = decode_predictions(log_probs)
                for i in range(B):
                    true_len = lengths[i].item()
                    true_chars = [CRNN_INDEX_TO_CHAR.get(idx.item(), '') for idx in y[i][:true_len]]
                    true_str = "".join(true_chars)
                    pred_str = decoded_preds[i]
                    
                    # Pad strings to same length for character-wise metrics comparison
                    max_len = max(len(true_str), len(pred_str))
                    t_pad = true_str.ljust(max_len, ' ')
                    p_pad = pred_str.ljust(max_len, ' ')
                    
                    all_true_chars.extend(list(t_pad))
                    all_pred_chars.extend(list(p_pad))
                        
        val_loss = val_loss_sum / val_samples
        
        # Calculate Metrics (macro average for balanced evaluation across characters)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_chars, all_pred_chars, average='macro', zero_division=0
        )
        
        t_elapsed = time.time() - t0
        print(f"Epoch [{epoch}/{epochs}] - Time: {t_elapsed:.1f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        print(f"  -> Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(base_dir, "best_model_1.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved to {save_path} (Val Loss decreased)")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

if __name__ == "__main__":
    import sys
    resume_flag = True
    if len(sys.argv) > 1:
        if sys.argv[1] == '0':
            resume_flag = False
        elif sys.argv[1] == '1':
            resume_flag = True
            
    train(resume=resume_flag)
