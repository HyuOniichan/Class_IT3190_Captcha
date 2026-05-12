import os
import torch

from model.cnn import build_model


def load_inference_model(model_path="model/saved/captcha_cnn.pt", num_classes=10, device=None):
    #Sau này nhớ thay số class khi update dataset
    #Lỗi khi để num_classes = 36. Đoán là do chỉ có số nên chỉ lưu 10 class

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model, _ = build_model(num_classes=num_classes, device=device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"[Warning] Model file {model_path} not found. Using untrained model.")

    model.eval()

    return model, device