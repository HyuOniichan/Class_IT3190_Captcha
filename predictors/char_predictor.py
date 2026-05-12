import torch
from datalayer.build_dataset import INDEX_TO_CHAR

class CharacterPredictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_char(self, char_img):
        """
        Predict a single character image.
        Input:
            char_img -> numpy array (28x28)
        Output:
            predicted character
        """
        self.model.eval()
        with torch.no_grad():
            char_tensor = (
                torch.from_numpy(char_img).float().unsqueeze(0).unsqueeze(0) / 255.0
            )
            char_tensor = char_tensor.to(self.device)
            outputs = self.model(char_tensor)
            pred_idx = outputs.argmax(1).item()
        return INDEX_TO_CHAR[pred_idx]

