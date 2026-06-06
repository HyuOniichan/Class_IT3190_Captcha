import torch
import torch.nn as nn
try:
    from .preprocess import preprocess_image
except ImportError:
    from preprocess import preprocess_image


class PaperCRNN(nn.Module):
    """
    CRNN Model based on Algorithm-I: Numeric Captcha Solver.
    Input image size: Grayscale, 80x200 (H x W)
    """
    def __init__(self, num_classes=37, hidden_size=128):
        super(PaperCRNN, self).__init__()
        
        # 7. Convolution + Sigmoid (Layer 1) & 8. Max-Pooling (Layer 1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.sig1 = nn.Sigmoid()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 9. Convolution + Sigmoid (Layer 2) & 10. Max-Pooling (Layer 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.sig2 = nn.Sigmoid()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After two 2x2 max poolings, H=80 -> 40 -> 20. W=200 -> 100 -> 50.
        # Channels = 64. So the feature map P(2) is (B, 64, 20, 50).
        # Sequence Reshaping (Step 11):
        # We need to reshape this to (T, B, H' * C) = (50, B, 20 * 64) for LSTM input
        
        # 12, 13, 14, 15. Bidirectional LSTM Layers
        lstm_input_size = 20 * 64
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=1, # Paper specifies a forward and backward pass, typically 1 layer of BiLSTM
            bidirectional=True,
            batch_first=False # PyTorch LSTM expects (seq_len, batch, input_size) by default
        )
        
        # 16. Fully Connected Layer with Softmax (17. Linear Transformation)
        # Bidirectional means output size is hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        """
        x shape: (B, 1, 80, 200) representing Grayscale images.
        """
        # CNN Phase
        # Layer 1
        x = self.conv1(x)
        x = self.sig1(x)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.sig2(x)
        x = self.pool2(x)
        
        # 11. Sequence Reshaping
        # Current shape of x: (B, C=64, H=20, W=50)
        B, C, H, W = x.size()
        
        # Interpret each column of P(2) as a feature vector in a sequence
        # We want sequence of length T = W.
        # First, permute to (B, W, C, H)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Flatten C and H into a single vector of size C*H
        x = x.view(B, W, C * H)
        
        # RNN expects input of shape (seq_len, batch, input_size)
        x = x.permute(1, 0, 2).contiguous() # Now shape is (W, B, C*H)
        
        # RNN Phase
        x, _ = self.lstm(x) # output x shape: (W, B, hidden_size * 2)
        
        # Fully Connected Layer Phase
        x = self.fc(x) # shape: (W, B, num_classes)
        
        # 17. Softmax Activation
        # For CTC Loss in PyTorch, log_softmax is applied along the class dimension (dim=2)
        log_probs = torch.nn.functional.log_softmax(x, dim=2)
        
        return log_probs

if __name__ == "__main__":
    # Quick Verification
    model = PaperCRNN(num_classes=37)
    print(model)
    
    # Verify preprocess module integration
    from PIL import Image
    dummy_img = Image.new("RGB", (300, 150), color="white")
    processed_tensor = preprocess_image(dummy_img)
    print("Preprocessed tensor shape:", processed_tensor.shape)
    assert processed_tensor.shape == (1, 1, 80, 200), "Preprocess shape mismatch!"
    
    # Dummy input test through the model
    output = model(processed_tensor)
    
    # Expected output shape: (W=50, Batch=1, NumClasses=37)
    print("Output shape:", output.shape)
    assert output.shape == (50, 1, 37), "Shape mismatch!"
    print("Verification Passed!")
