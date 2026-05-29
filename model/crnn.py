import torch
import torch.nn as nn

class CaptchaCRNN(nn.Module):
    """
    CRNN (CNN + RNN + CTC Loss):
    Dùng cho CAPTCHA có nhiễu gạch ngang. CNN trích xuất tính năng,
    LSTM học thứ tự ký tự, và CTC Loss giúp mô hình tự học mà không cần label từng vị trí cắt.
    """
    def __init__(self, num_classes=37):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )

        # RNN sequence modeling
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.3,
        )

        # Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (B, 1, H, W)
        conv = self.cnn(x)           # (B, 256, 1, W_seq)
        conv = conv.squeeze(2)       # (B, 256, W_seq)
        conv = conv.permute(2, 0, 1) # (T, B, 256)

        rnn_out, _ = self.rnn(conv)  # (T, B, 256)
        output = self.fc(rnn_out)    # (T, B, num_classes)

        log_probs = torch.nn.functional.log_softmax(output, dim=2)
        return log_probs


def build_model(num_classes=37, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaptchaCRNN(num_classes=num_classes).to(device)
    return model, device
