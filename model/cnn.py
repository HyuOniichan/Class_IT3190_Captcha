import torch
import torch.nn as nn


class CaptchaCNN(nn.Module):
    """
    Compact CNN for 28×28 single-channel character classification.

    Architecture:
        Conv(1→32, 3) → ReLU → MaxPool(2)
        Conv(32→64, 3) → ReLU → MaxPool(2)
        Conv(64→128, 3) → ReLU
        Flatten → FC(512) → ReLU → Dropout → FC(num_classes)

    After two 2×2 max-pools the spatial size is 5×5 (28→13→5 with valid padding
    inside each pool window), giving 128*5*5 = 3200 features before the FC layers.
    """

    def __init__(self, num_classes=36):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=36, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaptchaCNN(num_classes=num_classes).to(device)
    return model, device
