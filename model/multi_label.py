import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CaptchaMultiLabelCNN(nn.Module):
    """
    CNN Multi-label (Cải tiến):
    Dùng kiến trúc mạnh hơn (ResNet backbone) để dự đoán đồng thời nhiều vị trí ký tự.
    """
    def __init__(self, num_classes=36, num_chars=5):
        super().__init__()
        self.num_chars = num_chars
        self.num_classes = num_classes

        self.conv = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet Backbone
        self.layer1 = ResNetBlock(64, 64, stride=1)
        self.layer2 = ResNetBlock(64, 128, stride=2)
        self.layer3 = ResNetBlock(128, 256, stride=2)
        self.layer4 = ResNetBlock(256, 512, stride=2)

        # Adaptive pooling to handle various input sizes
        self.pool = nn.AdaptiveAvgPool2d((2, 8))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Heads for each character position
        self.heads = nn.ModuleList([
            nn.Linear(1024, num_classes) for _ in range(num_chars)
        ])

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.fc(x)

        # Return stacked outputs of shape (B, num_chars, num_classes)
        outputs = [head(x) for head in self.heads]
        return torch.stack(outputs, dim=1)


def build_model(num_classes=36, device=None, num_chars=5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaptchaMultiLabelCNN(num_classes=num_classes, num_chars=num_chars).to(device)
    return model, device
