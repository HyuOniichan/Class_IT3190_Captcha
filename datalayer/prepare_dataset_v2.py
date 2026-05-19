import os
import random
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    from captcha.image import ImageCaptcha
except ImportError:  # pragma: no cover
    ImageCaptcha = None

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
    optim = None
    DataLoader = None
    Dataset = None
    transforms = None

CHARSET = [str(d) for d in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def add_random_lines(image: np.ndarray, line_count: int = 3, thickness_range: Tuple[int, int] = (1, 2)) -> np.ndarray:
    """Thêm đường gạch ngẫu nhiên vào ảnh để mô phỏng CAPTCHA có nhiễu đứt đoạn."""
    img = image.copy()
    h, w = img.shape[:2]
    for _ in range(line_count):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        x2 = random.randint(0, w - 1)
        y2 = random.randint(0, h - 1)
        thickness = random.randint(thickness_range[0], thickness_range[1])
        color = 0 if len(img.shape) == 2 else (0, 0, 0)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def add_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.02, salt_vs_pepper: float = 0.5) -> np.ndarray:
    """Thêm nhiễu muối tiêu vào ảnh grayscale."""
    img = image.copy()
    h, w = img.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt noise
    num_salt = int(num_pixels * salt_vs_pepper)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    img[coords[0], coords[1]] = 255

    # Pepper noise
    num_pepper = num_pixels - num_salt
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    img[coords[0], coords[1]] = 0

    return img


def augment_dataset_with_noise(
    input_dir: str = "dataset/raw/1k_pbm",
    output_dir: str = "dataset_v2/raw",
    copies_per_image: int = 3,
    line_prob: float = 0.8,
    salt_pepper_prob: float = 0.8,
    line_range: Tuple[int, int] = (1, 4),
    noise_amount: float = 0.02,
):
    """
    Tăng cường dữ liệu bằng cách tạo nhiều bản biến thể từ dataset nhóm 1.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pbm")]
    count = 0

    for filename in input_files:
        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Copy gốc vào thư mục đích để giữ label
        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_orig.png"), image)
        count += 1

        for i in range(copies_per_image):
            aug = image.copy()
            if random.random() < line_prob:
                line_count = random.randint(line_range[0], line_range[1])
                aug = add_random_lines(aug, line_count=line_count)
            if random.random() < salt_pepper_prob:
                aug = add_salt_and_pepper_noise(aug, amount=noise_amount)

            out_name = f"{base_name}_noise_{i}.png"
            cv2.imwrite(os.path.join(output_dir, out_name), aug)
            count += 1

    print(f"[Dataset V2] Generated {count} noisy images into {output_dir}")
    return output_dir


def generate_captcha_library(
    output_dir: str = "dataset_v2/raw",
    count: int = 200,
    width: int = 128,
    height: int = 64,
    length: int = 4,
    noise_dots: int = 300,
    max_lines: int = 4,
):
    """
    Sinh CAPTCHA từ thư viện captcha và bổ sung nhiễu để đạt độ khó tương đương nhóm 2.
    """
    if ImageCaptcha is None or Image is None or ImageDraw is None:
        raise ImportError(
            "Thiếu thư viện captcha hoặc Pillow. Cài đặt bằng: pip install captcha pillow"
        )

    os.makedirs(output_dir, exist_ok=True)
    generator = ImageCaptcha(width=width, height=height)
    count = int(count)
    generated = 0

    for i in range(count):
        text = "".join(random.choices(CHARSET, k=length))
        pil_img = generator.generate_image(text)
        draw = ImageDraw.Draw(pil_img)

        # Thêm nhiễu dòng ngang / chéo vào CAPTCHA
        line_count = random.randint(1, max_lines)
        for _ in range(line_count):
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            x2 = random.randint(0, width - 1)
            y2 = random.randint(0, height - 1)
            thickness = random.randint(1, 2)
            draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=thickness)

        # Thêm nhiễu điểm muối tiêu
        for _ in range(noise_dots):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            color = (255, 255, 255) if random.random() > 0.5 else (0, 0, 0)
            draw.point((x, y), fill=color)

        filename = f"{text}_lib_{i}.png"
        filepath = os.path.join(output_dir, filename)
        pil_img.convert("RGB").save(filepath)
        generated += 1

    print(f"[Dataset V2] Generated {generated} images using captcha library into {output_dir}")
    return output_dir


def prepare_gan_training_data(
    input_dir: str = "dataset/raw/1k_pbm",
    output_dir: str = "dataset_v2/gan_data",
    target_size: Tuple[int, int] = (128, 64),
):
    """
    Chuẩn bị dữ liệu cho GAN từ dataset nhóm 1.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pbm")]
    count = 0

    for filename in files:
        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, target_size)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(output_path, image)
        count += 1

    print(f"[Dataset V2] Prepared {count} GAN training images into {output_dir}")
    return output_dir


if torch is not None and nn is not None:
    class GANDataset(Dataset):
        def __init__(self, folder: str, transform):
            self.folder = folder
            self.transform = transform
            self.files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            path = os.path.join(self.folder, self.files[idx])
            img = Image.open(path).convert("L")
            return self.transform(img)

    class SimpleGenerator(nn.Module):
        def __init__(self, z_dim: int = 100, out_channels: int = 1):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(z_dim, 128, 4, 1, 0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, x):
            return self.net(x)

    class SimpleDiscriminator(nn.Module):
        def __init__(self, in_channels: int = 1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 32, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Linear(128 * 16 * 8, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    def train_gan(
        data_dir: str = "dataset_v2/gan_data",
        output_dir: str = "dataset_v2/raw",
        epochs: int = 3,
        batch_size: int = 16,
        z_dim: int = 100,
        lr: float = 0.0002,
    ) -> Optional[str]:
        """
        Huấn luyện một GAN đơn giản để tạo thêm ảnh CAPTCHA biến thể.
        """
        if torch is None:
            print("PyTorch chưa cài đặt - bỏ qua GAN training.")
            return None
        os.makedirs(output_dir, exist_ok=True)

        target_size = (64, 128)
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = GANDataset(data_dir, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = SimpleGenerator(z_dim=z_dim).to(device)
        discriminator = SimpleDiscriminator().to(device)
        criterion = nn.BCELoss()
        g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)
        for epoch in range(epochs):
            for real in dataloader:
                real = real.to(device)
                batch_size_curr = real.size(0)

                # Train discriminator
                noise = torch.randn(batch_size_curr, z_dim, 1, 1, device=device)
                fake = generator(noise)
                if fake.shape[-2:] != real.shape[-2:]:
                    fake = F.interpolate(
                        fake,
                        size=real.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                real_labels = torch.ones(batch_size_curr, 1, device=device)
                fake_labels = torch.zeros(batch_size_curr, 1, device=device)

                d_optimizer.zero_grad()
                real_loss = criterion(discriminator(real), real_labels)
                fake_loss = criterion(discriminator(fake.detach()), fake_labels)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                d_optimizer.step()

                # Train generator
                g_optimizer.zero_grad()
                g_loss = criterion(discriminator(fake), real_labels)
                g_loss.backward()
                g_optimizer.step()

        with torch.no_grad():
            generated = generator(fixed_noise)
            if generated.shape[-2:] != target_size:
                generated = F.interpolate(
                    generated,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            generated = generated.cpu()
            for i in range(generated.size(0)):
                img_tensor = generated[i].squeeze(0)
                img_tensor = (img_tensor + 1) / 2
                img_np = img_tensor.numpy() * 255
                img_np = img_np.astype(np.uint8)
                save_path = os.path.join(output_dir, f"gan_{i}.png")
                cv2.imwrite(save_path, img_np)

        print(f"[Dataset V2] Saved GAN-generated samples to {output_dir}")
        return output_dir

else:
    def train_gan(*args, **kwargs):
        print("PyTorch chưa cài đặt. Bỏ qua bước GAN.")
        return None


def split_metadata(
    raw_dir: str = "dataset_v2/raw",
    meta_dir: str = "dataset_v2/meta",
    split_ratio: float = 0.8,
    seed: int = 36,
):
    """
    Tạo file train.csv và test.csv từ raw dataset bằng cách chia ngẫu nhiên.
    """
    os.makedirs(meta_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir) if f.lower().endswith((".png", ".pbm"))]
    random.Random(seed).shuffle(files)
    split_idx = int(split_ratio * len(files))

    train_files = files[:split_idx]
    test_files = files[split_idx:]

    with open(os.path.join(meta_dir, "train.csv"), "w") as f:
        for file in train_files:
            f.write(file + "\n")

    with open(os.path.join(meta_dir, "test.csv"), "w") as f:
        for file in test_files:
            f.write(file + "\n")

    print(f"[Dataset V2] Created metadata at {meta_dir} with {len(train_files)} train and {len(test_files)} test files.")
    return meta_dir


def prepare_dataset_v2(
    source_dir: str = "dataset/raw/1k_pbm",
    output_dir: str = "dataset_v2/raw",
    meta_dir: str = "dataset_v2/meta",
    use_gan: bool = True,
    captcha_count: int = 200,
    augment_copies: int = 3,
):
    """
    Chuẩn bị dataset V2 bằng cách tăng cường và sinh dữ liệu.

    Tasks:
    1. Tăng cường ảnh từ dataset nhóm 1 với noise, đường gạch.
    2. Sinh ảnh CAPTCHA mới bằng captcha library.
    3. Chuẩn bị dữ liệu cho GAN và thử nghiệm nếu có PyTorch.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    print("[Dataset V2] Bắt đầu chuẩn bị dữ liệu...")
    augment_dataset_with_noise(
        input_dir=source_dir,
        output_dir=output_dir,
        copies_per_image=augment_copies,
    )

    if ImageCaptcha is not None and Image is not None:
        generate_captcha_library(
            output_dir=output_dir,
            count=captcha_count,
            width=128,
            height=64,
            length=4,
            noise_dots=250,
            max_lines=4,
        )
    else:
        print("[Dataset V2] Bỏ qua captcha library vì thiếu thư viện captcha/Pillow.")

    prepare_gan_training_data(input_dir=source_dir, output_dir=os.path.join(output_dir, "gan_data"))

    if use_gan:
        train_gan(
            data_dir=os.path.join(output_dir, "gan_data"),
            output_dir=output_dir,
            epochs=2,
            batch_size=16,
        )
    else:
        print("[Dataset V2] Gan không được bật. Nếu muốn, chạy train_gan() riêng.")

    split_metadata(raw_dir=output_dir, meta_dir=meta_dir)
    print("[Dataset V2] Chuẩn bị dữ liệu hoàn thành.")
    return output_dir, meta_dir


if __name__ == "__main__":
    prepare_dataset_v2()
