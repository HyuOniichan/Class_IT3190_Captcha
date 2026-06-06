from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path_or_pil):
    """
    Algorithm-I (Steps 1-5): Image Preprocessing
    1. Prepare input image
    2. Grayscale Conversion
    3. Resizing to 200x80 (Width x Height) -> (80, 200) in tensor shape
    4. Normalization (0-1 range)
    5. Matrix Transposition to match CNN input format
    """
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")
        
    transform = transforms.Compose([
        # 2. Grayscale Conversion
        transforms.Grayscale(num_output_channels=1),
        # 3. Resizing (H=80, W=200)
        transforms.Resize((80, 200)),
        # 4. Normalization (ToTensor automatically scales pixels to 0-1)
        transforms.ToTensor(),
    ])
    
    # Apply transforms (shape: 1, 80, 200)
    tensor = transform(img)
    
    # 5. Matrix Transposition: add batch dimension -> (1, 1, 80, 200)
    tensor = tensor.unsqueeze(0)
    return tensor
