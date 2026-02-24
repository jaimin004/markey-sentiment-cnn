import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= FEATURE EXTRACTOR =================
class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Identity()  # remove classifier

    def forward(self, x):
        features = self.mobilenet(x)  # Output: [batch, 1280]
        return features


# ================= IMAGE TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================= HELPER FUNCTION =================
@torch.no_grad()
def extract_visual_features(image_path):
    model = MobileNetV2FeatureExtractor().to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    features = model(image)  # shape: [1, 1280]
    return features
