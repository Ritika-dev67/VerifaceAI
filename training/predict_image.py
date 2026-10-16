import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Same preprocessing used during testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "../backend/image_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# Load image
image_path = input("Enter image path: ").strip().strip('"')
image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)

# Prediction
with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(
        probabilities,
        1
    )

# IMPORTANT:
# Check this from your dataset:
# print(train_dataset.class_to_idx)

from torchvision import datasets

dataset = datasets.ImageFolder(
    "../dataset/images/train"
)

classes = dataset.classes

print("Classes:", classes)
print("Mapping:", dataset.class_to_idx)

prediction = classes[predicted.item()]

print("Prediction:", prediction.upper())
print(
    "Confidence:",
    f"{confidence.item() * 100:.2f}%"
)

