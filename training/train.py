import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load training images
train_dataset = datasets.ImageFolder(
    "../dataset/images/train",
    transform=transform
)

# Load validation images
val_dataset = datasets.ImageFolder(
    "../dataset/images/validate",
    transform=transform
)

# Load test images
test_dataset = datasets.ImageFolder(
    "../dataset/images/test",
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
from collections import Counter

print("Train class distribution:")
print(Counter(train_dataset.targets))

print("Validation class distribution:")
print(Counter(val_dataset.targets))

print("Test class distribution:")
print(Counter(test_dataset.targets))
# Display information
print("Classes:", train_dataset.classes)
print("Class mapping:", train_dataset.class_to_idx)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))

# Test loading one batch
images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print(train_dataset.class_to_idx)

