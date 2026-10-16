
from torchvision import datasets
from collections import Counter

for split in ["train", "val", "test"]:
    path = f"../dataset/video_frames/{split}"
    dataset = datasets.ImageFolder(path)

    print(f"\n{split.upper()}")
    print("Classes:", dataset.classes)
    print("Mapping:", dataset.class_to_idx)
    print("Total frames:", len(dataset))
    print("Class counts:", Counter(dataset.targets))
