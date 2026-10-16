import os
import random
import shutil

random.seed(42)

SOURCE = "../dataset/video_dataset"
DEST = "../dataset/video_split"

classes = ["real", "fake"]

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

for class_name in classes:

    source_folder = os.path.join(SOURCE, class_name)

    videos = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    random.shuffle(videos)

    total = len(videos)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_videos = videos[:train_end]
    val_videos = videos[train_end:val_end]
    test_videos = videos[val_end:]

    for split, split_videos in [
        ("train", train_videos),
        ("val", val_videos),
        ("test", test_videos)
    ]:

        destination = os.path.join(
            DEST,
            split,
            class_name
        )

        os.makedirs(destination, exist_ok=True)

        for video in split_videos:

            source_path = os.path.join(
                source_folder,
                video
            )

            destination_path = os.path.join(
                destination,
                video
            )

            shutil.copy2(
                source_path,
                destination_path
            )

        print(
            class_name,
            split,
            len(split_videos)
        )

print("\nVideo splitting completed!")