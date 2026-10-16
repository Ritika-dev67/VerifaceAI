import cv2
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "../backend/video_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


classes = ["fake", "real"]


video_path = input(
    "Enter video path: "
)


cap = cv2.VideoCapture(
    video_path
)


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)


print("Total frames:", total_frames)
print("FPS:", fps)


frame_predictions = []

frame_number = 0

while True:

    success, frame = cap.read()

    if not success:
        break


    # Process every 10th frame
    if frame_number % 10 == 0:

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            frame_rgb
        )

        image = transform(
            image
        ).unsqueeze(0)

        image = image.to(device)


        with torch.no_grad():

            output = model(image)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                1
            )


        predicted_class = classes[
            prediction.item()
        ]

        frame_predictions.append(
            prediction.item()
        )


        print(
            f"Frame {frame_number}: "
            f"{predicted_class.upper()} "
            f"({confidence.item() * 100:.2f}%)"
        )


    frame_number += 1


cap.release()


if len(frame_predictions) == 0:

    print("No frames were processed.")

else:

    fake_count = frame_predictions.count(0)
    real_count = frame_predictions.count(1)


    print("\n----------------------")
    print("VIDEO RESULT")
    print("----------------------")

    val_accuracy = (
        100.0 * max(fake_count, real_count) / (fake_count + real_count)
        if (fake_count + real_count) > 0
        else 0.0
    )

    print(
        "Fake frames:",
        fake_count
    )

    print(
        "Real frames:",
        real_count
    )

    if fake_count > real_count:
        print("Final Prediction: FAKE")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
    else:
        print("Final Prediction: REAL")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
