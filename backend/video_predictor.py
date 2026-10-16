import cv2
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

classes = ["fake", "real"]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "video_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "error": "Could not open video"
        }

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_predictions = []
    confidences = []

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Analyze every 10th frame
        if frame_number % 10 == 0:

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                frame_rgb
            )

            image = transform(image).unsqueeze(0)

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

            frame_predictions.append(
                prediction.item()
            )

            confidences.append(
                confidence.item()
            )

        frame_number += 1

    cap.release()

    if not frame_predictions:

        return {
            "error": "No frames could be processed"
        }

    fake_count = frame_predictions.count(0)
    real_count = frame_predictions.count(1)

    if fake_count > real_count:
        final_prediction = "FAKE"
    else:
        final_prediction = "REAL"

    average_confidence = (
        sum(confidences) /
        len(confidences)
    ) * 100

    return {
        "prediction": final_prediction,
        "confidence": round(
            average_confidence,
            2
        ),
        "fake_frames": fake_count,
        "real_frames": real_count,
        "total_analyzed_frames": len(frame_predictions),
        "total_video_frames": total_frames
    }
if __name__ == "__main__":

    video_path = input(
        "Enter video path: "
    )

    result = predict_video(video_path)

    print(result)