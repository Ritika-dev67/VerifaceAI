import cv2
import os

SOURCE = "../dataset/video_split"
DEST = "../dataset/video_frames"

splits = ["train", "val", "test"]
classes = ["real", "fake"]

FRAMES_PER_VIDEO = 10

for split in splits:

    for class_name in classes:

        source_folder = os.path.join(
            SOURCE,
            split,
            class_name
        )

        destination_folder = os.path.join(
            DEST,
            split,
            class_name
        )

        os.makedirs(destination_folder, exist_ok=True)

        videos = os.listdir(source_folder)

        for video_name in videos:

            video_path = os.path.join(
                source_folder,
                video_name
            )

            cap = cv2.VideoCapture(video_path)

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if total_frames <= 0:
                print("Could not read:", video_name)
                cap.release()
                continue

            frame_positions = [
                int(i * total_frames / FRAMES_PER_VIDEO)
                for i in range(FRAMES_PER_VIDEO)
            ]

            video_id = os.path.splitext(video_name)[0]

            for frame_number, position in enumerate(frame_positions):

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    position
                )

                success, frame = cap.read()

                if success:

                    filename = (
                        f"{video_id}_frame_{frame_number}.jpg"
                    )

                    output_path = os.path.join(
                        destination_folder,
                        filename
                    )

                    cv2.imwrite(
                        output_path,
                        frame
                    )

            cap.release()

print("\nFrame extraction completed!")