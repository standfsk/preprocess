from pathlib import Path
import cv2
import os
from tqdm import tqdm

video_paths = sorted(Path("dataset/piw/videos").glob("*/**/*.mp4"))
image_counts = {}
for video_path in video_paths:
    modality = video_path.parts[2]
    annotation_type = video_path.parts[3]
    output_path = Path("dataset/piw")/modality/annotation_type/"train/images"
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_interval = 10
    frame_count = 0
    image_number = 1

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(range(frames), total=frames, desc=f"Processing {video_path}")
    for i in pbar:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            if modality == "rgb":
                frame = cv2.resize(frame, (1920, 1080))
            filename = f"{modality}__{annotation_type}__{video_path.stem}__{image_number:04d}.jpg"
            save_path = output_path / filename
            cv2.imwrite(str(save_path), frame)
            image_number += 1
        frame_count += 1
