from pathlib import Path
import cv2
from tqdm import tqdm

frame_count = 1
frame_interval = 2  # Save every 2nd frame

video_paths = Path("dataset/dddd").glob("*")
for video_path in video_paths:
    cap = cv2.VideoCapture(str(video_path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(range(frames), total=frames, desc=f"Processing {video_path.name}")
    for i in pbar:
        ret, frame = cap.read()
        if not ret:
            break

        if i % frame_interval == 0:
            cv2.imwrite(f"dataset/ddd/train/images/frame_{frame_count:08d}.jpg", frame)
            frame_count += 1

    cap.release()
