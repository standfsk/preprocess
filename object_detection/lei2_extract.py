from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import cv2
from utils import xyxy2xywh

input_path = Path("archive")
output_path = Path("Lei2")
os.makedirs(output_path/"images", exist_ok=True)
os.makedirs(output_path/"labels", exist_ok=True)
label_paths = sorted(input_path.glob("*/Annotation_files/*.txt"))
pbar = tqdm(label_paths, total=len(label_paths))
color = [0, 255, 255]

for index, label_path in enumerate(pbar):
    pbar.set_description(f"Processing {label_path}")

    video_path = (label_path.parent.with_name("Videos") / label_path.name).with_suffix(".avi")
    cap = cv2.VideoCapture(str(video_path))
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    video_num = label_path.stem.split(" ")[-1].strip("()")

    with open(label_path, "r") as label_file:
        label_data = label_file.read().splitlines()
        label_data = [label for label in label_data if label.strip() and len(label.split(","))>=2]
        for label in label_data:
            frame_count = int(label.split(",")[0])
            class_id = label.split(",")[1]
            bbox = label.split(",")[2:]
            bbox = list(map(int, bbox))
            if sum(bbox) == 0:
                new_label = "\n"
            else:
                bbox = xyxy2xywh(bbox)
                bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
                bbox = list(map(str, bbox))
                new_label = f"{class_id} {' '.join(bbox)}\n"

            label_save_path = output_path/"labels"/f"{label_path.parents[1].stem}_{video_num}_{frame_count:04d}.txt"
            with open(label_save_path, "w") as new_label_file:
                new_label_file.write(new_label)

    frame_count = 1
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        image_save_path = output_path/"images"/f"{label_path.parents[1].stem}_{video_num}_{frame_count:04d}.jpg"
        cv2.imwrite(str(image_save_path), frame)
        frame_count += 1