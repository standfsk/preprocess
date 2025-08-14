from pathlib import Path
import cv2
import numpy as np
import os
import time
from tqdm import tqdm
import shutil

output_path = Path("dataset/ms_coco_pose/head_less")
label_paths = sorted(Path("dataset/ms_coco_pose").glob("*/labels/*.txt"))
for label_path in tqdm(label_paths):
    move = False
    image_path = (label_path.parent.with_name("images")/label_path.name).with_suffix(".jpg")

    subset = label_path.parts[2]
    os.makedirs(output_path/subset/"images", exist_ok=True)
    os.makedirs(output_path/subset/"labels", exist_ok=True)
    with open(str(label_path), "r") as label_file:
        label_data = label_file.read().splitlines()
        for label in label_data:
            class_id = label.split(" ")[0]
            bbox = label.split(" ")[1:5]
            keypoints = label.split(" ")[5:]
            keypoints = np.array(keypoints).reshape(-1, 3)
            if np.array_equal(keypoints[0], ["0.0", "0.0", "0"]):
                move = True
                break
    if move:
        shutil.move(image_path, output_path/subset/"images"/image_path.name)
        shutil.move(label_path, output_path/subset/"labels"/label_path.name)
