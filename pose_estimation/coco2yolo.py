import pickle
import os
import cv2  # Only needed to get image sizes
from pathlib import Path
import json
from tqdm import tqdm
import shutil
import numpy as np
from collections import defaultdict


def convert(x, shape):
    # top_x, top_y, w, h -> cx,cy,w,h
    y = x.copy()
    y[0] = (y[0] + y[2] / 2) / shape[1]
    y[1] = (y[1] + y[3] / 2) / shape[0]
    y[2] = y[2] / shape[1]
    y[3] = y[3] / shape[0]
    return y


input_path = Path("dataset/ms_coco_pose")

json_paths = Path(input_path/"annotations").glob("*.json")
for json_path in json_paths:
    split = "train" if "train" in json_path.name else "val"
    os.makedirs(input_path/split/"images", exist_ok=True)
    os.makedirs(input_path/split/"labels", exist_ok=True)
    image_data = defaultdict(list)
    label_data = defaultdict(list)
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
        for image_info in json_data['images']:
            image_id = int(image_info['id'])
            image_file_name = image_info['file_name']
            image_height = int(image_info['height'])
            image_width = int(image_info['width'])
            image_data[image_id] = [image_file_name, [image_height, image_width]]

        for annotation_info in json_data['annotations']:
            image_matching_id = int(annotation_info['image_id'])
            bbox = annotation_info['bbox']
            keypoints = annotation_info['keypoints']
            category_id = int(annotation_info['category_id'])

            image_file_name, image_shape = image_data[image_matching_id]
            scaled_bbox = convert(bbox, image_shape)
            scaled_keypoints = []
            keypoints = np.array(keypoints).reshape(-1, 3)
            keypoints = np.delete(keypoints, [1,2,3,4], axis=0)
            if keypoints.any():
                for x, y, vis in keypoints:
                    scaled_x = x / image_shape[1]
                    scaled_y = y / image_shape[0]
                    scaled_keypoints.extend([scaled_x, scaled_y, vis])

                line = f"0 {' '.join(list(map(str, scaled_bbox)))} {' '.join(list(map(str, scaled_keypoints)))}\n"
                label_data[image_file_name].append(line)

        # save
        for image_file_name, lines in tqdm(label_data.items(), desc=f"processing {split}"):
            # label
            label_save_path = input_path / split / "labels" / image_file_name
            with open(label_save_path.with_suffix(".txt"), "w") as label_file:
                for line in lines:
                    label_file.write(line)

