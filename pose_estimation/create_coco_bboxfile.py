from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image
import shutil
import json
from collections import defaultdict

categories = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]

json_path = "val.json"

annotations = []
ann_id = 1
# {"bbox": [249.8199079291458, 175.21093805640606, 74.00419360691592, 55.626325589288854], "category_id": 1, "image_id": 532481, "score": 0.9992738366127014}

image_data = defaultdict(list)
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
        label_data = {'bbox': bbox, 'category_id': category_id, 'image_id': image_matching_id, 'score': 1.0, 'keypoints': keypoints}
        annotations.append(label_data)

with open("test.json", 'w') as f:
    json.dump(annotations, f)


