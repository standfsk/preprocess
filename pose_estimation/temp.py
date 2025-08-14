import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
import random
from pathlib import Path
import argparse
from utils import xywh2xyxy
random.seed(42)
import json

if __name__ == "__main__":
    color = [0, 0, 255]
    image_path = Path("SNC_2024_09_05_17_11_10_00014.jpg")
    label_path = image_path.with_suffix(".json")
    image = cv2.imread(str(image_path))
    image_height, image_width = image.shape[:2]
    with open(label_path, 'r', encoding='utf-8') as label_file:
        label_data = json.load(label_file)
        for annotation_info in label_data['annotation_info']:
            x = int(float(annotation_info['x']) * image_width)
            y = int(float(annotation_info['y']) * image_height)
            label = annotation_info['label']
            if (x,y) == (0,0):
                continue
            else:
                cv2.putText(image, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(image, (x,y), 4, color, -1)

        #     cv2.rectangle(img, (bbox[0] - 1, bbox[1] - 20), (bbox[0] + 5 * 12, bbox[1]+5), color, -1)
        #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        #     cv2.putText(img, text, (bbox[0] + 5, bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite("res.jpg", image)
