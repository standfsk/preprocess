from pathlib import Path
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
import cv2
import os
from tqdm import tqdm
import shutil
import json
from utils import xyxy2xywh, enlarge_bbox


def preprocess(input_path, output_path):
    skeleton_map = {
        'nose': 0,
        'leftShoulder': 1,
        'rightShoulder': 2,
        'leftElbow': 3,
        'rightElbow': 4,
        'leftWrist': 5,
        'rightWrist': 6,
        'leftHip': 7,
        'rightHip': 8,
        'leftKnee': 9,
        'rightKnee': 10,
        'leftAnkle': 11,
        'rightAnkle': 12
    }

    annotation_paths = sorted(input_path.glob("annotations/*.json"))
    for annotation_path in tqdm(annotation_paths, desc="preprocessing uav human dataset"):
        with open(annotation_path, "r") as annotation_file:
            annotation_data = json.load(annotation_file)

        new_label_data = []
        for annotation in annotation_data['completions']:
            keypoints = np.zeros((13, 2), dtype=np.float16)
            for annotation_info in annotation['result']:
                if "value" not in list(annotation_info.keys()):
                    continue
                image_height = annotation_info['original_height']
                image_width = annotation_info['original_width']

                kpt_name = annotation_info['value']['keypointlabels'][0]
                if kpt_name in ['leftEye', 'rightEye', 'leftEar', 'rightEar']:
                    continue
                kpt_id = skeleton_map[kpt_name]
                if "x" not in list(annotation_info['value'].keys()):
                    kpt_x = 0.
                    kpt_y = 0.
                elif "y" not in list(annotation_info['value'].keys()):
                    kpt_x = 0.
                    kpt_y = 0.
                elif annotation_info['value']['width'] == None:
                    kpt_x = 0.
                    kpt_y = 0.
                else:
                    kpt_x = annotation_info['value']['x'] / 100 * image_width
                    kpt_y = annotation_info['value']['y'] / 100 * image_height
                keypoints[kpt_id] = [kpt_x, kpt_y]

            bbox_x1 = np.min(keypoints[:, 0]) / image_width
            bbox_y1 = np.min(keypoints[:, 1]) / image_height
            bbox_x2 = np.max(keypoints[:, 0]) / image_width
            bbox_y2 = np.max(keypoints[:, 1]) / image_height
            bbox = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
            bbox = xyxy2xywh(bbox)
            bbox = enlarge_bbox(bbox, [image_height, image_width], ratio=0.15)
            bbox = list(map(str, bbox))

            new_keypoints = keypoints.copy()
            new_keypoints[:, 0] = new_keypoints[:, 0] / image_width
            new_keypoints[:, 1] = new_keypoints[:, 1] / image_height
            vis = np.full([new_keypoints.shape[0], 1], 2.)
            new_keypoints = np.hstack((new_keypoints, vis))
            new_keypoints = list(map(str, new_keypoints.reshape(-1)))

            new_label = f"0 {' '.join(bbox)} {' '.join(new_keypoints)}\n"
            new_label_data.append(new_label)

        new_label_path = (output_path/"labels"/annotation_path.name).with_suffix(".txt")
        with open(new_label_path, "w") as new_label_file:
            for new_label in new_label_data:
                new_label_file.write(new_label)


if __name__ == "__main__":
    input_path = Path("dataset/uav_human")
    output_path = input_path/"train"
    os.makedirs(output_path/"labels", exist_ok=True)
    if os.path.exists(input_path/"images"):
        shutil.move(input_path/"images", output_path)
    preprocess(input_path, output_path)
