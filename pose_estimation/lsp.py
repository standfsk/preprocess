from pathlib import Path
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
import cv2
import os
from tqdm import tqdm
import shutil
from utils import xyxy2xywh, enlarge_bbox

def preprocess(input_path):
    flip_index = np.array([[0,12], [1,10], [2,8,], [3,7], [4,9], [5,11], [6,6], [7,4], [8,2], [9,1], [10,3], [11,5], [13,0]])

    annotation_data = loadmat(str(input_path/"joints.mat"))
    for frame_id in tqdm(range(annotation_data['joints'].shape[2]), desc="preprocessing lsp.."):
        image_path = input_path/"train/images"/f"im{frame_id+1:05}.jpg"
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        keypoints = annotation_data['joints'][..., frame_id]
        keypoints = np.clip(keypoints, 0, 100000)
        keypoints[..., 2][np.where(keypoints[:, 2] == 1.)] = 2.
        new_keypoints = np.zeros((13, 3))
        new_keypoints[flip_index[:, 1]] = keypoints[flip_index[:, 0]]

        # bbox
        new_keypoints_without_none = new_keypoints[np.where(new_keypoints[:, 2] != 0.)]
        bbox_x1 = np.min(new_keypoints_without_none[:, 0]) / image_width
        bbox_y1 = np.min(new_keypoints_without_none[:, 1]) / image_height
        bbox_x2 = np.max(new_keypoints_without_none[:, 0]) / image_width
        bbox_y2 = np.max(new_keypoints_without_none[:, 1]) / image_height
        bbox = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
        bbox = xyxy2xywh(bbox)
        bbox = enlarge_bbox(bbox, [image_height, image_width], ratio=0.15)
        bbox = list(map(str, bbox))

        # rescale keypoints
        new_keypoints[:, 0] = new_keypoints[:, 0] / image_width
        new_keypoints[:, 1] = new_keypoints[:, 1] / image_height
        new_keypoints = new_keypoints.reshape(-1)
        new_keypoints = list(map(str, new_keypoints))
        new_label = f"0 {' '.join(bbox)} {' '.join(new_keypoints)}\n"

        new_label_path = (image_path.parent.with_name("labels")/image_path.name).with_suffix(".txt")
        with open(new_label_path, "w") as new_label_file:
            new_label_file.write(new_label)

if __name__ == "__main__":
    input_path = Path("dataset/lsp")
    preprocess(input_path)
