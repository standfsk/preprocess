from pathlib import Path
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
import cv2
import os
from tqdm import tqdm
import shutil

def preprocess(input_path):
    # label info: [head, right wrist, left wrist, right elbow, left elbow, right shoulder, left shoulder]
    flip_index = np.array([[0, 0], [1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1]])

    annotation_data = loadmat(str(input_path/"YouTube_Pose_dataset.mat"))['data'][0]
    for annotation_info in tqdm(annotation_data):
        video_name = annotation_info[1].item()
        annotations = annotation_info[2]
        frame_ids = list(annotation_info[3][0])

        for frame_index, frame_id in zip(range(annotations.shape[2]), frame_ids):
            # image info
            image_path = input_path / "frames" / video_name / f"frame_{frame_id:06d}.jpg"
            image = cv2.imread(image_path)
            image_height, image_width = image.shape[:2]

            # label info
            keypoints = annotations[..., frame_index]
            bbox_x1 = np.min(keypoints[:, 0]) / image_width
            bbox_y1 = np.min(keypoints[:, 1]) / image_height
            bbox_x2 = np.max(keypoints[:, 0]) / image_width
            bbox_y2 = np.max(keypoints[:, 1]) / image_height
            bbox = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
            bbox = list(map(str, bbox))

            new_keypoints = np.zeros((2, 13))
            new_keypoints[:, flip_index[:, 1]] = keypoints[:, flip_index[:, 0]]
            new_keypoints[0, :] = new_keypoints[0, :] / image_width
            new_keypoints[1, :] = new_keypoints[1, :] / image_height
            vis = np.full((1, new_keypoints.shape[-1]), 2.)
            new_keypoints = np.vstack((new_keypoints, vis))
            new_keypoints = new_keypoints.T.reshape(-1)
            new_keypoints = list(map(str, new_keypoints))
            new_label = f"0 {' '.join(bbox)} {' '.join(new_keypoints)}\n"

            with open(image_path.with_suffix(".txt"), "w") as new_label_file:
                new_label_file.write(new_label)

def consolidate_files(input_path):
    dirs = sorted(input_path.glob("frames/*"))
    for dir in tqdm(dirs):
        video_name = dir.stem
        image_paths = dir.glob("*.jpg")
        for image_index, image_path in enumerate(image_paths, start=1):
            image_save_path = input_path/"train/images"/f"{video_name}-{image_index:04d}.jpg"
            shutil.copy(image_path, image_save_path)

            label_path = image_path.with_suffix(".txt")
            label_save_path = (image_save_path.parent.with_name("labels")/image_save_path.name).with_suffix(".txt")
            shutil.copy(label_path, label_save_path)




if __name__ == "__main__":
    input_path = Path("dataset/upper_body")
    preprocess(input_path)
    consolidate_files(input_path)
