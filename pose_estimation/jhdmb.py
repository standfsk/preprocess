import json
from collections import defaultdict, deque
import numpy as np
import shutil
from pathlib import Path
import os
from tqdm import tqdm
import cv2

def scale_and_center_stack(keypoints_stack):
    # Make a copy to avoid modifying the original
    keypoints_stack_copy = keypoints_stack.copy()

    # find values of [0,0,0] in keypoints_stack_copy's rows and build mask based on value
    zero_mask = np.all(keypoints_stack_copy == [0, 0, 0], axis=2)

    # Calculate zero_point and module_keypoint based on the mean of keypoints across all frames
    zero_points = (keypoints_stack[:, 1, :2] + keypoints_stack[:, 2, :2]) / 2  # left/right shoulder -> neck
    module_keypoints = (keypoints_stack[:, 7, :2] + keypoints_stack[:, 8, :2]) / 2  # hip midpoint

    # hstack zero_points with 0 to fit the size
    # zero_points_for_zero_mask refers to matrix data that has size of (13, 2) which contains value of zero_points
    zero_points_for_zero_mask = np.hstack((zero_points, np.zeros((30, 1))))
    # keypoints_stack_copy[zero_mask] refers to keypoints_stack_copy's position where values are [0, 0, 0]
    # np.where(zero_mask)[0] refers to row position of zero_mask matrix
    # zero_points_for_zero_mask[np.where(zero_mask)[0]] refers to zero_points_for_zero_mask with size of zero_mask's row position
    keypoints_stack_copy[zero_mask] = zero_points_for_zero_mask[np.where(zero_mask)[0]]

    #scale_mags = np.array([np.linalg.norm(x - y) if np.linalg.norm(x - y) > 1 else 1 for x, y in zip(zero_points, module_keypoints)])

    # Center and scale keypoints
    keypoints_stack_copy[:, :, :2] = (keypoints_stack_copy[:, :, :2] - zero_points[:, None])# / scale_mags[:, None, None]

    # min_x = np.min(keypoints_stack_copy[:, :, 0], axis=1)
    # max_x = np.max(keypoints_stack_copy[:, :, 0], axis=1)
    # min_y = np.min(keypoints_stack_copy[:, :, 1], axis=1)
    # max_y = np.max(keypoints_stack_copy[:, :, 1], axis=1)
    #
    # scale_x = max_x - min_x
    # scale_y = max_y - min_y
    #
    # scale_xy = np.where(scale_x < scale_y, 1 / scale_y, 1 / scale_x)

    min_x = np.min(keypoints_stack_copy[:, :, 0])
    max_x = np.max(keypoints_stack_copy[:, :, 0])
    min_y = np.min(keypoints_stack_copy[:, :, 1])
    max_y = np.max(keypoints_stack_copy[:, :, 1])

    scale_x = max_x - min_x
    scale_y = max_y - min_y

    if scale_x < scale_y:
        scale_xy = 1 / scale_y
    else:
        scale_xy = 1 / scale_x

    # Normalize each frame within the entire range
    #keypoints_stack_copy[:, :, 0] = (keypoints_stack_copy[:, :, 0] - min_x) / scale_x
    #keypoints_stack_copy[:, :, 1] = (keypoints_stack_copy[:, :, 1] - min_y) / scale_y
    # keypoints_stack_copy[:, :, 0] = (keypoints_stack_copy[:, :, 0] - min_x[:, np.newaxis]) * scale_xy[:, np.newaxis] + ((1 - scale_x * scale_xy) / 2)[:, np.newaxis]
    # keypoints_stack_copy[:, :, 1] = (keypoints_stack_copy[:, :, 1] - min_y[:, np.newaxis]) * scale_xy[:, np.newaxis] + ((1 - scale_y * scale_xy) / 2)[:, np.newaxis]
    keypoints_stack_copy[:, :, 0] = (keypoints_stack_copy[:, :, 0] - min_x) * scale_xy + (1 - scale_x * scale_xy) / 2
    keypoints_stack_copy[:, :, 1] = (keypoints_stack_copy[:, :, 1] - min_y) * scale_xy + (1 - scale_y * scale_xy) / 2

    zero_points = (keypoints_stack_copy[:, 1, :2] + keypoints_stack_copy[:, 2, :2]) / 2

    # Calculate pairwise distances for each frame using the global zero_point
    for i in range(keypoints_stack_copy.shape[0]):
        keypoints_stack_copy[i, 0, 2] = np.linalg.norm(keypoints_stack_copy[i, 0, :2] - zero_points[0])  # head to zero_point
        keypoints_stack_copy[i, 1, 2] = np.linalg.norm(keypoints_stack_copy[i, 1, :2] - zero_points[1])  # left shoulder to zero_point
        keypoints_stack_copy[i, 2, 2] = np.linalg.norm(keypoints_stack_copy[i, 2, :2] - zero_points[2])  # right shoulder to zero_point
        keypoints_stack_copy[i, 3, 2] = np.linalg.norm(keypoints_stack_copy[i, 3, :2] - keypoints_stack_copy[i, 1, :2])  # left elbow to shoulder
        keypoints_stack_copy[i, 4, 2] = np.linalg.norm(keypoints_stack_copy[i, 4, :2] - keypoints_stack_copy[i, 2, :2])  # right elbow to shoulder
        keypoints_stack_copy[i, 5, 2] = np.linalg.norm(keypoints_stack_copy[i, 5, :2] - keypoints_stack_copy[i, 3, :2])  # left wrist to elbow
        keypoints_stack_copy[i, 6, 2] = np.linalg.norm(keypoints_stack_copy[i, 6, :2] - keypoints_stack_copy[i, 4, :2])  # right wrist to elbow
        keypoints_stack_copy[i, 7, 2] = np.linalg.norm(keypoints_stack_copy[i, 7, :2] - zero_points[7])  # left hip to zero_point
        keypoints_stack_copy[i, 8, 2] = np.linalg.norm(keypoints_stack_copy[i, 8, :2] - zero_points[8])  # right hip to zero_point
        keypoints_stack_copy[i, 9, 2] = np.linalg.norm(keypoints_stack_copy[i, 9, :2] - keypoints_stack_copy[i, 7, :2])  # left knee to hip
        keypoints_stack_copy[i, 10, 2] = np.linalg.norm(keypoints_stack_copy[i, 10, :2] - keypoints_stack_copy[i, 8, :2])  # right knee to hip
        keypoints_stack_copy[i, 11, 2] = np.linalg.norm(keypoints_stack_copy[i, 11, :2] - keypoints_stack_copy[i, 9, :2])  # left ankle to knee
        keypoints_stack_copy[i, 12, 2] = np.linalg.norm(keypoints_stack_copy[i, 12, :2] - keypoints_stack_copy[i, 10, :2])  # right ankle to knee

    keypoints_stack_copy[zero_mask] = [0, 0, 0]
    return keypoints_stack_copy


def convert(x, shape):
    # top_x, top_y, w, h -> cx,cy,w,h
    y = x.copy()
    y[0] = (y[0] + y[2] / 2) / shape[1]
    y[1] = (y[1] + y[3] / 2) / shape[0]
    y[2] = y[2] / shape[1]
    y[3] = y[3] / shape[0]
    return y

def preprocess(json_path, input_path):
    image_data = defaultdict(list)
    label_data = defaultdict(list)
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

        for image_info in json_data['images']:
            file_name = '/'.join(image_info['file_name'].split("/")[1:])
            image_id = image_info['frame_id']
            image_width = image_info['width']
            image_height = image_info['height']
            image_data[image_id] = [file_name, [image_height, image_width]]

        for annotation_info in json_data['annotations']:
            image_matching_id = annotation_info['image_id']
            image_file_name, image_shape = image_data[image_matching_id]

            keypoints = annotation_info['keypoints']
            bbox = annotation_info['bbox']

            scaled_bbox = convert(bbox, image_shape)
            scaled_keypoints = []
            keypoints = np.delete(np.array(keypoints).reshape(-1, 3), [0, 1], axis=0)
            new_keypoints = np.zeros([13, 3])
            if keypoints.any():
                for old_index, new_index in flip_index:
                    new_keypoints[new_index] = keypoints[old_index]
                for x, y, vis in new_keypoints:
                    scaled_x = x / image_shape[1]
                    scaled_y = y / image_shape[0]
                    scaled_keypoints.extend([scaled_x, scaled_y, vis])

                line = f"0 {' '.join(list(map(str, scaled_bbox)))} {' '.join(list(map(str, scaled_keypoints)))}\n"
                label_data[image_file_name].append(line)

    for file_name, label in tqdm(label_data.items(), desc=str(json_path)):
        image_path = input_path/"images"/file_name
        new_label_path = image_path.with_suffix(".txt")
        with open(new_label_path, "w") as new_label_file:
            for line in label:
                new_label_file.write(line)

def remove_invalid_files(input_path):
    dirs = sorted(input_path.glob("images/*/*"))
    for dir in tqdm(dirs, desc="removing invalid files"):
        if os.path.isdir(dir):
            if not len(sorted(dir.glob("*.txt"))):
                shutil.rmtree(dir)
        else:
            os.remove(dir)

def create_action_features(input_path, output_path):
    dirs = sorted(input_path.glob("images/*/*"))

    for dir in tqdm(dirs, desc="creating action features.."):
        action_feature = deque()
        action = dir.parts[3]

        label_paths = dir.glob("*.txt")
        for label_path in label_paths:
            image_path = label_path.with_suffix(".png")
            image = cv2.imread(str(image_path))
            image_height, image_width = image.shape[:2]

            with open(label_path, "r") as label_file:
                label_data = label_file.read().splitlines()
                for label in label_data:
                    skeleton_data = []
                    keypoints = label.split(" ")[5:]
                    keypoints = np.array(keypoints).reshape(-1, 3)
                    for kpt_x, kpt_y, kpt_vis in keypoints:
                        scaled_kpt_x = float(kpt_x) * image_width
                        scaled_kpt_y = float(kpt_y) * image_height
                        skeleton_data.append([scaled_kpt_x, scaled_kpt_y, 0])
                    action_feature.append(skeleton_data)

                    if len(action_feature) == 30:
                        action_feature_data = np.array(action_feature)
                        action_feature_data = scale_and_center_stack(action_feature_data).transpose(1, 0, 2)
                        file_num = len([x for x in os.listdir(output_path) if action in x and x.endswith(".npy")]) + 1
                        np.save(output_path / f"{action}-{file_num:06d}.npy", action_feature_data)

                        action_feature.popleft()




if __name__ == "__main__":
    flip_index = [[0, 0], [1, 2], [2, 1], [3, 8], [4, 7], [5, 4], [6, 3], [7, 10], [8, 9], [9, 6], [10, 5], [11, 12], [12, 11]]
    input_path = Path("dataset/JHMDB")
    for json_path in input_path.glob("annotations/*.json"):
        preprocess(json_path, input_path)

    remove_invalid_files(input_path)

    output_path = Path("dataset/action_features")
    os.makedirs(output_path, exist_ok=True)
    create_action_features(input_path, output_path)
