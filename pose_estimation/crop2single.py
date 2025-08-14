from pathlib import Path
import numpy as np
import cv2
import os
from tqdm import tqdm
from itertools import combinations
from utils import xywh2xyxy

def get_iou(boxA, boxB):
    x1_A, y1_A, x2_A, y2_A = boxA
    x1_B, y1_B, x2_B, y2_B = boxB

    x_left = max(x1_A, x1_B)
    y_top = max(y1_A, y1_B)
    x_right = min(x2_A, x2_B)
    y_bottom = min(y2_A, y2_B)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_A = (x2_A - x1_A) * (y2_A - y1_A)
    area_B = (x2_B - x1_B) * (y2_B - y1_B)

    union_area = area_A + area_B - intersection_area
    iou = intersection_area / union_area
    return iou

if __name__ == "__main__":
    image_paths = sorted(Path("dataset").glob("*/*/images/*.jpg"))
    output_path = Path("dataset/single")
    for image_path in tqdm(image_paths):
        dataset_name = image_path.parts[1]
        subset = image_path.parts[2]
        save_path = output_path/"train/images"
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path.with_name("labels"), exist_ok=True)

        if os.path.exists(save_path / f"{image_path.stem}_001.jpg"):
            continue
        elif dataset_name in ["cropped", "coco", "single", "dd", "JHMDB"]:
            continue

        image = cv2.imread(str(image_path))
        image_height, image_width = image.shape[:2]

        label_path = (image_path.parent.with_name("labels")/image_path.name).with_suffix(".txt")
        try:
            with open(label_path, "r") as label_file:
                label_data = label_file.read().splitlines()

                if len(label_data) >= 2:
                    bbox_pairs = list(combinations(range(len(label_data)), 2))
                    bboxes = []
                    for label in label_data:
                        bbox = label.split(" ")[1:5]
                        bbox = list(map(float, bbox))
                        bbox[0] = bbox[0] * image_width
                        bbox[1] = bbox[1] * image_height
                        bbox[2] = bbox[2] * image_width
                        bbox[3] = bbox[3] * image_height
                        bbox = xywh2xyxy(bbox)
                        bbox = list(map(int, bbox))

                        x1 = max(0, bbox[0])
                        y1 = max(0, bbox[1])
                        x2 = min(bbox[2], image_width)
                        y2 = min(bbox[3], image_height)
                        bboxes.append([x1, y1, x2, y2])

                    overlaps = []
                    mask = np.ones(len(bboxes), dtype=bool)
                    bboxes = np.array(bboxes)
                    for bbox_pair in bbox_pairs:
                        b1, b2 = bboxes[list(bbox_pair)]
                        iou = get_iou(b1, b2)
                        if iou > 0.1:
                            overlaps.extend(bbox_pair)

                    mask[overlaps] = False
                    label_data = label_data[mask]

                for label_index, label in enumerate(label_data, start=1):
                    class_id = label.split(" ")[0]
                    bbox = label.split(" ")[1:5]
                    keypoints = label.split(" ")[5:]

                    bbox = list(map(float, bbox))
                    bbox[0] = bbox[0] * image_width
                    bbox[1] = bbox[1] * image_height
                    bbox[2] = bbox[2] * image_width
                    bbox[3] = bbox[3] * image_height
                    bbox = xywh2xyxy(bbox)
                    bbox = list(map(int, bbox))

                    x1 = max(0, bbox[0])
                    y1 = max(0, bbox[1])
                    x2 = min(bbox[2], image_width)
                    y2 = min(bbox[3], image_height)

                    image_ = image.copy()
                    cropped_image = image_[y1: y2, x1:x2]
                    cropped_height, cropped_width = cropped_image.shape[:2]

                    adjusted_keypoints = []
                    keypoints = np.array(keypoints).astype(np.float16).reshape(13, 3)
                    for kp in keypoints:
                        if kp.any():
                            kp_x, kp_y, kp_conf = kp
                            kp_x = kp_x * image_width
                            kp_y = kp_y * image_height

                            if x1 <= kp_x <= x2 and y1 <= kp_y <= y2:
                                kp_x = (kp_x - x1) / cropped_width
                                kp_y = (kp_y - y1) / cropped_height

                                adjusted_keypoints.append(str(kp_x))
                                adjusted_keypoints.append(str(kp_y))
                                adjusted_keypoints.append(str(kp_conf))
                            else:
                                adjusted_keypoints.extend(('0.', '0.', '0.'))
                        else:
                            adjusted_keypoints.extend(('0.', '0.', '0.'))

                    if adjusted_keypoints:
                        cv2.imwrite(str(save_path / f"{image_path.stem}_{label_index:03d}.jpg"), cropped_image)

                        new_label = f"{class_id} 0.5 0.5 1.0 1.0 {' '.join(adjusted_keypoints)}\n"
                        with open(save_path.with_name("labels")/f"{image_path.stem}_{label_index:03d}.txt", "w") as new_label_file:
                            new_label_file.write(new_label)
        except:
            continue
