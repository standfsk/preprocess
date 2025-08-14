import argparse
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Set
import json

import imagehash
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from collections import defaultdict
import numpy as np


class Preprocessor:
    def __init__(self, directories: List[str]):
        """
        Parameters:
            directories: directories to run preprocessing methods
        """
        self.directories = directories
        self.image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        self.set_image_format()
        self.remove_invalid_labels()

    def set_image_format(self, quality=95) -> None:
        """
        set image format to jpg
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            image_paths = input_path.glob("*/images/*")
            for image_path in image_paths:
                if image_path.suffix == ".jpg":
                    continue
                print(f"Setting image format to jpg {image_path.name} -> {image_path.stem}.jpg")
                image = Image.open(str(image_path)).convert("RGB")
                new_image_path = image_path.with_suffix(".jpg")
                image.save(str(new_image_path), "JPEG", quality=quality)
                os.remove(image_path)

    def remove_invalid_labels(self) -> None:
        """
        remove label if no matching image
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            label_paths = input_path.glob("*/labels/*.txt")
            for label_path in label_paths:
                image_path = (label_path.parent.with_name("images")/label_path.name).with_suffix(".jpg")
                if not os.path.exists(image_path):
                    print(f"Removing invalid label: {label_path}")
                    os.remove(str(label_path))

    def merge(self) -> None:
        """
        merge (val, test) directories into train
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            for subset in ["val", "test"]:
                if subset == "val":
                    if not os.path.exists(input_path/subset):
                        subset = "valid"
                image_paths = input_path.glob(f"{subset}/images/*.jpg")
                for image_path in image_paths:
                    new_image_path = input_path/"train/images"/image_path.name
                    os.rename(image_path, new_image_path)

                    label_path = (image_path.parent.with_name("labels")/image_path.name).with_suffix(".txt")
                    new_label_path = input_path/"train/labels"/label_path.name
                    os.rename(label_path, new_label_path)

                if os.path.exists(input_path/subset):
                    shutil.rmtree(input_path/subset)

            merged_file_num = sorted((input_path/"train/images").glob("*.jpg"))
            print(f"merged file num: {len(merged_file_num)}")

    def split(self) -> None:
        """
        split directories into (train(90%), val(10%))
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            image_paths = sorted((input_path/"train").glob("images/*.jpg"))
            train_image_paths, val_image_paths = train_test_split(image_paths, test_size=0.1, random_state=42)
            print(f"train file num: {len(train_image_paths)}")
            print(f"val file num: {len(val_image_paths)}")
            for subset, subset_image_paths in [["val", val_image_paths]]:
                os.makedirs(input_path/subset/"images", exist_ok=True)
                os.makedirs(input_path/subset/"labels", exist_ok=True)

                for index, image_path in enumerate(subset_image_paths):
                    label_path = (image_path.parent.with_name("labels")/image_path.name).with_suffix(".txt")
                    os.rename(image_path, input_path/subset/"images"/image_path.name)
                    os.rename(label_path, input_path/subset/"labels"/label_path.name)

    def get_image_hash(self, image_path: str) -> Optional[imagehash.ImageHash]:
        """
        get image hash
        """
        try:
            img = Image.open(image_path)
            return imagehash.average_hash(img)
        except Exception as e:
            print(f"Error hashing image {image_path}: {e}")
            return None

    def remove_duplicates(self, threshold=0):
        """
        remove duplicates

        Parameters:
            threshold (int): threshold to check image hash difference
        """
        hash_dict = {}
        for base_dir in self.directories:
            for subset in ["train", "val", "test"]:
                images_path = Path(base_dir) / subset / "images"

                if not os.path.exists(images_path):
                    print(f"Skipping non-existent path: {images_path}")
                    continue

                for pth in tqdm(sorted(images_path.rglob("*")), desc=f"Remove {base_dir}/{subset} duplicates"):
                    if pth.is_file():
                        if any(pth.name.lower().endswith(ext) for ext in self.image_extensions):
                            image_hash = self.get_image_hash(str(pth))

                            if image_hash is None:
                                continue

                            # Remove similar image hashes
                            for (existing_filename, existing_hash), existing_path in hash_dict.items():
                                if abs(image_hash - existing_hash) <= threshold:
                                    if os.path.exists(pth):
                                        os.remove(pth)

                                    # Remove corresponding label
                                    label_path = (pth.parent.with_name("labels") / pth.name).with_suffix(".txt")
                                    if os.path.exists(label_path):
                                        os.remove(label_path)
                            else:
                                # If no similar hash found, add to hash dictionary
                                hash_dict[(pth.name, image_hash)] = pth

    def resize(self, target_image_size: List[int]) -> None:
        """
        resize image

        Parameters:
            target_image_size (List[int]): target image size [Width, Height]
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            image_paths = input_path.glob("*/images/*.jpg")
            for image_path in image_paths:
                image = cv2.imread(str(image_path))
                resized_image = cv2.resize(image, target_image_size)
                cv2.imwrite(str(image_path), resized_image)

        print(f"resized images in {self.directories} to {target_image_size}")

    def yolo2coco(self, output_path: str) -> None:
        """
        yolo format to coco format

        Parameters:
            output_path (str): path to save converted format file
        """
        categories = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]

        output_path = Path(output_path)
        os.makedirs(output_path/"annotations", exist_ok=True)
        for base_dir in self.directories:
            input_path = Path(base_dir)
            for subset in ['train', 'val']:
                os.makedirs(output_path/subset, exist_ok=True)
                images = []
                annotations = []
                ann_id = 1

                image_paths = sorted(input_path.glob(f"{subset}/images/*.jpg"))
                for image_id, image_path in tqdm(enumerate(image_paths, start=1), total=len(image_paths), desc=f"Converting {base_dir}/{subset} to COCO format"):
                    width, height = Image.open(image_path).size

                    images.append({
                        "id": image_id,
                        "file_name": image_path.name,
                        "width": width,
                        "height": height
                    })

                    label_path = (image_path.parent.with_name("labels")/image_path.name).with_suffix(".txt")
                    with open(str(label_path), 'r') as label_file:
                        for line in label_file:
                            parts = list(map(float, line.strip().split()))
                            class_id = int(parts[0])
                            x_center, y_center, w, h = parts[1:5]
                            kp_data = parts[5:]

                            x = (x_center - w / 2) * width
                            y = (y_center - h / 2) * height
                            bbox = [x, y, w * width, h * height]

                            keypoints = []
                            num_kps = 0
                            for i in range(0, len(kp_data), 3):
                                kp_x = kp_data[i] * width
                                kp_y = kp_data[i + 1] * height
                                v = 0 if int(kp_data[i + 2]) == 3 else int(kp_data[i + 2])
                                keypoints.extend([kp_x, kp_y, v])
                                if v > 0:
                                    num_kps += 1

                            annotations.append({
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": bbox,
                                "keypoints": keypoints,
                                "num_keypoints": num_kps,
                                "iscrowd": 0,
                                "area": bbox[2] * bbox[3]
                            })

                            ann_id += 1
                    shutil.copy(str(image_path), output_path/subset/image_path.name)

                coco_dict = {
                    "images": images,
                    "annotations": annotations,
                    "categories": [{"id": i, "name": name} for i, name in enumerate(categories)]
                }

                with open(output_path/"annotations"/f"{subset}.json", 'w') as f:
                    json.dump(coco_dict, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images")
    parser.add_argument("--dirs", required=True, nargs="+", help="directory paths")
    parser.add_argument("--merge", action="store_true", help="merge directories")
    parser.add_argument("--split", action="store_true", help="split directories")
    parser.add_argument("--remove-dup", action="store_true", help="remove duplicates")
    parser.add_argument("--resize", nargs=2, type=int, help="resize image: [Width, Height]")
    parser.add_argument("--yolo2coco", action="store_true", help="yolo format to coco format")


    args = parser.parse_args()
    preprocessor = Preprocessor(args.dirs)
    if args.merge:
        preprocessor.merge()

    if args.split:
        preprocessor.split()

    if args.remove_dup:
        preprocessor.remove_duplicates()

    if args.resize:
        preprocessor.resize(args.resize)

    if args.yolo2coco:
        preprocessor.yolo2coco(output_path="dataset/coco")