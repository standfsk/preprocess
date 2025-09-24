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
        # self.verify_labels()

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

    def verify_labels(self) -> None:
        """
        verify label
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            label_paths = sorted(input_path.glob("*/labels/*.txt"))
            for label_path in tqdm(label_paths, desc="Verifying labels"):
                with open(str(label_path), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"{label_path} has label with wrong format: {line}")
                            f.close()
                            image_path = (label_path.parent.with_name("images")/label_path.name).with_suffix(".jpg")
                            os.remove(str(image_path))
                            os.remove(str(label_path))
                            break
                        _, x_center, y_center, w, h = map(float, parts)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            print(f"{label_path} has label outside the range(0 ~ 1): {line}")

    def merge(self) -> None:
        """
        merge (valid, test) directories into train
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            for subset in ["valid", "test"]:
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
        split directories into (train(90%), valid(10%))
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            image_paths = sorted((input_path/"train").glob("images/*.jpg"))
            train_image_paths, valid_image_paths = train_test_split(image_paths, test_size=0.1, random_state=42)
            print(f"train file num: {len(train_image_paths)}")
            print(f"valid file num: {len(valid_image_paths)}")
            for subset, subset_image_paths in [["valid", valid_image_paths]]:
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
            for subset in ["train", "valid", "test"]:
                images_path = Path(base_dir)/subset/"images"

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

    def change_class(self, current_class: List[int], target_class: List[int]) -> None:
        """
        change class by input

        Parameters:
            current_class (List[int]): current class list
            target_class (List[int]): target class list

        Example:
            current_class: [0, 1]
            target_class: [1, 0]

            result:
                class id 0 becomes 1
                class id 1 becomes 0
        """
        correct_class = {x: y for x, y in zip(current_class, target_class)}
        for base_dir in self.directories:
            input_path = Path(base_dir)
            label_paths = sorted(input_path.glob("*/labels/*.txt"))
            for label_path in tqdm(label_paths, total=len(label_paths), desc="Changing class"):
                new_label_data = []
                with open(str(label_path), "r") as label_file:
                    label_data = label_file.read().splitlines()
                    label_data = [label for label in label_data if label.strip()]
                    for label in label_data:
                        current_class_id = label.split(" ")[0]
                        bbox = label.split(" ")[1:]

                        target_class_id = correct_class[int(current_class_id)]
                        new_label = f"{target_class_id} {' '.join(bbox)}\n"
                        new_label_data.append(new_label)

                with open(str(label_path), "w") as new_label_file:
                    for new_label in new_label_data:
                        new_label_file.write(new_label)

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

    def get_info(self) -> None:
        """
        get dataset statistics per class

        The method performs the following:
            1. counts number of images by class
            2. counts number of annotations by class
            3. calculates average image size by class
            4. calculates average bbox size by class

        Requirements:
            - A file named "classes.txt" must be placed in each dataset directory
            - Annotations are assumed to be in YOLO format (x,y,w,h)
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)

            with open(str(input_path/"classes.txt"), "r") as class_file:
                classes = class_file.read().splitlines()
                class_map = {cls_id: cls for cls_id, cls in enumerate(classes)}

            for subset in ["train", "valid", "*"]:
                images_num = 0
                labels_num = defaultdict(int)
                images_heights = []
                images_widths = []
                labels_heights = defaultdict(list)
                labels_widths = defaultdict(list)

                image_paths = sorted(input_path.glob(f"{subset}/images/*.jpg"))
                if not len(image_paths):
                    continue

                for image_path in tqdm(image_paths, total=len(image_paths), desc=f"Calculating {subset} data statistics"):
                    image = cv2.imread(str(image_path))
                    image_height, image_width = image.shape[:2]

                    images_widths.append(image_width)
                    images_heights.append(image_height)

                    images_num += 1

                    label_path = (image_path.parent.with_name("labels")/image_path.name).with_suffix(".txt")
                    with open(str(label_path), "r") as label_file:
                        label_data = label_file.read().splitlines()
                        for label in label_data:
                            class_id = int(label.split(" ")[0])
                            bbox = label.split(" ")[1:]
                            label_width = float(bbox[2]) * image_width
                            label_height = float(bbox[3]) * image_height

                            labels_widths[class_map[class_id]].append(label_width)
                            labels_heights[class_map[class_id]].append(label_height)

                            labels_num[class_map[class_id]] += 1

                print(f"-----{subset} info------")
                print(f"number of images: {images_num}")
                print(f"image average size: {int(np.mean(images_widths))} x {int(np.mean(images_heights))}")
                print(f"number of labels: {sum(labels_num.values())}", {k: v for k, v in labels_num.items()})
                if sum(labels_num.values()):
                    average_label_widths = []
                    average_label_heights = []
                    for cls in classes:
                        if labels_widths[cls]:
                            average_widths = int(sum(labels_widths[cls]) / len(labels_widths[cls]))
                            average_heights = int(sum(labels_heights[cls]) / len(labels_heights[cls]))
                            print(f"{cls} average size: {average_widths} x {average_heights}")

                            average_label_widths.append(average_widths)
                            average_label_heights.append(average_heights)
                        else:
                            print(f"there is no {cls} label")
                    print(f"label average size: {int(np.mean(average_label_widths))} x {int(np.mean(average_label_heights))}")

    def filter_negative(self, output_path: str) -> None:
        """
        move negative samples to output_path

        Parameters:
            output_path (str): target path to move negative samples
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            output_path = Path(output_path)
            os.makedirs(output_path/"train/images", exist_ok=True)
            os.makedirs(output_path/"train/labels", exist_ok=True)
            label_paths = sorted(input_path.glob("**/labels/*.txt"))
            for label_path in tqdm(label_paths, desc="filtering negative samples"):
                with open(str(label_path), "r") as label_file:
                    label_data = label_file.read().splitlines()

                if label_data == []:
                    image_path = (label_path.parent.with_name("images")/label_path.name).with_suffix(".jpg")
                    os.rename(image_path, output_path/"train/images"/image_path.name)
                    os.rename(label_path, output_path/"train/labels"/label_path.name)

        print(f"moved negative samples from {self.directories} to {output_path}")

    def yolo2coco(self, output_path: str) -> None:
        """
        yolo format to coco format

        Parameters:
            output_path (str) : path to save converted format file
        """

        output_path = Path(output_path)
        os.makedirs(output_path/"annotations", exist_ok=True)
        for base_dir in self.directories:
            input_path = Path(base_dir)
            with open(input_path/"classes.txt", "r") as f:
                categories = f.read().splitlines()
            for subset in ["train", "valid"]:
                os.makedirs(output_path/subset, exist_ok=True)
                images = []
                annotations = []
                ann_id = 1

                image_paths = sorted(input_path.glob(f"{subset}/images/*.jpg"))
                for image_id, image_path in tqdm(enumerate(image_paths, start=1), total=len(image_paths),
                                                 desc=f"Converting {base_dir}/{subset} to COCO format"):
                    width, height = Image.open(image_path).size

                    images.append({
                        "id": image_id,
                        "file_name": image_path.name,
                        "width": width,
                        "height": height
                    })

                    label_path = (image_path.parent.with_name("labels") / image_path.name).with_suffix(".txt")
                    with open(str(label_path), 'r') as label_file:
                        for line in label_file:
                            parts = list(map(float, line.strip().split()))
                            class_id = int(parts[0])
                            x_center, y_center, w, h = parts[1:]

                            x = (x_center - w / 2) * width
                            y = (y_center - h / 2) * height
                            bbox = [x, y, w * width, h * height]

                            annotations.append({
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": bbox,
                                "iscrowd": 0,
                                "area": bbox[2] * bbox[3]
                            })

                            ann_id += 1
                    shutil.copy(str(image_path), output_path / subset / image_path.name)

                coco_dict = {
                    "images": images,
                    "annotations": annotations,
                    "categories": [{"id": i, "name": name} for i, name in enumerate(categories)]
                }

                with open(output_path / "annotations" / f"{subset}.json", 'w') as f:
                    json.dump(coco_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images")
    parser.add_argument("--dirs", required=True, nargs="+", help="directory paths")
    parser.add_argument("--merge", action="store_true", help="merge directories")
    parser.add_argument("--split", action="store_true", help="split directories")
    parser.add_argument("--remove-dup", action="store_true", help="remove duplicates")
    parser.add_argument("--change-class", action="store_true", help="change class")
    parser.add_argument("--info", action="store_true", help="get dataset statistics")
    parser.add_argument("--resize", nargs=2, type=int, help="resize image: [Width, Height]")
    parser.add_argument("--filter-neg", action="store_true", help="filter negative samples")
    parser.add_argument("--yolo2coco", action="store_true", help="yolo to coco")

    args = parser.parse_args()
    preprocessor = Preprocessor(args.dirs)
    if args.merge:
        preprocessor.merge()

    if args.split:
        preprocessor.split()

    if args.remove_dup:
        preprocessor.remove_duplicates()

    if args.change_class:
        preprocessor.change_class([0, 1], [1, 0])

    if args.info:
        preprocessor.get_info()

    if args.resize:
        preprocessor.resize(args.resize)

    if args.filter_neg:
        preprocessor.filter_negative(output_path="dataset/negative")

    if args.yolo2coco:
        preprocessor.yolo2coco(output_path="dataset/coco")
