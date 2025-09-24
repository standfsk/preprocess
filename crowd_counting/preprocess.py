import argparse
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import imagehash
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import cv2
from collections import defaultdict


class Preprocessor:
    def __init__(self, directories: List[str]):
        """
        Parameters:
            directories: directories to run preprocessing methods
        """
        self.directories = directories
        self.set_image_format()

    def set_image_format(self, quality=95) -> None:
        """
        set image format to jpg
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            image_paths = input_path.glob("*/*")
            for image_path in image_paths:
                if image_path.suffix in [".jpg", ".npy"]:
                    continue
                print(f"Setting image format to jpg {image_path.name} -> {image_path.stem}.jpg")
                image = Image.open(str(image_path)).convert("RGB")
                new_image_path = image_path.with_suffix(".jpg")
                image.save(str(new_image_path), "JPEG", quality=quality)
                os.remove(image_path)

    def verify_labels(self) -> None:
        """
        verify labels
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            label_paths = input_path.glob("*/*.npy")
            for label_path in label_paths:
                image_path = label_path.with_suffix(".jpg")
                image = cv2.imread(str(image_path))
                image_height, image_width = image.shape[:2]

                label_data = np.load(label_path)
                for label in label_data:
                    if label[0] > image_width or label[0] < 0 or label[1] > image_height or label[1] < 0:
                        print(f"{label_path} has invalid label: {label}")

    def merge(self):
        """
        merge (valid, test) directories into train
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            for subset in ["valid", "test"]:
                image_paths = sorted((input_path/subset).glob("*.jpg"))
                for index, image_path in enumerate(image_paths):
                    new_image_path = input_path/"train"/image_path.name
                    os.rename(image_path, new_image_path)

                    label_path = image_path.with_suffix(".npy")
                    new_label_path = input_path/"train"/label_path.name
                    os.rename(label_path, new_label_path)

                if os.path.exists(input_path/subset):
                    shutil.rmtree(input_path/subset)

            merged_file_num = sorted((input_path/"train").glob("*.jpg"))
            print(f"merged file num: {len(merged_file_num)}")

    def split(self):
        """
        split directories into (train(80%), valid(10%), test(10%))
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            image_paths = sorted((input_path/"train").glob("*.jpg"))
            train_image_paths, valid_image_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
            valid_image_paths, test_image_paths = train_test_split(valid_image_paths, test_size=0.5, random_state=42)
            print(f"train file num: {len(train_image_paths)}")
            print(f"valid file num: {len(valid_image_paths)}")
            print(f"test file num: {len(test_image_paths)}")
            for subset, image_paths in [["valid", valid_image_paths], ["test", test_image_paths]]:
                os.makedirs(input_path/subset, exist_ok=True)

                for index, image_path in enumerate(image_paths):
                    label_path = image_path.with_suffix(".npy")
                    os.rename(image_path, input_path/subset/image_path.name)
                    os.rename(label_path, input_path/subset/label_path.name)

    def scale_data(
            self,
            image: np.ndarray,
            points: np.ndarray,
            min_size: int = 512,
            max_size: int = 1920,
            base: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        resizes input image and scales the associated labels

        Parameters:
            image (np.ndarray): input image
            points (np.ndarray): points label associated with the image
            min_size (int): minimum size of width and height for image
            max_size (int): maximum size of width and height for image
            base (int): base value for size

        Returns:
             resized_image (np.ndarray): resized image
             resized_points (np.ndarray): scaled points associated with the image
        """
        img_w, img_h = image.shape[1], image.shape[0]
        aspect_ratios = (img_w/img_h, img_h/img_w)
        if min_size/max_size <= min(aspect_ratios) <= max(aspect_ratios) <= max_size/min_size:
            if min_size <= min(img_w, img_h) <= max(img_w,img_h) <= max_size:  # already within the range, no need to resize
                ratio = 1.
            elif min(img_w, img_h) < min_size:  # smaller than the minimum size, resize to the minimum size
                ratio = min_size/min(img_w, img_h)
            else:  # larger than the maximum size, resize to the maximum size
                ratio = max_size/max(img_w, img_h)
    
            new_w, new_h = int(round(img_w * ratio/base) * base), int(round(img_h * ratio/base) * base)
            new_w = max(min_size, min(max_size, new_w))
            new_h = max(min_size, min(max_size, new_h))
        else:
            return self.scale_data(image, points, min_size, float("inf"), base)
    
        resized_image = cv2.resize(image, (new_w, new_h))
        resized_points = points * np.array([[new_w/img_w, new_h/img_h]]) if len(points) > 0 and (new_w, new_h) != (img_w, img_h) else points
        resized_points = resized_points.astype(np.int32)
        return resized_image, resized_points

    def resize(self, min_size: int = 512, max_size:int = 1920) -> None:
        """
        save resized image and scaled labels
        """
        for base_dir in self.directories:
            input_path = Path(base_dir)
            for subset in ["train", "valid", "test"]:
                image_paths = sorted((input_path/subset).glob("*.jpg"))
                for image_path in tqdm(image_paths, total=len(image_paths), desc="Resizing"):
                    image = cv2.imread(str(image_path))
                    label_path = image_path.with_suffix(".npy")
                    label_data = np.load(str(label_path))
    
                    image_scaled, label_data_scaled = self.scale_data(image, label_data, min_size, max_size)
                    cv2.imwrite(str(image_path), image_scaled)
                    np.save(str(label_path), label_data_scaled)

    def get_info(self) -> None:
        """
        get dataset statistics

        The method performs the following:
           1. counts number of images
           2. counts number of annotations
           3. calculates average image size
           4. counts labels by range
            - 0 ~ 100
            - 100 ~ 500
            - 500 ~ 1000
            - 1000 ~ more
        """

        for base_dir in self.directories:
            input_path = Path(base_dir)
            for subset in ["train", "valid", "test", "*"]:
                images_num = 0
                labels_num = []
                images_heights = []
                images_widths = []

                image_paths = sorted(input_path.glob(f"{subset}/*.jpg"))
                for image_path in tqdm(image_paths, total=len(image_paths), desc=f"Calculating {subset} data statistics"):
                    image = cv2.imread(str(image_path))
                    image_height, image_width = image.shape[:2]

                    images_widths.append(image_width)
                    images_heights.append(image_height)

                    images_num += 1

                    label_path = image_path.with_suffix(".npy")
                    label_data = np.load(label_path)

                    labels_num.append(len(label_data))

                print(f"-----{subset} info------")
                print(f"number of images: {images_num}")
                print(f"image average size: {int(np.mean(images_widths))} x {int(np.mean(images_heights))}")
                labels_num = np.array(labels_num, dtype=np.int16)
                print(f"total number of labels: {np.sum(labels_num)}")
                print(f"average number of labels: {round(np.mean(labels_num), 2)}")

                label_group1 = labels_num[labels_num < 100]
                label_group2 = labels_num[(labels_num >= 100) & (labels_num < 500)]
                label_group3 = labels_num[(labels_num >= 500) & (labels_num < 1000)]
                label_group4 = labels_num[labels_num >= 1000]
                print(f"labels range 0 ~ 100: {len(label_group1)}({int(len(label_group1)/images_num*100)}%)")
                print(f"labels range 100 ~ 500: {len(label_group2)}({int(len(label_group2)/images_num*100)}%)")
                print(f"labels range 500 ~ 1000: {len(label_group3)}({int(len(label_group3)/images_num*100)}%)")
                print(f"labels range 1000 ~ : {len(label_group4)}({int(len(label_group4)/images_num*100)}%)")
                print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images")
    parser.add_argument("--dirs", required=True, nargs="+", help="directory paths")
    parser.add_argument("--merge", action="store_true", help="merge directories")
    parser.add_argument("--split", action="store_true", help="split directories")
    parser.add_argument("--resize", action="store_true", help="resize images and labels")
    parser.add_argument("--info", action="store_true", help="get dataset statistics")

    args = parser.parse_args()
    preprocessor = Preprocessor(args.dirs)
    if args.merge:
        preprocessor.merge()

    if args.split:
        preprocessor.split()

    if args.resize:
        preprocessor.resize(min_size=512, max_size=1920)

    if args.info:
        preprocessor.get_info()
