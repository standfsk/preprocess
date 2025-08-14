from pathlib import Path
import os
import cv2
from tqdm import tqdm
import numpy as np
import shutil


input_path = Path("dataset/dd/train")
image_paths = sorted(input_path.glob("images/*.jpg"))
for image_path in tqdm(image_paths):
    image = cv2.imread(str(image_path))
    image_height, image_width = image.shape[:2]

    new_label_data = []
    label_path = (image_path.parent.with_name("labels")/image_path.name).with_suffix(".txt")
    with open(label_path, "r") as label_file:
        label_data = label_file.read().splitlines()
        for label in label_data:
            class_id = label.split(" ")[0]
            bbox = label.split(" ")[1:5]
            bbox = list(map(float, bbox))

            center_x, center_y = bbox[0] * image_width, bbox[1] * image_height
            new_label_data.append([center_x, center_y])

    new_label_data = np.array(new_label_data).astype(np.int16)
    os.rename(image_path, input_path/image_path.name)
    np.save((input_path/image_path.name).with_suffix(".npy"), new_label_data)

shutil.rmtree(input_path/"images")
shutil.rmtree(input_path/"labels")








