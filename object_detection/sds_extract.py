import glob
import os
from pathlib import Path
import json
import numpy as np
import shutil
from tqdm import tqdm

def convert(x, shape):
    # top_x, top_y, w, h -> cx,cy,w,h
    y = x.copy()
    y[0] = (y[0] + y[2] / 2) / shape[1]
    y[1] = (y[1] + y[3] / 2) / shape[0]
    y[2] = y[2] / shape[1]
    y[3] = y[3] / shape[0]
    return y

def extract(split, input_path, output_path):
    os.makedirs(output_path / split / "images", exist_ok=True)
    os.makedirs(output_path / split / "labels", exist_ok=True)

    image_data = {}
    label_data = {}
    with open(input_path/f"annotations/instances_{split}.json", "r") as json_file:
        json_data = json.load(json_file)
        for image_info in json_data['images']:
            image_id = int(image_info['id'])
            image_file_name = image_info['file_name']
            image_height = int(image_info['height'])
            image_width = int(image_info['width'])
            image_data[image_id] = [image_file_name, [image_height, image_width]]

        for annotation_info in json_data['annotations']:
            image_matching_id = int(annotation_info['image_id'])
            bbox = annotation_info['bbox']
            category_id = int(annotation_info['category_id'])

            image_file_name, image_shape = image_data[image_matching_id]
            bbox = convert(bbox, image_shape)
            line = f"{category_id} {' '.join(list(map(str, bbox)))}\n"
            if not image_file_name in label_data.keys():
                label_data[image_file_name] = []
            label_data[image_file_name].append(line)

        # save
        for image_file_name, lines in tqdm(label_data.items(), desc=f"processing {split}"):
            # image
            shutil.copy(input_path/f"images/{split}/{image_file_name}",
                        output_path / split / "images" / image_file_name)

            # label
            with open(output_path / split / "labels" / image_file_name.replace(".jpg", ".txt"), "w") as label_file:
                for line in lines:
                    label_file.write(line)

if __name__ == "__main__":
    input_path = Path("dataset/Compressed Version")
    output_path = Path("dataset/SeaDroneSee")

    for split in ["train", "val"]:
        extract(split, input_path, output_path)













