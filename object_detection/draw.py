import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
import random
from pathlib import Path
import argparse
from utils import xywh2xyxy, empty_directory
random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw")
    parser.add_argument("--dirs", required=True, nargs="+", help="directory paths")

    args = parser.parse_args()

    for directory_path in args.dirs:
        input_path = Path(directory_path)
        output_path = Path("res")
        empty_directory(output_path)

        if not os.path.exists(input_path/"classes.txt"):
            class_mapping = {0: "person"}
        else:
            with open(input_path/"classes.txt", "r") as classes_txt:
                classes = classes_txt.read().splitlines()
            class_mapping = {x:y for x,y in enumerate(classes)}

        colors = [[int(random.random() * 255) for _ in range(3)] for _ in range(1000)]
        image_paths = sorted(input_path.glob("**/images/*"))
        pbar = tqdm(image_paths, total=len(image_paths))
        for index, image_path in enumerate(pbar):
            pbar.set_description(f"Drawing {image_path}")
            label_path = (image_path.parent.with_name("labels") / image_path.name).with_suffix(".txt")
            img = cv2.imread(str(image_path))
            with open(label_path, 'r') as f:
                outputs = f.read().splitlines()
                for object_id, output in enumerate(outputs):
                    output = output.split(" ")
                    class_id = int(output[0])
                    bbox = np.array(output[1:5]).astype(np.float32)
                    bbox = [bbox[0]*img.shape[1], bbox[1]*img.shape[0], bbox[2]*img.shape[1], bbox[3]*img.shape[0]]
                    bbox = xywh2xyxy(bbox)
                    bbox = list(map(int, bbox))

                    cv2.rectangle(img, (bbox[0] - 1, bbox[1] - 20), (bbox[0] + 5 * 12, bbox[1]+5), colors[class_id], -1)
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[class_id], 2)
                    cv2.putText(img, class_mapping[class_id], (bbox[0] + 5, bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(str(output_path/image_path.name), img)


