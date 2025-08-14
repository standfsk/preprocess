import cv2
import os
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
import argparse


def empty_directory(pth):
    if os.path.exists(pth):
        shutil.rmtree(pth)
    os.makedirs(pth, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw")
    parser.add_argument("--dirs", required=True, nargs="+", help="directory paths")
    parser.add_argument("--single", action="store_true", help="draw single image")

    args = parser.parse_args()

    for directory_path in args.dirs:
        input_path = Path(directory_path)
        output_path = Path("res")
        empty_directory(output_path)
        image_paths = sorted(input_path.glob("*/*.jpg"))
        if args.single:
            image_paths = image_paths[0]
        pbar = tqdm(image_paths, total=len(image_paths))
        for image_path in pbar:
            pbar.set_description(f"Drawing {image_path}")
            image = cv2.imread(str(image_path))

            label_path = image_path.with_suffix(".npy")
            label_data = np.load(label_path)

            for label in label_data:
                cv2.circle(image, label, 2, (0, 0, 255), -1)

            cv2.putText(image, f"count: {len(label_data)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
            cv2.imwrite(str(output_path/image_path.name), image)
