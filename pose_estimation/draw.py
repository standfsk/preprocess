import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
import random
from pathlib import Path
import argparse
from utils import xywh2xyxy
random.seed(42)

def empty_directory(pth):
    if os.path.exists(pth):
        shutil.rmtree(pth)
    os.makedirs(pth, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw")
    parser.add_argument("--dirs", required=True, nargs="+", help="directory paths")

    args = parser.parse_args()

    for directory_path in args.dirs:
        input_path = Path(directory_path)
        output_path = Path("res")
        empty_directory(output_path)

        color = [0, 0, 255]
        image_paths = sorted(input_path.glob("**/images/*.jpg"))
        pbar = tqdm(image_paths, total=len(image_paths))
        for index, image_path in enumerate(pbar):
            pbar.set_description(f"Drawing {image_path}")
            label_path = (image_path.parent.with_name("labels") / image_path.name).with_suffix(".txt")
            img = cv2.imread(str(image_path))
            with open(label_path, 'r') as f:
                outputs = f.read().splitlines()
                for person_id, output in enumerate(outputs):
                    output = output.split()
                    bbox = np.array(output[1:5]).astype(np.float32)
                    bbox = [bbox[0]*img.shape[1], bbox[1]*img.shape[0], bbox[2]*img.shape[1], bbox[3]*img.shape[0]]
                    bbox = list(map(int, bbox))
                    bbox = xywh2xyxy(bbox)
                    text = f"person-{person_id}"

                    if output[5:]:
                        keypoints = np.array(output[5:]).reshape(-1, 3).astype(np.float32)
                        for keypoint_id, (kpts) in enumerate(keypoints):
                            x,y, score = kpts
                            x = int(float(x)*img.shape[1])
                            y = int(float(y)*img.shape[0])
                            if (x,y) == (0,0):
                                continue
                            else:
                                cv2.putText(img, f'{keypoint_id}', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                cv2.circle(img, (x,y), 4, color, -1)

                        coords = [[int(x*img.shape[1]), int(y*img.shape[0])] for x,y in keypoints[:, :2]]
                        for kpt_id, comb in enumerate([[1,2], [1,3], [3,5], [2,4], [4,6], [1,7], [2,8], [7,8], [7,9], [9,11], [8,10], [10,12]]):
                            comb1, comb2 = comb
                            x1, y1 = coords[comb1]
                            x2, y2 = coords[comb2]

                            if (x1, y1) == (0,0) or (x2, y2) == (0,0):
                                continue
                            cv2.line(img, (x1,y1), (x2,y2), color, 3)

                    cv2.rectangle(img, (bbox[0] - 1, bbox[1] - 20), (bbox[0] + 5 * 12, bbox[1]+5), color, -1)
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(img, text, (bbox[0] + 5, bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imwrite(f"res/{os.path.basename(image_path)}", img)
