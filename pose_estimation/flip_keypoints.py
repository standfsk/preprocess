from pathlib import Path
import numpy as np
from tqdm import tqdm

flip_index = [[0,12], [1,10], [2,8], [3,7], [4,9], [5,11], [9,0], [10,6], [11,4], [12,2], [13,1], [14,3], [15,5]]

label_paths = sorted(Path("dataset/aihub").glob("*/*/*.txt"))
for label_path in tqdm(label_paths):
    new_label_data = []
    with open(label_path, "r") as label_file:
        label_data = label_file.read().splitlines()
        for label in label_data:

            class_id = label.split(" ")[0]
            bbox = label.split(" ")[1:5]
            keypoints = label.split(" ")[5:]

            keypoints = np.array(keypoints).reshape(-1, 3)
            new_keypoints = np.zeros([13, 3])
            for old_index, new_index in flip_index:
                new_keypoints[new_index] = keypoints[old_index]
            new_label_data.append(f"{class_id} {' '.join(bbox)} {' '.join(new_keypoints.flatten().astype(str))}\n")

    with open(label_path, "w") as new_label_file:
        for new_label in new_label_data:
            new_label_file.write(new_label)

