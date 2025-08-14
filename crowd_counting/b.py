from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
from collections import defaultdict

output_path = Path("dataset/uptec_crowd_domain_2025/train")

count = defaultdict(int) # crowd_rgb, crowd_ir, serious_rgb, serious_ir
image_paths = sorted(Path("dataset/dd").glob("*/*.jpg"))
for image_path in tqdm(image_paths):
    label_path = image_path.with_suffix(".npy")

    image_sensor_type = image_path.stem.split("_")[-1].split(" ")[0].lower()
    num_people = len(np.load(label_path))
    if num_people >= 100 and num_people < 500:
        data_composition = f"crowded_{image_sensor_type}"
        count[data_composition] += 1
    elif num_people >= 500:
        data_composition = f"serious_{image_sensor_type}"
        count[data_composition] += 1
    else:
        raise ValueError()
    file_save_name = f"{data_composition}_{count[data_composition]:04d}"

    shutil.copy(image_path, output_path/f"{file_save_name}.jpg")
    shutil.copy(label_path, output_path/f"{file_save_name}.npy")

