from pathlib import Path
import numpy as np
import os

label_paths = Path("dataset/dd").glob("*/*.npy")
for label_path in label_paths:
    image_path = label_path.with_suffix(".jpg")

    num_people = len(np.load(label_path))
    if num_people < 100:
        os.remove(image_path)
        os.remove(label_path)