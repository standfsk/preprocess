from pathlib import Path
import numpy as np
import os

label_paths = Path("dataset/head").glob("*.npy")
for label_path in label_paths:
    num_people = len(np.load(label_path))
    print(label_path, num_people)

