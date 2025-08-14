from pathlib import Path
import os
from tqdm import tqdm

input_path = Path("dataset/pose/d")
label_paths = sorted(input_path.glob("**/labels/*.txt"))
for label_path in tqdm(label_paths):

    invalid_labels = False
    new_label_data = []
    with open(str(label_path), "r") as label_file:
        label_data = label_file.read().splitlines()

        for label in label_data:
            class_id = label.split(" ")[0]
            bbox = label.split(" ")[1:]
            if class_id in ["2"]:
                new_label = f"0 {' '.join(bbox)}\n"
                new_label_data.append(new_label)
            else:
                invalid_labels = True
    if invalid_labels:
        image_path = (label_path.parent.with_name("images")/label_path.name).with_suffix(".jpg")
        os.remove(str(image_path))
        os.remove(str(label_path))

    else:
        with open(str(label_path), "w") as new_label_file:
            for new_label in new_label_data:
                new_label_file.write(new_label)





