from pathlib import Path
import os

image_paths = sorted(Path("dataset/person_head/working/dd").glob("images/*.jpg"))
for image_index, image_path in enumerate(image_paths, start=1):
    label_path = (image_path.parent.with_name("labels")/image_path.name).with_suffix(".txt")

    head_count = 0
    with open(label_path, "r") as label_file:
        label_data = label_file.read().splitlines()
        for label in label_data:
            class_id = int(label.split(" ")[0])
            if class_id == 0:
                head_count += 1

    if head_count >= 90:
        print(label_path, head_count)

    # os.rename(image_path, image_path.parent/f"frame_{image_index:04d}.jpg")
    # os.rename(label_path, label_path.parent/f"frame_{image_index:04d}.txt")




