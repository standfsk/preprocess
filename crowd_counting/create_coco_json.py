import json
import glob
import os
import numpy as np
import cv2

# Define the dataset structure
coco_data = {
    "images": [],
    "categories": [],
    "annotations": []
}

# Add a category
coco_data["categories"].append({
    "id": 1,
    "name": "Point",
    "supercategory": "Point"
})

image_paths = sorted(glob.glob(os.path.join("dataset", "temp", "*.jpg")))
for image_idx, image_path in enumerate(image_paths):
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    width, height = image.shape[1], image.shape[0]
    label_path = image_path.replace(".jpg", ".npy")
    label_data = np.load(label_path)

    coco_data["images"].append({
        "id": image_idx,
        "file_name": image_name,
        "width": width,
        "height": height
    })

    coco_data["annotations"].append({
        "id": image_idx+1,
        "image_id": image_idx,
        "category_id": 1,
        "points": [[int(x), int(y)] for x,y in label_data]
    })

# Save the JSON file
output_file = "annotations.json"
with open(output_file, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"COCO JSON file saved as {output_file}")
