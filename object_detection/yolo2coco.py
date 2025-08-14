import os
import json
from PIL import Image
from tqdm import tqdm
import shutil

def yolo_to_coco(annotation_folder, image_folder, output_json, categories, save_path):
    """
    Converts YOLO format annotations to COCO format.

    Parameters:
        annotation_folder (str): Path to the folder containing YOLO .txt files.
        image_folder (str): Path to the folder containing images.
        output_json (str): Path to save the COCO .json file.
        categories (list): List of categories as dictionaries, e.g., [{"id": 0, "name": "cat"}, ...].
    """
    images = []
    annotations = []
    annotation_id = 1  # COCO annotations require unique IDs

    # Iterate over YOLO annotation files
    for txt_file in tqdm(os.listdir(annotation_folder)):
        if not txt_file.endswith(".txt"):
            continue

        # Get image details
        base_name = os.path.splitext(txt_file)[0]
        image_path = os.path.join(image_folder, base_name + ".jpg")
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found. Skipping.")
            continue

        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        # Add image metadata to COCO
        image_id = len(images) + 1
        images.append({
            "id": image_id,
            "file_name": base_name + ".jpg",
            "width": width,
            "height": height
        })

        # Read YOLO annotations
        txt_path = os.path.join(annotation_folder, txt_file)
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                # Convert normalized coordinates to absolute coordinates
                x_center_abs = x_center * width
                y_center_abs = y_center * height
                bbox_width_abs = bbox_width * width
                bbox_height_abs = bbox_height * height

                # Convert YOLO bbox format to COCO bbox format: [x_min, y_min, width, height]
                x_min = x_center_abs - (bbox_width_abs / 2)
                y_min = y_center_abs - (bbox_height_abs / 2)

                # Add annotation to COCO
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),  # COCO category IDs are integers
                    "bbox": [x_min, y_min, bbox_width_abs, bbox_height_abs],
                    "area": bbox_width_abs * bbox_height_abs,
                    "iscrowd": 0,
                    "segmentation": []
                })
                annotation_id += 1

        shutil.copy(image_path, os.path.join(save_path, os.path.basename(image_path)))

    # Create COCO dictionary
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

# Example usage
if __name__ == "__main__":
    if os.path.exists("coco"):
        shutil.rmtree("coco")
    os.makedirs("coco/images/train", exist_ok=True)
    os.makedirs("coco/images/val", exist_ok=True)
    os.makedirs("coco/annotations", exist_ok=True)

    data_path = "dataset/VisDrone"
    for mode in ["train", "val"]:
        annotation_folder = os.path.join(data_path, mode, "labels")
        image_folder = annotation_folder.replace("labels", "images")
        output_json = os.path.join("coco", "annotations", f"instances_{mode}.json")
        save_path = os.path.join("coco", "images", mode)

        # Define categories (example: 3 classes)
        categories = [
            {"id": 0, "name": "pedestrian"},
            {"id": 1, "name": "people"},
            {"id": 2, "name": "bicycle"},
            {"id": 3, "name": "car"},
            {"id": 4, "name": "van"},
            {"id": 5, "name": "truck"},
            {"id": 6, "name": "tricycle"},
            {"id": 7, "name": "awning-tricycle"},
            {"id": 8, "name": "bus"},
            {"id": 9, "name": "motor"}
        ]

        yolo_to_coco(annotation_folder, image_folder, output_json, categories, save_path)
