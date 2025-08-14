from PIL import Image, ImageDraw
from pathlib import Path
import os
import numpy as np
from utils import xywh2xyxy
from shapely.geometry import Polygon, box

def draw_gray_diagonal_band(image, polygons, gray_level=128):
    w, h = image.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))  # transparent overlay
    draw = ImageDraw.Draw(overlay)

    for polygon in polygons:
        # Draw solid gray with full opacity
        draw.polygon(polygon, fill=(gray_level, gray_level, gray_level, 255))

    # Convert image to RGBA if not already
    image = image.convert("RGBA")

    # Composite: paste gray overlay on top of image
    result = Image.alpha_composite(image, overlay)
    return result

pts = [(1424, 583), (1454, 587), (1461, 653), (1417, 650)]
polygons = [pts]

# Example usage
output_path = Path("dataset/person_head/working/dd/train/images")
os.makedirs(output_path, exist_ok=True)
image_paths = Path("dataset/person_head/working/dd/train/images").glob("*")
for image_path in image_paths:
    img = Image.open(str(image_path))
    w, h = img.size
    result = draw_gray_diagonal_band(img, polygons)

    background = Image.new("RGB", result.size, (255, 255, 255))  # white background
    background.paste(result, mask=result.split()[3])  # paste with alpha channel as mask
    background.save(output_path/image_path.name)

    label_path = (image_path.parent.with_name("labels")/image_path.name).with_suffix(".txt")
    # Process labels
    new_label_data = []
    with open(label_path, "r") as label_file:
        label_data = label_file.read().splitlines()
        for label in label_data:
            class_id = int(label.split(" ")[0])
            bbox = label.split(" ")[1:5]  # still normalized xywh

            # Convert YOLO normalized xywh → absolute xywh
            bbox_ = np.array(bbox, dtype=np.float32)
            bbox_ = [bbox_[0] * w, bbox_[1] * h, bbox_[2] * w, bbox_[3] * h]

            # Convert xywh → xyxy
            bbox_ = xywh2xyxy(bbox_)
            bbox_ = list(map(int, bbox_))

            # Create shapely box
            rect = box(*bbox_)

            # Keep only if NOT inside any polygon
            if not any(poly.contains(rect) for poly in polygons):
                new_label_data.append(f"{class_id} {' '.join(bbox)}\n")

    # Save filtered labels
    with open(label_path, "w") as new_label_file:
        new_label_file.writelines(new_label_data)

