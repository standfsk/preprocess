import glob
import os
import xml.etree.ElementTree as ET
import cv2

def yolo_to_voc(label_path, img_width, img_height, xml_path, image_path):
    # Create root XML structure
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = image_path
    ET.SubElement(annotation, "path").text = image_path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    # Read YOLO annotation
    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])

            # Convert normalized YOLO coordinates to absolute pixel values
            xmin = int((x_center - width / 2) * img_width)
            ymin = int((y_center - height / 2) * img_height)
            xmax = int((x_center + width / 2) * img_width)
            ymax = int((y_center + height / 2) * img_height)

            # Ensure bounding box is within image dimensions
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_width, xmax)
            ymax = min(img_height, ymax)

            # Create object node
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = f"{class_id}"  # Get class name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

    # Write to XML file
    tree = ET.ElementTree(annotation)
    with open(xml_path, "wb") as f:
        tree.write(f)


if __name__ == "__main__":
    os.makedirs(os.path.join("dataset", "dataset_100000", "train", "xmls"), exist_ok=True)
    label_paths = glob.glob(os.path.join("dataset", "dataset_100000", "train", "labels", "*.txt"))
    for label_path in label_paths:
        image_path = label_path.replace("labels", "images").replace(".txt", ".jpg")
        if not os.path.exists(image_path):
            continue

        xml_path = os.path.join("dataset", "dataset_100000", "train", "xmls", os.path.basename(label_path).replace(".txt", ".xml"))

        image = cv2.imread(image_path)
        image_width = image.shape[1]
        image_height = image.shape[0]
        yolo_to_voc(label_path, image_width, image_height, xml_path, image_path)
