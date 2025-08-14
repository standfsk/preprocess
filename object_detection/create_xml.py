import xml.etree.ElementTree as ET
from datetime import datetime
from xml.dom import minidom
import glob, os
import numpy as np
from PIL import Image
from utils import xywh2xyxy

def xywh2xyxy(x):
    """Converts bbox format from [x, y, w, h] to [x1, y1, x2, y2], supporting torch.Tensor and np.ndarray."""
    y = np.copy(x)
    y[0] = round(x[0] - x[2] / 2, 3)  # top left x
    y[1] = round(x[1] - x[3] / 2, 3)  # top left y
    y[2] = round(x[0] + x[2] / 2, 3)  # bottom right x
    y[3] = round(x[1] + x[3] / 2, 3)  # bottom right y
    return y

def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    task = root.find("meta/job")
    task_id = task.find("id").text
    task_size = task.find("size").text
    task_stop_frame = task.find("stop_frame").text

    segment = task.find("segments/segment")
    segment_id = segment.find("id").text
    segment_stop = segment.find("stop").text
    segment_url = segment.find("url").text

    return task_id, task_size, task_stop_frame, segment_id, segment_stop, segment_url

if __name__ == "__main__":
    info = read_xml("annotations.xml")

    # Create the root element
    root = ET.Element("annotations")

    # Add version element
    version = ET.SubElement(root, "version")
    version.text = "1.1"

    # Add meta element
    meta = ET.SubElement(root, "meta")

    # Add job element inside meta
    job = ET.SubElement(meta, "job")
    ET.SubElement(job, "id").text = info[0]
    ET.SubElement(job, "size").text = info[1]
    ET.SubElement(job, "mode").text = "annotation"
    ET.SubElement(job, "overlap").text = "0"
    ET.SubElement(job, "bugtracker")
    ET.SubElement(job, "created").text = datetime.utcnow().isoformat() + "+00:00"
    ET.SubElement(job, "updated").text = datetime.utcnow().isoformat() + "+00:00"
    ET.SubElement(job, "subset").text = "default"
    ET.SubElement(job, "start_frame").text = "0"
    ET.SubElement(job, "stop_frame").text = info[2]
    ET.SubElement(job, "frame_filter")

    # Add segments element inside job
    segments = ET.SubElement(job, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "id").text = info[3]
    ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = info[4]
    ET.SubElement(segment, "url").text = info[5]

    # Add owner element inside job
    owner = ET.SubElement(job, "owner")
    ET.SubElement(owner, "username").text = "user"
    ET.SubElement(owner, "email")

    # Add assignee element inside job
    ET.SubElement(job, "assignee")

    # Add labels element inside job
    labels = ET.SubElement(job, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = "point"
    ET.SubElement(label, "color").text = "#eea580"
    ET.SubElement(label, "type").text = "points"
    ET.SubElement(label, "attributes")

    # Add dumped element inside meta
    ET.SubElement(meta, "dumped").text = datetime.utcnow().isoformat() + "+00:00"

    classes = {0: "head", 1: "person", 2: "car", 3: "bird", 4: "animal"}

    image_paths = sorted(glob.glob(os.path.join("dataset", "person_head/working/head02/images", "*.jpg")))
    for image_index, image_path in enumerate(image_paths):
        label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
        image = Image.open(image_path)
        image_width, image_height = image.size

        annotations = []
        if os.path.exists(label_path):
            with open(str(label_path), "r") as label_file:
                label_data = label_file.read().splitlines()
                for label in label_data:
                    class_id = label.split(" ")[0]
                    cls = classes[int(class_id)]

                    bbox = label.split(" ")[1:5]
                    bbox = list(map(float, bbox))
                    bbox[0] = bbox[0] * image_width
                    bbox[1] = bbox[1] * image_height
                    bbox[2] = bbox[2] * image_width
                    bbox[3] = bbox[3] * image_height
                    bbox = xywh2xyxy(bbox)
                    bbox = list(map(str, bbox))
                    annotations.append([*bbox, cls])

        image_element = ET.SubElement(root, "image", {
            "id": str(image_index),
            "name": os.path.basename(image_path),
            "width": str(image_width),
            "height": str(image_height)
        })
        for annotation_info in annotations:
            ET.SubElement(image_element, "box", {
                "label": annotation_info[-1],
                "source": "manual",
                "occluded": "0",
                "xtl": annotation_info[0],
                "ytl": annotation_info[1],
                "xbr": annotation_info[2],
                "ybr": annotation_info[3],
                "z_order": "0"
            })

    xml_str = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
    with open("output.xml", "w", encoding="utf-8") as f:
        f.write(pretty_xml)