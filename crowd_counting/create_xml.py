import xml.etree.ElementTree as ET
from datetime import datetime
from xml.dom import minidom
import glob, os
import cv2
import numpy as np

# Create the root element
root = ET.Element("annotations")

# Add version element
version = ET.SubElement(root, "version")
version.text = "1.1"

# Add meta element
meta = ET.SubElement(root, "meta")

# Add job element inside meta
job = ET.SubElement(meta, "job")
ET.SubElement(job, "id").text = "58"
ET.SubElement(job, "size").text = "543"
ET.SubElement(job, "mode").text = "annotation"
ET.SubElement(job, "overlap").text = "0"
ET.SubElement(job, "bugtracker")
ET.SubElement(job, "created").text = datetime.utcnow().isoformat() + "+00:00"
ET.SubElement(job, "updated").text = datetime.utcnow().isoformat() + "+00:00"
ET.SubElement(job, "subset").text = "default"
ET.SubElement(job, "start_frame").text = "0"
ET.SubElement(job, "stop_frame").text = "542"
ET.SubElement(job, "frame_filter")

# Add segments element inside job
segments = ET.SubElement(job, "segments")
segment = ET.SubElement(segments, "segment")
ET.SubElement(segment, "id").text = "58"
ET.SubElement(segment, "start").text = "0"
ET.SubElement(segment, "stop").text = "542"
ET.SubElement(segment, "url").text = "http://localhost:8080/api/jobs/57"

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

image_paths = sorted(glob.glob(os.path.join("dataset", "uptec_crowd_domain_2025/train", "*.jpg")))
for image_index, image_path in enumerate(image_paths):
    label_path = image_path.replace("images", "labels").replace(".jpg", ".npy")
    image = cv2.imread(image_path)
    image_width, image_height = image.shape[1], image.shape[0]

    points = []
    if os.path.exists(label_path):
        npy_data = np.load(label_path)
        for point in npy_data:
            x = point[0]
            y = point[1]
            points.append(f"{x},{y}")
        points = ";".join(points)

    image_element = ET.SubElement(root, "image", {
        "id": str(image_index),
        "name": os.path.basename(image_path),
        "width": str(image_width),
        "height": str(image_height)
    })

    if points:
        ET.SubElement(image_element, "points", {
            "label": "point",
            "source": "manual",
            "occluded": "0",
            "points": points,
            "z_order": "0"
        })

xml_str = ET.tostring(root, encoding="utf-8")
pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
with open("output.xml", "w", encoding="utf-8") as f:
    f.write(pretty_xml)