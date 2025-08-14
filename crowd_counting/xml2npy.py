from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

output_path = Path("dataset/dd/train")
annotation_by_image = {}
tree = ET.parse('annotations.xml')  # Or use ET.fromstring(xml_string) if you have it as a string
root = tree.getroot()

# Get all points under the <image> tag
for image in root.findall('image'):
    image_name = image.attrib.get('name')

    points = []
    for point in image.findall('points'):
        label = point.attrib.get('label')
        coords = point.attrib.get('points')
        points.extend([tuple(map(int, map(float, pair.split(",")))) for pair in coords.split(";")])

    if points:
        np.save((output_path/image_name).with_suffix(".npy"), np.array(points))
