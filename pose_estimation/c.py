import cv2
import json
import numpy as np

def convert(x):
    y = np.copy(x)
    y[0] = x[0]
    y[1] = x[1]
    y[2] = x[0] + x[2]
    y[3] = x[1] + x[3]
    return y


image = cv2.imread("sample.jpg")
with open("sample.json", "r", encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    for label in json_data['labels']['b_box']:
        bbox = convert(list(label['coord'].values()))
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)

    for keypoint_info in json_data['labels']['keypoint'][0]['points']:
        kpt_x = keypoint_info['x']
        kpt_y = keypoint_info['y']
        cv2.circle(image, (kpt_x, kpt_y), 3, (255, 255, 0), -1)

cv2.imwrite("res.jpg", image)

