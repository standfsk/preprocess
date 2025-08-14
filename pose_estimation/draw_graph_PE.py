import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

precision_by_epoch = {}
recall_by_epoch = {}
map50_by_epoch = {}
map_by_epoch = {}
pose_model_pth = "pose_estimation_d6v5"
with open(f"결과/pose_estimation/{pose_model_pth}/results.txt", "r") as txt_file:
    txt_data = txt_file.read().splitlines()
    for data in txt_data:
        data = [x for x in data.split(" ") if x != ""]
        epoch = int(data[0].split("/")[0])
        precision = data[10]
        recall = data[11]
        map50 = data[12]
        map = data[13]
        precision_by_epoch[epoch] = round(float(precision), 4)
        recall_by_epoch[epoch] = round(float(recall), 4)
        map50_by_epoch[epoch] = round(float(map50), 4)
        map_by_epoch[epoch] = round(float(map), 4)

plt.clf()
plt.plot(precision_by_epoch.keys(), precision_by_epoch.values())
plt.xticks([0, 150, 300])
min_precision = min(list(precision_by_epoch.values()))
max_precision = max(list(precision_by_epoch.values()))
plt.yticks(np.linspace(min_precision, max_precision, 5))
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision')
plt.savefig(os.path.join("결과", "pose_estimation", pose_model_pth, 'precision.jpg'))

plt.clf()
plt.plot(recall_by_epoch.keys(), recall_by_epoch.values())
plt.xticks([0, 150, 300])
min_recall = min(list(recall_by_epoch.values()))
max_recall = max(list(recall_by_epoch.values()))
plt.yticks(np.linspace(min_recall, max_recall, 5))
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall')
plt.savefig(os.path.join("결과", "pose_estimation", pose_model_pth, 'recall.jpg'))

plt.clf()
plt.plot(map50_by_epoch.keys(), map50_by_epoch.values())
plt.xticks([0, 150, 300])
min_map50 = min(list(map50_by_epoch.values()))
max_map50 = max(list(map50_by_epoch.values()))
plt.yticks(np.linspace(min_map50, max_map50, 5))
plt.xlabel('Epoch')
plt.ylabel('mAP@.5')
plt.title('mAP@.5')
plt.savefig(os.path.join("결과", "pose_estimation", pose_model_pth, 'map50.jpg'))

plt.clf()
plt.plot(map_by_epoch.keys(), map_by_epoch.values())
plt.xticks([0, 150, 300])
min_map = min(list(map_by_epoch.values()))
max_map = max(list(map_by_epoch.values()))
plt.yticks(np.linspace(min_map, max_map, 5))
plt.xlabel('Epoch')
plt.ylabel('mAP@.5:.95')
plt.title('mAP@.5:.95')
plt.savefig(os.path.join("결과", "pose_estimation", pose_model_pth, 'map.jpg'))

idx = list(map50_by_epoch.values()).index(max_map50)
print(f"P:{precision_by_epoch[idx]} R: {recall_by_epoch[idx]} map50: {map50_by_epoch[idx]} map: {map_by_epoch[idx]}")
