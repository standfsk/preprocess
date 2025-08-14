import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

train_mae_by_epoch = {}
train_mse_by_epoch = {}
valid_mae_by_epoch = {}
valid_mse_by_epoch = {}
pose_model_pth = "v3"
with open(f"결과/{pose_model_pth}/debug.log", "r") as txt_file:
    txt_data = txt_file.read().splitlines()
    for data in txt_data:

        if "Train," in data:
            epoch = int(data.split("Epoch ")[-1].split(" ")[0])
            train_mse = float(data.split("MSE: ")[-1].split(" ")[0])
            train_mae = float(data.split("MSE: ")[-1].split(" ")[2].split(",")[0])
            train_mse_by_epoch[epoch] = train_mse
            train_mae_by_epoch[epoch] = train_mae
        elif "Val," in data:
            epoch = int(data.split("Epoch ")[-1].split(" ")[0])
            valid_mse = float(data.split("MSE: ")[-1].split(" ")[0])
            valid_mae = float(data.split("MSE: ")[-1].split(" ")[2].split(",")[0])
            valid_mse_by_epoch[epoch] = valid_mse
            valid_mae_by_epoch[epoch] = valid_mae


plt.clf()
plt.plot(train_mse_by_epoch.keys(), train_mse_by_epoch.values())
plt.xticks(np.linspace(0, max(list(train_mse_by_epoch.keys())), 3, dtype=int))
min_train_mse = min(list(train_mse_by_epoch.values()))
max_train_mse = max(list(train_mse_by_epoch.values()))
plt.yticks(np.linspace(min_train_mse, max_train_mse, 5))
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE')
plt.savefig(os.path.join("결과", pose_model_pth, 'train_mse.jpg'))

plt.clf()
plt.plot(train_mae_by_epoch.keys(), train_mae_by_epoch.values())
plt.xticks(np.linspace(0, max(list(train_mae_by_epoch.keys())), 3, dtype=int))
min_train_mae = min(list(train_mae_by_epoch.values()))
max_train_mae = max(list(train_mae_by_epoch.values()))
plt.yticks(np.linspace(min_train_mae, max_train_mae, 5))
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE')
plt.savefig(os.path.join("결과", pose_model_pth, 'train_mae.jpg'))

plt.clf()
plt.plot(valid_mse_by_epoch.keys(), valid_mse_by_epoch.values())
plt.xticks(np.linspace(0, max(list(valid_mse_by_epoch.keys())), 3, dtype=int))
min_valid_mse = min(list(valid_mse_by_epoch.values()))
max_valid_mse = max(list(valid_mse_by_epoch.values()))
plt.yticks(np.linspace(min_valid_mse, max_valid_mse, 5))
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE')
plt.savefig(os.path.join("결과", pose_model_pth, 'valid_mse.jpg'))

plt.clf()
plt.plot(valid_mae_by_epoch.keys(), valid_mae_by_epoch.values())
plt.xticks(np.linspace(0, max(list(valid_mae_by_epoch.keys())), 3, dtype=int))
min_valid_mae = min(list(valid_mae_by_epoch.values()))
max_valid_mae = max(list(valid_mae_by_epoch.values()))
plt.yticks(np.linspace(min_valid_mae, max_valid_mae, 5))
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE')
plt.savefig(os.path.join("결과", pose_model_pth, 'valid_mae.jpg'))

idx = list(valid_mae_by_epoch.values()).index(min_valid_mae)
print(f"train-MSE:{train_mse_by_epoch[idx]} train-MAE: {train_mae_by_epoch[idx]} valid-MSE: {valid_mse_by_epoch[idx]} valid-MAE: {valid_mae_by_epoch[idx]}")
