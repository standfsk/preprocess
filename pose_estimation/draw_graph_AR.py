import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

train_accuracy = []
train_loss = []
test_accuracy = []
test_loss = []
action_model_pth = "resnet18_6_c8"
with open(f"결과/action_recognition/{action_model_pth}/train.log", "r") as txt_file:
    txt_data = txt_file.read().splitlines()
    for data in txt_data:
        if "Train Epoch:" in data:
            acc_loss = data.split("Accuracy: ")[1].split(" ")
            train_accuracy.append(round(float(acc_loss[0]), 4))
            train_loss.append(round(float(acc_loss[-1]), 4))
        elif "Test Accuracy:" in data:
            acc_loss = data.split("Accuracy: ")[1].split(" ")
            test_accuracy.append(round(float(acc_loss[0]), 4))
            test_loss.append(round(float(acc_loss[-1]), 4))

train_accuracy = {i:x for i, x in enumerate(train_accuracy)}
train_loss = {i:x for i, x in enumerate(train_loss)}
test_accuracy = {i:x for i, x in enumerate(test_accuracy)}
test_loss = {i:x for i, x in enumerate(test_loss)}

plt.clf()
plt.plot(train_accuracy.keys(), train_accuracy.values())
plt.xticks([0, 50, 100])
min_train_accuracy = min(list(train_accuracy.values()))
max_train_accuracy = max(list(train_accuracy.values()))
plt.yticks(np.linspace(min_train_accuracy, max_train_accuracy, 5))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Accuracy')
plt.savefig(os.path.join("결과", "action_recognition", action_model_pth, 'train_accuracy.jpg'))

plt.clf()
plt.plot(train_loss.keys(), train_loss.values())
plt.xticks([0, 50, 100])
min_train_loss = min(list(train_loss.values()))
max_train_loss = max(list(train_loss.values()))
plt.yticks(np.linspace(min_train_loss, max_train_loss, 5))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.savefig(os.path.join("결과", "action_recognition", action_model_pth, 'train_loss.jpg'))

plt.clf()
plt.plot(test_accuracy.keys(), test_accuracy.values())
plt.xticks([0, 50, 100])
min_test_accuracy = min(list(test_accuracy.values()))
max_test_accuracy = max(list(test_accuracy.values()))
plt.yticks(np.linspace(min_test_accuracy, max_test_accuracy, 5))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Val Accuracy')
plt.savefig(os.path.join("결과", "action_recognition", action_model_pth, 'val_accuracy.jpg'))

plt.clf()
plt.plot(test_loss.keys(), test_loss.values())
plt.xticks([0, 50, 100])
min_test_loss = min(list(test_loss.values()))
max_test_loss = max(list(test_loss.values()))
plt.yticks(np.linspace(min_test_loss, max_test_loss, 5))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Val Loss')
plt.savefig(os.path.join("결과", "action_recognition", action_model_pth, 'val_loss.jpg'))

idx = list(test_loss.values()).index(min_test_loss)
print(f"train_acc:{train_accuracy[idx]} train_loss: {train_loss[idx]} val_acc: {test_accuracy[idx]} val_loss: {test_loss[idx]}")

