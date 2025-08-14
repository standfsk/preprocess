import numpy as np
import matplotlib.pyplot as plt
import os

# Load keypoints
a = np.load("sample1.npy")  # shape: (13, 30, 3)

# Define bones (adjust as needed)
bones = [
    (0, 1),  # head to left shoulder
    (0, 2),  # head to right shoulder
    (1, 3),  # left shoulder to left elbow
    (3, 5),  # left elbow to left wrist
    (2, 4),  # right shoulder to right elbow
    (4, 6),  # right elbow to right wrist
    (1, 7),  # left shoulder to left hip
    (2, 8),  # right shoulder to right hip
    (7, 9),  # left hip to left knee
    (9, 11), # left knee to left ankle
    (8, 10), # right hip to right knee
    (10, 12),# right knee to right ankle
    (7, 8)   # left hip to right hip (pelvis)
]
# Create output folder
os.makedirs("skeleton_frames", exist_ok=True)

# Loop through frames
for frame_idx in range(a.shape[1]):
    keypoints = a[:, frame_idx, :]  # (13, 3)

    plt.figure(figsize=(5, 5))

    # Draw bones
    for start, end in bones:
        x = [keypoints[start, 0], keypoints[end, 0]]
        y = [keypoints[start, 1], keypoints[end, 1]]
        plt.plot(x, y, 'b-', linewidth=2)

    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        plt.plot(x, y, 'ro' if conf > 0.5 else 'ko', markersize=5)
        plt.text(x, y - 0.009, str(i), fontsize=8, ha='center', color='black')  # Label just above point

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()

    # Save to file
    filename = f"skeleton_frames/frame_{frame_idx:03d}.jpg"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close to avoid memory issues
