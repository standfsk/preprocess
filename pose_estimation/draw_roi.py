import cv2
import glob
import os
import numpy as np
from tqdm import tqdm
def denormalize(roi, img_size):
    denormalized_roi = [[0, 0], [0, 0], [0, 0], [0, 0]]
    for point_id, point in enumerate(roi):
        if point_id in [0, 2, 4, 6]:
            denormalized_roi[int(point_id / 2)][0] = int(float(point) * img_size[1])
        else:
            denormalized_roi[int(point_id / 2)][1] = int(float(point) * img_size[0])
    return denormalized_roi

def draw_roi(input_video_path, output_path):
    label_path = input_video_path.replace(".mp4", ".txt")
    with open(label_path, "r") as txt_file:
        txt_data = txt_file.read().split(" ")

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open input video.")
    else:
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi files
        out = cv2.VideoWriter(os.path.join(output_path, os.path.basename(input_video_path)), fourcc, fps, (width, height))

        # Define the rectangle coordinates and color
        color = (0, 255, 0)  # Rectangle color (Green in BGR format)
        thickness = 2  # Thickness of the rectangle border

        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            roi = denormalize(txt_data[4:], frame.shape)
            roi = np.array(roi, np.int32)
            roi = roi.reshape((-1, 1, 2))

            # Draw the rectangle on the frame
            cv2.polylines(frame, [roi], isClosed=True, color=color, thickness=thickness)

            # Write the modified frame to the output video
            out.write(frame)

        # Release video objects
        cap.release()
        out.release()

output_path = "res"
os.makedirs(output_path, exist_ok=True)
video_paths = glob.glob(os.path.join("test", "intrusion*.mp4")) + glob.glob(os.path.join("test", "loitering*.mp4"))
for video_path in tqdm(video_paths, total=len(video_paths), desc="drawing roi..."):
    draw_roi(video_path, output_path)