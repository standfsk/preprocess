import glob
import os
import cv2
import shutil
import random

def count_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video file
    video.release()

    return total_frames

random.seed(42)
# climb_video_paths = random.sample(glob.glob(os.path.join("train_videos", "climb*.mp4")), 50)
# crawl_video_paths = glob.glob(os.path.join("train_videos", "crawl*.mp4"))
# raise_video_paths = random.sample(glob.glob(os.path.join("train_videos", "raise*.mp4")), 36)
# intrusion_video_paths = random.sample(glob.glob(os.path.join("train_videos", "intrusion*.mp4")), 9)
# loitering_video_paths = random.sample(glob.glob(os.path.join("train_videos", "loitering*.mp4")), 11)
#
# video_paths = climb_video_paths + crawl_video_paths + raise_video_paths + intrusion_video_paths + loitering_video_paths
video_paths = glob.glob(os.path.join("test", "*.mp4"))

count_by_action = {'intrusion':0, 'loitering':0, 'climb':0, 'crawl':0, 'raise':0}
for video_path in video_paths:
    action = os.path.basename(video_path).split("_")[0]
    # os.makedirs(os.path.join("train", action), exist_ok=True)
    count_by_action[action] += count_frames(video_path)
    # shutil.copy(video_path, os.path.join("train", action, os.path.basename(video_path)))

print(count_by_action)