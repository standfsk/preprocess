import xml.etree.ElementTree as ET
import glob
import os
import shutil
from tqdm import tqdm
import cv2

def change2int(elm):
    return int(float(elm))

def count_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video file
    video.release()

    return total_frames



gt_info_by_video = dict()
with open("gt_roi.txt", "r") as txt_file:
    txt_data = txt_file.read().splitlines()
    for data in txt_data:
        video_name, intrusion_gt, loitering_gt, no_of_frames = data.split("\t")
        gt_info_by_video[video_name] = [intrusion_gt, loitering_gt, no_of_frames]

output_path = "test"

# Parse the XML file
tree = ET.parse('annotations.xml')  # Replace 'annotations.xml' with your XML file path
root = tree.getroot()

action_mapping = {"intrusion": "3", "loitering": "3", "climb": "0", "crawl": "1", "raise": "2"}
for image_info in tqdm(root.findall("image")):
    image_id = image_info.attrib["id"]
    image_name = image_info.attrib["name"]
    video_name = image_name.split(".jpg")[0]
    action = video_name.split("_")[0]
    image_width = int(image_info.attrib["width"])
    image_height = int(image_info.attrib["height"])
    for polygon_info in image_info.findall("polygon"):
        points_in_line = []
        points = polygon_info.attrib["points"].split(";")
        class_id = action_mapping[action]
        points_in_line.append(class_id)
        intrusion_time, loitering_time, no_of_frames = gt_info_by_video[image_name.replace(".jpg", ".mp4")]
        if ',' in intrusion_time:
            if "0," in no_of_frames or ",0" in no_of_frames:
                frame_counted = []
                for no_of_frame in no_of_frames.split(","):
                    frame_counted.append(count_frames(os.path.join("test", image_name.replace(".jpg", ".mp4"))))
            else:
                frame_counted = no_of_frames.split(",")
            points_in_line1, points_in_line2 = [], []
            points_in_line1.append(class_id)
            points_in_line1.append(intrusion_time.split(",")[0])
            points_in_line1.append(loitering_time.split(",")[0])
            points_in_line1.append(str(frame_counted[0]))
            points_in_line2.append(class_id)
            points_in_line2.append(intrusion_time.split(",")[1])
            points_in_line2.append(loitering_time.split(",")[1])
            points_in_line2.append(str(frame_counted[1]))
            for point in points:
                x, y = point.split(",")
                x_scaled, y_scaled = float(x)/image_width, float(y)/image_height
                points_in_line1.append(f"{x_scaled} {y_scaled}")
                points_in_line2.append(f"{x_scaled} {y_scaled}")
            with open(f"{output_path}/{video_name}.txt", "w") as txt_file:
                txt_file.write(" ".join(points_in_line1) + "\n")
                txt_file.write(" ".join(points_in_line2))

        else:
            if int(no_of_frames) == 0:
                frame_counted = count_frames(os.path.join("test", image_name.replace(".jpg", ".mp4")))
            else:
                frame_counted = no_of_frames
            points_in_line.append(intrusion_time)
            points_in_line.append(loitering_time)
            points_in_line.append(str(frame_counted))
            for point in points:
                x, y = point.split(",")
                x_scaled, y_scaled = float(x)/image_width, float(y)/image_height
                points_in_line.append(f"{x_scaled} {y_scaled}")
            with open(f"{output_path}/{video_name}.txt", "w") as txt_file:
                txt_file.write(" ".join(points_in_line))

    if image_name.replace(".jpg", ".mp4") in ["loitering_domain_rgb_004.mp4"]:
        with open(f"{output_path}/{video_name}.txt", "a") as txt_file:
            points_in_line = [action_mapping[action], "0", "0", "114", "0", "0", "0", "0", "0", "0", "0", "0"]
            txt_file.write("\n" + " ".join(points_in_line))

video_paths = glob.glob(os.path.join("test", "*.mp4"))
for video_path in video_paths:
    video_name = os.path.basename(video_path)
    action = video_name.split("_")[0]
    if action in ["intrusion", "loitering", "other"]:
        continue
    no_of_frames = count_frames(video_path)
    no_of_action_frames = no_of_frames

    if video_name in ["crawl_domain_ir_001.mp4", "crawl_domain_ir_002.mp4"]:
        points_in_line = [action_mapping[action], "0", "0", f"{no_of_action_frames}", "0", "0", "0", "0", "0", "0", "0", "0"]
        with open(f"{output_path}/{video_name.split('.')[0]}.txt", "w") as txt_file:
            txt_file.write(" ".join(points_in_line) + "\n")
            txt_file.write(" ".join(points_in_line))

    else:
        points_in_line = [action_mapping[action], "0", "0", f"{no_of_action_frames}", "0", "0", "0", "0", "0", "0", "0", "0"]
        with open(f"{output_path}/{video_name.split('.')[0]}.txt", "w") as txt_file:
            txt_file.write(" ".join(points_in_line))

