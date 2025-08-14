from pathlib import Path
import cv2
import os
from tqdm import tqdm

video_paths = sorted(Path("dataset/action/ava-kinetics").glob("*/*.mp4"))
for video_path in tqdm(video_paths, total=len(video_paths)):
    output_path = Path("dataset/action/ava-kinetics/dd")/video_path.parent.stem
    os.makedirs(output_path, exist_ok=True)
    file_name = '_'.join(video_path.stem.split("_")[:-2]) + ".mp4"
    os.rename(video_path, output_path/file_name)
    # start_time = int(video_path.stem.split("_")[-2])
    # end_time = int(video_path.stem.split("_")[-1])
    # cap = cv2.VideoCapture(str(video_path))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    # start_frame = int(start_time * fps)
    # end_frame = int(end_time * fps)
    #
    # if start_frame >= total_frames or end_frame > total_frames:
    #     cap.release()
    #     continue
    # elif total_frames < 50:
    #     cap.release()
    #     continue
    #
    # # Set video position to start_frame
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    #
    # # Prepare output video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # save_path = output_path / file_name
    # out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    #
    # current_frame = start_frame
    # while current_frame < end_frame:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     out.write(frame)
    #     current_frame += 1
    #
    # cap.release()
    # out.release()

