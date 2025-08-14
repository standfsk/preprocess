import csv
import os
from pathlib import Path


target_action_ids = {"c150", "c151", "c154"}

# Open the CSV file in read mode ('r')
with open('dataset/action/Charades/Charades_v1_train.csv', 'r', newline='') as csvfile:
    # Create a csv.reader object
    csv_reader = csv.reader(csvfile)

    # If your CSV has a header, you might want to skip it
    header = next(csv_reader)
    print(f"Header: {header}")

    # Iterate over each row in the CSV file
    for row in csv_reader:
        file_name = row[0]
        action_ids = {x.split(" ")[0] for x in row[9].split(";")}
        if target_action_ids & action_ids:
            if os.path.exists(f"dataset/action/dd/{file_name}.mp4"):
                os.rename(f"dataset/action/dd/{file_name}.mp4", f"dataset/action/d/{file_name}.mp4")


