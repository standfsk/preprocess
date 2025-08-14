from pathlib import Path
import json

label_paths = Path("dataset/dd").glob("*/*.json")
for label_path in label_paths:
    subset = label_path.parts[2]

    with open(label_path, "r") as label_file:
        label_data = json.load(label_file)
        print("a")
