from pathlib import Path
import requests
import os
from tqdm import tqdm

file_list_urls = [
    ["train", "https://s3.amazonaws.com/ava-dataset/trainval/", "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt"],
    ["test", "https://s3.amazonaws.com/ava-dataset/test/", "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_test_v2.1.txt"]
]
output_path = Path("dataset/action/AVA")
for file_list_type, download_url, file_list_url in file_list_urls:
    response = requests.get(file_list_url)
    file_list = response.text.splitlines()
    for file_name in tqdm(file_list, desc=f"{file_list_type} downloading.."):
        file_url = download_url + file_name
        file_response = requests.get(file_url, stream=True)
        file_response.raise_for_status()

        with open(output_path/file_list_type/file_name, "wb") as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)