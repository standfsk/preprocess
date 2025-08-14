import os
from pathlib import Path
from typing import Optional

import imagehash
from PIL import Image
from tqdm import tqdm


def get_image_hash(image_path: str) -> Optional[imagehash.ImageHash]:
    try:
        img = Image.open(image_path)
        return imagehash.average_hash(img)
    except Exception as e:
        print(f"Error hashing image {image_path}: {e}")
        return None

def remove_duplicates(directories, threshold=0):
    hash_dict = {}
    for directory in tqdm(directories):
        image_paths = Path(directory).glob("*")
        for image_path in image_paths:
            if image_path.is_file():
                if any(image_path.name.lower().endswith(ext) for ext in image_extensions):
                    image_hash = get_image_hash(str(image_path))
                    if image_hash is None:
                        continue

                    # Remove similar image hashes
                    for (existing_filename, existing_hash), existing_path in hash_dict.items():
                        if abs(image_hash - existing_hash) <= threshold:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                    else:
                        # If no similar hash found, add to hash dictionary
                        hash_dict[(image_path.name, image_hash)] = image_path
if __name__ == "__main__":
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

    input_path = Path(".")
    directories = sorted(input_path.glob("TS*")) + sorted(input_path.glob("VS*"))
    remove_duplicates(directories)

