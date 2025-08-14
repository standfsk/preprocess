from PIL import Image, ImageDraw
from pathlib import Path
import os

def draw_gray_diagonal_band(image, gray_level=128):
    w, h = image.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))  # transparent overlay
    draw = ImageDraw.Draw(overlay)

    # Define polygon points for diagonal band (parallelogram)
    pts = [(1424, 583), (1454, 587), (1461, 653), (1417, 650)]

    # Draw solid gray with full opacity
    draw.polygon(pts, fill=(gray_level, gray_level, gray_level, 255))

    # Convert image to RGBA if not already
    image = image.convert("RGBA")

    # Composite: paste gray overlay on top of image
    result = Image.alpha_composite(image, overlay)
    return result

# Example usage
output_path = Path("dataset/person_head/working/dd/train/images")
os.makedirs(output_path, exist_ok=True)
image_paths = Path("dataset/person_head/working/dd/train/images").glob("*")
for image_path in image_paths:
    img = Image.open(str(image_path))
    w, h = img.size
    result = draw_gray_diagonal_band(img)

    background = Image.new("RGB", result.size, (255, 255, 255))  # white background
    background.paste(result, mask=result.split()[3])  # paste with alpha channel as mask
    background.save(output_path/image_path.name)

