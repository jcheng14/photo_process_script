import os
import cv2

# Define the path to your folder
folder_path = "."

# Get all image paths in the folder
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Load each image using OpenCV
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is not None:
        print(f"Loaded {image_path}")
    else:
        print(f"Failed to load {image_path}")

