import os
from PIL import Image, UnidentifiedImageError
#import imagehash
import cv2
import sys
import numpy as np

# Define the path to your folder
folder_path = "/Users/felixcheng/Downloads/15"

# Function to calculate the hash of an image
def calculate_hash(image_path):
    image = cv2.imread(image_path)
    return cv2.imencode('.png', image)[1].tobytes()

# Function to identify and display duplicates
def identify_duplicates(folder_path):
    hashes = {}
    duplicates = []

    # Get all image paths in the folder
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_path in image_paths:
        try:
            image_hash = calculate_hash(image_path)
            if image_hash in hashes:
                duplicates.append((image_path, hashes[image_hash]))
            else:
                hashes[image_hash] = image_path
        except Exception as e:  # <--- Update the exception handling
            print(f"Skipping file: {image_path}, due to error: {e}")

    return duplicates

# Function to display duplicate images for verification
def show_duplicates(duplicates):
    for duplicate, original in duplicates:
        print(f"\nDuplicate: {duplicate}")
        print(f"Original: {original}")
        
        # Load images
        dup_img = cv2.imread(duplicate)
        orig_img = cv2.imread(original)
        
        # Display images side by side
        combined = cv2.hconcat([orig_img, dup_img])
        cv2.imshow('Original vs Duplicate', combined)
        cv2.waitKey(0)  # Wait for a key press to show next
        cv2.destroyAllWindows()

# Run the function to identify duplicates
duplicates = identify_duplicates(folder_path)

if duplicates:
    print(f"Found {len(duplicates)} duplicates.")
    show_duplicates(duplicates)
else:
    print("No duplicates found.")

