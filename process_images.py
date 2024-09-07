import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import re
import matplotlib
from itertools import combinations
import time
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_images(folder):
    """
    Load images from a specified folder.
    
    Args:
    folder (str): Path to the folder containing images.
    
    Returns:
    dict: A dictionary with filenames as keys and image arrays as values.
    """
    images = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                images[filename] = img
            else:
                logging.warning(f"Failed to load image: {filename}")
    logging.info(f"Loaded {len(images)} images from {folder}")
    return images

def compare_images(img1, img2):
    """
    Compare two images using Histogram Comparison.
    
    Args:
    img1, img2 (numpy.ndarray): Input images to compare.
    
    Returns:
    float: Histogram Comparison score.
    """
    # Downsample images for histogram comparison
    scale_percent = 50 # percent of original size
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)
    img1_downsampled = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    img2_downsampled = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    
    # Histogram Comparison
    hist1 = cv2.calcHist([img1_downsampled], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2_downsampled], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    hist_comp = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return hist_comp

def is_similar(img1, img2, hist_threshold=0.9):
    """
    Check if two images are similar based on Histogram Comparison threshold.
    
    Args:
    img1, img2 (numpy.ndarray): Input images to compare.
    hist_threshold (float): Threshold for Histogram Comparison similarity.
    
    Returns:
    tuple: Boolean indicating similarity and Histogram Comparison score.
    """
    hist_comp = compare_images(img1, img2)
    return hist_comp >= hist_threshold, hist_comp

def extract_shortip(filename):
    """
    Extract the first three digits from a filename.
    
    Args:
    filename (str): Input filename.
    
    Returns:
    str or None: First three digits if found, None otherwise.
    """
    match = re.match(r'^(\d{3})', filename)
    return match.group(1) if match else None

def display_comparison(img1, img2, title1, title2, hist_comp, hist_threshold):
    """
    Display and save a comparison of two images with their similarity metrics.
    
    Args:
    img1, img2 (numpy.ndarray): Input images to compare.
    title1, title2 (str): Titles for the images.
    hist_comp: Histogram Comparison score.
    hist_threshold: Threshold for Histogram Comparison similarity.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original color images
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"{title1}")
    ax1.axis('off')
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"{title2}")
    ax2.axis('off')
    
    is_similar = hist_comp >= hist_threshold
    similarity_status = "Similar" if is_similar else "Different"
    plt.suptitle(f"Histogram Comparison: {hist_comp:.4f} (threshold: {hist_threshold:.2f})\n"
                 f"Status: {similarity_status}")
    plt.tight_layout()
    
    # Create 'compared_images' folder if it doesn't exist
    os.makedirs('compared_images', exist_ok=True)
    
    # Save the figure in the 'compared_images' folder
    plt.savefig(os.path.join('compared_images', f"comparison_{title1}_{title2}.png"))
    plt.close(fig)  # Close the figure after saving

def process_images(baseline_folder, target_folder, hist_threshold=0.9):
    """
    Process images by comparing target images with baseline images.
    
    Args:
    baseline_folder (str): Path to the folder containing baseline images.
    target_folder (str): Path to the folder containing target images.
    hist_threshold (float): Threshold for Histogram Comparison similarity.
    
    Returns:
    list: List of relevant image names from the target folder.
    """
    baseline_images = load_images(baseline_folder)
    target_images = load_images(target_folder)
    
    relevant_images = []
    
    # Group images by ShortIP
    baseline_grouped = {}
    target_grouped = {}
    
    for name, img in baseline_images.items():
        shortip = extract_shortip(name)
        if shortip:
            baseline_grouped.setdefault(shortip, {})[name] = img
    
    for name, img in target_images.items():
        shortip = extract_shortip(name)
        if shortip:
            target_grouped.setdefault(shortip, {})[name] = img
    
    # Define the number of comparisons to display
    num_comparisons_to_display = 200

    # Estimate total comparisons
    total_comparisons = sum(len(target_group) * len(baseline_grouped.get(shortip, {})) for shortip, target_group in target_grouped.items())
    logging.info(f"Estimated total comparisons with baseline: {total_comparisons}")
    
    # Compare with baseline
    logging.info("Comparing target images with baseline")
    comparison_count = 0
    start_time = time.time()
    for shortip, target_group in target_grouped.items():
        baseline_group = baseline_grouped.get(shortip, {})
        for target_name, target_img in target_group.items():
            is_relevant = False
            for baseline_name, baseline_img in baseline_group.items():
                comparison_count += 1
                similar, hist_comp = is_similar(target_img, baseline_img, hist_threshold)
                logging.debug(f"Comparison {comparison_count}/{total_comparisons}: {target_name} vs {baseline_name} - Hist Comp: {hist_comp:.4f}")
                if comparison_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    logging.info(f"Progress: {comparison_count}/{total_comparisons} comparisons, Time taken: {elapsed_time:.2f} seconds")
                if comparison_count <= num_comparisons_to_display:
                    display_comparison(target_img, baseline_img, f"Target_{target_name}", f"Baseline_{baseline_name}", hist_comp, hist_threshold)
                if similar:
                    is_relevant = True
                    break
            if is_relevant:
                relevant_images.append(target_name)
    
    logging.info(f"Processing complete. Relevant images: {len(relevant_images)}")
    return relevant_images

# Main execution
if __name__ == "__main__":
    logging.info("Script started")
    baseline_folder = os.path.join(os.getcwd(), 'selected_front')
    target_folder = os.path.join(os.getcwd(), '8_30_truck_random')

    # Set threshold for similarity
    hist_threshold = 0.9  # Histogram Comparison threshold

    relevant_images = process_images(baseline_folder, target_folder, hist_threshold)

    print(f"Number of relevant images in target folder: {len(relevant_images)}")

    logging.info("Script completed")
