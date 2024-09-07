import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import torch
import shutil
import subprocess
import logging
import random

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# YOLOv4-tiny model configuration
YOLO_CONFIG = 'yolov4-tiny.cfg'
YOLO_WEIGHTS = 'yolov4-tiny.weights'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

def load_yolo():
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def process_image(image, net, output_layers):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    return indexes, boxes, confidences, class_ids

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(class_id)
    color = (0, 255, 0)  # Green color for bounding box
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, f"{label} {confidence:.2f}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def interactive_labeling(image_folder):
    labeled_folder = os.path.join(os.path.dirname(image_folder), "labeled_images")
    if not os.path.exists(labeled_folder):
        os.makedirs(labeled_folder)
    
    # Load existing labels
    labels_file = os.path.join(labeled_folder, "labels.json")
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    else:
        labels = {}
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if filename in labels:
                logging.info(f"Skipping {filename} as it's already labeled.")
                continue
            
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            clone = image.copy()
            cv2.namedWindow("Image")
            drawing = False
            ix, iy = -1, -1
            rect = []

            def draw_rectangle(event, x, y, flags, param):
                nonlocal drawing, ix, iy, image, clone, rect

                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    ix, iy = x, y
                    rect = []

                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        image_copy = clone.copy()
                        cv2.rectangle(image_copy, (ix, iy), (x, y), (0, 255, 0), 2)
                        cv2.imshow("Image", image_copy)

                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
                    rect = [ix, iy, x, y]
                    cv2.imshow("Image", image)

            cv2.setMouseCallback("Image", draw_rectangle)

            while True:
                cv2.imshow("Image", image)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("r"):  # Reset the image
                    image = clone.copy()
                    rect = []
                elif key == ord("c"):  # Confirm the bounding box
                    break
                elif key == ord("n"):  # No timber logs in this image
                    logging.info(f"No timber logs in {filename}")
                    break

            if rect:  # If a bounding box was drawn
                logging.info(f"Bounding box drawn for timber logs in {filename}")
                output_path = os.path.join(labeled_folder, f"labeled_{filename}")
                cv2.imwrite(output_path, image)
                
                # Convert to YOLOv4-tiny format
                x, y, x2, y2 = rect
                x_center = (x + x2) / 2 / width
                y_center = (y + y2) / 2 / height
                w = abs(x2 - x) / width
                h = abs(y2 - y) / height
                
                # Save in YOLOv4-tiny format
                yolo_label = f"0 {x_center} {y_center} {w} {h}"
                label_file = os.path.join(labeled_folder, f"{os.path.splitext(filename)[0]}.txt")
                with open(label_file, 'w') as f:
                    f.write(yolo_label)
                
                labels[filename] = rect
            else:
                labels[filename] = None  # Mark as processed but no timber logs
            
            cv2.destroyAllWindows()
            
            # Save labels after each image
            with open(labels_file, 'w') as f:
                json.dump(labels, f)

    # Create classes.txt file
    with open(os.path.join(labeled_folder, 'classes.txt'), 'w') as f:
        f.write('timber_pile')

def split_dataset(labeled_folder):
    # Split dataset into train and eval
    all_files = [f for f in os.listdir(labeled_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_files)
    split_index = int(0.7 * len(all_files))
    
    train_files = all_files[:split_index]
    eval_files = all_files[split_index:]

    # Create train and eval folders
    train_folder = os.path.join(labeled_folder, 'train')
    eval_folder = os.path.join(labeled_folder, 'eval')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(eval_folder, exist_ok=True)

    # Move files to respective folders
    for file in train_files:
        shutil.move(os.path.join(labeled_folder, file), os.path.join(train_folder, file))
        txt_file = os.path.splitext(file)[0] + '.txt'
        if os.path.exists(os.path.join(labeled_folder, txt_file)):
            shutil.move(os.path.join(labeled_folder, txt_file), os.path.join(train_folder, txt_file))

    for file in eval_files:
        shutil.move(os.path.join(labeled_folder, file), os.path.join(eval_folder, file))
        txt_file = os.path.splitext(file)[0] + '.txt'
        if os.path.exists(os.path.join(labeled_folder, txt_file)):
            shutil.move(os.path.join(labeled_folder, txt_file), os.path.join(eval_folder, txt_file))

    logging.info(f"Dataset split: {len(train_files)} train images, {len(eval_files)} eval images")

def create_yolo_txt(image_filename, annotation, output_folder):
    logging.info(f"Creating YOLO txt file for {image_filename}")
    # Create a .txt file for each image
    txt_filename = os.path.splitext(image_filename)[0] + '.txt'
    txt_filepath = os.path.join(output_folder, txt_filename)

    # Get the image path
    image_path = os.path.join(output_folder, image_filename)
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        logging.error(f"Error: Image file not found: {image_path}")
        return

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Error: Unable to read image file: {image_path}")
        return

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Extract bounding box information
    x, y, x2, y2 = annotation
    width = abs(x2 - x)
    height = abs(y2 - y)

    # Calculate center coordinates and normalized width/height
    x_center = (x + x2) / 2 / img_width
    y_center = (y + y2) / 2 / img_height
    w = width / img_width
    h = height / img_height

    # Ensure values are within [0, 1] range
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    w = max(0, min(1, w))
    h = max(0, min(1, h))

    try:
        with open(txt_filepath, 'w') as txt_file:
            # Write the YOLO format line to the .txt file (assuming class_id is 0 for timber_pile)
            txt_file.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        # Verify the file was written
        if os.path.getsize(txt_filepath) > 0:
            logging.info(f"Successfully created YOLO txt file: {txt_filepath}")
        else:
            logging.error(f"Error: YOLO txt file is empty: {txt_filepath}")
    except IOError as e:
        logging.error(f"Error writing to file {txt_filepath}: {e}")

def convert_json_to_yolo_txt(json_file_path, output_folder):
    logging.info(f"Converting JSON to YOLO txt format: {json_file_path}")
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        labels_data = json.load(f)

    # Process each image in the JSON data
    for image_filename, annotation in labels_data.items():
        if annotation is not None:
            logging.debug(f"Processing annotation for {image_filename}: {annotation}")
            create_yolo_txt(image_filename, annotation, output_folder)
        else:
            logging.info(f"Skipping {image_filename} as it has no annotation")

    logging.info(f"Conversion complete. Label files saved in {output_folder}")

    # Create train.txt and val.txt (assuming 70% train, 30% val split)
    train_folder = os.path.join(output_folder, 'train')
    eval_folder = os.path.join(output_folder, 'eval')
    
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    logging.info("Creating train.txt file...")
    with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
        for image_file in os.listdir(train_folder):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                f.write(os.path.abspath(os.path.join(train_folder, image_file)) + '\n')
    logging.info(f"train.txt created with {len(os.listdir(train_folder))} entries")
    
    logging.info("Creating val.txt file...")
    with open(os.path.join(data_dir, 'val.txt'), 'w') as f:
        for image_file in os.listdir(eval_folder):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                f.write(os.path.abspath(os.path.join(eval_folder, image_file)) + '\n')
    logging.info(f"val.txt created with {len(os.listdir(eval_folder))} entries")

    # Create obj.names with the correct class name
    class_names = ["timber_pile"]
    logging.info("Creating obj.names file...")
    with open(os.path.join(data_dir, 'obj.names'), 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    logging.info("obj.names file created")

    # Create obj.data
    logging.info("Creating obj.data file...")
    with open(os.path.join(data_dir, 'obj.data'), 'w') as f:
        f.write(f"classes = {len(class_names)}\n")
        f.write(f"train = {os.path.abspath(os.path.join(data_dir, 'train.txt'))}\n")
        f.write(f"valid = {os.path.abspath(os.path.join(data_dir, 'val.txt'))}\n")
        f.write(f"names = {os.path.abspath(os.path.join(data_dir, 'obj.names'))}\n")
        f.write(f"backup = {os.path.abspath('backup/')}\n")
    logging.info("obj.data file created")

    logging.info(f"Created additional files in {data_dir}")

# Add this function to create the YAML file
def create_yaml_file(data_dir, class_names, labeled_folder):
    yaml_path = os.path.join(data_dir, 'data.yaml')
    labeled_yaml_path = os.path.join(labeled_folder, 'data.yaml')
    logging.info(f"Creating YAML file: {yaml_path}")
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.abspath(os.path.join(data_dir, 'train.txt'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(data_dir, 'val.txt'))}\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    # Create the same YAML file in the labeled_images folder
    shutil.copy(yaml_path, labeled_yaml_path)

    logging.info(f"YAML file created successfully: {yaml_path}")
    logging.info(f"YAML file copied to: {labeled_yaml_path}") 

def fine_tune_model(labeled_folder):
    logging.info(f"Starting fine-tuning process for labeled folder: {labeled_folder}")
    # Step 1: Prepare your data
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Debug: Print current working directory and labeled folder path
    logging.debug(f"Current working directory: {os.getcwd()}")
    logging.debug(f"Labeled folder path: {os.path.abspath(labeled_folder)}")
    
    # Debug: Check if labeled folder exists
    if os.path.exists(labeled_folder):
        logging.debug(f"Labeled folder found: {labeled_folder}")
    else:
        logging.error(f"Labeled folder not found: {labeled_folder}")
        raise FileNotFoundError(f"Labeled folder not found: {labeled_folder}")
    
    # Load labels from labels.json
    labels_file = os.path.join(labeled_folder, "labels.json")
    logging.debug(f"Attempting to open file: {labels_file}")

    if os.path.exists(labels_file):
        logging.debug(f"File found: {labels_file}")
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        logging.info("Successfully loaded labels from labels.json")

        # Convert JSON to YOLO txt format and create additional files
        convert_json_to_yolo_txt(labels_file, labeled_folder)

        # Create YAML file
        class_names = ["timber_pile"]
        create_yaml_file(data_dir, class_names, labeled_folder)
    else:
        logging.error(f"File not found: {labels_file}")
        raise FileNotFoundError(f"labels.json not found in {labels_file}")
    
    # Step 2: Ask for confirmation before proceeding
    user_input = input("Data preparation complete. Do you want to proceed with fine-tuning? (y/n): ")
    if user_input.lower() != 'y':
        logging.info("Fine-tuning process cancelled by user.")
        return

    # Step 3: Modify the YOLO configuration
    cfg_path = 'yolov4-tiny-custom.cfg'
    shutil.copy('yolov4-tiny.cfg', cfg_path)
    
    # Modify the cfg file for your custom classes
    with open(cfg_path, 'r') as f:
        cfg_content = f.read()
    
    cfg_content = cfg_content.replace('classes=80', f'classes={len(class_names)}')
    cfg_content = cfg_content.replace('filters=255', f'filters={(len(class_names) + 5) * 3}')
    
    with open(cfg_path, 'w') as f:
        f.write(cfg_content)

    # Step 4: Run the fine-tuning process
    command = [
        './darknet/darknet', 'detector', 'train',
        os.path.abspath(os.path.join(data_dir, 'obj.data')),
        os.path.abspath(cfg_path),
        os.path.abspath('yolov4-tiny.weights'),
        '-dont_show'
    ]

    # Execute the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # Initialize epoch counter
    epoch = 0

    # Read output line by line
    for line in process.stdout:
        logging.info(line.strip())  # Log the line
        if "avg" in line:  # This typically indicates the end of an epoch
            epoch += 1
            logging.info(f"Completed epoch {epoch}")

    # Wait for the process to complete
    process.wait()

    logging.info("Training completed.")

    # Specify the correct path to the backup directory
    darknet_path = os.path.join(os.path.dirname(__file__), "darknet")
    backup_dir = os.path.join(darknet_path, "backup")

    if not os.path.exists(backup_dir):
        logging.warning(f"Warning: Backup directory not found at {backup_dir}")
        return None

    weights_files = [f for f in os.listdir(backup_dir) if f.endswith('.weights')]
    if weights_files:
        latest_weights = max(weights_files, key=lambda f: os.path.getmtime(os.path.join(backup_dir, f)))
        weights_path = os.path.join(backup_dir, latest_weights)
        logging.info(f"Fine-tuned model weights are available at: {weights_path}")
        return weights_path
    else:
        raise FileNotFoundError(f"No weights file found in the backup directory: {backup_dir}")

def detect_timber_trucks(image_path, model='original', output_folder='output'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the model
    if model == 'original':
        net, output_layers = load_yolo()
    else:
        net = cv2.dnn.readNet(model, YOLO_CONFIG)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    if os.path.isdir(image_path):
        # If image_path is a directory, process all images in the folder
        for filename in os.listdir(image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_path, filename)
                image = cv2.imread(img_path)
                indexes, boxes, confidences, class_ids = process_image(image, net, output_layers)
                
                # Process results
                if len(indexes) > 0:
                    logging.info(f"Timber pile detected in {filename}")
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        label = f'Timber Pile: {confidences[i]:.2f}'
                        draw_prediction(image, class_ids[i], confidences[i], x, y, x+w, y+h)
                    
                    output_path = os.path.join(output_folder, f"detected_{filename}")
                    cv2.imwrite(output_path, image)
                else:
                    logging.info(f"No timber piles detected in {filename}")
                
                # Display the result
                cv2.imshow("Detection Result", image)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
    else:
        # If image_path is a single image, process it directly
        image = cv2.imread(image_path)
        indexes, boxes, confidences, class_ids = process_image(image, net, output_layers)
        
        # Process results
        if len(indexes) > 0:
            logging.info(f"Timber pile detected in {image_path}")
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = f'Timber Pile: {confidences[i]:.2f}'
                draw_prediction(image, class_ids[i], confidences[i], x, y, x+w, y+h)
            
            output_path = os.path.join(output_folder, f"detected_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, image)
        else:
            logging.info(f"No timber piles detected in {image_path}")
        
        # Display the result
        cv2.imshow("Detection Result", image)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.withdraw()
    
    # Use the existing labeled_images folder
    labeled_folder = os.path.join(os.getcwd(), "labeled_images")
    
    choice = messagebox.askyesno("YOLOv4-tiny Processing", "Would you like to label more data?\nYes: Label more data\nNo: Use existing labeled data")
    
    if choice:
        messagebox.showinfo("YOLOv4-tiny Processing", "Select the folder containing images for labeling.")
        image_folder = filedialog.askdirectory(title="Select Image Folder")
        
        if image_folder:
            messagebox.showinfo("Instructions", "For each image:\n"
                                "- Left-click and drag to draw a bounding box\n"
                                "- Release left-click to finish drawing\n"
                                "- Press 'c' to confirm the bounding box\n"
                                "- Press 'n' if there are no timber piles\n"
                                "- Press 'r' to reset the image")
            interactive_labeling(image_folder)
    
    # Split the dataset regardless of whether new images were labeled
    split_dataset(labeled_folder)

    # Ask user if they want to fine-tune the YOLOv4-tiny model
    fine_tune_choice = messagebox.askyesno("YOLOv4-tiny Processing", "Do you want to fine-tune the YOLOv4-tiny model?")
    
    if fine_tune_choice:
        messagebox.showinfo("YOLOv4-tiny Processing", "Starting fine-tuning of YOLOv4-tiny model...")
        fine_tuned_model = fine_tune_model(labeled_folder)
        messagebox.showinfo("YOLOv4-tiny Processing", "Fine-tuning of YOLOv4-tiny model complete.")
    else:
        fine_tuned_model = None

    # Ask user if they want to run detection
    detection_choice = messagebox.askyesno("YOLOv4-tiny Processing", "Do you want to run detection?")
    
    if detection_choice:
        messagebox.showinfo("YOLOv4-tiny Processing", "Select the folder containing images for detection.")
        detection_folder = filedialog.askdirectory(title="Select Detection Folder")
    else:
        detection_folder = None
    
    if detection_folder:
        output_folder = os.path.join(os.path.dirname(detection_folder), "detected_timber_piles")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        messagebox.showinfo("YOLOv4-tiny Processing", "Running detection with both original and fine-tuned YOLOv4-tiny models...")
        
        # Detect using original YOLOv4-tiny model
        detect_timber_trucks(detection_folder, model='original', output_folder=os.path.join(output_folder, 'original'))
        
        # Detect using fine-tuned YOLOv4-tiny model if available
        if fine_tuned_model:
            detect_timber_trucks(detection_folder, model=fine_tuned_model, output_folder=os.path.join(output_folder, 'fine_tuned'))
        
        messagebox.showinfo("YOLOv4-tiny Processing", f"Detection complete. Results saved in {output_folder}")

if __name__ == "__main__":
    main()
