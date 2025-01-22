# script for getting green bounding box pixel coordinates

import cv2
import numpy as np

def extract_green_bounding_boxes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = image[:, :500, :] # Crop the image to the left half
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define green color range in HSV
    lower_green = np.array([40, 40, 40])  # Adjust as necessary
    upper_green = np.array([80, 255, 255])  # Adjust as necessary

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store the corner coordinates
    bounding_boxes = []

    for contour in contours:
        # Get a bounding rectangle (minAreaRect for rotated boxes)
        x, y, w, h = cv2.boundingRect(contour)
        top_left = (x, y)
        top_right = (x + w, y)
        bottom_left = (x, y + h)
        bottom_right = (x + w, y + h)
        bounding_boxes.append([top_left, top_right, bottom_left, bottom_right])

        # Optionally: Draw rectangles on the image for visualization
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save or display the image with rectangles (optional)
    cv2.imwrite("output_with_bounding_boxes.png", image)
    return bounding_boxes

if __name__ == "__main__":
    image_path = "data/segmentation_dataset/train/combined_frame_0.jpg"
    bounding_boxes = extract_green_bounding_boxes(image_path)
