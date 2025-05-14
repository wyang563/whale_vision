import cv2
import numpy as np
import os
import shutil
from pathlib import Path

def create_edge_dataset():
    # Define paths
    base_dir = Path('whale_vision/data')
    yolo_dir = base_dir / 'yolo_dataset'
    edges_dir = base_dir / 'yolo_edges_dataset'
    
    # Create new dataset directory structure
    edges_images_dir = edges_dir / 'images'
    edges_labels_dir = edges_dir / 'labels'
    
    # Create directories if they don't exist
    edges_images_dir.mkdir(parents=True, exist_ok=True)
    edges_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for img_path in (yolo_dir / 'images').glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Convert edges to 3-channel image to maintain compatibility
            edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Save edge-detected image
            output_path = edges_images_dir / img_path.name
            cv2.imwrite(str(output_path), edges_3channel)
            
            # Copy corresponding label file if it exists
            label_path = yolo_dir / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, edges_labels_dir / label_path.name)

if __name__ == "__main__":
    create_edge_dataset()
