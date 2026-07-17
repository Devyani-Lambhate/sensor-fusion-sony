import os
import cv2
import numpy as np
from pathlib import Path


def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    return mse


def resize_to_common_size(img1, img2):
    """Resize both images to the smaller common dimension."""
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    # Use the minimum dimensions
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    
    # Resize both images
    img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    return img1_resized, img2_resized


def process_image_pairs(dir1, dir2, output_file):
    """Process all image pairs with exact filename match from two directories."""
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)
    
    # Get all files from dir1
    files1 = sorted([f for f in dir1_path.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}])
    
    results = []
    
    for file1 in files1:
        filename = file1.name  # Use exact filename including extension
        # Find exact matching filename in dir2
        matching_file2 = dir2_path / filename
        
        if matching_file2.exists():
            # Read images as grayscale
            img1 = cv2.imread(str(file1), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(matching_file2), cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                print(f"Warning: Could not read {file1} or {matching_file2}")
                continue
            
            try:
                # Resize both images to common smaller size
                img1_resized, img2_resized = resize_to_common_size(img1, img2)
                mse_value = calculate_mse(img1_resized, img2_resized)
                results.append((filename, mse_value))
                print(f"{filename}: MSE = {mse_value:.4f} (resized to {img1_resized.shape})")
            except ValueError as e:
                print(f"Error processing {filename}: {e}")
    
    # Save results to .csv file
    if results:
        with open(output_file, 'w') as f:
            f.write("filename,mse\n")
            for filename, mse in results:
                f.write(f'"{filename}",{mse}\n')
        print(f"\nResults saved to {output_file}")
    else:
        print("No matching image pairs found")


# Usage example:
process_image_pairs("/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet/data/nuscenes-mini-clean/sweeps/CAM_FRONT", 
                   "/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet/data/nuscenes/v1.0-mini-dense-snow/sweeps/CAM_FRONT", 
                   "mse_camera_sweeps.csv")
