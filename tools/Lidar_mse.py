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


def calculate_lidar_mse(pc1, pc2):
    """Calculate MSE between two LiDAR point clouds (x,y,z,i normalized)."""
    if pc1.shape != pc2.shape:
        # Truncate/pad to common size
        min_pts = min(pc1.shape[0], pc2.shape[0])
        pc1 = pc1[:min_pts]
        pc2 = pc2[:min_pts]
    
    # Normalize coordinates to [0,1] range for stable MSE
    pc1_norm = (pc1 - pc1.min(axis=0)) / (pc1.max(axis=0) - pc1.min(axis=0) + 1e-8)
    pc2_norm = (pc2 - pc2.min(axis=0)) / (pc2.max(axis=0) - pc2.min(axis=0) + 1e-8)
    
    mse_xyz = np.mean((pc1_norm[:, :3] - pc2_norm[:, :3]) ** 2)
    mse_i = np.mean((pc1_norm[:, 3] - pc2_norm[:, 3]) ** 2)
    
    return mse_xyz, mse_i, mse_xyz + 0.1 * mse_i  # Combined MSE


def process_lidar_pairs(dir1, dir2, output_file):
    """Process LiDAR point clouds with exact filename match."""
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)
    print(dir1)
    files1 = sorted([f for f in dir1_path.iterdir()])
    
    results = []
    print(files1)
    for file1 in files1:
        print(file1)
        filename = file1.name
        matching_file2 = dir2_path / filename
        
        if matching_file2.exists():
            try:
                # Load nuScenes PCD files (x,y,z,i format)
                pc1 = np.fromfile(str(file1), dtype=np.float32).reshape(-1, 5)[:, :4]  # x,y,z,i
                pc2 = np.fromfile(str(matching_file2), dtype=np.float32).reshape(-1, 5)[:, :4]
                
                mse_xyz, mse_i, mse_total = calculate_lidar_mse(pc1, pc2)
                results.append((filename, mse_xyz, mse_i, mse_total))
                print(f"{filename}: XYZ_MSE={mse_xyz:.4f}, I_MSE={mse_i:.4f}, TOTAL={mse_total:.4f}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Save results
    if results:
        with open(output_file, 'w') as f:
            f.write("filename,mse_xyz,mse_intensity,mse_total\n")
            for filename, mse_xyz, mse_i, mse_total in results:
                f.write(f'"{filename}",{mse_xyz},{mse_i},{mse_total}\n')
        print(f"\nLiDAR MSE results saved to {output_file}")


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
                h1, w1 = img1.shape
                h2, w2 = img2.shape
                target_h, target_w = min(h1, h2), min(w1, w2)
                
                img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
                img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
                
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


# Usage examples for nuScenes:
# Camera sweeps (existing)
#process_image_pairs("/home/user/.../nuscenes-mini-clean/sweeps", 
#                   "/home/user/.../v1.0-mini-dense-snow/sweeps", 
#                   "mse_camera_sweeps.csv")

# LiDAR point clouds (NEW)
process_lidar_pairs("/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet/data/nuscenes-mini-clean/sweeps/LIDAR_TOP", 
                   "/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet/data/nuscenes/v1.0-mini-dense-snow/sweeps/LIDAR_TOP", 
                   "mse_lidar_sweeps.csv")
