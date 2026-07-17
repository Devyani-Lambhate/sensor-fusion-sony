import os
from PIL import Image
import argparse

def resize_images_interpolation(root_dir, output_dir, target_size=(1600, 900)):
    """Resize all images to exact target size using Lanczos interpolation."""
    os.makedirs(output_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            print(file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, root_dir)
                output_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    img = Image.open(input_path)
                    # Resize to exact dimensions with Lanczos interpolation (high quality)
                    resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    resized.save(output_path, quality=95, optimize=True)
                    print(f"Resized: {rel_path} -> {target_size}")
                except Exception as e:
                    print(f"Error processing {rel_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path to input images")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--size", nargs=2, type=int, default=[1600, 900], 
                       help="Target height width (default: 1600 900)")
    args = parser.parse_args()
    
    target_size = tuple(args.size)
    print(args.input_dir, args.output_dir, target_size)
    resize_images_interpolation(args.input_dir, args.output_dir, target_size)
