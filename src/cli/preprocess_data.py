"""
Data preprocessing script for IC SEM defect detection dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse


def preprocess_dataset(data_dir, output_dir):
    """
    Preprocess the dataset by resizing images and normalizing
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    (output_dir / "images").mkdir(exist_ok=True, parents=True)
    (output_dir / "masks").mkdir(exist_ok=True, parents=True)
    
    # Process images and masks
    img_exts = ['.jpg', '.jpeg', '.png']
    
    # Find all images in the source directory
    for img_path in data_dir.rglob('*'):
        if img_path.suffix.lower() in img_exts:
            # Read and preprocess image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Resize image to standard size (adjust as needed)
            target_size = (512, 512)
            img_resized = cv2.resize(img, target_size)
            
            # Save processed image
            rel_path = img_path.relative_to(data_dir)
            out_path = output_dir / "images" / rel_path.with_suffix('.jpg')
            out_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(out_path), img_resized)
            
            # Process corresponding mask if exists
            mask_candidates = [
                img_path.with_name(img_path.stem + '_mask' + img_path.suffix),
                img_path.with_name(img_path.stem.replace('_img', '_mask') + img_path.suffix),
            ]
            
            for mask_path in mask_candidates:
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                        mask_out_path = output_dir / "masks" / mask_path.relative_to(data_dir).with_suffix('.png')
                        mask_out_path.parent.mkdir(exist_ok=True, parents=True)
                        cv2.imwrite(str(mask_out_path), (mask_resized > 127).astype(np.uint8) * 255)
                        break


def main():
    parser = argparse.ArgumentParser(description='Preprocess IC SEM defect detection dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the raw dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to save the preprocessed dataset')
    
    args = parser.parse_args()
    
    preprocess_dataset(args.data_dir, args.output_dir)
    print(f"Dataset preprocessed and saved to {args.output_dir}")


if __name__ == "__main__":
    main()