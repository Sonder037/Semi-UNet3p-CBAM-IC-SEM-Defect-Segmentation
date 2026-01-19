"""
Script to split dataset into train/validation/test sets
"""

import os
import shutil
import random
from pathlib import Path
import argparse


def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Split dataset into train, validation and test sets
    """
    random.seed(seed)
    
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    # Create destination directories
    for split in ['train', 'val', 'test']:
        (dest_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dest_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    all_images = []
    
    for ext in img_extensions:
        all_images.extend(list(source_dir.rglob(f'*{ext}')))
    
    # Filter to only image files (not masks)
    image_files = [f for f in all_images if not '_mask' in f.name]
    
    # Shuffle the files
    random.shuffle(image_files)
    
    # Calculate split indices
    total_count = len(image_files)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)
    
    # Split the files
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        print(f"Processing {split_name}: {len(files)} files")
        
        for img_path in files:
            # Copy image
            dest_img_path = dest_dir / split_name / 'images' / img_path.name
            shutil.copy2(img_path, dest_img_path)
            
            # Try to find and copy corresponding mask
            mask_candidates = [
                img_path.with_name(img_path.stem + '_mask' + img_path.suffix),
                img_path.with_name(img_path.stem.replace('_img', '_mask') + img_path.suffix),
            ]
            
            for mask_path in mask_candidates:
                if mask_path.exists():
                    dest_mask_path = dest_dir / split_name / 'masks' / mask_path.name
                    shutil.copy2(mask_path, dest_mask_path)
                    break
            else:
                # Create empty mask if none exists
                print(f"Warning: No mask found for {img_path.name}, creating empty mask")
                create_empty_mask(dest_dir / split_name / 'masks' / (img_path.stem + '_mask.png'))


def create_empty_mask(mask_path):
    """
    Create an empty mask file
    """
    import numpy as np
    import cv2
    
    # Assuming a standard size, adjust as needed
    empty_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.imwrite(str(mask_path), empty_mask)


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/validation/test sets')
    parser.add_argument('--source-dir', type=str, required=True,
                        help='Path to the source dataset directory')
    parser.add_argument('--dest-dir', type=str, required=True,
                        help='Path to save the split dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of training data (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Ratio of validation data (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    split_dataset(
        args.source_dir, 
        args.dest_dir, 
        args.train_ratio, 
        args.val_ratio, 
        args.seed
    )
    
    print(f"Dataset split completed:")
    print(f"- Train: {args.train_ratio*100}%")
    print(f"- Validation: {args.val_ratio*100}%")
    print(f"- Test: {(1-args.train_ratio-args.val_ratio)*100}%")


if __name__ == "__main__":
    main()