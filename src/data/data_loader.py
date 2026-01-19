"""
Dataset loader for IC SEM defect detection
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path


class ICSEMDataset(Dataset):
    """
    Dataset for IC SEM defect detection
    """
    
    def __init__(self, image_dir, mask_dir=None, transform=None):
        """
        Initialize dataset
        
        Args:
            image_dir: Path to the directory containing images
            mask_dir: Path to the directory containing masks (optional for unlabeled data)
            transform: Transformations to apply to the images
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        
        # Find all image files
        self.images = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.images.extend(list(self.image_dir.rglob(f'*{ext}')))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            # If image can't be loaded, try next one
            return self[(idx + 1) % len(self)]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mask_dir:
            # Look for corresponding mask
            mask_path = None
            stem = img_path.stem
            
            # Try different naming conventions for masks
            possible_mask_names = [
                self.mask_dir / f"{stem}_mask.png",
                self.mask_dir / f"{stem}_mask.jpg",
                self.mask_dir / f"{stem}.png",  # Same name, different extension
                self.mask_dir / f"{stem}.jpg",
            ]
            
            for mp in possible_mask_names:
                if mp.exists():
                    mask_path = mp
                    break
            
            if mask_path and mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.float32)  # Convert to binary mask
            else:
                # Create empty mask if not exists
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            # For unlabeled data, return zeros as mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            
        if self.transform:
            # Apply transforms if available (requires albumentations library)
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Normalize image
            image = image.astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)  # Change to CHW format
            
        return torch.from_numpy(image), torch.from_numpy(mask)


def get_data_loaders(config):
    """
    Create data loaders from config
    
    Args:
        config: Configuration dictionary with data settings
    
    Returns:
        Dictionary containing train, val and test data loaders
    """
    # Construct paths from config
    dataset_path = Path(config['data']['dataset_path'])
    
    # Define splits paths
    train_img_dir = dataset_path / config['data']['train_folder'] / 'images'
    train_mask_dir = dataset_path / config['data']['train_folder'] / 'masks'
    val_img_dir = dataset_path / config['data']['test_folder'] / 'images'  # Using test as validation temporarily
    val_mask_dir = dataset_path / config['data']['test_folder'] / 'masks'
    
    # If train/val/test splits exist, use those instead
    split_train_img = dataset_path / 'train' / 'images'
    split_train_mask = dataset_path / 'train' / 'masks'
    split_val_img = dataset_path / 'val' / 'images'
    split_val_mask = dataset_path / 'val' / 'masks'
    split_test_img = dataset_path / 'test' / 'images'
    split_test_mask = dataset_path / 'test' / 'masks'
    
    # Use split data if available, otherwise fall back to original structure
    if split_train_img.exists():
        train_img_dir, train_mask_dir = split_train_img, split_train_mask
    if split_val_img.exists():
        val_img_dir, val_mask_dir = split_val_img, split_val_mask
    
    # Create datasets
    train_dataset = ICSEMDataset(train_img_dir, train_mask_dir)
    val_dataset = ICSEMDataset(val_img_dir, val_mask_dir)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False, 
        num_workers=config['data']['num_workers']
    )
    
    # If test data exists, create test loader
    if split_test_img.exists() and split_test_mask.exists():
        test_dataset = ICSEMDataset(split_test_img, split_test_mask)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['data']['batch_size'], 
            shuffle=False, 
            num_workers=config['data']['num_workers']
        )
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    return {
        'train': train_loader,
        'val': val_loader
    }