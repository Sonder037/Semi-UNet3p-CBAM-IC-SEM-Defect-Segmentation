"""
Training script for Semi-UNet3+-CBAM model
"""

import os
import sys
import yaml

# Add the project root to the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

from models import get_model
from data.data_loader import get_data_loaders
from utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train Semi-UNet3+-CBAM model')
    parser.add_argument('--config', type=str, default='../configs/default_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for images, masks in tqdm(dataloader, desc='Training', leave=False):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), masks)
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    total_loss = 0.0
    iou_scores = []
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation', leave=False):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs.squeeze(), masks)
            total_loss += loss.item()
            
            # Calculate metrics
            preds = (outputs > 0.5).float()
            iou = calculate_iou(preds, masks)
            dice = calculate_dice_coeff(preds, masks)
            
            iou_scores.append(iou)
            dice_scores.append(dice)
    
    avg_loss = total_loss / len(dataloader)
    mean_iou = torch.stack(iou_scores).mean().item()
    mean_dice = torch.stack(dice_scores).mean().item()
    
    return avg_loss, mean_iou, mean_dice


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device)
    
    # Create model
    model = get_model(
        'semiunet3plus_cbam', 
        n_channels=config['model']['n_channels'],
        n_classes=config['model']['n_classes'],
        feature_scale=config['model']['feature_scale'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=config['training']['patience'], 
        factor=0.5,
        min_lr=config['training']['min_lr']
    )
    
    # Create data loaders
    dataloaders = get_data_loaders(config)
    
    # Create checkpoint directory
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    
    # TensorBoard writer
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    writer = SummaryWriter(log_dir=config['training']['log_dir'])
    
    # Training loop
    best_val_loss = float('inf')
    best_iou_score = 0.0
    
    for epoch in range(config['training']['epochs']):
        print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}]')
        
        # Train
        train_loss = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_iou, val_dice = validate(
            model, dataloaders['val'], criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metrics/IOU', val_iou, epoch)
        writer.add_scalar('Metrics/Dice', val_dice, epoch)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Val IOU: {val_iou:.4f}, Val Dice: {val_dice:.4f}')
        
        # Save best model
        if val_iou > best_iou_score:
            best_iou_score = val_iou
            checkpoint_path = os.path.join(config['training']['save_dir'], 'best_model.pth')
            save_checkpoint(
                model, optimizer, epoch, val_iou, val_dice, checkpoint_path
            )
            print(f'Saved best model with IOU: {val_iou:.4f}')
        
        # Save model periodically
        if (epoch + 1) % 10 == 0:
            epoch_path = os.path.join(
                config['training']['save_dir'], 
                f'model_epoch_{epoch+1}.pth'
            )
            save_checkpoint(
                model, optimizer, epoch, val_iou, val_dice, epoch_path
            )
    
    writer.close()
    print("Training completed!")
    print(f"Best IOU score: {best_iou_score:.4f}")


if __name__ == '__main__':
    main()