"""
Utility functions for the Semi-UNet3+-CBAM project
"""

import torch
import numpy as np


def calculate_iou(preds, targets, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) for segmentation
    """
    intersection = (preds * targets).sum()
    union = (preds + targets).sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def calculate_dice_coeff(preds, targets, smooth=1e-6):
    """
    Calculate Dice coefficient for segmentation
    """
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice


def calculate_pixel_accuracy(preds, targets):
    """
    Calculate pixel accuracy for segmentation
    """
    correct_pixels = (preds == targets).sum()
    total_pixels = targets.numel()
    accuracy = correct_pixels.float() / total_pixels
    return accuracy


def calculate_precision_recall_f1(preds, targets, smooth=1e-6):
    """
    Calculate precision, recall, and F1 score for segmentation
    """
    true_positives = (preds * targets).sum()
    false_positives = (preds * (1 - targets)).sum()
    false_negatives = ((1 - preds) * targets).sum()
    
    precision = (true_positives + smooth) / (true_positives + false_positives + smooth)
    recall = (true_positives + smooth) / (true_positives + false_negatives + smooth)
    f1_score = (2 * precision * recall) / (precision + recall + smooth)
    
    return precision, recall, f1_score


def visualize_results(original_image, ground_truth_mask, predicted_mask, title="Segmentation Result"):
    """
    Visualize segmentation results
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image.permute(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(ground_truth_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(predicted_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_checkpoint(model, optimizer, epoch, val_iou, val_dice, filepath):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_iou': val_iou,
        'val_dice': val_dice,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    val_iou = checkpoint['val_iou']
    val_dice = checkpoint['val_dice']
    
    return epoch, val_iou, val_dice