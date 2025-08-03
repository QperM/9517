import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time
import json
from tqdm import tqdm

# Set matplotlib backend
plt.switch_backend('Agg')

def compute_iou(y_true, y_pred, threshold=0.5):
    """Compute IoU for binary segmentation"""
    y_pred = (y_pred > threshold).astype(np.uint8)
    y_true = y_true.astype(np.uint8)
    
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def compute_dice_coefficient(y_true, y_pred, threshold=0.5):
    """Compute Dice coefficient (F1 score) for binary segmentation"""
    y_pred = (y_pred > threshold).astype(np.uint8)
    y_true = y_true.astype(np.uint8)
    
    intersection = np.logical_and(y_true, y_pred).sum()
    total = y_true.sum() + y_pred.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2 * intersection) / total

def compute_pixel_accuracy(y_true, y_pred, threshold=0.5):
    """Compute pixel accuracy"""
    y_pred = (y_pred > threshold).astype(np.uint8)
    y_true = y_true.astype(np.uint8)
    
    return np.mean(y_true == y_pred)

class DeadTreeDataset(Dataset):
    """Dataset class for loading images and masks"""
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get image file list
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        
        # Create image and mask path lists, ensure pairing
        self.image_paths = []
        self.mask_paths = []
        
        for img_file in self.image_files:
            img_path = os.path.join(image_dir, img_file)
            mask_file = img_file.replace('NRG_', 'mask_')
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid image-mask pairs found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} valid image-mask pairs")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def create_unet_plus_plus_model(device):
    """Create UNet++ model with the same parameters as in the notebook"""
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
        decoder_use_batchnorm=True,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_use_attention=True
    ).to(device)
    
    return model

def main():
    print("Starting UNet++ Comprehensive Evaluation...")
    
    # Configuration
    IMG_SIZE = 256
    BATCH_SIZE = 16
    SAVE_DIR = 'Enhancemodle/Unet++'
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    # Load test data
    print("Loading test data...")
    test_dataset = DeadTreeDataset("usa_split/images/val", "usa_split/masks/val", transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model and generate predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_unet_plus_plus_model(device)
    
    # Load model weights
    model_path = os.path.join(SAVE_DIR, 'unet_plus_plus_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("Using untrained model for evaluation")
    
    model.eval()
    
    # Generate predictions
    print("Generating predictions...")
    all_predictions = []
    all_masks = []
    all_images = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Generating predictions", ncols=100):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            all_predictions.append(predictions.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            all_images.append(images.cpu().numpy())
    
    # Concatenate all predictions and masks
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    
    print(f"Generated predictions for {len(all_predictions)} images")
    
    # Compute metrics for each image
    print("Computing metrics...")
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    
    for i in range(len(all_predictions)):
        pred = all_predictions[i, 0]  # Remove channel dimension
        mask = all_masks[i, 0]  # Remove channel dimension
        
        iou = compute_iou(mask, pred)
        dice = compute_dice_coefficient(mask, pred)
        pixel_acc = compute_pixel_accuracy(mask, pred)
        
        iou_scores.append(iou)
        dice_scores.append(dice)
        pixel_accuracies.append(pixel_acc)
    
    # Compute overall metrics
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)
    
    # Compute confusion matrix for overall metrics
    all_pred_flat = (all_predictions.reshape(-1) > 0.5).astype(np.uint8)
    all_mask_flat = all_masks.reshape(-1).astype(np.uint8)
    
    cm = confusion_matrix(all_mask_flat, all_pred_flat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Compute precision, recall, f1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Performance categories
    excellent = sum(1 for iou in iou_scores if iou >= 0.8)
    good = sum(1 for iou in iou_scores if 0.6 <= iou < 0.8)
    fair = sum(1 for iou in iou_scores if 0.4 <= iou < 0.6)
    poor = sum(1 for iou in iou_scores if iou < 0.4)
    
    total_images = len(iou_scores)
    
    # Save comprehensive results
    print("Saving comprehensive evaluation results...")
    with open(os.path.join(SAVE_DIR, 'comprehensive_evaluation_results.txt'), 'w') as f:
        f.write(f'=== UNet++ Comprehensive Evaluation Results ===\n\n')
        
        f.write(f'1. BASIC INFORMATION\n')
        f.write(f'{"-" * 30}\n')
        f.write(f'Number of test images: {total_images}\n')
        f.write(f'Image size: {IMG_SIZE}x{IMG_SIZE} pixels\n')
        f.write(f'Model: UNet++ with ResNet34 encoder\n')
        f.write(f'Input channels: 3 (RGB)\n')
        f.write(f'Output channels: 1 (Binary segmentation)\n\n')
        
        f.write(f'2. SEGMENTATION METRICS\n')
        f.write(f'{"-" * 30}\n')
        f.write(f'IoU (Intersection over Union):\n')
        f.write(f'  Mean IoU: {mean_iou:.4f}\n')
        f.write(f'  Min IoU: {min(iou_scores):.4f}\n')
        f.write(f'  Max IoU: {max(iou_scores):.4f}\n')
        f.write(f'  Median IoU: {np.median(iou_scores):.4f}\n')
        f.write(f'  Std IoU: {np.std(iou_scores):.4f}\n\n')
        
        f.write(f'Dice Coefficient (F1 Score):\n')
        f.write(f'  Mean Dice: {mean_dice:.4f}\n')
        f.write(f'  Min Dice: {min(dice_scores):.4f}\n')
        f.write(f'  Max Dice: {max(dice_scores):.4f}\n')
        f.write(f'  Median Dice: {np.median(dice_scores):.4f}\n')
        f.write(f'  Std Dice: {np.std(dice_scores):.4f}\n\n')
        
        f.write(f'Pixel Accuracy:\n')
        f.write(f'  Mean Pixel Accuracy: {mean_pixel_acc:.4f}\n')
        f.write(f'  Min Pixel Accuracy: {min(pixel_accuracies):.4f}\n')
        f.write(f'  Max Pixel Accuracy: {max(pixel_accuracies):.4f}\n')
        f.write(f'  Median Pixel Accuracy: {np.median(pixel_accuracies):.4f}\n')
        f.write(f'  Std Pixel Accuracy: {np.std(pixel_accuracies):.4f}\n\n')
        
        f.write(f'3. CLASSIFICATION METRICS (Dead Tree Class)\n')
        f.write(f'{"-" * 40}\n')
        f.write(f'Confusion Matrix:\n')
        f.write(f'  True Negatives (TN): {tn:,}\n')
        f.write(f'  False Positives (FP): {fp:,}\n')
        f.write(f'  False Negatives (FN): {fn:,}\n')
        f.write(f'  True Positives (TP): {tp:,}\n\n')
        
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1-Score: {f1_score:.4f}\n\n')
        
        f.write(f'4. PERFORMANCE CATEGORIES\n')
        f.write(f'{"-" * 30}\n')
        f.write(f'Excellent (IoU ≥ 0.8): {excellent} images ({excellent/total_images*100:.1f}%)\n')
        f.write(f'Good (0.6 ≤ IoU < 0.8): {good} images ({good/total_images*100:.1f}%)\n')
        f.write(f'Fair (0.4 ≤ IoU < 0.6): {fair} images ({fair/total_images*100:.1f}%)\n')
        f.write(f'Poor (IoU < 0.4): {poor} images ({poor/total_images*100:.1f}%)\n\n')
        
        f.write(f'5. INDIVIDUAL IMAGE RESULTS\n')
        f.write(f'{"-" * 30}\n')
        f.write(f'Image\tIoU\t\tDice\t\tPixel_Acc\n')
        f.write(f'{"-" * 50}\n')
        
        for i in range(total_images):
            f.write(f'{i+1}\t{iou_scores[i]:.4f}\t\t{dice_scores[i]:.4f}\t\t{pixel_accuracies[i]:.4f}\n')
        
        f.write(f'\n6. SUMMARY\n')
        f.write(f'{"-" * 30}\n')
        f.write(f'Overall Performance: {"Excellent" if mean_iou >= 0.7 else "Good" if mean_iou >= 0.5 else "Fair"}\n')
        f.write(f'Primary Metric (Mean IoU): {mean_iou:.4f}\n')
        f.write(f'Balanced Metric (F1-Score): {f1_score:.4f}\n')
        f.write(f'Pixel Accuracy: {mean_pixel_acc:.4f}\n')
        f.write(f'Model Architecture: UNet++ with attention mechanisms\n')
    
    # Save individual results for further analysis
    individual_results = {
        'image_indices': list(range(1, total_images + 1)),
        'iou_scores': iou_scores,
        'dice_scores': dice_scores,
        'pixel_accuracies': pixel_accuracies
    }
    
    with open(os.path.join(SAVE_DIR, 'individual_results.json'), 'w') as f:
        json.dump(individual_results, f, indent=2)
    
    # Create performance distribution plots
    plt.figure(figsize=(15, 5))
    
    # IoU distribution
    plt.subplot(1, 3, 1)
    plt.hist(iou_scores, bins=20, alpha=0.7, color='blue')
    plt.axvline(mean_iou, color='red', linestyle='--', label=f'Mean: {mean_iou:.3f}')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.title('IoU Score Distribution')
    plt.legend()
    
    # Dice distribution
    plt.subplot(1, 3, 2)
    plt.hist(dice_scores, bins=20, alpha=0.7, color='green')
    plt.axvline(mean_dice, color='red', linestyle='--', label=f'Mean: {mean_dice:.3f}')
    plt.xlabel('Dice Score')
    plt.ylabel('Frequency')
    plt.title('Dice Score Distribution')
    plt.legend()
    
    # Pixel accuracy distribution
    plt.subplot(1, 3, 3)
    plt.hist(pixel_accuracies, bins=20, alpha=0.7, color='orange')
    plt.axvline(mean_pixel_acc, color='red', linestyle='--', label=f'Mean: {mean_pixel_acc:.3f}')
    plt.xlabel('Pixel Accuracy')
    plt.ylabel('Frequency')
    plt.title('Pixel Accuracy Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'performance_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Background', 'Dead Tree']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], ',d'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n=== UNet++ Evaluation Summary ===")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"\nPerformance Categories:")
    print(f"  Excellent (IoU ≥ 0.8): {excellent} images ({excellent/total_images*100:.1f}%)")
    print(f"  Good (0.6 ≤ IoU < 0.8): {good} images ({good/total_images*100:.1f}%)")
    print(f"  Fair (0.4 ≤ IoU < 0.6): {fair} images ({fair/total_images*100:.1f}%)")
    print(f"  Poor (IoU < 0.4): {poor} images ({poor/total_images*100:.1f}%)")
    
    print(f"\nResults saved to {SAVE_DIR}")

if __name__ == "__main__":
    main() 