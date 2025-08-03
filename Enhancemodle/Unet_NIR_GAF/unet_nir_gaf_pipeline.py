import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Parameters
RGB_IMAGE_DIR = 'USA_segmentation/RGB_images'
NRG_IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
SAVE_DIR = 'Enhancemodle/Unet_NIR_GAF'
IMG_SIZE = 256
TRAIN_SPLIT = 0.8
BATCH_SIZE = 4  # Reduced due to dual input
EPOCHS = 50
LEARNING_RATE = 0.001

os.makedirs(SAVE_DIR, exist_ok=True)

# Data loading functions
def read_rgb_image(path):
    """Read and preprocess RGB image"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[-1] == 3:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0
    else:
        raise ValueError(f"{path} is not a 3-channel RGB image!")

def read_nrg_image(path):
    """Read and preprocess NRG image (4 channels: NIR, Red, Green, Blue)"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.shape[-1] == 4:
            # 4-channel NRG image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            return img.astype(np.float32) / 255.0
        elif img.shape[-1] == 3:
            # 3-channel RGB image - convert to 4-channel by adding a zero channel
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Add a zero channel as NIR (placeholder)
            nir_channel = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
            img_4ch = np.concatenate([nir_channel, img], axis=2)
            return img_4ch.astype(np.float32) / 255.0
        else:
            raise ValueError(f"{path} has {img.shape[-1]} channels, expected 3 or 4!")
    else:
        raise ValueError(f"Could not read image: {path}")

def read_mask(path):
    """Read and preprocess mask"""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)

# Improved Cross-Modal Feature Calibration Module
class CrossModalCalibration(nn.Module):
    def __init__(self, channels):
        super(CrossModalCalibration, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x

# Simplified NIR-Guided Attention Module
class ImprovedNIRGuidedAttention(nn.Module):
    def __init__(self, in_channels):
        super(ImprovedNIRGuidedAttention, self).__init__()
        self.rgb_calibration = CrossModalCalibration(in_channels)
        self.nir_calibration = CrossModalCalibration(in_channels)
        
        # Cross-modal fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb_feat, nir_feat):
        # Calibrate features
        rgb_calibrated = self.rgb_calibration(rgb_feat)
        nir_calibrated = self.nir_calibration(nir_feat)
        
        # Cross-modal fusion
        combined = torch.cat([rgb_calibrated, nir_calibrated], dim=1)
        attention_weights = self.fusion(combined)
        
        # Apply attention to RGB features
        attended_rgb = rgb_feat * attention_weights
        return attended_rgb, attention_weights

# Improved NIR-Guided Attention Fusion UNet
class ImprovedNIRGAFUNet(nn.Module):
    def __init__(self, rgb_channels=3, nir_channels=4, out_channels=1):
        super(ImprovedNIRGAFUNet, self).__init__()
        
        # RGB Encoder
        self.rgb_encoder1 = nn.Sequential(
            nn.Conv2d(rgb_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.rgb_encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.rgb_encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.rgb_encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # NIR Encoder
        self.nir_encoder1 = nn.Sequential(
            nn.Conv2d(nir_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.nir_encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.nir_encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.nir_encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Reduced attention modules (only 2 levels instead of 4)
        self.attention2 = ImprovedNIRGuidedAttention(64)   # Level 2
        self.attention4 = ImprovedNIRGuidedAttention(256)  # Level 4
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),  # 512 = 256 + 256 (skip connection)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),   # 256 = 128 + 128 (skip connection)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),   # 128 = 64 + 64 (skip connection)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),  # 64 = 32 + 32 (skip connection)
            nn.Sigmoid()
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, rgb_input, nir_input):
        # RGB Encoder
        rgb1 = self.rgb_encoder1(rgb_input)
        rgb2 = self.rgb_encoder2(self.pool(rgb1))
        rgb3 = self.rgb_encoder3(self.pool(rgb2))
        rgb4 = self.rgb_encoder4(self.pool(rgb3))
        
        # NIR Encoder
        nir1 = self.nir_encoder1(nir_input)
        nir2 = self.nir_encoder2(self.pool(nir1))
        nir3 = self.nir_encoder3(self.pool(nir2))
        nir4 = self.nir_encoder4(self.pool(nir3))
        
        # Selective NIR-Guided Attention Fusion (only at levels 2 and 4)
        attended_rgb2, att2 = self.attention2(rgb2, nir2)
        attended_rgb4, att4 = self.attention4(rgb4, nir4)
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(attended_rgb4))
        
        # Decoder with skip connections
        dec4 = self.decoder4(bottleneck)
        dec4 = torch.cat([dec4, attended_rgb4], dim=1)
        
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat([dec3, rgb3], dim=1)  # Use original RGB features
        
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat([dec2, attended_rgb2], dim=1)
        
        dec1 = self.decoder1(dec2)
        dec1 = torch.cat([dec1, rgb1], dim=1)  # Use original RGB features
        
        # Final output
        output = self.final(dec1)
        
        return output, [att2, att4]

# Improved Loss Function with IoU Loss
class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return IoU loss (1 - IoU)
        return 1 - iou

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCELoss()
        self.iou_loss = IoULoss()
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        iou = self.iou_loss(pred, target)
        return self.alpha * bce + self.beta * iou

# Dataset class for dual input
class DualInputSegDataset(Dataset):
    def __init__(self, rgb_paths, nrg_paths, mask_paths):
        self.rgb_paths = rgb_paths
        self.nrg_paths = nrg_paths
        self.mask_paths = mask_paths
        
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        rgb_img = read_rgb_image(self.rgb_paths[idx])
        nrg_img = read_nrg_image(self.nrg_paths[idx])
        mask = read_mask(self.mask_paths[idx])
        
        # Convert to tensor format
        rgb_img = torch.tensor(rgb_img.transpose(2, 0, 1), dtype=torch.float32)  # HWC -> CHW
        nrg_img = torch.tensor(nrg_img.transpose(2, 0, 1), dtype=torch.float32)  # HWC -> CHW
        mask = torch.tensor(mask[None, ...], dtype=torch.float32)  # H,W -> 1,H,W
        
        return rgb_img, nrg_img, mask

# Metrics computation
def compute_iou(y_true, y_pred):
    """Compute Intersection over Union"""
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_dice_coefficient(y_true, y_pred):
    """Compute Dice coefficient (F1 score for segmentation)"""
    intersection = np.logical_and(y_true, y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2 * intersection) / union

def compute_pixel_accuracy(y_true, y_pred):
    """Compute pixel accuracy"""
    return np.mean(y_true == y_pred)

def compute_classification_metrics(y_true, y_pred):
    """Compute precision, recall, F1-score for dead tree class"""
    from sklearn.metrics import confusion_matrix
    
    # Flatten arrays for sklearn metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1]).ravel()
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def main():
    print("Starting Improved NIR-Guided Attention Fusion UNet Pipeline...")
    
    # 1. Load and match data files
    print("Loading image files...")
    rgb_files = sorted([os.path.join(RGB_IMAGE_DIR, f) for f in os.listdir(RGB_IMAGE_DIR) 
                       if f.endswith(('.tif', '.tiff', '.png'))])
    nrg_files = sorted([os.path.join(NRG_IMAGE_DIR, f) for f in os.listdir(NRG_IMAGE_DIR) 
                       if f.endswith(('.tif', '.tiff', '.png'))])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) 
                        if f.endswith(('.png', '.tif', '.tiff'))])
    
    # Match files by base name
    rgb_base_names = [os.path.splitext(os.path.basename(f))[0].replace('RGB_', '') for f in rgb_files]
    nrg_base_names = [os.path.splitext(os.path.basename(f))[0].replace('NRG_', '') for f in nrg_files]
    mask_base_names = [os.path.splitext(os.path.basename(f))[0].replace('mask_', '') for f in mask_files]
    
    # Find common base names
    common_names = set(rgb_base_names) & set(nrg_base_names) & set(mask_base_names)
    print(f"Found {len(common_names)} matching image sets")
    
    # Create matched file lists
    matched_rgb_files = []
    matched_nrg_files = []
    matched_mask_files = []
    
    for name in sorted(common_names):
        rgb_idx = rgb_base_names.index(name)
        nrg_idx = nrg_base_names.index(name)
        mask_idx = mask_base_names.index(name)
        
        matched_rgb_files.append(rgb_files[rgb_idx])
        matched_nrg_files.append(nrg_files[nrg_idx])
        matched_mask_files.append(mask_files[mask_idx])
    
    # Split data
    train_rgb, test_rgb, train_nrg, test_nrg, train_masks, test_masks = train_test_split(
        matched_rgb_files, matched_nrg_files, matched_mask_files, 
        train_size=TRAIN_SPLIT, random_state=42
    )
    
    # Save test image paths
    with open(os.path.join(SAVE_DIR, 'test_imgs.txt'), 'w') as f:
        for rgb_p, nrg_p in zip(test_rgb, test_nrg):
            f.write(f"{rgb_p},{nrg_p}\n")
    
    # 2. Create datasets and dataloaders
    train_dataset = DualInputSegDataset(train_rgb, train_nrg, train_masks)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = ImprovedNIRGAFUNet(rgb_channels=3, nir_channels=4, out_channels=1).to(device)
    criterion = CombinedLoss(alpha=0.6, beta=0.4)  # Combined BCE + IoU loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training loop
    print('Starting training...')
    train_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (rgb_images, nrg_images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')):
            rgb_images = rgb_images.to(device)
            nrg_images = nrg_images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs, attention_weights = model(rgb_images, nrg_images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')
    
    # 5. Save model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'improved_unet_nir_gaf_model.pth'))
    print('Model saved!')
    
    # 6. Enhanced Evaluation
    print('Evaluating model with comprehensive metrics...')
    model.eval()
    predictions = []
    
    # Load test data for evaluation
    test_dataset = DualInputSegDataset(test_rgb, test_nrg, test_masks)
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            rgb_img, nrg_img, mask = test_dataset[i]
            rgb_img = rgb_img.unsqueeze(0).to(device)
            nrg_img = nrg_img.unsqueeze(0).to(device)
            
            pred, _ = model(rgb_img, nrg_img)
            pred = pred.cpu().numpy()[0, 0]  # Remove batch and channel dimensions
            predictions.append(pred)
    
    y_pred = np.array(predictions)
    
    # Load ground truth masks
    y_test = np.stack([read_mask(p) for p in test_masks])
    
    # Calculate comprehensive metrics
    y_pred_binary = (y_pred > 0.5).astype(np.uint8)
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    
    for i in range(len(y_test)):
        iou = compute_iou(y_test[i], y_pred_binary[i])
        dice = compute_dice_coefficient(y_test[i], y_pred_binary[i])
        pixel_acc = compute_pixel_accuracy(y_test[i], y_pred_binary[i])
        
        iou_scores.append(iou)
        dice_scores.append(dice)
        pixel_accuracies.append(pixel_acc)
    
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)
    
    # Calculate classification metrics
    y_test_all = y_test.flatten()
    y_pred_all = y_pred_binary.flatten()
    class_metrics = compute_classification_metrics(y_test_all, y_pred_all)
    
    print(f'Mean IoU: {mean_iou:.4f}')
    print(f'Mean Dice: {mean_dice:.4f}')
    print(f'Precision: {class_metrics["precision"]:.4f}')
    print(f'Recall: {class_metrics["recall"]:.4f}')
    print(f'F1-Score: {class_metrics["f1_score"]:.4f}')
    
    # Save comprehensive results
    with open(os.path.join(SAVE_DIR, 'improved_training_results.txt'), 'w') as f:
        f.write(f'=== Improved NIR-Guided Attention Fusion UNet Training Results ===\n')
        f.write(f'Training epochs: {EPOCHS}\n')
        f.write(f'Learning rate: {LEARNING_RATE}\n')
        f.write(f'Batch size: {BATCH_SIZE}\n')
        f.write(f'Final training loss: {train_losses[-1]:.4f}\n\n')
        
        f.write(f'=== Model Improvements ===\n')
        f.write(f'1. Reduced attention modules from 4 to 2 levels\n')
        f.write(f'2. Added Cross-Modal Feature Calibration\n')
        f.write(f'3. Combined BCE + IoU Loss\n')
        f.write(f'4. Selective attention at levels 2 and 4 only\n\n')
        
        f.write(f'=== Evaluation Metrics ===\n')
        f.write(f'Mean IoU: {mean_iou:.4f}\n')
        f.write(f'Mean Dice: {mean_dice:.4f}\n')
        f.write(f'Mean Pixel Accuracy: {mean_pixel_acc:.4f}\n')
        f.write(f'Precision: {class_metrics["precision"]:.4f}\n')
        f.write(f'Recall: {class_metrics["recall"]:.4f}\n')
        f.write(f'F1-Score: {class_metrics["f1_score"]:.4f}\n')
        f.write(f'True Positives: {class_metrics["tp"]:,}\n')
        f.write(f'False Positives: {class_metrics["fp"]:,}\n')
        f.write(f'True Negatives: {class_metrics["tn"]:,}\n')
        f.write(f'False Negatives: {class_metrics["fn"]:,}\n')
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Improved NIR-GAF UNet Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(SAVE_DIR, 'improved_training_loss.png'))
    plt.close()
    
    print('Training completed!')

if __name__ == '__main__':
    main() 