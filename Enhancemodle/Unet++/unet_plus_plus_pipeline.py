import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import shutil
import random
import time
import json
import cv2
from tqdm import tqdm

# Set matplotlib backend
plt.switch_backend('Agg')

# Configuration
IMG_SIZE = 256
BATCH_SIZE = 20
LEARNING_RATE = 1e-4
EPOCHS = 100
SAVE_DIR = 'Enhancemodle/Unet++'
PATIENCE = 10
NUM_WORKERS = 0  # 减少worker数量以避免GPU等待数据

# Early stopping configuration
EARLY_STOPPING_MONITOR = 'val_loss'  # 'val_loss' 或 'miou'
EARLY_STOPPING_MIN_DELTA = 1e-4  # 最小改善阈值

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

def prepare_dataset():
    """Prepare dataset, split original data into training and validation sets"""
    print("Preparing dataset...")
    
    # Set random seed
    random.seed(42)
    
    # Source data paths
    source_images = "USA_segmentation/NRG_images"
    source_masks = "USA_segmentation/masks"
    
    # Check if source data exists
    if not os.path.exists(source_images):
        raise FileNotFoundError(f"Source images directory not found: {source_images}")
    if not os.path.exists(source_masks):
        raise FileNotFoundError(f"Source masks directory not found: {source_masks}")
    
    # Target paths
    target_base = "usa_split"
    train_images = f"{target_base}/images/train"
    train_masks = f"{target_base}/masks/train"
    val_images = f"{target_base}/images/val"
    val_masks = f"{target_base}/masks/val"
    
    # Clear and recreate directories
    for path in [train_images, train_masks, val_images, val_masks]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(source_images) if f.endswith('.png')]
    print(f"Found {len(image_files)} image files in source directory")
    
    # Random split: 80% training, 20% validation
    random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    
    # Copy training data
    train_valid_pairs = 0
    for filename in train_files:
        src_img = os.path.join(source_images, filename)
        dst_img = os.path.join(train_images, filename)
        shutil.copy2(src_img, dst_img)
        
        mask_filename = filename.replace('NRG_', 'mask_')
        src_mask = os.path.join(source_masks, mask_filename)
        dst_mask = os.path.join(train_masks, mask_filename)
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, dst_mask)
            train_valid_pairs += 1
        else:
            print(f"Warning: No mask found for {filename}")
    
    # Copy validation data
    val_valid_pairs = 0
    for filename in val_files:
        src_img = os.path.join(source_images, filename)
        dst_img = os.path.join(val_images, filename)
        shutil.copy2(src_img, dst_img)
        
        mask_filename = filename.replace('NRG_', 'mask_')
        src_mask = os.path.join(source_masks, mask_filename)
        dst_mask = os.path.join(val_masks, mask_filename)
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, dst_mask)
            val_valid_pairs += 1
        else:
            print(f"Warning: No mask found for {filename}")
    
    print(f"Training valid pairs: {train_valid_pairs}")
    print(f"Validation valid pairs: {val_valid_pairs}")
    print("Dataset preparation completed!")

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

def compute_miou(preds, masks, threshold=0.5):
    """Compute mIoU"""
    preds = (preds > threshold).int().cpu().numpy().flatten()
    masks = masks.int().cpu().numpy().flatten()
    
    cm = confusion_matrix(masks, preds, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0
    
    tn, fp, fn, tp = cm.ravel()
    iou_fg = tp / (tp + fp + fn + 1e-8)
    iou_bg = tn / (tn + fn + fp + 1e-8)
    
    return (iou_fg + iou_bg) / 2

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

def train_model(train_loader, val_loader, device, num_epochs=EPOCHS):
    """Train UNet++ model"""
    print(f"\n=== Training UNet++ ===")
    
    # Create model
    model = create_unet_plus_plus_model(device)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training settings
    best_val_loss = float("inf")
    best_miou = 0.0
    patience_counter = 0
    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # Early stopping settings
    min_delta = EARLY_STOPPING_MIN_DELTA
    patience = PATIENCE
    verbose = True
    monitor = EARLY_STOPPING_MONITOR
    mode = 'min' if monitor == 'val_loss' else 'max'  # 监控模式
    
    print(f"Early stopping configuration:")
    print(f"  Monitor: {monitor}")
    print(f"  Mode: {mode}")
    print(f"  Patience: {patience}")
    print(f"  Min delta: {min_delta}")
    
    # Record training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'miou': [],
        'epochs': []
    }
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # 添加训练进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} [Train]', 
                         leave=False, ncols=100)
        
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # 更新进度条
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds, all_masks = [], []
        
        # 添加验证进度条
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} [Val]', 
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                all_preds.append(torch.sigmoid(outputs))
                all_masks.append(masks)
                
                # 更新进度条
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        miou = compute_miou(all_preds, all_masks)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['miou'].append(miou)
        history['epochs'].append(epoch)

        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | mIoU: {miou:.4f}")

        # Early stopping check
        improved = False
        current_metric = avg_val_loss if monitor == 'val_loss' else miou
        best_metric = best_val_loss if monitor == 'val_loss' else best_miou
        
        # Check if monitored metric improved
        if mode == 'min':
            if current_metric < (best_metric - min_delta):
                improved = True
        else:  # mode == 'max'
            if current_metric > (best_metric + min_delta):
                improved = True
        
        if improved:
            best_val_loss = avg_val_loss
            best_miou = miou
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'unet_plus_plus_model.pth'))
            if verbose:
                print(f"✓ New best model saved! Val Loss: {avg_val_loss:.4f}, mIoU: {miou:.4f}")
        else:
            patience_counter += 1
            if verbose:
                print(f"✗ No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"Best mIoU: {best_miou:.4f}")
                break
    
    training_time = time.time() - start_time
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Return results
    results = {
        'model_type': 'unet_plus_plus',
        'best_val_loss': best_val_loss,
        'best_miou': best_miou,
        'training_time': training_time,
        'epochs_trained': epoch,
        'history': history,
        'total_params': total_params
    }
    
    print(f"Training completed for UNet++!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    return results

def main():
    print("Starting UNet++ Pipeline...")
    
    # 1. Prepare dataset
    if not os.path.exists("usa_split"):
        prepare_dataset()
    else:
        print("Dataset already exists!")
    
    # 2. Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    # 3. Create datasets
    train_dataset = DeadTreeDataset("usa_split/images/train", "usa_split/masks/train", transform)
    val_dataset = DeadTreeDataset("usa_split/images/val", "usa_split/masks/val", transform)
    
    # 4. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # 5. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 6. Train model
    results = train_model(train_loader, val_loader, device, num_epochs=EPOCHS)
    
    # 7. Save comprehensive results
    with open(os.path.join(SAVE_DIR, 'training_results.txt'), 'w') as f:
        f.write(f'=== UNet++ Training Results ===\n')
        f.write(f'Training epochs: {EPOCHS}\n')
        f.write(f'Learning rate: {LEARNING_RATE}\n')
        f.write(f'Batch size: {BATCH_SIZE}\n')
        f.write(f'Final training loss: {results["history"]["train_loss"][-1]:.4f}\n')
        f.write(f'Best validation loss: {results["best_val_loss"]:.4f}\n')
        f.write(f'Best mIoU: {results["best_miou"]:.4f}\n')
        f.write(f'Training time: {results["training_time"]:.2f} seconds\n')
        f.write(f'Total parameters: {results["total_params"]:,}\n\n')
        
        f.write(f'=== Model Architecture ===\n')
        f.write(f'Model: UNet++ with ResNet34 encoder\n')
        f.write(f'Input channels: 3 (RGB)\n')
        f.write(f'Output channels: 1 (Binary segmentation)\n')
        f.write(f'Decoder attention type: scse\n')
        f.write(f'Decoder channels: (256, 128, 64, 32, 16)\n')
        f.write(f'Decoder use attention: True\n')
        f.write(f'Decoder use batchnorm: True\n\n')
        
        f.write(f'=== Training Configuration ===\n')
        f.write(f'Loss function: BCEWithLogitsLoss\n')
        f.write(f'Optimizer: Adam\n')
        f.write(f'Learning rate: {LEARNING_RATE}\n')
        f.write(f'Batch size: {BATCH_SIZE}\n')
        f.write(f'Image size: {IMG_SIZE}x{IMG_SIZE}\n')
        f.write(f'Early stopping patience: {PATIENCE}\n')
    
    # 8. Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(results['history']['train_loss'], label='Training Loss')
    plt.plot(results['history']['val_loss'], label='Validation Loss')
    plt.title('UNet++ Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'training_loss.png'))
    plt.close()
    
    # 9. Plot mIoU
    plt.figure(figsize=(10, 6))
    plt.plot(results['history']['miou'])
    plt.title('UNet++ Validation mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.savefig(os.path.join(SAVE_DIR, 'training_miou.png'))
    plt.close()
    
    print(f"Training completed! Results saved to {SAVE_DIR}")
    print(f"Best mIoU: {results['best_miou']:.4f}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")

if __name__ == "__main__":
    main() 