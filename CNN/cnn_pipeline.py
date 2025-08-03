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

# Parameters
IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
SAVE_DIR = 'CNN'
IMG_SIZE = 256
TRAIN_SPLIT = 0.8
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001

os.makedirs(SAVE_DIR, exist_ok=True)

# Data loading functions
def read_image(path):
    """Read and preprocess image"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[-1] == 3:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0
    else:
        raise ValueError(f"{path} is not a 3-channel image!")

def read_mask(path):
    """Read and preprocess mask"""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)

# Simple CNN Architecture for Segmentation
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsample and decode
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final output layer
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Dataset class
class SegDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = self.X[idx].transpose(2, 0, 1)  # HWC -> CHW
        mask = self.y[idx][None, ...]         # H,W -> 1,H,W
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# IoU calculation
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
    print("Starting CNN Forest Segmentation Pipeline...")
    
    # 1. Load and split data
    print("Loading image files...")
    image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                         if f.endswith(('.tif', '.tiff', '.png'))])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) 
                        if f.endswith(('.png', '.tif', '.tiff'))])
    
    assert len(image_files) == len(mask_files), 'Number of images and masks do not match!'
    
    # Split data
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_files, mask_files, train_size=TRAIN_SPLIT, random_state=42
    )
    
    # Save test image paths
    with open(os.path.join(SAVE_DIR, 'test_imgs.txt'), 'w') as f:
        for p in test_imgs:
            f.write(p + '\n')
    
    # 2. Load data
    print('Loading training data...')
    X_train = np.stack([read_image(p) for p in tqdm(train_imgs)])
    y_train = np.stack([read_mask(p) for p in tqdm(train_masks)])
    
    print('Loading test data...')
    X_test = np.stack([read_image(p) for p in tqdm(test_imgs)])
    y_test = np.stack([read_mask(p) for p in tqdm(test_masks)])
    
    # Note: Not saving y_test.npy since feature extraction is fast
    
    print(f'Training set shape: {X_train.shape}, {y_train.shape}')
    print(f'Test set shape: {X_test.shape}, {y_test.shape}')
    
    # 3. Create datasets and dataloaders
    train_dataset = SegDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = SimpleCNN(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training loop
    print('Starting training...')
    train_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')
    
    # 6. Save model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'cnn_model.pth'))
    print('Model saved!')
    
    # 7. Enhanced Evaluation
    print('Evaluating model with comprehensive metrics...')
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            img = torch.tensor(X_test[i:i+1].transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
            pred = model(img)
            pred = pred.cpu().numpy()[0, 0]  # Remove batch and channel dimensions
            predictions.append(pred)
    
    y_pred = np.array(predictions)
    # Note: Not saving y_pred.npy since feature extraction is fast
    
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
    with open(os.path.join(SAVE_DIR, 'training_results.txt'), 'w') as f:
        f.write(f'=== CNN Training Results ===\n')
        f.write(f'Training epochs: {EPOCHS}\n')
        f.write(f'Learning rate: {LEARNING_RATE}\n')
        f.write(f'Batch size: {BATCH_SIZE}\n')
        f.write(f'Final training loss: {train_losses[-1]:.4f}\n\n')
        
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
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(SAVE_DIR, 'training_loss.png'))
    plt.close()
    
    print('Training completed!')

if __name__ == '__main__':
    main() 