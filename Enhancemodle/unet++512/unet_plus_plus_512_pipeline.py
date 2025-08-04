import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
import time
import json
from sklearn.model_selection import train_test_split

# Set matplotlib backend
plt.switch_backend('Agg')

class DeadTreeDataset512(Dataset):
    """Dataset class for 512x512 images"""
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        if len(self.image_paths) == 0:
            raise ValueError("No valid image-mask pairs found.")
        print(f"Found {len(self.image_paths)} valid image-mask pairs")
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

def compute_iou(pred, target, threshold=0.5):
    """Compute IoU between prediction and target"""
    pred = (pred > threshold).int()
    target = target.int()
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection.float() / union.float()).item()

def create_unet_plus_plus_512():
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
        decoder_use_batchnorm=True,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_use_attention=True
    )
    return model

def prepare_data(image_dir, mask_dir, test_size=0.2):
    """Prepare and split data into training and validation sets."""
    print("Preparing and splitting data...")
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_paths = []
    mask_paths = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        mask_file = img_file.replace('NRG_', 'mask_')
        mask_path = os.path.join(mask_dir, mask_file)
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
    if len(image_paths) == 0:
        raise ValueError(f"No valid image-mask pairs found in {image_dir}")
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=42
    )
    print(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")
    return train_images, val_images, train_masks, val_masks

def train_unet_plus_plus_512(num_epochs=100, batch_size=8, learning_rate=1e-4):
    print("=== Training UNet++ for 512x512 Images ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    # 自动划分数据集
    train_images, val_images, train_masks, val_masks = prepare_data(
        "USA_segmentation/NRG_images", "USA_segmentation/masks", test_size=0.2
    )
    train_dataset = DeadTreeDataset512(train_images, train_masks, transform)
    val_dataset = DeadTreeDataset512(val_images, val_masks, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    model = create_unet_plus_plus_512().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_miou = 0.0
    patience = 15
    patience_counter = 0
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = script_dir  # Save directly in the script directory
    history = {'train_loss': [], 'val_loss': [], 'miou': [], 'epochs': []}
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        all_preds, all_masks = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                all_preds.append(torch.sigmoid(outputs))
                all_masks.append(masks)
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        miou_scores = []
        for i in range(all_preds.shape[0]):
            pred = all_preds[i:i+1]
            mask = all_masks[i:i+1]
            miou = compute_iou(pred, mask)
            miou_scores.append(miou)  # miou is already a float
        miou = np.mean(miou_scores)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['miou'].append(miou)
        history['epochs'].append(epoch)
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | mIoU: {miou:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_miou = miou
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "unet_plus_plus_model.pth"))
            print(f"New best model saved! Val Loss: {avg_val_loss:.4f}, mIoU: {miou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
    training_time = time.time() - start_time
    with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(history['epochs'], history['miou'], label='mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Mean IoU')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()
    results = {
        'model_type': 'unet_plus_plus_512',
        'best_val_loss': best_val_loss,
        'best_miou': best_miou,
        'training_time': training_time,
        'epochs_trained': epoch,
        'history': history,
        'total_params': total_params,
        'image_size': '512x512',
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    with open(os.path.join(save_dir, "training_results.txt"), 'w') as f:
        f.write("UNet++ 512x512 Training Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")
        f.write(f"Best mIoU: {best_miou:.4f}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Epochs trained: {epoch}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Image size: 512x512\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
    print(f"Training completed for UNet++ 512x512!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    return results

if __name__ == "__main__":
    results = train_unet_plus_plus_512(num_epochs=100, batch_size=8, learning_rate=1e-4)
    print("Training completed!") 