import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from collections import Counter

# Parameters
IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
IMG_SIZE = 256
TRAIN_SPLIT = 0.8

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

def extract_simple_features(image, mask):
    """Extract simple but effective features"""
    features_list = []
    labels_list = []
    
    # Sample every 32nd pixel to reduce computation
    step = 32
    
    for i in range(0, IMG_SIZE, step):
        for j in range(0, IMG_SIZE, step):
            # Get 8x8 patch around the pixel
            patch_size = 8
            i_start = max(0, i - patch_size//2)
            i_end = min(IMG_SIZE, i + patch_size//2)
            j_start = max(0, j - patch_size//2)
            j_end = min(IMG_SIZE, j + patch_size//2)
            
            patch = image[i_start:i_end, j_start:j_end]
            patch_mask = mask[i_start:i_end, j_start:j_end]
            
            # Get label for this pixel
            label = mask[i, j]
            
            # Extract simple features
            features = []
            
            # Color features (mean of each channel)
            for c in range(3):
                features.append(np.mean(patch[:, :, c]))
            
            # Grayscale mean and std
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            features.extend([np.mean(gray_patch), np.std(gray_patch)])
            
            # Position features
            features.extend([i/IMG_SIZE, j/IMG_SIZE])
            
            features_list.append(features)
            labels_list.append(label)
    
    return np.array(features_list), np.array(labels_list)

def main():
    print("Debugging SVM Data...")
    
    # 1. Load and split data
    print("Loading image files...")
    image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                         if f.endswith(('.tif', '.tiff', '.png'))])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) 
                        if f.endswith(('.png', '.tif', '.tiff'))])
    
    print(f"Total images: {len(image_files)}")
    
    # Split data
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_files, mask_files, train_size=TRAIN_SPLIT, random_state=42
    )
    
    print(f"Training images: {len(train_imgs)}")
    print(f"Test images: {len(test_imgs)}")
    
    # 2. Check mask statistics
    print("\nChecking mask statistics...")
    all_labels = []
    
    for mask_path in train_masks:
        mask = read_mask(mask_path)
        labels = mask.flatten()
        all_labels.extend(labels)
    
    label_counts = Counter(all_labels)
    print(f"Label distribution: {label_counts}")
    print(f"Background pixels: {label_counts[0]:,}")
    print(f"Dead tree pixels: {label_counts[1]:,}")
    print(f"Dead tree percentage: {label_counts[1]/(label_counts[0]+label_counts[1])*100:.2f}%")
    
    # 3. Check features for a few samples
    print("\nChecking features for first 3 training images...")
    
    for i in range(min(3, len(train_imgs))):
        image = read_image(train_imgs[i])
        mask = read_mask(train_masks[i])
        
        features, labels = extract_simple_features(image, mask)
        
        print(f"\nImage {i+1}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label distribution: {Counter(labels)}")
        print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Feature mean: {features.mean():.3f}")
        print(f"  Feature std: {features.std():.3f}")
        
        # Check if there are any dead tree pixels
        dead_tree_indices = np.where(labels == 1)[0]
        if len(dead_tree_indices) > 0:
            print(f"  Dead tree features (first 5):")
            for j in range(min(5, len(dead_tree_indices))):
                idx = dead_tree_indices[j]
                print(f"    {features[idx]}")
        else:
            print(f"  No dead tree pixels in this image!")

if __name__ == '__main__':
    main() 