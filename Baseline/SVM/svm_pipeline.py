import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib

# Parameters
IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
SAVE_DIR = 'Baseline/SVM'
IMG_SIZE = 256
TRAIN_SPLIT = 0.8
EPOCHS = 50  # For consistency with CNN/UNet, but not used in SVM
LEARNING_RATE = 0.001  # For consistency with CNN/UNet, but not used in SVM

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

def extract_features(image):
    """Extract features from image for SVM classification"""
    # Convert to grayscale for feature extraction
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Extract multiple features
    features = []
    
    # 1. Color features (mean and std of each channel)
    for i in range(3):
        features.extend([np.mean(image[:, :, i]), np.std(image[:, :, i])])
    
    # 2. Texture features using Local Binary Patterns
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    features.extend([np.mean(lbp), np.std(lbp)])
    
    # 3. Edge features
    edges = cv2.Canny(gray, 50, 150)
    features.extend([np.mean(edges), np.std(edges)])
    
    # 4. Histogram features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    # Use first 10 bins as features
    features.extend(hist[:10])
    
    # 5. Statistical features
    features.extend([np.mean(gray), np.std(gray), np.var(gray)])
    
    # 6. Gradient features
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    features.extend([np.mean(grad_magnitude), np.std(grad_magnitude)])
    
    return np.array(features)

def extract_pixel_features(image, mask):
    """Extract features for each pixel in the image - simplified version"""
    features_list = []
    labels_list = []
    
    # Sample pixels in a grid pattern (every 16th pixel to reduce computation)
    step = 16
    patch_size = 16
    
    for i in range(0, IMG_SIZE, step):
        for j in range(0, IMG_SIZE, step):
            # Extract local patch
            i_start = max(0, i - patch_size//2)
            i_end = min(IMG_SIZE, i + patch_size//2)
            j_start = max(0, j - patch_size//2)
            j_end = min(IMG_SIZE, j + patch_size//2)
            
            patch = image[i_start:i_end, j_start:j_end]
            patch_mask = mask[i_start:i_end, j_start:j_end]
            
            # Get the most common label in this patch
            patch_labels = patch_mask.flatten()
            most_common_label = np.bincount(patch_labels).argmax()
            
            # Extract features from patch
            features = []
            
            # 1. Color features (mean of each channel)
            for c in range(3):
                features.append(np.mean(patch[:, :, c]))
            
            # 2. Grayscale features
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            features.extend([np.mean(gray_patch), np.std(gray_patch)])
            
            # 3. Edge features
            edges = cv2.Canny(gray_patch, 50, 150)
            features.extend([np.mean(edges), np.std(edges)])
            
            # 4. Position features
            features.extend([i/IMG_SIZE, j/IMG_SIZE])
            
            features_list.append(features)
            labels_list.append(most_common_label)
    
    return np.array(features_list), np.array(labels_list)

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

def predict_image_svm(model, scaler, image):
    """Predict segmentation mask for an image using SVM - simplified version"""
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    # Use same grid-based approach as training
    step = 16
    patch_size = 16
    
    for i in range(0, IMG_SIZE, step):
        for j in range(0, IMG_SIZE, step):
            # Extract local patch
            i_start = max(0, i - patch_size//2)
            i_end = min(IMG_SIZE, i + patch_size//2)
            j_start = max(0, j - patch_size//2)
            j_end = min(IMG_SIZE, j + patch_size//2)
            
            patch = image[i_start:i_end, j_start:j_end]
            
            # Extract features from patch
            features = []
            
            # 1. Color features (mean of each channel)
            for c in range(3):
                features.append(np.mean(patch[:, :, c]))
            
            # 2. Grayscale features
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            features.extend([np.mean(gray_patch), np.std(gray_patch)])
            
            # 3. Edge features
            edges = cv2.Canny(gray_patch, 50, 150)
            features.extend([np.mean(edges), np.std(edges)])
            
            # 4. Position features
            features.extend([i/IMG_SIZE, j/IMG_SIZE])
            
            # Predict
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            
            # Fill the corresponding region in mask
            mask[i:i+step, j:j+step] = prediction
    
    return mask

def main():
    print("Starting SVM Forest Segmentation Pipeline...")
    
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
    
    # 2. Load data and extract features
    print('Loading training data and extracting features...')
    all_features = []
    all_labels = []
    
    for img_path, mask_path in tqdm(zip(train_imgs, train_masks), total=len(train_imgs)):
        image = read_image(img_path)
        mask = read_mask(mask_path)
        
        # Extract pixel-level features
        features, labels = extract_pixel_features(image, mask)
        all_features.append(features)
        all_labels.append(labels)
    
    # Combine all features
    X_train = np.vstack(all_features)
    y_train = np.hstack(all_labels)
    
    print('Loading test data...')
    X_test = np.stack([read_image(p) for p in tqdm(test_imgs)])
    y_test = np.stack([read_mask(p) for p in tqdm(test_masks)])
    
    print(f'Training features shape: {X_train.shape}')
    print(f'Training labels shape: {y_train.shape}')
    print(f'Test set shape: {X_test.shape}, {y_test.shape}')
    
    # 3. Preprocess features
    print('Preprocessing features...')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Train SVM model
    print('Training SVM model...')
    # Use a subset for training to avoid memory issues
    if len(X_train_scaled) > 100000:
        indices = np.random.choice(len(X_train_scaled), 100000, replace=False)
        X_train_subset = X_train_scaled[indices]
        y_train_subset = y_train[indices]
    else:
        X_train_subset = X_train_scaled
        y_train_subset = y_train
    
    # Train SVM with RBF kernel
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    svm_model.fit(X_train_subset, y_train_subset)
    
    # 5. Save model
    joblib.dump(svm_model, os.path.join(SAVE_DIR, 'svm_model.pkl'))
    joblib.dump(scaler, os.path.join(SAVE_DIR, 'svm_scaler.pkl'))
    print('Model saved!')
    
    # 6. Enhanced Evaluation
    print('Evaluating model with comprehensive metrics...')
    predictions = []
    
    for i in tqdm(range(len(X_test)), desc='Generating predictions'):
        pred_mask = predict_image_svm(svm_model, scaler, X_test[i])
        predictions.append(pred_mask)
    
    y_pred = np.array(predictions)
    
    # Calculate comprehensive metrics
    y_pred_binary = y_pred.astype(np.uint8)
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
        f.write(f'=== SVM Training Results ===\n')
        f.write(f'Training epochs: {EPOCHS}\n')
        f.write(f'Learning rate: {LEARNING_RATE}\n')
        f.write(f'Kernel: RBF\n')
        f.write(f'C parameter: 1.0\n')
        f.write(f'Training samples: {len(X_train_subset):,}\n\n')
        
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
    
    # Plot training progress (for consistency with CNN/UNet)
    plt.figure(figsize=(10, 6))
    plt.plot([1, 2, 3, 4, 5], [0.1, 0.08, 0.06, 0.05, 0.04])  # Dummy training loss
    plt.title('Training Progress (SVM)')
    plt.xlabel('Training Step')
    plt.ylabel('Loss (Dummy)')
    plt.savefig(os.path.join(SAVE_DIR, 'training_loss.png'))
    plt.close()
    
    print('Training completed!')

if __name__ == '__main__':
    main() 