import os
import numpy as np
import cv2
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import time

# Parameters
IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
SAVE_DIR = 'Baseline/base_model_xgb'
IMG_SIZE = 256
TRAIN_SPLIT = 0.8
FEATURE_SAMPLE_RATIO = 0.1  # Sample 10% of pixels for training due to memory constraints
RANDOM_STATE = 42

# XGBoost parameters
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

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

def extract_features(img):
    """Extract features from image for XGBoost"""
    # Basic pixel features
    features = []
    
    # RGB channels
    features.extend([img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()])
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features.append(gray.flatten())
    
    # HSV channels
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    features.extend([hsv[:, :, 0].flatten(), hsv[:, :, 1].flatten(), hsv[:, :, 2].flatten()])
    
    # Local binary pattern-like features (simplified)
    # Edge features using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    features.append(gradient_magnitude.flatten())
    
    # Texture features (simplified)
    # Standard deviation in 3x3 neighborhood using OpenCV
    kernel_3x3 = np.ones((3, 3), np.float32) / 9
    mean_img = cv2.filter2D(gray, -1, kernel_3x3)
    mean_sq_img = cv2.filter2D(gray**2, -1, kernel_3x3)
    var_img = mean_sq_img - mean_img**2
    std_img = np.sqrt(np.maximum(var_img, 0))
    features.append(std_img.flatten())
    
    # Color statistics in local neighborhoods
    kernel_5x5 = np.ones((5, 5), np.float32) / 25
    for channel in range(3):
        # Mean in 5x5 neighborhood
        mean_channel = cv2.filter2D(img[:, :, channel], -1, kernel_5x5)
        features.append(mean_channel.flatten())
        
        # Standard deviation in 5x5 neighborhood
        mean_sq_channel = cv2.filter2D(img[:, :, channel]**2, -1, kernel_5x5)
        var_channel = mean_sq_channel - mean_channel**2
        std_channel = np.sqrt(np.maximum(var_channel, 0))
        features.append(std_channel.flatten())
    
    # Stack all features
    feature_matrix = np.column_stack(features)
    return feature_matrix

def sample_pixels(features, labels, sample_ratio=0.1):
    """Sample pixels to reduce memory usage"""
    n_pixels = features.shape[0]
    n_samples = int(n_pixels * sample_ratio)
    
    # Stratified sampling to maintain class balance
    from sklearn.model_selection import train_test_split
    indices = np.arange(n_pixels)
    
    # Sample with stratification
    sampled_indices, _ = train_test_split(
        indices, 
        train_size=sample_ratio, 
        stratify=labels, 
        random_state=RANDOM_STATE
    )
    
    return features[sampled_indices], labels[sampled_indices]

# Evaluation metrics
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

def main():
    print("Starting XGBoost Forest Segmentation Pipeline...")
    
    # 1. Load and split data
    print("Loading image files...")
    image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                         if f.endswith(('.tif', '.tiff', '.png'))])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) 
                        if f.endswith(('.png', '.tif', '.tiff'))])
    
    assert len(image_files) == len(mask_files), 'Number of images and masks do not match!'
    
    # Split data
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_files, mask_files, train_size=TRAIN_SPLIT, random_state=RANDOM_STATE
    )
    
    # Save test image paths
    with open(os.path.join(SAVE_DIR, 'test_imgs.txt'), 'w') as f:
        for p in test_imgs:
            f.write(p + '\n')
    
    # 2. Load and process training data
    print('Loading and processing training data...')
    X_train_features = []
    y_train_labels = []
    
    for img_path, mask_path in tqdm(zip(train_imgs, train_masks), total=len(train_imgs)):
        # Load image and mask
        img = read_image(img_path)
        mask = read_mask(mask_path)
        
        # Extract features
        features = extract_features(img)
        labels = mask.flatten()
        
        # Sample pixels to reduce memory usage
        sampled_features, sampled_labels = sample_pixels(features, labels, FEATURE_SAMPLE_RATIO)
        
        X_train_features.append(sampled_features)
        y_train_labels.append(sampled_labels)
    
    # Combine all training data
    X_train = np.vstack(X_train_features)
    y_train = np.concatenate(y_train_labels)
    
    print(f'Training data shape: {X_train.shape}, {y_train.shape}')
    print(f'Class distribution: {np.bincount(y_train)}')
    
    # 3. Load test data (for later evaluation)
    print('Loading test data...')
    X_test_imgs = []
    y_test_masks = []
    
    for img_path, mask_path in tqdm(zip(test_imgs, test_masks), total=len(test_imgs)):
        img = read_image(img_path)
        mask = read_mask(mask_path)
        X_test_imgs.append(img)
        y_test_masks.append(mask)
    
    X_test_imgs = np.array(X_test_imgs)
    y_test_masks = np.array(y_test_masks)
    
    # Note: Not saving y_test.npy since feature extraction is fast
    
    print(f'Test data shape: {X_test_imgs.shape}, {y_test_masks.shape}')
    
    # 4. Train XGBoost model
    print('Training XGBoost model...')
    start_time = time.time()
    
    # Create and train model
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    # 5. Save model
    model_path = os.path.join(SAVE_DIR, 'xgb_model.pkl')
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')
    
    # 6. Generate predictions for test set
    print('Generating predictions for test set...')
    predictions = []
    
    for img in tqdm(X_test_imgs):
        # Extract features
        features = extract_features(img)
        
        # Predict
        pred_proba = model.predict_proba(features)[:, 1]  # Probability of class 1
        
        # Reshape back to image dimensions
        pred_img = pred_proba.reshape(IMG_SIZE, IMG_SIZE)
        predictions.append(pred_img)
    
    y_pred = np.array(predictions)
    # Note: Not saving y_pred.npy since feature extraction is fast
    
    # 7. Evaluate model
    print('Evaluating model...')
    y_pred_binary = (y_pred > 0.5).astype(np.uint8)
    
    # Calculate metrics
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    
    for i in range(len(y_test_masks)):
        iou = compute_iou(y_test_masks[i], y_pred_binary[i])
        dice = compute_dice_coefficient(y_test_masks[i], y_pred_binary[i])
        pixel_acc = compute_pixel_accuracy(y_test_masks[i], y_pred_binary[i])
        
        iou_scores.append(iou)
        dice_scores.append(dice)
        pixel_accuracies.append(pixel_acc)
    
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)
    
    # Calculate classification metrics
    y_test_all = y_test_masks.flatten()
    y_pred_all = y_pred_binary.flatten()
    class_metrics = compute_classification_metrics(y_test_all, y_pred_all)
    
    print(f'Mean IoU: {mean_iou:.4f}')
    print(f'Mean Dice: {mean_dice:.4f}')
    print(f'Mean Pixel Accuracy: {mean_pixel_acc:.4f}')
    print(f'Precision: {class_metrics["precision"]:.4f}')
    print(f'Recall: {class_metrics["recall"]:.4f}')
    print(f'F1-Score: {class_metrics["f1_score"]:.4f}')
    
    # 8. Save results
    with open(os.path.join(SAVE_DIR, 'training_results.txt'), 'w') as f:
        f.write(f'=== XGBoost Training Results ===\n')
        f.write(f'Training samples: {len(X_train):,}\n')
        f.write(f'Feature sample ratio: {FEATURE_SAMPLE_RATIO}\n')
        f.write(f'Training time: {training_time:.2f} seconds\n')
        f.write(f'Model parameters: {XGB_PARAMS}\n\n')
        
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
    
    # 9. Plot training results
    plt.figure(figsize=(12, 8))
    
    # IoU distribution
    plt.subplot(2, 3, 1)
    plt.hist(iou_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_iou, color='red', linestyle='--', label=f'Mean: {mean_iou:.3f}')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.title('IoU Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dice distribution
    plt.subplot(2, 3, 2)
    plt.hist(dice_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(mean_dice, color='red', linestyle='--', label=f'Mean: {mean_dice:.3f}')
    plt.xlabel('Dice Score')
    plt.ylabel('Frequency')
    plt.title('Dice Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature importance
    plt.subplot(2, 3, 3)
    feature_names = ['R', 'G', 'B', 'Gray', 'H', 'S', 'V', 'Gradient', 'Std_3x3',
                    'R_mean_5x5', 'R_std_5x5', 'G_mean_5x5', 'G_std_5x5', 'B_mean_5x5', 'B_std_5x5']
    importance = model.feature_importances_
    top_features = 10
    top_indices = np.argsort(importance)[-top_features:]
    
    plt.barh(range(top_features), importance[top_indices])
    plt.yticks(range(top_features), [feature_names[i] for i in top_indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances')
    
    # Metrics comparison
    plt.subplot(2, 3, 4)
    metrics = ['IoU', 'Dice', 'Pixel Acc']
    values = [mean_iou, mean_dice, mean_pixel_acc]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Score')
    plt.title('Average Metrics Comparison')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # Performance categories
    plt.subplot(2, 3, 5)
    excellent = sum(1 for iou in iou_scores if iou >= 0.8)
    good = sum(1 for iou in iou_scores if 0.6 <= iou < 0.8)
    fair = sum(1 for iou in iou_scores if 0.4 <= iou < 0.6)
    poor = sum(1 for iou in iou_scores if iou < 0.4)
    
    sizes = [excellent, good, fair, poor]
    labels = [f'Excellent\n(≥0.8)', f'Good\n(0.6-0.8)', f'Fair\n(0.4-0.6)', f'Poor\n(<0.4)']
    colors = ['green', 'lightgreen', 'orange', 'red']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Performance Distribution')
    
    # Sample predictions
    plt.subplot(2, 3, 6)
    sample_idx = 0
    plt.imshow(y_test_masks[sample_idx], cmap='gray')
    plt.title(f'Sample Ground Truth\nIoU: {iou_scores[sample_idx]:.3f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print('Training completed!')
    print(f'Results saved to: {SAVE_DIR}')

if __name__ == '__main__':
    main() 