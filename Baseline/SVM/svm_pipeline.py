import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import time

# Parameters
IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
SAVE_DIR = 'Baseline/SVM'
IMG_SIZE = 256
TRAIN_SPLIT = 0.8
FEATURE_SAMPLE_RATIO = 0.01  # Sample 10% of pixels for training
RANDOM_STATE = 42

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
    """Extract features from image for SVM"""
    features = []
    # RGB channels
    features.extend([img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()])
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features.append(gray.flatten())
    # HSV channels
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    features.extend([hsv[:, :, 0].flatten(), hsv[:, :, 1].flatten(), hsv[:, :, 2].flatten()])
    # Edge features using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    features.append(gradient_magnitude.flatten())
    # Texture features (std in 3x3 neighborhood)
    kernel_3x3 = np.ones((3, 3), np.float32) / 9
    mean_img = cv2.filter2D(gray, -1, kernel_3x3)
    mean_sq_img = cv2.filter2D(gray**2, -1, kernel_3x3)
    var_img = mean_sq_img - mean_img**2
    std_img = np.sqrt(np.maximum(var_img, 0))
    features.append(std_img.flatten())
    # Color statistics in local neighborhoods
    kernel_5x5 = np.ones((5, 5), np.float32) / 25
    for channel in range(3):
        mean_channel = cv2.filter2D(img[:, :, channel], -1, kernel_5x5)
        features.append(mean_channel.flatten())
        mean_sq_channel = cv2.filter2D(img[:, :, channel]**2, -1, kernel_5x5)
        var_channel = mean_sq_channel - mean_channel**2
        std_channel = np.sqrt(np.maximum(var_channel, 0))
        features.append(std_channel.flatten())
    feature_matrix = np.column_stack(features)
    return feature_matrix

def sample_pixels(features, labels, sample_ratio=0.1):
    n_pixels = features.shape[0]
    n_samples = int(n_pixels * sample_ratio)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n_pixels)
    sampled_indices, _ = train_test_split(
        indices,
        train_size=sample_ratio,
        stratify=labels,
        random_state=RANDOM_STATE
    )
    return features[sampled_indices], labels[sampled_indices]

def compute_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_dice_coefficient(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2 * intersection) / union

def compute_pixel_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compute_classification_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1]).ravel()
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
    print("Starting SVM Forest Segmentation Pipeline...")
    # 1. Load and split data
    print("Loading image files...")
    image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                         if f.endswith(('.tif', '.tiff', '.png'))])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR)
                        if f.endswith(('.png', '.tif', '.tiff'))])
    print(f"Found {len(image_files)} images and {len(mask_files)} masks.")
    assert len(image_files) == len(mask_files), 'Number of images and masks do not match!'
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_files, mask_files, train_size=TRAIN_SPLIT, random_state=RANDOM_STATE
    )
    with open(os.path.join(SAVE_DIR, 'test_imgs.txt'), 'w') as f:
        for p in test_imgs:
            f.write(p + '\n')
    print(f"Training set: {len(train_imgs)} images, Test set: {len(test_imgs)} images.")

    # 2. Load and process training data
    print('Loading and processing training data...')
    X_train_features = []
    y_train_labels = []
    for img_path, mask_path in tqdm(zip(train_imgs, train_masks), total=len(train_imgs), desc="Processing Training Data"):
        img = read_image(img_path)
        mask = read_mask(mask_path)
        features = extract_features(img)
        labels = mask.flatten()
        features, labels = sample_pixels(features, labels, sample_ratio=FEATURE_SAMPLE_RATIO)
        X_train_features.append(features)
        y_train_labels.append(labels)
    X_train = np.vstack(X_train_features)
    y_train = np.hstack(y_train_labels)
    print(f"Final X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Remove PCA and use all 15 features
    print("Using all 15 features without PCA")

    # 3. Train SVM model
    print('Training SVM model with LinearSVC and probability calibration...')
    base_svm = LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=10000)
    svm = CalibratedClassifierCV(base_svm)  # Add probability support
    start_time = time.time()

    # Add logging to track training progress
    print(f"Starting SVM training with {X_train.shape[0]} samples and {X_train.shape[1]} features...")
    try:
        svm.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f'Training completed in {train_time:.2f} seconds.')
    except Exception as e:
        print(f"Error during training: {e}")
        return

    joblib.dump(svm, os.path.join(SAVE_DIR, 'svm_model.pkl'))
    print('Model saved!')

    # 4. Load and process test data
    print('Loading and processing test data...')
    X_test_features = []
    y_test_labels = []
    for img_path, mask_path in tqdm(zip(test_imgs, test_masks), total=len(test_imgs), desc="Processing Test Data"):
        img = read_image(img_path)
        mask = read_mask(mask_path)
        features = extract_features(img)
        X_test_features.append(features)
        y_test_labels.append(mask.flatten())
    X_test = np.vstack(X_test_features)
    y_test = np.hstack(y_test_labels)
    print(f"Final X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 5. Predict and evaluate
    print('Predicting on test data...')
    start_time = time.time()
    try:
        y_pred_prob = svm.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob > 0.5).astype(np.uint8)
        inference_time = (time.time() - start_time) / len(X_test)
        print(f'Average inference time per pixel: {inference_time:.6f} seconds')
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # 6. Metrics
    try:
        iou = compute_iou(y_test, y_pred)
        dice = compute_dice_coefficient(y_test, y_pred)
        pixel_acc = compute_pixel_accuracy(y_test, y_pred)
        class_metrics = compute_classification_metrics(y_test, y_pred)
        print(f'IoU: {iou:.4f}')
        print(f'Dice: {dice:.4f}')
        print(f'Pixel Accuracy: {pixel_acc:.4f}')
        print(f'Precision: {class_metrics["precision"]:.4f}')
        print(f'Recall: {class_metrics["recall"]:.4f}')
        print(f'F1-Score: {class_metrics["f1_score"]:.4f}')
    except Exception as e:
        print(f"Error during metrics computation: {e}")
        return

    # 7. Save results
    try:
        with open(os.path.join(SAVE_DIR, 'training_results.txt'), 'w') as f:
            f.write(f'=== SVM Training Results ===\n')
            f.write(f'Training samples: {X_train.shape[0]}\n')
            f.write(f'Training time: {train_time:.2f} seconds\n')
            f.write(f'Average inference time per pixel: {inference_time:.6f} seconds\n')
            f.write(f'IoU: {iou:.4f}\n')
            f.write(f'Dice: {dice:.4f}\n')
            f.write(f'Pixel Accuracy: {pixel_acc:.4f}\n')
            f.write(f'Precision: {class_metrics["precision"]:.4f}\n')
            f.write(f'Recall: {class_metrics["recall"]:.4f}\n')
            f.write(f'F1-Score: {class_metrics["f1_score"]:.4f}\n')
            f.write(f'True Positives: {class_metrics["tp"]:,}\n')
            f.write(f'False Positives: {class_metrics["fp"]:,}\n')
            f.write(f'True Negatives: {class_metrics["tn"]:,}\n')
            f.write(f'False Negatives: {class_metrics["fn"]:,}\n')
        print('Results saved!')
    except Exception as e:
        print(f"Error during saving results: {e}")
        return

if __name__ == '__main__':
    main()