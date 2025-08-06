import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import joblib
from svm_pipeline import extract_features, compute_iou, compute_dice_coefficient, compute_pixel_accuracy, compute_classification_metrics
import matplotlib.pyplot as plt
import time

# Paths
IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
MODEL_PATH = 'Baseline/SVM/svm_model.pkl'
RESULTS_PATH = 'Baseline/SVM/comprehensive_evaluation_results.txt'
IMG_SIZE = 256

# Load model
print("Loading SVM model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
svm = joblib.load(MODEL_PATH)

# Load test data
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[-1] == 3:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0
    else:
        raise ValueError(f"{path} is not a 3-channel image!")

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)

print("Loading test data...")
image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('.tif', '.tiff', '.png'))])
mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith(('.png', '.tif', '.tiff'))])

X_test = np.stack([read_image(p) for p in image_files])
y_test = np.stack([read_mask(p) for p in mask_files])

# Extract features
print("Extracting features...")
X_test_features = [extract_features(img) for img in X_test]
X_test_features = np.vstack(X_test_features)

# Predict
print("Making predictions...")
y_pred_prob = svm.predict_proba(X_test_features)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(np.uint8)

# Reshape predictions to match image dimensions
y_pred = y_pred.reshape(y_test.shape)

# Compute metrics
print("Computing metrics...")
iou_scores = [compute_iou(y_test[i], y_pred[i]) for i in range(len(y_test))]
dice_scores = [compute_dice_coefficient(y_test[i], y_pred[i]) for i in range(len(y_test))]
pixel_accuracies = [compute_pixel_accuracy(y_test[i], y_pred[i]) for i in range(len(y_test))]

mean_iou = np.mean(iou_scores)
mean_dice = np.mean(dice_scores)
mean_pixel_acc = np.mean(pixel_accuracies)

# Classification metrics
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()
class_metrics = compute_classification_metrics(y_test_flat, y_pred_flat)

def measure_inference_time(model, test_data, num_runs=3):
    """Measure average inference time per image with reduced runs and sampling"""
    sampled_data = test_data[:min(100, len(test_data))]  # Limit to 100 samples for timing
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        for features in sampled_data:
            _ = model.predict([features])
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times) / len(sampled_data)

def count_model_parameters(model):
    """Count total number of parameters"""
    if hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'coef_'):
        base_model = model.base_estimator
        return sum(p.size for p in base_model.coef_) + len(base_model.intercept_)
    else:
        print("Warning: The model does not have accessible parameters. Skipping parameter count.")
        return 0

# Measure inference time
print("Measuring inference time...")
inference_time = measure_inference_time(svm, X_test_features)

# Count model parameters
num_params = count_model_parameters(svm)

# Save comprehensive results
print("Saving comprehensive evaluation results...")
with open(RESULTS_PATH, 'w') as f:
    f.write("SVM Segmentation Model - Comprehensive Evaluation Results\n")
    f.write("=" * 80 + "\n\n")

    # Basic Information
    f.write("1. BASIC INFORMATION\n")
    f.write("-" * 30 + "\n")
    f.write(f"Number of test images: {len(y_test)}\n")
    f.write(f"Image size: {y_test.shape[1]}x{y_test.shape[2]} pixels\n")
    f.write(f"Model parameters: {num_params:,}\n")
    f.write(f"Average inference time: {inference_time:.4f} seconds per image\n\n")

    # Segmentation Metrics
    f.write("2. SEGMENTATION METRICS\n")
    f.write("-" * 30 + "\n")
    f.write(f"IoU (Intersection over Union):\n")
    f.write(f"  Mean IoU: {mean_iou:.4f}\n")
    f.write(f"  Min IoU: {np.min(iou_scores):.4f}\n")
    f.write(f"  Max IoU: {np.max(iou_scores):.4f}\n\n")

    f.write(f"Dice Coefficient (F1 Score):\n")
    f.write(f"  Mean Dice: {mean_dice:.4f}\n")
    f.write(f"  Min Dice: {np.min(dice_scores):.4f}\n")
    f.write(f"  Max Dice: {np.max(dice_scores):.4f}\n\n")

    f.write(f"Pixel Accuracy:\n")
    f.write(f"  Mean Pixel Accuracy: {mean_pixel_acc:.4f}\n")
    f.write(f"  Min Pixel Accuracy: {np.min(pixel_accuracies):.4f}\n")
    f.write(f"  Max Pixel Accuracy: {np.max(pixel_accuracies):.4f}\n\n")

    # Classification Metrics
    f.write("3. CLASSIFICATION METRICS\n")
    f.write("-" * 30 + "\n")
    f.write(f"Precision: {class_metrics['precision']:.4f}\n")
    f.write(f"Recall: {class_metrics['recall']:.4f}\n")
    f.write(f"F1-Score: {class_metrics['f1_score']:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(f"  True Positives (TP): {class_metrics['tp']}\n")
    f.write(f"  False Positives (FP): {class_metrics['fp']}\n")
    f.write(f"  True Negatives (TN): {class_metrics['tn']}\n")
    f.write(f"  False Negatives (FN): {class_metrics['fn']}\n\n")

# Generate visualizations
print("Generating visualizations...")
plt.figure(figsize=(10, 6))
plt.hist(iou_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.title('IoU Score Distribution')
plt.xlabel('IoU Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('Baseline/SVM/iou_distribution.png')
plt.close()

print("Evaluation completed and visualizations saved.")