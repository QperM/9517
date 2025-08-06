import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import joblib
from svm_pipeline import extract_features, compute_iou, compute_dice_coefficient, compute_pixel_accuracy, compute_classification_metrics
import matplotlib.pyplot as plt
import time

# Paths and constants
IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
MODEL_PATH = 'Baseline/SVM/svm_model.pkl'
RESULTS_PATH = 'Baseline/SVM/comprehensive_evaluation_results.txt'
TEST_IMGS_TXT = 'Baseline/CNN/test_imgs.txt'
IMG_SIZE = 256

# Load SVM model
print("Loading SVM model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
svm = joblib.load(MODEL_PATH)

# Load test image filenames from txt
print("Loading test image list from test_imgs.txt...")
test_imgs = []
if os.path.exists(TEST_IMGS_TXT):
    with open(TEST_IMGS_TXT, 'r') as f:
        test_imgs = [line.strip() for line in f.readlines()]
print(f"Number of test images in txt: {len(test_imgs)}")

def read_image(path):
    # Read and preprocess image as float32 normalized to [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image file not found or cannot be read: {path}")
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0
    else:
        raise ValueError(f"{path} is not a 3-channel image!")

def read_mask(path):
    # Read and preprocess mask as binary uint8
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found: {path}")
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask file could not be read: {path}")
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)

print("Loading test images and masks...")
# Construct full paths for images
image_files = [os.path.join(IMAGE_DIR, os.path.basename(p)) for p in test_imgs]

# Construct mask file paths, removing 'NRG_' prefix from basename for mask naming
mask_files = []
for img_name in test_imgs:
    base_name = os.path.basename(img_name)
    if base_name.startswith('NRG_'):
        base_name = base_name[len('NRG_'):]  # remove prefix
    mask_name = 'mask_' + base_name
    mask_path = os.path.join(MASK_DIR, mask_name)
    mask_files.append(mask_path)

print(f"Total test images: {len(image_files)}")
print(f"Total mask files: {len(mask_files)}")

# Read all images and masks into numpy arrays
X_test = np.stack([read_image(p) for p in image_files])
y_test = np.stack([read_mask(p) for p in mask_files])

print("Extracting features from test images...")
X_test_features = [extract_features(img) for img in X_test]
X_test_features = np.vstack(X_test_features)

print("Predicting with SVM model...")
y_pred_prob = svm.predict_proba(X_test_features)[:, 1]
y_pred_bin = (y_pred_prob > 0.5).astype(np.uint8)
y_pred = y_pred_prob.reshape(y_test.shape)
y_pred_bin = y_pred_bin.reshape(y_test.shape)

print("Computing evaluation metrics...")
iou_scores = [compute_iou(y_test[i], y_pred_bin[i]) for i in range(len(y_test))]
dice_scores = [compute_dice_coefficient(y_test[i], y_pred_bin[i]) for i in range(len(y_test))]
pixel_accuracies = [compute_pixel_accuracy(y_test[i], y_pred_bin[i]) for i in range(len(y_test))]

mean_iou = np.mean(iou_scores)
mean_dice = np.mean(dice_scores)
mean_pixel_acc = np.mean(pixel_accuracies)

# Classification metrics on flattened arrays
y_test_flat = y_test.flatten()
y_pred_flat = y_pred_bin.flatten()
class_metrics = compute_classification_metrics(y_test_flat, y_pred_flat)

def measure_inference_time(model, features_list, num_runs=3):
    # Measure average inference time per image on a subset of data
    sampled_data = features_list[:min(100, len(features_list))]
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        for features in sampled_data:
            _ = model.predict([features])
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times) / len(sampled_data)

print("Measuring inference time...")
inference_time = measure_inference_time(svm, X_test_features)

print("Saving evaluation results to file...")
with open(RESULTS_PATH, 'w') as f:
    f.write("SVM Segmentation Model - Comprehensive Evaluation Results\n")
    f.write("=" * 80 + "\n\n")
    f.write("1. BASIC INFORMATION\n")
    f.write("-" * 30 + "\n")
    f.write(f"Number of test images: {len(y_test)}\n")
    f.write(f"Image size: {y_test.shape[1]}x{y_test.shape[2]} pixels\n")
    f.write(f"Average inference time: {inference_time:.4f} seconds per image\n\n")

    f.write("2. SEGMENTATION METRICS\n")
    f.write("-" * 30 + "\n")
    f.write(f"Mean IoU: {mean_iou:.4f}\n")
    f.write(f"Mean Dice: {mean_dice:.4f}\n")
    f.write(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}\n\n")

    f.write("3. CLASSIFICATION METRICS\n")
    f.write("-" * 30 + "\n")
    f.write(f"Precision: {class_metrics['precision']:.4f}\n")
    f.write(f"Recall: {class_metrics['recall']:.4f}\n")
    f.write(f"F1-Score: {class_metrics['f1_score']:.4f}\n")
    f.write(f"TP: {class_metrics['tp']}\n")
    f.write(f"FP: {class_metrics['fp']}\n")
    f.write(f"TN: {class_metrics['tn']}\n")
    f.write(f"FN: {class_metrics['fn']}\n")

print("Generating visualizations...")

# Histogram and scatter plots (unchanged)
plt.figure(figsize=(10, 6))
plt.hist(iou_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.title('IoU Score Distribution')
plt.xlabel('IoU Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(dice_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.title('Dice Score Distribution')
plt.xlabel('Dice Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(pixel_accuracies, bins=20, alpha=0.7, color='orange', edgecolor='black')
plt.title('Pixel Accuracy Distribution')
plt.xlabel('Pixel Accuracy')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(iou_scores, dice_scores, alpha=0.7)
plt.xlabel('IoU Score')
plt.ylabel('Dice Score')
plt.title('IoU vs Dice Score Scatter')
plt.grid(True)
plt.show()

# Add detailed visualization of prediction results
num_viz = 5  # number of images to visualize
for i in range(num_viz):
    img = None
    if i < len(test_imgs):
        try:
            img_path = test_imgs[i]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = cv2.resize(img, (y_test.shape[2], y_test.shape[1]))
                img = img.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    plt.figure(figsize=(20, 4))

    # Original image
    plt.subplot(1, 5, 1)
    if img is not None:
        plt.imshow(img[..., ::-1])  # BGR to RGB
        plt.title('Original Image')
    else:
        plt.text(0.5, 0.5, 'Image not available', ha='center', va='center')
        plt.title('Original Image')
    plt.axis('off')

    # Ground truth mask
    plt.subplot(1, 5, 2)
    plt.imshow(y_test[i], cmap='gray')
    plt.title(f'Ground Truth\nIoU: {iou_scores[i]:.3f}')
    plt.axis('off')

    # Predicted mask (binary)
    plt.subplot(1, 5, 3)
    plt.imshow(y_pred_bin[i], cmap='gray')
    plt.title(f'Prediction (Binary)\nDice: {dice_scores[i]:.3f}')
    plt.axis('off')

    # Predicted mask (probability)
    plt.subplot(1, 5, 4)
    plt.imshow(y_pred[i], cmap='viridis')
    plt.colorbar()
    plt.title('Prediction (Probability)')
    plt.axis('off')

    # Error visualization
    plt.subplot(1, 5, 5)
    error_map = np.zeros_like(y_test[i], dtype=np.uint8)
    error_map[np.logical_and(y_test[i] == 1, y_pred_bin[i] == 0)] = 1  # False Negative (red)
    error_map[np.logical_and(y_test[i] == 0, y_pred_bin[i] == 1)] = 2  # False Positive (blue)
    error_map[np.logical_and(y_test[i] == 1, y_pred_bin[i] == 1)] = 3  # True Positive (green)

    colors = ['black', 'red', 'blue', 'green']
    error_img = np.zeros((*error_map.shape, 3))
    for j, color in enumerate(colors):
        mask = error_map == j
        if j == 0:  # black
            error_img[mask] = [0, 0, 0]
        elif j == 1:  # red
            error_img[mask] = [1, 0, 0]
        elif j == 2:  # blue
            error_img[mask] = [0, 0, 1]
        elif j == 3:  # green
            error_img[mask] = [0, 1, 0]

    plt.imshow(error_img)
    plt.title('Error Analysis\nRed=FN, Blue=FP, Green=TP')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

print("Evaluation and visualization completed.")
