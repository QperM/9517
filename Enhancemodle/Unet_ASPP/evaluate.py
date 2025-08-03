import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import time
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from unet_aspp_pipeline import UNetASPP, compute_iou

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

def measure_inference_time(model, test_data, device, num_runs=10):
    """Measure average inference time per image"""
    model.eval()
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            for i in range(len(test_data)):
                img = torch.tensor(test_data[i:i+1].transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
                _ = model(img)
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time_per_image = np.mean(times) / len(test_data)
    return avg_time_per_image

def count_model_parameters(model):
    """Count total number of parameters"""
    return sum(p.numel() for p in model.parameters())

def main():
    """Evaluate UNet+ASPP model performance with comprehensive metrics"""
    print("Evaluating UNet+ASPP Forest Segmentation Model...")
    
    # Load test data and predictions - regenerate since not saved
    print("Regenerating test data and predictions...")
    
    # Load and split data again
    image_files = sorted([os.path.join('USA_segmentation/NRG_images', f) for f in os.listdir('USA_segmentation/NRG_images') 
                         if f.endswith(('.tif', '.tiff', '.png'))])
    mask_files = sorted([os.path.join('USA_segmentation/masks', f) for f in os.listdir('USA_segmentation/masks') 
                        if f.endswith(('.png', '.tif', '.tiff'))])
    
    # Split data with same random state
    from sklearn.model_selection import train_test_split
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_files, mask_files, train_size=0.8, random_state=42
    )
    
    # Load test data
    def read_image(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[-1] == 3:
            img = cv2.resize(img, (256, 256))
            return img.astype(np.float32) / 255.0
        else:
            raise ValueError(f"{path} is not a 3-channel image!")

    def read_mask(path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8)
    
    print('Loading test data...')
    X_test = np.stack([read_image(p) for p in test_imgs])
    y_test = np.stack([read_mask(p) for p in test_masks])
    
    # Load model and generate predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetASPP(n_channels=3, n_classes=1, bilinear=False).to(device)
    
    # Load model weights
    model_path = 'Enhancemodle/Unet_ASPP/unet_aspp_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded")
    else:
        print("Warning: Model weights not found. Please run training first.")
        return
    
    # Generate predictions
    print('Generating predictions...')
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            img = torch.tensor(X_test[i:i+1].transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
            pred = model(img)
            pred = pred.cpu().numpy()[0, 0]  # Remove batch and channel dimensions
            predictions.append(pred)
    
    y_pred = np.array(predictions)
    
    # Convert predictions to binary
    y_pred_bin = (y_pred > 0.5).astype(np.uint8)
    
    # 1. Basic Segmentation Metrics
    print("Computing segmentation metrics...")
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    
    for i in range(len(y_test)):
        iou = compute_iou(y_test[i], y_pred_bin[i])
        dice = compute_dice_coefficient(y_test[i], y_pred_bin[i])
        pixel_acc = compute_pixel_accuracy(y_test[i], y_pred_bin[i])
        
        iou_scores.append(iou)
        dice_scores.append(dice)
        pixel_accuracies.append(pixel_acc)
    
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)
    mean_dice = np.mean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)
    
    # 2. Classification Metrics
    print("Computing classification metrics...")
    # Aggregate all images for overall metrics
    y_test_all = y_test.flatten()
    y_pred_all = y_pred_bin.flatten()
    
    class_metrics = compute_classification_metrics(y_test_all, y_pred_all)
    
    # 3. Efficiency Metrics
    print("Computing efficiency metrics...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetASPP(n_channels=3, n_classes=1, bilinear=False).to(device)
    
    # Load model weights if available
    model_path = 'Enhancemodle/Unet_ASPP/unet_aspp_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded for inference time measurement")
    
    # Measure inference time
    try:
        inference_time = measure_inference_time(model, y_test, device)
    except Exception as e:
        print(f"Could not measure inference time: {e}")
        inference_time = None
    
    # Count parameters
    num_params = count_model_parameters(model)
    
    # 4. Save comprehensive results
    print("Saving comprehensive evaluation results...")
    with open('Enhancemodle/Unet_ASPP/comprehensive_evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write("UNet+ASPP Forest Segmentation Model - Comprehensive Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic Information
        f.write("1. BASIC INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of test images: {len(y_test)}\n")
        f.write(f"Image size: {y_test.shape[1]}x{y_test.shape[2]} pixels\n")
        f.write(f"Model parameters: {num_params:,}\n")
        if inference_time:
            f.write(f"Average inference time: {inference_time:.4f} seconds per image\n")
        f.write("\n")
        
        # Segmentation Metrics
        f.write("2. SEGMENTATION METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"IoU (Intersection over Union):\n")
        f.write(f"  Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}\n")
        f.write(f"  Min IoU: {np.min(iou_scores):.4f}\n")
        f.write(f"  Max IoU: {np.max(iou_scores):.4f}\n")
        f.write(f"  Median IoU: {np.median(iou_scores):.4f}\n\n")
        
        f.write(f"Dice Coefficient (F1 Score):\n")
        f.write(f"  Mean Dice: {mean_dice:.4f}\n")
        f.write(f"  Min Dice: {np.min(dice_scores):.4f}\n")
        f.write(f"  Max Dice: {np.max(dice_scores):.4f}\n\n")
        
        f.write(f"Pixel Accuracy:\n")
        f.write(f"  Mean Pixel Accuracy: {mean_pixel_acc:.4f}\n")
        f.write(f"  Min Pixel Accuracy: {np.min(pixel_accuracies):.4f}\n")
        f.write(f"  Max Pixel Accuracy: {np.max(pixel_accuracies):.4f}\n\n")
        
        # Classification Metrics
        f.write("3. CLASSIFICATION METRICS (Dead Tree Class)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precision: {class_metrics['precision']:.4f}\n")
        f.write(f"Recall: {class_metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {class_metrics['f1_score']:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"  True Positives (TP): {class_metrics['tp']:,}\n")
        f.write(f"  False Positives (FP): {class_metrics['fp']:,}\n")
        f.write(f"  True Negatives (TN): {class_metrics['tn']:,}\n")
        f.write(f"  False Negatives (FN): {class_metrics['fn']:,}\n\n")
        
        # Performance Categories
        f.write("4. PERFORMANCE CATEGORIES\n")
        f.write("-" * 30 + "\n")
        excellent = sum(1 for iou in iou_scores if iou >= 0.8)
        good = sum(1 for iou in iou_scores if 0.6 <= iou < 0.8)
        fair = sum(1 for iou in iou_scores if 0.4 <= iou < 0.6)
        poor = sum(1 for iou in iou_scores if iou < 0.4)
        
        f.write(f"Excellent (IoU ≥ 0.8): {excellent} images ({excellent/len(y_test)*100:.1f}%)\n")
        f.write(f"Good (0.6 ≤ IoU < 0.8): {good} images ({good/len(y_test)*100:.1f}%)\n")
        f.write(f"Fair (0.4 ≤ IoU < 0.6): {fair} images ({fair/len(y_test)*100:.1f}%)\n")
        f.write(f"Poor (IoU < 0.4): {poor} images ({poor/len(y_test)*100:.1f}%)\n\n")
        
        # Individual Results
        f.write("5. INDIVIDUAL IMAGE RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write("Image\tIoU\t\tDice\t\tPixel_Acc\n")
        f.write("-" * 50 + "\n")
        for i in range(len(y_test)):
            f.write(f"{i+1}\t{iou_scores[i]:.4f}\t\t{dice_scores[i]:.4f}\t\t{pixel_accuracies[i]:.4f}\n")
        f.write("\n")
        
        # Efficiency Analysis
        f.write("6. EFFICIENCY ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model Parameters: {num_params:,}\n")
        if inference_time:
            f.write(f"Inference Time: {inference_time:.4f} seconds per image\n")
            f.write(f"Real-time Performance: {'✓' if inference_time < 2.0 else '✗'} (< 2s requirement)\n")
        f.write("\n")
        
        # Failure Analysis
        f.write("7. FAILURE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        worst_indices = np.argsort(iou_scores)[:5]
        f.write("Worst performing images (lowest IoU):\n")
        for i, idx in enumerate(worst_indices):
            f.write(f"  {i+1}. Image {idx+1}: IoU = {iou_scores[idx]:.4f}\n")
        f.write("\n")
        
        # Summary
        f.write("8. SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall Performance: {'Excellent' if mean_iou >= 0.8 else 'Good' if mean_iou >= 0.6 else 'Fair' if mean_iou >= 0.4 else 'Poor'}\n")
        f.write(f"Primary Metric (Mean IoU): {mean_iou:.4f}\n")
        f.write(f"Balanced Metric (F1-Score): {class_metrics['f1_score']:.4f}\n")
        if inference_time:
            f.write(f"Efficiency: {'Satisfactory' if inference_time < 2.0 else 'Needs Improvement'}\n")
    
    print(f"Comprehensive evaluation completed!")
    print(f"Results saved to: Enhancemodle/Unet_ASPP/comprehensive_evaluation_results.txt")
    
    # Print summary to console
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Precision: {class_metrics['precision']:.4f}")
    print(f"Recall: {class_metrics['recall']:.4f}")
    print(f"F1-Score: {class_metrics['f1_score']:.4f}")
    print(f"Model Parameters: {num_params:,}")
    if inference_time:
        print(f"Inference Time: {inference_time:.4f}s per image")
    
    # 5. Enhanced Visualizations
    print("Generating enhanced visualizations...")
    
    # Load test image paths
    test_imgs_txt = 'Enhancemodle/Unet_ASPP/test_imgs.txt'
    test_imgs = []
    if os.path.exists(test_imgs_txt):
        with open(test_imgs_txt, 'r') as f:
            test_imgs = [line.strip() for line in f.readlines()]
    
    # Create detailed visualization for first 5 images
    num_viz = min(5, len(y_test))
    
    for i in range(num_viz):
        # Load original image if available
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
        
        # Create comprehensive subplot
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
        labels = ['Background', 'False Negative', 'False Positive', 'True Positive']
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
    
    # Create comprehensive summary visualization
    plt.figure(figsize=(16, 12))
    
    # IoU distribution
    plt.subplot(3, 3, 1)
    plt.hist(iou_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_iou, color='red', linestyle='--', label=f'Mean: {mean_iou:.3f}')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.title('IoU Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dice distribution
    plt.subplot(3, 3, 2)
    plt.hist(dice_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(mean_dice, color='red', linestyle='--', label=f'Mean: {mean_dice:.3f}')
    plt.xlabel('Dice Score')
    plt.ylabel('Frequency')
    plt.title('Dice Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # IoU vs Dice correlation
    plt.subplot(3, 3, 3)
    plt.scatter(iou_scores, dice_scores, alpha=0.6)
    plt.xlabel('IoU Score')
    plt.ylabel('Dice Score')
    plt.title('IoU vs Dice Correlation')
    plt.grid(True, alpha=0.3)
    
    # IoU vs image index
    plt.subplot(3, 3, 4)
    plt.plot(range(1, len(iou_scores) + 1), iou_scores, 'o-', alpha=0.7)
    plt.axhline(mean_iou, color='red', linestyle='--', label=f'Mean: {mean_iou:.3f}')
    plt.xlabel('Test Image Index')
    plt.ylabel('IoU Score')
    plt.title('IoU Scores by Image')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confusion matrix visualization
    plt.subplot(3, 3, 5)
    cm = np.array([[class_metrics['tn'], class_metrics['fp']], 
                   [class_metrics['fn'], class_metrics['tp']]])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Background', 'Dead Tree'], rotation=45)
    plt.yticks(tick_marks, ['Background', 'Dead Tree'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Best prediction
    best_idx = np.argmax(iou_scores)
    plt.subplot(3, 3, 6)
    plt.imshow(y_test[best_idx], cmap='gray')
    plt.title(f'Best Prediction\nIoU: {iou_scores[best_idx]:.3f}')
    plt.axis('off')
    
    # Worst prediction
    worst_idx = np.argmin(iou_scores)
    plt.subplot(3, 3, 7)
    plt.imshow(y_test[worst_idx], cmap='gray')
    plt.title(f'Worst Prediction\nIoU: {iou_scores[worst_idx]:.3f}')
    plt.axis('off')
    
    # Performance categories pie chart
    plt.subplot(3, 3, 8)
    excellent = sum(1 for iou in iou_scores if iou >= 0.8)
    good = sum(1 for iou in iou_scores if 0.6 <= iou < 0.8)
    fair = sum(1 for iou in iou_scores if 0.4 <= iou < 0.6)
    poor = sum(1 for iou in iou_scores if iou < 0.4)
    
    sizes = [excellent, good, fair, poor]
    labels = [f'Excellent\n(≥0.8)', f'Good\n(0.6-0.8)', f'Fair\n(0.4-0.6)', f'Poor\n(<0.4)']
    colors = ['green', 'lightgreen', 'orange', 'red']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Performance Distribution')
    
    # Metrics comparison
    plt.subplot(3, 3, 9)
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
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Create confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    cm = np.array([[class_metrics['tn'], class_metrics['fp']], 
                   [class_metrics['fn'], class_metrics['tp']]])
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix Heatmap', fontsize=14)
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Background', 'Dead Tree'], rotation=45)
    plt.yticks(tick_marks, ['Background', 'Dead Tree'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    print(f"Enhanced visualizations completed!")
    print(f"Results saved to: Enhancemodle/Unet_ASPP/comprehensive_evaluation_results.txt")

if __name__ == '__main__':
    main() 