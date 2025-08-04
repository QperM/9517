#!/usr/bin/env python3
"""
Enhanced Random Forest for Dead Tree Segmentation
Advanced implementation with feature engineering, augmentation, and ensemble methods
"""

import os
import sys
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from scipy import ndimage
from scipy.stats import randint, uniform
import albumentations as A
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EnhancedRandomForestSegmentation:
    """Enhanced Random Forest based dead tree segmentation model"""
    
    def __init__(self, n_estimators=200, max_depth=None, random_state=42):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        # Initialize ensemble models
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True,
            oob_score=True
        )
        
        self.et_model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True
        )
        
        # Ensemble voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('et', self.et_model)
            ],
            voting='soft'
        )
        
        # Feature preprocessing
        self.scaler = RobustScaler()
        self.pca = None
        self.feature_selector = None
        
        # Performance tracking
        self.feature_importance = None
        self.training_time = None
        self.prediction_time = None
        self.best_params = None
        
    def extract_advanced_features(self, rgb_patch, nrg_patch):
        """Extract advanced features including texture, shape, and spectral features"""
        features = []
        
        # 1. Basic RGB features
        rgb_mean = np.mean(rgb_patch, axis=(0, 1))
        rgb_std = np.std(rgb_patch, axis=(0, 1))
        rgb_median = np.median(rgb_patch, axis=(0, 1))
        rgb_min = np.min(rgb_patch, axis=(0, 1))
        rgb_max = np.max(rgb_patch, axis=(0, 1))
        
        # 2. NRG features
        nrg_mean = np.mean(nrg_patch, axis=(0, 1))
        nrg_std = np.std(nrg_patch, axis=(0, 1))
        nrg_median = np.median(nrg_patch, axis=(0, 1))
        
        # 3. Spectral indices
        if nrg_patch.shape[2] >= 4:
            nir = nrg_patch[:, :, 3]
            red = nrg_patch[:, :, 0]
            green = nrg_patch[:, :, 1]
            
            # NDVI (Normalized Difference Vegetation Index)
            ndvi = np.mean((nir - red) / (nir + red + 1e-8))
            
            # GNDVI (Green Normalized Difference Vegetation Index)
            gndvi = np.mean((nir - green) / (nir + green + 1e-8))
            
            # EVI (Enhanced Vegetation Index)
            evi = np.mean(2.5 * (nir - red) / (nir + 6 * red - 7.5 * green + 1 + 1e-8))
        else:
            ndvi = gndvi = evi = 0.0
        
        # 4. Advanced texture features
        gray_patch = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
        
        # GLCM features
        glcm_features = self._compute_glcm_features(gray_patch)
        
        # LBP features
        lbp_features = self._compute_lbp_features(gray_patch)
        
        # Gabor filter features
        gabor_features = self._compute_gabor_features(gray_patch)
        
        # 5. Edge and gradient features
        edge_features = self._compute_edge_features(gray_patch)
        
        # 6. Shape and morphological features
        shape_features = self._compute_shape_features(gray_patch)
        
        # 7. Color histogram features
        hist_features = self._compute_histogram_features(rgb_patch, nrg_patch)
        
        # 8. Statistical moments
        moments = self._compute_statistical_moments(gray_patch)
        
        # Combine all features
        features.extend(rgb_mean)
        features.extend(rgb_std)
        features.extend(rgb_median)
        features.extend(rgb_min)
        features.extend(rgb_max)
        features.extend(nrg_mean)
        features.extend(nrg_std)
        features.extend(nrg_median)
        features.extend([ndvi, gndvi, evi])
        features.extend(glcm_features)
        features.extend(lbp_features)
        features.extend(gabor_features)
        features.extend(edge_features)
        features.extend(shape_features)
        features.extend(hist_features)
        features.extend(moments)
        
        return np.array(features)
    
    def _compute_glcm_features(self, gray_patch):
        """Compute Gray Level Co-occurrence Matrix features"""
        # Simplified GLCM features
        # In practice, you might want to use skimage.feature.graycomatrix
        
        # Local variance
        local_var = ndimage.variance(gray_patch)
        
        # Local entropy
        hist, _ = np.histogram(gray_patch, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        
        # Contrast
        contrast = np.std(gray_patch)
        
        # Homogeneity (simplified)
        homogeneity = np.mean(1 / (1 + gray_patch**2))
        
        return [local_var, entropy, contrast, homogeneity]
    
    def _compute_lbp_features(self, gray_patch):
        """Compute Local Binary Pattern features"""
        lbp = self._compute_lbp(gray_patch)
        lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        lbp_hist = lbp_hist / np.sum(lbp_hist)
        
        # LBP statistics
        lbp_mean = np.mean(lbp)
        lbp_std = np.std(lbp)
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-8))
        
        return np.concatenate([lbp_hist, [lbp_mean, lbp_std, lbp_entropy]])
    
    def _compute_lbp(self, gray_patch):
        """Compute Local Binary Pattern"""
        lbp = np.zeros_like(gray_patch)
        for i in range(1, gray_patch.shape[0] - 1):
            for j in range(1, gray_patch.shape[1] - 1):
                center = gray_patch[i, j]
                code = 0
                code |= (gray_patch[i-1, j-1] > center) << 7
                code |= (gray_patch[i-1, j] > center) << 6
                code |= (gray_patch[i-1, j+1] > center) << 5
                code |= (gray_patch[i, j+1] > center) << 4
                code |= (gray_patch[i+1, j+1] > center) << 3
                code |= (gray_patch[i+1, j] > center) << 2
                code |= (gray_patch[i+1, j-1] > center) << 1
                code |= (gray_patch[i, j-1] > center) << 0
                lbp[i, j] = code
        return lbp
    
    def _compute_gabor_features(self, gray_patch):
        """Compute Gabor filter features"""
        features = []
        
        # Gabor filter parameters
        angles = [0, 45, 90, 135]
        frequencies = [0.1, 0.3, 0.5]
        
        for angle in angles:
            for freq in frequencies:
                # Simplified Gabor-like filter
                kernel = cv2.getGaborKernel((15, 15), 3, np.radians(angle), 2*np.pi*freq, 0.5, 0)
                filtered = cv2.filter2D(gray_patch, cv2.CV_8UC3, kernel)
                features.extend([np.mean(filtered), np.std(filtered)])
        
        return features
    
    def _compute_edge_features(self, gray_patch):
        """Compute edge and gradient features"""
        # Sobel gradients
        grad_x = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)
        
        # Canny edges
        edges = cv2.Canny(gray_patch, 50, 150)
        
        # Edge statistics
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        grad_mag_mean = np.mean(grad_magnitude)
        grad_mag_std = np.std(grad_magnitude)
        grad_dir_mean = np.mean(grad_direction)
        grad_dir_std = np.std(grad_direction)
        
        return [edge_density, grad_mag_mean, grad_mag_std, grad_dir_mean, grad_dir_std]
    
    def _compute_shape_features(self, gray_patch):
        """Compute shape and morphological features"""
        # Threshold to create binary image
        _, binary = cv2.threshold(gray_patch, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Shape features
            areas = [cv2.contourArea(c) for c in contours]
            perimeters = [cv2.arcLength(c, True) for c in contours]
            
            # Compactness
            compactness = np.mean([4 * np.pi * area / (perimeter**2 + 1e-8) 
                                 for area, perimeter in zip(areas, perimeters)])
            
            # Aspect ratio
            aspect_ratios = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratios.append(w / (h + 1e-8))
            
            avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 0
        else:
            compactness = 0
            avg_aspect_ratio = 0
        
        return [compactness, avg_aspect_ratio]
    
    def _compute_histogram_features(self, rgb_patch, nrg_patch):
        """Compute color histogram features"""
        features = []
        
        # RGB histogram
        try:
            rgb_hist = cv2.calcHist([rgb_patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            rgb_hist = rgb_hist.flatten() / (np.sum(rgb_hist) + 1e-8)
        except:
            rgb_hist = np.zeros(512)
        
        # NRG histogram
        try:
            nrg_hist = cv2.calcHist([nrg_patch], [0, 1, 2, 3], None, [6, 6, 6, 6], 
                                   [0, 256, 0, 256, 0, 256, 0, 256])
            nrg_hist = nrg_hist.flatten() / (np.sum(nrg_hist) + 1e-8)
        except:
            nrg_hist = np.zeros(1296)
        
        features.extend(rgb_hist)
        features.extend(nrg_hist)
        
        return features
    
    def _compute_statistical_moments(self, gray_patch):
        """Compute statistical moments"""
        # Central moments
        mean = np.mean(gray_patch)
        variance = np.var(gray_patch)
        skewness = self._compute_skewness(gray_patch, mean)
        kurtosis = self._compute_kurtosis(gray_patch, mean)
        
        return [mean, variance, skewness, kurtosis]
    
    def _compute_skewness(self, data, mean):
        """Compute skewness"""
        n = len(data.flatten())
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum(((data - mean) / std) ** 3) / n
    
    def _compute_kurtosis(self, data, mean):
        """Compute kurtosis"""
        n = len(data.flatten())
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum(((data - mean) / std) ** 4) / n - 3
    
    def augment_data(self, X, y, augmentation_factor=2):
        """Apply data augmentation to increase training samples"""
        print(f"Applying data augmentation (factor: {augmentation_factor})...")
        
        augmented_X = []
        augmented_y = []
        
        # Define augmentation pipeline
        augmenter = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        ])
        
        for i in tqdm(range(len(X)), desc="Augmenting data"):
            # Original sample
            augmented_X.append(X[i])
            augmented_y.append(y[i])
            
            # Augmented samples
            for _ in range(augmentation_factor - 1):
                # For feature vectors, we'll add noise instead of image augmentation
                noise = np.random.normal(0, 0.01, X[i].shape)
                augmented_sample = X[i] + noise
                augmented_X.append(augmented_sample)
                augmented_y.append(y[i])
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def feature_selection(self, X, y, method='kbest', n_features=100):
        """Perform feature selection"""
        print(f"Performing feature selection using {method}...")
        
        if method == 'kbest':
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        elif method == 'rfe':
            # Use Random Forest for RFE
            base_rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            self.feature_selector = RFE(estimator=base_rf, n_features_to_select=n_features)
        
        X_selected = self.feature_selector.fit_transform(X, y)
        print(f"Selected {X_selected.shape[1]} features from {X.shape[1]} original features")
        
        return X_selected
    
    def train(self, X_train, y_train, tune_hyperparameters=True, use_augmentation=True, 
              use_feature_selection=True, use_ensemble=True):
        """Train the enhanced random forest model"""
        print("Training enhanced random forest model...")
        start_time = time.time()
        
        # Data augmentation
        if use_augmentation:
            X_train, y_train = self.augment_data(X_train, y_train, augmentation_factor=2)
        
        # Feature preprocessing
        print("Preprocessing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Feature selection
        if use_feature_selection:
            X_train_selected = self.feature_selection(X_train_scaled, y_train, method='kbest', n_features=100)
        else:
            X_train_selected = X_train_scaled
        
        # PCA for dimensionality reduction (optional)
        if X_train_selected.shape[1] > 50:
            print("Applying PCA for dimensionality reduction...")
            self.pca = PCA(n_components=min(50, X_train_selected.shape[1]), random_state=self.random_state)
            X_train_pca = self.pca.fit_transform(X_train_selected)
            print(f"Reduced features from {X_train_selected.shape[1]} to {X_train_pca.shape[1]}")
            X_train_final = X_train_pca
        else:
            X_train_final = X_train_selected
        
        # Hyperparameter tuning
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            if use_ensemble:
                self.best_params = self._tune_ensemble_hyperparameters(X_train_final, y_train)
            else:
                self.best_params = self._tune_single_model_hyperparameters(X_train_final, y_train)
        
        # Train final model
        if use_ensemble:
            print("Training ensemble model...")
            self.ensemble_model.fit(X_train_final, y_train)
            self.model = self.ensemble_model
        else:
            print("Training single Random Forest model...")
            self.rf_model.fit(X_train_final, y_train)
            self.model = self.rf_model
        
        self.training_time = time.time() - start_time
        if use_ensemble:
            # For ensemble, use the first estimator's feature importance
            self.feature_importance = self.model.estimators_[0].feature_importances_
        else:
            self.feature_importance = self.model.feature_importances_
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        print(f"Feature importance shape: {self.feature_importance.shape}")
        
        return self
    
    def _tune_single_model_hyperparameters(self, X, y):
        """Tune hyperparameters for single Random Forest model"""
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': [10, 20, 30, None],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')
        random_search = RandomizedSearchCV(
            rf, param_dist, n_iter=20, cv=3, scoring='f1', 
            random_state=self.random_state, n_jobs=-1, verbose=1
        )
        
        random_search.fit(X, y)
        self.rf_model = random_search.best_estimator_
        return random_search.best_params_
    
    def _tune_ensemble_hyperparameters(self, X, y):
        """Tune hyperparameters for ensemble model"""
        # Tune Random Forest
        rf_params = {
            'n_estimators': randint(100, 300),
            'max_depth': [10, 20, None],
            'min_samples_split': randint(2, 10)
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')
        rf_search = RandomizedSearchCV(
            rf, rf_params, n_iter=10, cv=3, scoring='f1',
            random_state=self.random_state, n_jobs=-1
        )
        rf_search.fit(X, y)
        
        # Tune Extra Trees
        et_params = {
            'n_estimators': randint(100, 300),
            'max_depth': [10, 20, None],
            'min_samples_split': randint(2, 10)
        }
        
        et = ExtraTreesClassifier(random_state=self.random_state, class_weight='balanced')
        et_search = RandomizedSearchCV(
            et, et_params, n_iter=10, cv=3, scoring='f1',
            random_state=self.random_state, n_jobs=-1
        )
        et_search.fit(X, y)
        
        # Create new ensemble with best models
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_search.best_estimator_),
                ('et', et_search.best_estimator_)
            ],
            voting='soft'
        )
        
        return {
            'rf_params': rf_search.best_params_,
            'et_params': et_search.best_params_
        }
    
    def predict(self, X):
        """Predict"""
        start_time = time.time()
        
        # Apply same preprocessing as training
        X_scaled = self.scaler.transform(X)
        
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        if self.pca is not None:
            X_pca = self.pca.transform(X_selected)
            predictions = self.model.predict(X_pca)
        else:
            predictions = self.model.predict(X_selected)
        
        self.prediction_time = time.time() - start_time
        return predictions
    
    def predict_proba(self, X):
        """Predict probabilities"""
        # Apply same preprocessing as training
        X_scaled = self.scaler.transform(X)
        
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        if self.pca is not None:
            X_pca = self.pca.transform(X_selected)
            return self.model.predict_proba(X_pca)
        else:
            return self.model.predict_proba(X_selected)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating enhanced model performance...")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        iou = self._calculate_iou(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"IoU: {iou:.4f}")
        print(f"Training time: {self.training_time:.2f} seconds")
        print(f"Prediction time: {self.prediction_time:.2f} seconds")
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'iou': iou,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return results
    
    def _calculate_iou(self, y_true, y_pred):
        """Calculate IoU (Jaccard index)"""
        intersection = np.logical_and(y_true, y_pred)
        union = np.logical_or(y_true, y_pred)
        
        if np.sum(union) == 0:
            return 1.0 if np.sum(intersection) == 0 else 0.0
        
        return np.sum(intersection) / np.sum(union)
    
    def post_process_segmentation(self, segmentation_map, kernel_size=3):
        """Apply post-processing to improve segmentation quality"""
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Opening (erosion followed by dilation) to remove noise
        opening = cv2.morphologyEx(segmentation_map, cv2.MORPH_OPEN, kernel)
        
        # Closing (dilation followed by erosion) to fill holes
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        # Remove small objects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)
        
        # Remove components smaller than threshold
        min_size = 50
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                closing[labels == i] = 0
        
        return closing
    
    def segment_image(self, rgb_img, nrg_img, patch_size=32):
        """Segment the entire image with enhanced features"""
        h, w = rgb_img.shape[:2]
        segmentation_map = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(0, h - patch_size, patch_size // 2):
            for j in range(0, w - patch_size, patch_size // 2):
                rgb_patch = rgb_img[i:i+patch_size, j:j+patch_size]
                nrg_patch = nrg_img[i:i+patch_size, j:j+patch_size]
                
                feature_vector = self.extract_advanced_features(rgb_patch, nrg_patch)
                
                prediction = self.predict([feature_vector])[0]
                
                segmentation_map[i:i+patch_size, j:j+patch_size] = prediction * 255
        
        # Apply post-processing
        segmentation_map = self.post_process_segmentation(segmentation_map)
        
        return segmentation_map
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("Model not trained, cannot show feature importance.")
            return
        
        indices = np.argsort(self.feature_importance)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(top_n), self.feature_importance[indices])
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title(f'Enhanced Random Forest Feature Importance (Top {top_n})')
        plt.xticks(range(top_n), [f'Feature_{i}' for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Enhanced Random Forest Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def save_model(self, filepath):
        """Save model and preprocessing components"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_selector': self.feature_selector,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Enhanced model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load model and preprocessing components"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.feature_selector = model_data['feature_selector']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        print(f"Enhanced model loaded from {filepath}")

def main():
    """Main function to run enhanced Random Forest"""
    print("=" * 60)
    print("Enhanced Random Forest Dead Tree Segmentation")
    print("=" * 60)
    
    try:
        # Import enhanced data loader
        from data_loader import EnhancedForestDataLoader
        
        # Check if dataset exists
        if not os.path.exists("USA_segmentation"):
            print("Error: Dataset directory 'USA_segmentation' not found.")
            print("Please make sure the dataset is extracted in the project root directory.")
            return
        
        # 1. Data loading
        print("\n1. Loading data...")
        data_loader = EnhancedForestDataLoader("USA_segmentation")
        
        if len(data_loader.rgb_files) == 0:
            print("Error: No RGB image files found.")
            return
        
        print(f"Found {len(data_loader.rgb_files)} RGB images")
        print(f"Found {len(data_loader.nrg_files)} NRG images")
        print(f"Found {len(data_loader.mask_files)} mask images")
        
        # 2. Prepare dataset
        print("\n2. Preparing enhanced dataset...")
        X_train, X_test, y_train, y_test = data_loader.prepare_dataset(
            test_size=0.2, 
            random_state=42
        )
        
        print(f"Train set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Train label distribution: {np.bincount(y_train)}")
        print(f"Test label distribution: {np.bincount(y_test)}")
        
        # 3. Train enhanced model
        print("\n3. Training Enhanced Random Forest model...")
        enhanced_rf = EnhancedRandomForestSegmentation(
            n_estimators=200,
            max_depth=20,
            random_state=42
        )
        
        enhanced_rf.train(
            X_train, y_train, 
            tune_hyperparameters=True,
            use_augmentation=True,
            use_feature_selection=True,
            use_ensemble=True
        )
        
        # 4. Evaluate enhanced model
        print("\n4. Evaluating enhanced model performance...")
        results = enhanced_rf.evaluate(X_test, y_test)
        
        # 5. Show results
        print("\n" + "=" * 40)
        print("Enhanced Model Results:")
        print("=" * 40)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print("="*40)
        print(f"IoU (Jaccard) score: {results['iou']:.4f}")
        print("="*40)
        print(f"Training time: {results['training_time']:.2f} s")
        print(f"Prediction time: {results['prediction_time']:.2f} s")
        
        # 6. Save enhanced model
        print("\n5. Saving enhanced model...")
        os.makedirs("Enhancemodle/Enhanced_Random_Forest", exist_ok=True)
        model_path = "Enhancemodle/Enhanced_Random_Forest/enhanced_random_forest_model.pkl"
        enhanced_rf.save_model(model_path)
        print(f"Enhanced model saved to: {model_path}")
        
        print("\nEnhanced training completed!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please make sure all required packages are installed:")
        print("pip install scikit-learn opencv-python matplotlib seaborn albumentations tqdm")
        
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 