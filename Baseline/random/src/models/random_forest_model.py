import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

class RandomForestSegmentation:
    """Random Forest based dead tree segmentation model"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.feature_importance = None
        self.training_time = None
        self.prediction_time = None
    
    def train(self, X_train, y_train, tune_hyperparameters=False):
        """Train the random forest model"""
        print("Training random forest model...")
        start_time = time.time()
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42, class_weight='balanced'),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.feature_importance = self.model.feature_importances_
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        print(f"Feature importance shape: {self.feature_importance.shape}")
        
        return self
    
    def predict(self, X):
        """Predict"""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = time.time() - start_time
        return predictions
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model performance...")
        
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
        plt.title(f'Random Forest Feature Importance (Top {top_n})')
        plt.xticks(range(top_n), [f'Feature_{i}' for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def save_model(self, filepath):
        """Save model"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def segment_image(self, rgb_img, nrg_img, patch_size=32):
        """Segment the entire image"""
        h, w = rgb_img.shape[:2]
        segmentation_map = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(0, h - patch_size, patch_size // 2):
            for j in range(0, w - patch_size, patch_size // 2):
                rgb_patch = rgb_img[i:i+patch_size, j:j+patch_size]
                nrg_patch = nrg_img[i:i+patch_size, j:j+patch_size]
                
                feature_vector = self._compute_patch_features(rgb_patch, nrg_patch)
                
                prediction = self.model.predict([feature_vector])[0]
                
                segmentation_map[i:i+patch_size, j:j+patch_size] = prediction * 255
        
        return segmentation_map
    
    def _compute_patch_features(self, rgb_patch, nrg_patch):
        """Compute features for a single image patch (same as in data loader)"""
        features = []
        
        rgb_mean = np.mean(rgb_patch, axis=(0, 1))
        rgb_std = np.std(rgb_patch, axis=(0, 1))
        try:
            rgb_hist = cv2.calcHist([rgb_patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            rgb_hist = rgb_hist.flatten() / (np.sum(rgb_hist) + 1e-8)
        except:
            rgb_hist = np.zeros(512)
        
        nrg_mean = np.mean(nrg_patch, axis=(0, 1))
        nrg_std = np.std(nrg_patch, axis=(0, 1))
        try:
            nrg_hist = cv2.calcHist([nrg_patch], [0, 1, 2, 3], None, [6, 6, 6, 6], 
                                   [0, 256, 0, 256, 0, 256, 0, 256])
            nrg_hist = nrg_hist.flatten() / (np.sum(nrg_hist) + 1e-8)
        except:
            nrg_hist = np.zeros(1296)
        
        gray_patch = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
        lbp = self._compute_lbp(gray_patch)
        lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        lbp_hist = lbp_hist / np.sum(lbp_hist)
        
        grad_x = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)
        
        grad_mag_mean = np.mean(grad_magnitude)
        grad_mag_std = np.std(grad_magnitude)
        grad_dir_mean = np.mean(grad_direction)
        grad_dir_std = np.std(grad_direction)
        
        edges = cv2.Canny(gray_patch, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        features.extend(rgb_mean)
        features.extend(rgb_std)
        features.extend(rgb_hist)
        features.extend(nrg_mean)
        features.extend(nrg_std)
        features.extend(nrg_hist)
        features.extend(lbp_hist)
        features.extend([grad_mag_mean, grad_mag_std, grad_dir_mean, grad_dir_std])
        features.append(edge_density)
        
        return np.array(features)
    
    def _compute_lbp(self, gray_patch):
        """Compute Local Binary Pattern (LBP)"""
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