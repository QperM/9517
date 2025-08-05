#!/usr/bin/env python3
"""
Random Forest with Feature Selection and Hyperparameter Tuning
"""

import os
import sys
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

class RandomForestFeatureSelectionHyperparamTuning:
    """Random Forest with Feature Selection and Hyperparameter Tuning"""
    
    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.training_time = 0
        self.prediction_time = 0
        self.selected_features = None
        self.best_params = None
        
    def feature_selection(self, X, y, method='kbest', n_features=100):
        """Advanced feature selection"""
        print(f"Performing {method} feature selection...")
        
        if method == 'kbest':
            # SelectKBest with f_classif
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = self.feature_selector.get_support()
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            base_rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            self.feature_selector = RFE(estimator=base_rf, n_features_to_select=n_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = self.feature_selector.support_
            
        elif method == 'variance':
            # Variance threshold
            from sklearn.feature_selection import VarianceThreshold
            self.feature_selector = VarianceThreshold(threshold=0.01)
            X_selected = self.feature_selector.fit_transform(X)
            self.selected_features = self.feature_selector.get_support()
            
        print(f"Selected {X_selected.shape[1]} features from {X.shape[1]} original features")
        return X_selected
    
    def hyperparameter_tuning(self, X, y):
        """Advanced hyperparameter tuning using RandomizedSearchCV"""
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Create base model
        base_rf = RandomForestClassifier(random_state=self.random_state)
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_grid,
            n_iter=20,  # Number of parameter settings sampled
            cv=3,
            scoring='f1_weighted',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Fit the search
        random_search.fit(X, y)
        
        # Get best parameters
        self.best_params = random_search.best_params_
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def train(self, X_train, y_train, use_feature_selection=True, use_hyperparameter_tuning=True):
        """Train enhanced model with feature selection and hyperparameter tuning"""
        print("Training Enhanced Random Forest V2...")
        start_time = time.time()
        
        # Step 1: Feature Selection
        if use_feature_selection:
            X_train_selected = self.feature_selection(X_train, y_train, method='kbest', n_features=100)
        else:
            X_train_selected = X_train
            self.selected_features = np.ones(X_train.shape[1], dtype=bool)
        
        # Step 2: Hyperparameter Tuning
        if use_hyperparameter_tuning:
            self.model = self.hyperparameter_tuning(X_train_selected, y_train)
        else:
            # Use default parameters
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self.model.fit(X_train_selected, y_train)
        
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        # Print feature importance
        if hasattr(self.model, 'feature_importances_'):
            top_features = np.argsort(self.model.feature_importances_)[-10:]
            print("Top 10 feature importances:")
            for i, feature_idx in enumerate(reversed(top_features)):
                print(f"  Feature {feature_idx}: {self.model.feature_importances_[feature_idx]:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X)
        else:
            X_selected = X
        
        start_time = time.time()
        predictions = self.model.predict(X_selected)
        self.prediction_time = time.time() - start_time
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X)
        else:
            X_selected = X
        
        return self.model.predict_proba(X_selected)
    
    def calculate_iou(self, y_true, y_pred):
        """Calculate IoU (Jaccard index)"""
        intersection = np.logical_and(y_true, y_pred)
        union = np.logical_or(y_true, y_pred)
        
        if np.sum(union) == 0:
            return 1.0 if np.sum(intersection) == 0 else 0.0
        
        return np.sum(intersection) / np.sum(union)
    
    def calculate_dice(self, y_true, y_pred):
        """Calculate Dice coefficient (F1 score)"""
        intersection = np.logical_and(y_true, y_pred)
        return 2 * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred))
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating Random Forest with Feature Selection and Hyperparameter Tuning performance...")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        iou = self.calculate_iou(y_test, y_pred)
        dice = self.calculate_dice(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"IoU Score: {iou:.4f}")
        print(f"Dice Score: {dice:.4f}")
        print(f"Training time: {self.training_time:.2f} seconds")
        print(f"Prediction time: {self.prediction_time:.2f} seconds")
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'iou': iou,
            'dice': dice,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': self.best_params,
            'selected_features': self.selected_features
        }
        
        return results
    
    def save_model(self, filepath):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'best_params': self.best_params,
            'training_time': self.training_time
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to: {filepath}")

def main():
    """Main function to run Random Forest with Feature Selection and Hyperparameter Tuning"""
    print("=" * 60)
    print("Random Forest with Feature Selection and Hyperparameter Tuning")
    print("=" * 60)
    
    try:
        # Import data loader from current directory
        from data_loader import EnhancedForestDataLoader
        
        # Check if dataset exists
        dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'USA_segmentation')
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset directory '{dataset_path}' not found.")
            return
        
        # 1. Data loading
        print("\n1. Loading data...")
        data_loader = EnhancedForestDataLoader(dataset_path)
        
        if len(data_loader.rgb_files) == 0:
            print("Error: No RGB image files found.")
            return
        
        print(f"Found {len(data_loader.rgb_files)} RGB images")
        print(f"Found {len(data_loader.nrg_files)} NRG images")
        print(f"Found {len(data_loader.mask_files)} mask images")
        
        # 2. Prepare dataset
        print("\n2. Preparing dataset...")
        X_train, X_test, y_train, y_test = data_loader.prepare_dataset(
            test_size=0.2, 
            random_state=42
        )
        
        if X_train is None:
            print("Error: Failed to prepare dataset.")
            return
        
        print(f"Train set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Train label distribution: {np.bincount(y_train)}")
        print(f"Test label distribution: {np.bincount(y_test)}")
        
        # 3. Train Random Forest with Feature Selection and Hyperparameter Tuning
        print("\n3. Training Random Forest with Feature Selection and Hyperparameter Tuning...")
        rf_feature_hyperparam = RandomForestFeatureSelectionHyperparamTuning(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        
        rf_feature_hyperparam.train(
            X_train, y_train, 
            use_feature_selection=True,
            use_hyperparameter_tuning=True
        )
        
        # 4. Evaluate model
        print("\n4. Evaluating Random Forest with Feature Selection and Hyperparameter Tuning performance...")
        results = rf_feature_hyperparam.evaluate(X_test, y_test)
        
        # 5. Show results
        print("\n" + "=" * 40)
        print("Random Forest with Feature Selection and Hyperparameter Tuning Results:")
        print("=" * 40)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print("="*40)
        print(f"IoU (Jaccard) score: {results['iou']:.4f}")
        print(f"Dice coefficient: {results['dice']:.4f}")
        print("="*40)
        print(f"Training time: {results['training_time']:.2f} s")
        print(f"Prediction time: {results['prediction_time']:.2f} s")
        
        # 6. Save model
        print("\n5. Saving Random Forest with Feature Selection and Hyperparameter Tuning model...")
        os.makedirs("Enhancemodle/RandomForest_FeatureSelection_HyperparamTuning", exist_ok=True)
        model_path = "Enhancemodle/RandomForest_FeatureSelection_HyperparamTuning/random_forest_feature_selection_hyperparam_tuning_model.pkl"
        rf_feature_hyperparam.save_model(model_path)
        print(f"Random Forest with Feature Selection and Hyperparameter Tuning model saved to: {model_path}")
        
        print("\nRandom Forest with Feature Selection and Hyperparameter Tuning training completed!")
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