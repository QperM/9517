#!/usr/bin/env python3
"""
Random Forest with Ensemble Sampling Optimization
"""

import os
import sys
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

class RandomForestEnsembleSamplingOptimization:
    """Random Forest with Ensemble Sampling Optimization"""
    
    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.training_time = 0
        self.prediction_time = 0
        self.sampling_strategy = None
        self.ensemble_models = []
        
    def create_sampling_strategies(self, X, y):
        """Create different sampling strategies for ensemble"""
        print("Creating sampling strategies for ensemble...")
        
        strategies = []
        
        # Strategy 1: Bootstrap sampling (with replacement)
        n_samples = X.shape[0]
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        strategies.append(('bootstrap', X_bootstrap, y_bootstrap))
        
        # Strategy 2: Stratified sampling
        from sklearn.model_selection import train_test_split
        X_strat, _, y_strat, _ = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=self.random_state
        )
        strategies.append(('stratified', X_strat, y_strat))
        
        # Strategy 3: Random sampling without replacement
        random_indices = np.random.choice(n_samples, size=int(0.8*n_samples), replace=False)
        X_random = X[random_indices]
        y_random = y[random_indices]
        strategies.append(('random', X_random, y_random))
        
        # Strategy 4: Balanced sampling (if imbalanced)
        from sklearn.utils import resample
        if len(np.unique(y)) == 2:
            # Separate majority and minority classes
            majority_class = 0 if np.sum(y == 0) > np.sum(y == 1) else 1
            minority_class = 1 - majority_class
            
            X_majority = X[y == majority_class]
            X_minority = X[y == minority_class]
            y_majority = y[y == majority_class]
            y_minority = y[y == minority_class]
            
            # Upsample minority class
            X_minority_upsampled = resample(
                X_minority, 
                n_samples=len(X_majority), 
                random_state=self.random_state
            )
            y_minority_upsampled = np.full(len(X_minority_upsampled), minority_class)
            
            # Combine
            X_balanced = np.vstack([X_majority, X_minority_upsampled])
            y_balanced = np.hstack([y_majority, y_minority_upsampled])
            
            # Shuffle
            shuffle_idx = np.random.permutation(len(X_balanced))
            X_balanced = X_balanced[shuffle_idx]
            y_balanced = y_balanced[shuffle_idx]
            
            strategies.append(('balanced', X_balanced, y_balanced))
        
        print(f"Created {len(strategies)} sampling strategies")
        return strategies
    
    def create_ensemble_models(self, sampling_strategies):
        """Create ensemble models with different sampling strategies"""
        print("Creating ensemble models...")
        
        models = []
        
        for i, (strategy_name, X_strat, y_strat) in enumerate(sampling_strategies):
            print(f"Training model {i+1} with {strategy_name} sampling...")
            
            # Create base Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + i,  # Different random state for diversity
                max_samples=0.8,  # Use 80% of samples for each tree
                bootstrap=True
            )
            
            # Train the model
            rf_model.fit(X_strat, y_strat)
            
            # Evaluate individual model performance
            y_pred = rf_model.predict(X_strat)
            f1 = f1_score(y_strat, y_pred, average='weighted')
            
            print(f"  Model {i+1} F1 score: {f1:.4f}")
            
            models.append({
                'name': f'rf_{strategy_name}',
                'model': rf_model,
                'strategy': strategy_name,
                'f1_score': f1
            })
        
        return models
    
    def optimize_ensemble(self, ensemble_models, X_val, y_val):
        """Optimize ensemble by selecting best models and weights"""
        print("Optimizing ensemble composition...")
        
        # Evaluate each model on validation set
        model_scores = []
        for model_info in ensemble_models:
            model = model_info['model']
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted')
            model_scores.append((model_info, f1))
        
        # Sort by performance
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top models (keep at least 2, at most 3)
        n_models = min(3, max(2, len(model_scores)))
        selected_models = model_scores[:n_models]
        
        print(f"Selected {n_models} best models:")
        for i, (model_info, score) in enumerate(selected_models):
            print(f"  {i+1}. {model_info['name']}: F1={score:.4f}")
        
        # Create voting classifier
        estimators = []
        for model_info, _ in selected_models:
            estimators.append((model_info['name'], model_info['model']))
        
        # Use soft voting for better performance
        voting_classifier = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        return voting_classifier, selected_models
    
    def train(self, X_train, y_train, use_sampling_strategies=True, use_ensemble_optimization=True):
        """Train enhanced model with sampling strategies and ensemble optimization"""
        print("Training Random Forest with Ensemble Sampling Optimization...")
        start_time = time.time()
        
        # Split data for ensemble optimization
        from sklearn.model_selection import train_test_split
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )
        
        # Step 1: Create sampling strategies
        if use_sampling_strategies:
            sampling_strategies = self.create_sampling_strategies(X_train_main, y_train_main)
        else:
            # Use original data
            sampling_strategies = [('original', X_train_main, y_train_main)]
        
        # Step 2: Create ensemble models
        self.ensemble_models = self.create_ensemble_models(sampling_strategies)
        
        # Step 3: Optimize ensemble
        if use_ensemble_optimization and len(self.ensemble_models) > 1:
            self.ensemble_model, selected_models = self.optimize_ensemble(
                self.ensemble_models, X_val, y_val
            )
            
            # Train the final ensemble
            self.ensemble_model.fit(X_train_main, y_train_main)
            
        else:
            # Use the best individual model
            best_model = max(self.ensemble_models, key=lambda x: x['f1_score'])
            self.ensemble_model = best_model['model']
            print(f"Using best individual model: {best_model['name']}")
        
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        # Print ensemble information
        if hasattr(self.ensemble_model, 'estimators_'):
            print(f"Ensemble contains {len(self.ensemble_model.estimators_)} models")
        else:
            print("Using single optimized model")
    
    def predict(self, X):
        """Make predictions"""
        start_time = time.time()
        predictions = self.ensemble_model.predict(X)
        self.prediction_time = time.time() - start_time
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.ensemble_model.predict_proba(X)
    
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
        print("Evaluating Random Forest with Ensemble Sampling Optimization performance...")
        
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
            'ensemble_models': len(self.ensemble_models)
        }
        
        return results
    
    def save_model(self, filepath):
        """Save model to file"""
        model_data = {
            'ensemble_model': self.ensemble_model,
            'ensemble_models': self.ensemble_models,
            'training_time': self.training_time
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to: {filepath}")

def main():
    """Main function to run Random Forest with Ensemble Sampling Optimization"""
    print("=" * 60)
    print("Random Forest with Ensemble Sampling Optimization")
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
        
        # 3. Train Random Forest with Ensemble Sampling Optimization
        print("\n3. Training Random Forest with Ensemble Sampling Optimization...")
        rf_ensemble_sampling = RandomForestEnsembleSamplingOptimization(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        
        rf_ensemble_sampling.train(
            X_train, y_train, 
            use_sampling_strategies=True,
            use_ensemble_optimization=True
        )
        
        # 4. Evaluate model
        print("\n4. Evaluating Random Forest with Ensemble Sampling Optimization performance...")
        results = rf_ensemble_sampling.evaluate(X_test, y_test)
        
        # 5. Show results
        print("\n" + "=" * 40)
        print("Random Forest with Ensemble Sampling Optimization Results:")
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
        print(f"Ensemble models: {results['ensemble_models']}")
        
        # 6. Save model
        print("\n5. Saving Random Forest with Ensemble Sampling Optimization model...")
        os.makedirs("Enhancemodle/RandomForest_Ensemble_SamplingOptimization", exist_ok=True)
        model_path = "Enhancemodle/RandomForest_Ensemble_SamplingOptimization/random_forest_ensemble_sampling_optimization_model.pkl"
        rf_ensemble_sampling.save_model(model_path)
        print(f"Random Forest with Ensemble Sampling Optimization model saved to: {model_path}")
        
        print("\nRandom Forest with Ensemble Sampling Optimization training completed!")
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