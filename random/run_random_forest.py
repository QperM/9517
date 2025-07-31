#!/usr/bin/env python3
"""
Random Forest Dead Tree Segmentation - Quick Run Script
"""

import os
import sys
import numpy as np

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

def main():
    """Main function"""
    print("=" * 60)
    print("Random Forest Dead Tree Segmentation")
    print("=" * 60)
    
    try:
        # Import necessary modules
        from src.data.data_loader import ForestDataLoader
        from src.models.random_forest_model import RandomForestSegmentation
        
        # Check if dataset exists
        if not os.path.exists("USA_segmentation"):
            print("Error: Dataset directory 'USA_segmentation' not found.")
            print("Please make sure the dataset is extracted in the project root directory.")
            return
        
        # 1. Data loading
        print("\n1. Loading data...")
        data_loader = ForestDataLoader("USA_segmentation")
        
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
        
        print(f"Train set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Train label distribution: {np.bincount(y_train)}")
        print(f"Test label distribution: {np.bincount(y_test)}")
        
        # 3. Train model
        print("\n3. Training Random Forest model...")
        rf_model = RandomForestSegmentation(
            n_estimators=50,  # Fewer trees for faster training
            max_depth=10,     # Limit depth for faster training
            random_state=42
        )
        
        rf_model.train(X_train, y_train, tune_hyperparameters=False)
        
        # 4. Evaluate model
        print("\n4. Evaluating model performance...")
        results = rf_model.evaluate(X_test, y_test)
        
        # 5. Show results
        print("\n" + "=" * 40)
        print("Final Results:")
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
        
        # 6. Save model
        print("\n5. Saving model...")
        os.makedirs("results", exist_ok=True)
        model_path = "results/random_forest_model.pkl"
        rf_model.save_model(model_path)
        print(f"Model saved to: {model_path}")
        
        print("\nTraining completed!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please make sure all required packages are installed:")
        print("pip install -r requirements.txt")
        
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 