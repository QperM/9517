#!/usr/bin/env python3
"""
Enhanced Random Forest - Quick Run Script
"""

import os
import sys
import numpy as np

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

def main():
    """Main function"""
    print("=" * 60)
    print("Enhanced Random Forest Dead Tree Segmentation")
    print("=" * 60)
    
    try:
        # Import enhanced modules
        from enhanced_random_forest import EnhancedRandomForestSegmentation
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
        
        if X_train is None:
            print("Error: Failed to prepare dataset.")
            return
        
        print(f"Train set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Train label distribution: {np.bincount(y_train)}")
        print(f"Test label distribution: {np.bincount(y_test)}")
        
        # 3. Train enhanced model
        print("\n3. Training Enhanced Random Forest model...")
        enhanced_rf = EnhancedRandomForestSegmentation(
            n_estimators=100,  # Reduced for faster training
            max_depth=15,
            random_state=42
        )
        
        enhanced_rf.train(
            X_train, y_train, 
            tune_hyperparameters=False,  # Disable for faster training
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
        # Save model in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "enhanced_random_forest_model.pkl")
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