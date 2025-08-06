# COMP9517 25T2 Group Project: Dead Tree Segmentation

## Project Overview

This project is a dead tree segmentation system based on deep learning and machine learning, designed to identify and segment dead tree regions from aerial images. The project includes multiple baseline models and enhanced models, with comparative analysis of different method performances.

## Dataset

- **Dataset Name**: USA_segmentation
- **Image Size**: 256x256 pixels
- **Image Types**: RGB images, NRG images, mask images
- **Number of Test Images**: 89 images

## Project Structure

```
9517/
├── Baseline/                          # Baseline models
│   ├── base_model_unet/              # Basic UNet model
│   ├── base_model_xgb/               # XGBoost baseline model
│   ├── CNN/                          # CNN baseline model
│   ├── random/                       # Random baseline model
│   └── SVM/                          # SVM baseline model
├── Enhancemodle/                      # Enhanced models
│   ├── Enhanced_Random_Forest/       # Enhanced Random Forest model
│   ├── RandomForest_Ensemble_SamplingOptimization/  # Ensemble sampling optimization
│   ├── RandomForest_FeatureSelection_HyperparamTuning/  # Feature selection hyperparameter tuning
│   ├── Unet++/                       # UNet++ model
│   ├── Unet++_augmentation/          # Data augmentation UNet++
│   ├── Unet_ASPP/                    # UNet+ASPP model
│   └── Unet_NIR_GAF/                 # UNet+NIR+GAF model
├── USA_segmentation/                  # Dataset
│   ├── RGB_images/                   # RGB images
│   ├── NRG_images/                   # NRG images
│   └── masks/                        # Mask images
├── UNet_vs_UNetPlusPlus.ipynb       # Model comparison analysis
└── README.md                         # Project documentation
```
## How to Run the Models
For Non-Random Forest Models (Baseline & Enhanced UNet Variants)
​Applies to:​​

All models in Baseline/
UNet variants in Enhancemodle/ (except Random Forest folders)
Includes: Unet++/, Unet++_augmentation/, Unet_ASPP/, Unet_NIR_GAF/
​Steps:​​

Navigate to model directory: cd path/to/[folder_name]
Train the model: python *_pipeline.py (saves model/weights in current directory)
Evaluate: python evaluate.py (shows plots with plt.show; doesn't save images)
For Enhanced Random Forest Models
​For Enhanced_Random_Forest/:​​

Navigate: cd Enhancemodle/Enhanced_Random_Forest
Train: python run_enhanced_rf.py (saves model locally)
Generate evaluation report: python generate_evaluation_report.py (creates text report)
Generate visualizations: python generate_picture.py (saves example images)
​For RandomForest_Ensemble_SamplingOptimization/:​​

Navigate: cd Enhancemodle/RandomForest_Ensemble_SamplingOptimization
Train: python random_forest_ensemble_sampling_optimization.py (saves model locally)
Generate report: python generate_evaluation_report_ensemble_sampling_optimization.py (creates text report)
​For RandomForest_FeatureSelection_HyperparamTuning/:​​

Navigate: cd Enhancemodle/RandomForest_FeatureSelection_HyperparamTuning
Train: python random_forest_feature_selection_hyperparam_tuning.py (saves model locally)
Generate report: python generate_evaluation_report_feature_selection_hyperparam_tuning.py (creates text report)


## Model Performance Comparison

### Key Metrics Comparison

| Model | Mean IoU | Mean Dice | Mean Pixel Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|-------------------|-----------|--------|----------|
| **Enhanced Random Forest** | **0.5625** | **0.9796** | **0.9784** | **0.9816** | **0.9784** | **0.9796** |
| UNet++ | 0.3918 | 0.5450 | 0.9844 | 0.6099 | 0.5224 | 0.5627 |
| UNet (Baseline) | 0.3939 | 0.5462 | 0.9850 | 0.6293 | 0.5257 | 0.5728 |

### Performance Category Statistics

#### Enhanced Random Forest
- **Excellent** (IoU ≥ 0.8): 5 images (5.6%)
- **Good** (0.6 ≤ IoU < 0.8): 15 images (16.9%)
- **Fair** (0.4 ≤ IoU < 0.6): 45 images (50.6%)
- **Poor** (IoU < 0.4): 24 images (27.0%)

#### UNet++ (Enhanced Model)
- **Excellent** (IoU ≥ 0.8): 0 images (0.0%)
- **Good** (0.6 ≤ IoU < 0.8): 6 images (6.7%)
- **Fair** (0.4 ≤ IoU < 0.6): 39 images (43.8%)
- **Poor** (IoU < 0.4): 44 images (49.4%)

#### UNet (Baseline Model)
- **Excellent** (IoU ≥ 0.8): 1 image (1.1%)
- **Good** (0.6 ≤ IoU < 0.8): 6 images (6.7%)
- **Fair** (0.4 ≤ IoU < 0.6): 36 images (40.4%)
- **Poor** (IoU < 0.4): 46 images (51.7%)

## Model Details

### 1. Enhanced Random Forest (Best Performance)
- **Location**: `Enhancemodle/Enhanced_Random_Forest/`
- **Features**: 
  - Ensemble learning method (Random Forest + Extra Trees)
  - Advanced feature engineering (texture, shape, spectral features)
  - Data augmentation and feature selection
  - Post-processing optimization
- **Main Files**:
  - `run_enhanced_rf.py`: Quick run script
  - `enhanced_random_forest.py`: Core model implementation
  - `data_loader.py`: Data loader
  - `generate_figure2.py`: Visualization generation

### 2. UNet++ (Enhanced Model)
- **Location**: `Enhancemodle/Unet++/`
- **Features**:
  - Dense skip connections
  - Deep supervision
  - Improved encoder-decoder structure
- **Main Files**:
  - `unet_plus_plus_pipeline.py`: Model implementation
  - `evaluate.py`: Evaluation script

### 3. UNet (Baseline Model)
- **Location**: `Baseline/base_model_unet/`
- **Features**:
  - Standard UNet architecture
  - Encoder-decoder structure
  - Skip connections
- **Main Files**:
  - `unet_pipeline.py`: Model implementation
  - `evaluate.py`: Evaluation script

### 4. Other Enhanced Models
- **UNet+ASPP**: Atrous Spatial Pyramid Pooling
- **UNet+NIR+GAF**: Near-Infrared images and Gramian Angular Field
- **UNet++_augmentation**: Data augmentation version
- **unet++512**: 512-channel version

## Usage Instructions

### 1. Run Enhanced Random Forest (Recommended)

```bash
cd Enhancemodle/Enhanced_Random_Forest/
python run_enhanced_rf.py
```

### 2. Run UNet++ Model

```bash
cd Enhancemodle/Unet++/
python unet_plus_plus_pipeline.py
```

### 3. Run Baseline UNet Model

```bash
cd Baseline/base_model_unet/
python unet_pipeline.py
```

### 4. Generate Visualization Results

```bash
cd Enhancemodle/Enhanced_Random_Forest/
python generate_figure2.py
```

## Requirements

### Python Package Dependencies
```bash
pip install torch torchvision
pip install scikit-learn
pip install opencv-python
pip install matplotlib seaborn
pip install albumentations
pip install tqdm
pip install joblib
```

### Hardware Requirements
- **GPU**: Recommended NVIDIA GPU (for deep learning models)
- **Memory**: At least 8GB RAM
- **Storage**: At least 5GB available space

## Results Analysis

### Best Model: Enhanced Random Forest
- **Advantages**: 
  - Highest IoU score (0.5625)
  - Best Dice coefficient (0.9796)
  - Excellent classification metrics
  - Less overfitting issues
- **Use Cases**: 
  - Applications requiring high-precision segmentation
  - Environments with limited computational resources
  - Scenarios requiring interpretability

### Deep Learning Model Comparison
- **UNet++**: Slight improvement over baseline UNet
- **UNet**: Stable performance as baseline model
- **Enhanced versions**: Performance improvements through various techniques

## File Descriptions

### Core Files
- `*.py`: Python source code files
- `comprehensive_evaluation_results.txt`: Detailed evaluation results
- `training_results.txt`: Training results
- `*.png`: Visualization images
- `*.pkl`: Saved model files

### Dataset Files
- `RGB_images/`: Original RGB images
- `NRG_images/`: Near-infrared images
- `masks/`: Annotation masks

## Contributors

This project is a group project for COMP9517 25T2 course, demonstrating the implementation and comparative analysis of various dead tree segmentation methods.

## License

This project is for academic research purposes only.

---

**Note**: Please ensure the dataset is correctly placed in the project root directory and all dependencies are properly installed.