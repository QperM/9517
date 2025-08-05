# Enhanced Random Forest V2 - Feature Selection + Hyperparameter Tuning

This directory contains Enhanced Random Forest V2 implementation for dead tree segmentation in aerial forest images.

## Features

### 1. Advanced Feature Selection
- **SelectKBest**: Uses f_classif to select top k features based on statistical tests
- **Recursive Feature Elimination (RFE)**: Iteratively removes least important features
- **Variance Threshold**: Removes low-variance features
- **Feature Importance Analysis**: Shows top 10 most important features

### 2. Hyperparameter Tuning
- **RandomizedSearchCV**: Efficient random search for optimal parameters
- **Parameter Grid**:
  - n_estimators: [50, 100, 150, 200]
  - max_depth: [5, 10, 15, 20, None]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]
  - max_features: ['sqrt', 'log2', None]
  - bootstrap: [True, False]
- **Cross-validation**: 3-fold cross-validation
- **Scoring**: F1-weighted score for optimization

### 3. Model Architecture
- **Base Model**: Random Forest Classifier
- **Feature Selection**: Automatic selection of 100 best features
- **Hyperparameter Optimization**: 20 iterations of random search
- **Model Persistence**: Saves optimized model and feature selector

## Files

- `enhanced_random_forest_v2.py`: Main Enhanced Random Forest V2 implementation
- `README.md`: This documentation file

## Usage

### Quick Start

1. **Install Dependencies**:
```bash
pip install scikit-learn opencv-python matplotlib seaborn albumentations tqdm
```

2. **Run Training**:
```bash
python enhanced_random_forest_v2.py
```

### Expected Results

The enhanced model V2 should show improvements in:
- **Feature Efficiency**: Reduced feature dimensionality while maintaining performance
- **Model Optimization**: Better hyperparameters for improved accuracy
- **Training Speed**: Faster training with selected features
- **Prediction Accuracy**: Optimized model parameters

## Comparison with Baseline

| Aspect | Baseline RF | Enhanced RF V2 |
|--------|-------------|----------------|
| Features | All features | Selected 100 features |
| Hyperparameters | Default | Optimized via search |
| Feature Selection | None | SelectKBest + RFE |
| Cross-validation | None | 3-fold CV |
| Optimization | None | RandomizedSearchCV |

## Technical Details

### Feature Selection Process
1. **Statistical Testing**: Uses f_classif to rank features by statistical significance
2. **Top-K Selection**: Selects top 100 features from original feature set
3. **Feature Importance**: Analyzes and displays top 10 most important features

### Hyperparameter Tuning Process
1. **Parameter Grid**: Defines search space for 6 key parameters
2. **Random Search**: Performs 20 iterations of random parameter combinations
3. **Cross-validation**: Uses 3-fold CV to evaluate each parameter set
4. **Best Model**: Selects model with highest F1-weighted score

### Expected Performance
- **IoU Improvement**: 5-10% improvement over baseline
- **Training Time**: 2-5 minutes (including hyperparameter tuning)
- **Feature Reduction**: ~95% reduction in feature count
- **Accuracy**: Maintains or improves accuracy with fewer features

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce n_iter in RandomizedSearchCV
2. **Slow Training**: Disable hyperparameter tuning for faster training
3. **Feature Selection Errors**: Ensure sufficient samples for statistical tests

### Performance Tips
1. **Faster Training**: Set use_hyperparameter_tuning=False
2. **More Features**: Increase n_features in feature_selection()
3. **Quick Testing**: Reduce n_iter in hyperparameter tuning

## Future Enhancements

1. **Advanced Feature Selection**: Mutual information, correlation analysis
2. **Bayesian Optimization**: More efficient hyperparameter search
3. **Feature Engineering**: Domain-specific feature creation
4. **Model Interpretability**: SHAP values, feature importance visualization 