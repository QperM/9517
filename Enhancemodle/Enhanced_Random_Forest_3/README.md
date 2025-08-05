# Enhanced Random Forest V3 - Data Sampling Strategy + Ensemble Optimization

This directory contains Enhanced Random Forest V3 implementation for dead tree segmentation in aerial forest images.

## Features

### 1. Advanced Data Sampling Strategies
- **Bootstrap Sampling**: Random sampling with replacement for diversity
- **Stratified Sampling**: Maintains class distribution in samples
- **Random Sampling**: Random sampling without replacement
- **Balanced Sampling**: Upsamples minority class for balanced dataset
- **Max Samples Control**: Uses 80% of samples for each tree

### 2. Ensemble Optimization
- **Multiple Models**: Creates separate models for each sampling strategy
- **Performance Evaluation**: Evaluates each model on validation set
- **Model Selection**: Selects top 2-3 best performing models
- **Soft Voting**: Uses probability predictions for ensemble decisions
- **Ensemble Diversity**: Different random states for model diversity

### 3. Model Architecture
- **Base Models**: Multiple Random Forest classifiers
- **Voting Classifier**: Combines best models with soft voting
- **Validation Split**: Uses 20% of training data for ensemble optimization
- **Model Persistence**: Saves ensemble model and individual models

## Files

- `enhanced_random_forest_v3.py`: Main Enhanced Random Forest V3 implementation
- `README.md`: This documentation file

## Usage

### Quick Start

1. **Install Dependencies**:
```bash
pip install scikit-learn opencv-python matplotlib seaborn albumentations tqdm
```

2. **Run Training**:
```bash
python enhanced_random_forest_v3.py
```

### Expected Results

The enhanced model V3 should show improvements in:
- **Model Diversity**: Different sampling strategies create diverse models
- **Ensemble Stability**: Voting reduces variance and improves stability
- **Generalization**: Better performance on unseen data
- **Robustness**: Less sensitive to data distribution changes

## Comparison with Baseline

| Aspect | Baseline RF | Enhanced RF V3 |
|--------|-------------|----------------|
| Models | Single RF | Multiple RF + Voting |
| Sampling | Standard | Multiple strategies |
| Ensemble | None | Soft voting ensemble |
| Validation | None | Separate validation set |
| Diversity | Low | High (different sampling) |

## Technical Details

### Sampling Strategies Process
1. **Bootstrap Sampling**: Creates diverse datasets with replacement
2. **Stratified Sampling**: Maintains class balance in samples
3. **Random Sampling**: Simple random subset for diversity
4. **Balanced Sampling**: Addresses class imbalance issues

### Ensemble Optimization Process
1. **Model Creation**: Trains separate RF for each sampling strategy
2. **Performance Evaluation**: Tests each model on validation set
3. **Model Selection**: Ranks models by F1 score and selects top performers
4. **Ensemble Creation**: Combines best models with soft voting

### Expected Performance
- **IoU Improvement**: 3-8% improvement over baseline
- **Training Time**: 3-8 minutes (multiple models)
- **Ensemble Size**: 2-4 models in final ensemble
- **Stability**: Reduced variance in predictions

## Sampling Strategies Explained

### 1. Bootstrap Sampling
- **Purpose**: Creates diverse training sets
- **Method**: Random sampling with replacement
- **Benefit**: Increases model diversity

### 2. Stratified Sampling
- **Purpose**: Maintains class distribution
- **Method**: Stratified split with 70% training
- **Benefit**: Better representation of minority class

### 3. Random Sampling
- **Purpose**: Simple diversity through random subset
- **Method**: Random 80% of samples without replacement
- **Benefit**: Reduces overfitting

### 4. Balanced Sampling
- **Purpose**: Addresses class imbalance
- **Method**: Upsamples minority class to match majority
- **Benefit**: Better performance on minority class

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce n_estimators or number of sampling strategies
2. **Slow Training**: Disable ensemble optimization for faster training
3. **Imbalanced Data**: Ensure balanced sampling is enabled

### Performance Tips
1. **Faster Training**: Set use_ensemble_optimization=False
2. **More Models**: Increase number of sampling strategies
3. **Quick Testing**: Use fewer sampling strategies

## Future Enhancements

1. **Advanced Sampling**: SMOTE, ADASYN for better balancing
2. **Dynamic Ensemble**: Adaptive ensemble size based on performance
3. **Cross-validation**: K-fold CV for more robust evaluation
4. **Model Interpretability**: Individual model analysis and visualization 