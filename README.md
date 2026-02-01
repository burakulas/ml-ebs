# ML-EBS: Machine Learning for Eclipsing Binary Stars

A machine learning framework for predicting physical parameters and morphological classification of eclipsing binary stars from photometric light curves.


## Installation

### Requirements

- Python 3.7+
- 8GB+ RAM recommended
- Optional: CUDA-compatible GPU for faster Kepler/OGLE predictions (CuPy)

### Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration (optional):
```bash
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

Execute the combined script to run data preparation, feature extraction, and training sequentially:

```bash
python 123_extract_and_train.py
```

This script automatically runs:
1. `1_prepare_training_data.py` - Preprocesses light curves
2. `2_extract_training_features.py` - Extracts features
3. `3_train_models.py` - Trains RF and XGBoost models with 5-fold CV


### Option 2: Run Scripts Separately (Alternative)

If you need more control or want to modify individual steps:

```bash
# Step 1: Prepare training data (PCHIP interpolation to 1000 points)
python 1_prepare_training_data.py

# Step 2: Extract features (52 features per light curve)
python 2_extract_training_features.py

# Step 3: Train models (5-fold cross-validation)
python 3_train_models.py
```

### Make Predictions on New Data

After training, you can predict parameters for Kepler or OGLE light curves. Please refer to the download.txt files located in the [ogle_data](https://github.com/burakulas/ml-ebs/tree/main/ogle_data) and [kepler_data](https://github.com/burakulas/ml-ebs/tree/main/kepler_data) directories to access the necessary datasets.

```bash
# Predict OGLE catalog
python 4a_ogle_prediction.py

# Predict Kepler catalog
python 4b_kepler_prediction.py
```
Run the following command to predict parameters from custom light curve data. Ensure your CSV files are located in the [custom_data/](https://github.com/burakulas/ml-ebs/tree/main/custom_data) folder with two columns (phase and flux) and a phase range of 0.25 to 1.25. Prediction scripts use GPU acceleration (CuPy) if available, falling back to CPU otherwise.

```bash
python 4c_custom_prediction.py
```



## Output Files

### Training Pipeline Outputs

- `processed_data/training_data.pkl` - Preprocessed light curves
- `processed_data/training_features.pkl` - Extracted features
- `models/models_rf/` - Random Forest models (5 folds × 6 tasks)
- `models/models_xgb/` - XGBoost models (5 folds × 6 tasks)
- `models/models_rf/rf_cv_summary.csv` - Cross-validation results (RF)
- `models/models_xgb/xgb_cv_summary.csv` - Cross-validation results (XGB)

### Prediction Outputs

- `predictions/kepler_predictions/kepler_predictions.csv` 
- `predictions/ogle_predictions/ogle_predictions.csv` 
- `predictions/custom_predictions/custom_predictions.csv`


## License

MIT License - See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [burak.ulas@comu.edu.tr].

