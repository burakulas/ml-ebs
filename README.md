# ML-EBS: Machine Learning for Eclipsing Binary Stars

A machine learning framework for predicting physical parameters and morphological classification of eclipsing binary stars from photometric light curves.


## Installation

### Requirements

- Python 3.7+
- 8GB+ RAM recommended
- Optional: CUDA-compatible GPU for faster training and predictions (CuPy)

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

Execute the combined script to run data preparation, feature extraction, and training sequentially (models/held_out_data.pkl is included in this repository. When present, 3_train_models.py loads it directly to ensure the exact same 845/150 train/test split used in the paper. If deleted, a fresh stratified split will be created automatically).:

```bash
python 123_extract_and_train.py
```

This script automatically runs:
1. `1_prepare_training_data.py` - Preprocesses light curves
2. `2_extract_training_features.py` - Extracts features
3. `3_train_models.py` - Trains RF and XGBoost models with 5-fold CV

To additionally reproduce the held-out test results and figures from the paper:                                         
`python 5_held_out_evaluation.py`

### Option 2: Run Scripts Separately (Alternative)

If you need more control or want to modify individual steps:

```bash
# Step 1: Prepare training data (PCHIP interpolation to 1000 points)
python 1_prepare_training_data.py

# Step 2: Extract features (51 features per light curve)
python 2_extract_training_features.py

# Step 3: Train models (Trains RF and XGBoost models with 5-fold CV on 845 systems - 150 held out for final evaluation)
python 3_train_models.py

# Step 4: Evaluate on held-out test set                                 
python 5_held_out_evaluation.py
```

### Make Predictions on New Data

After training, you can predict parameters for OGLE and Kepler light curves. Please refer to the download.txt files located in the [ogle_data](https://github.com/burakulas/ml-ebs/tree/main/ogle_data) and [kepler_data](https://github.com/burakulas/ml-ebs/tree/main/kepler_data) directories to access the necessary datasets.

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
- 'models/held_out_data.pkl'  - Held-out test set (150 systems, pre-defined for reproducibility)

### Prediction Outputs

- `predictions/kepler_predictions/kepler_predictions.csv` 
- `predictions/ogle_predictions/ogle_predictions.csv` 
- `predictions/custom_predictions/custom_predictions.csv`


## License

MIT License - See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [burak.ulas@comu.edu.tr].

