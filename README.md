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

Execute the combined script to run data preparation, feature extraction, and training sequentially. `models/held_out_data.pkl` is included in this repository. When present, `3_train_models.py` loads it directly to ensure the exact same 845/150 train/test split used in the paper. If deleted, a fresh stratified split will be created automatically.

```bash
python 123_extract_and_train.py
```

This script automatically runs:
1. `1_prepare_training_data.py` - Preprocesses light curves
2. `2_extract_training_features.py` - Extracts 51 features per light curve
3. `3_train_models.py` - Trains RF and XGBoost models with 5-fold CV

### Option 2: Run Scripts Separately

If you need more control or want to modify individual steps:

```bash
# Step 1: Prepare training data (PCHIP interpolation to 1000 points)
python 1_prepare_training_data.py

# Step 2: Extract features (51 features per light curve)
python 2_extract_training_features.py

# Step 3: Train models (RF and XGBoost, 5-fold CV on 845 systems, 150 held out)
python 3_train_models.py
```

### Make Predictions on New Data

After training, predict parameters for OGLE, Kepler, or custom light curves. Prediction scripts use GPU acceleration (CuPy) if available, falling back to CPU otherwise. Refer to the `download.txt` files in [ogle_data](https://github.com/burakulas/ml-ebs/tree/main/ogle_data) and [kepler_data](https://github.com/burakulas/ml-ebs/tree/main/kepler_data) to access the necessary datasets.

```bash
# Predict OGLE catalog
python 4a_ogle_prediction.py

# Predict Kepler catalog
python 4b_kepler_prediction.py

# Predict custom light curves
python 4c_custom_prediction.py
```

For custom predictions, place your CSV files in the [custom_data/](https://github.com/burakulas/ml-ebs/tree/main/custom_data) folder. Each file must have two columns: `phase` and `flux` (with header row).

### Compute Mahalanobis Distance

After predictions, you can assess prediction reliability by computing Mahalanobis distance. This measures how far each system's features are from the training distribution; systems with high distance are out-of-distribution and predictions may be less reliable.

```bash
# Step 1: Extract features for distance computation
python 5a_extract_ogle_features.py
python 5b_extract_kepler_features.py

# Step 2: Compute distances and merge with predictions
python 6_compute_mahalanobis.py
```
### Evaluate on Held-Out Test Set                                                                                                          
                                                                                                                                             
After training, you can evaluate model performance on the 150-system held-out test set. This script loads the trained RF and XGBoost models, makes predictions on the held-out data, and reports R² scores and classification metrics.         
                                                                                                                                             
```bash                        
python 5_held_out_evaluation.py
```
## Output Files

### Training Pipeline

- `processed_data/training_data.pkl` - Preprocessed light curves
- `processed_data/training_features.pkl` - Extracted features
- `models/models_rf/` - Random Forest models (5 folds x 6 tasks)
- `models/models_xgb/` - XGBoost models (5 folds x 6 tasks)
- `models/models_rf/rf_cv_summary.csv` - Cross-validation results (RF)
- `models/models_xgb/xgb_cv_summary.csv` - Cross-validation results (XGB)
- `models/held_out_data.pkl` - Held-out test set (150 systems, pre-defined for reproducibility)
- `models/held_out_evaluation_results.csv` - Held-out R² and classification metrics

### Predictions

- `predictions/ogle_predictions/ogle_predictions.csv`
- `predictions/kepler_predictions/kepler_predictions.csv`
- `predictions/custom_predictions/custom_predictions.csv`

### Mahalanobis Distance

- `ogle_features.pkl` - OGLE feature vectors
- `kepler_features.pkl` - Kepler feature vectors
- `ogle_predictions_with_distance.csv` - OGLE predictions with Mahalanobis distance
- `kepler_predictions_with_distance.csv` - Kepler predictions with Mahalanobis distance
- `mahalanobis_summary.txt` - Distance summary statistics

## License

MIT License - See licence.txt for details.

## Contact

For questions or issues, please open a GitHub issue or contact [burak.ulas@comu.edu.tr].

