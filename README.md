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
2. `2_extract_training_features.py` - Extracts 52 features
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

After training, predict parameters for Kepler or OGLE light curves:

```bash
# Predict Kepler catalog
python 4_kepler_prediction.py

# Predict OGLE catalog
python 4a_ogle_prediction.py
```

**Note:** Prediction scripts use GPU acceleration (CuPy) if available, falling back to CPU otherwise.

## Training Data Format

Light curve files should be in `.dat` format with parameters encoded in filenames:

**Example filename:**
```
SystemName_m_3_i_82.4_q_0.481_t1_3664_t2_3700_p1_2.808_p2_2.808.dat
```

**File format (space-separated):**
```
phase1 flux1
phase2 flux2
...
```

**Required naming convention:**
- `m_X` - Morphology mode (2=detached, 3=contact, 5=semidetached)
- `i_XX.XX` - Inclination (degrees)
- `q_X.XXX` - Mass ratio
- `t1_XXXX` - Primary temperature (K or normalized)
- `t2_XXXX` - Secondary temperature (K or normalized)
- `p1_X.XXX` - Primary potential
- `p2_X.XXX` - Secondary potential

## Model Performance

### Regression (XGBoost, 5-fold CV)

| Parameter | R² | MAE |
|-----------|-----|-----|
| t2_t1 (temp ratio) | 0.914 ± 0.019 | 0.032 |
| p1 (primary Ω) | 0.896 ± 0.051 | 0.52 |
| p2 (secondary Ω) | 0.875 ± 0.077 | 0.54 |
| q (mass ratio) | 0.846 ± 0.044 | 0.069 |
| i (inclination) | 0.836 ± 0.030 | 1.83° |

### Classification (XGBoost)

- **Overall Accuracy:** 95.08% ± 1.07%
- **F1 Score (macro):** 94.53% ± 1.21%

**Per-class F1 scores:**
- Detached: 0.964 (421 systems)
- Contact: 0.965 (340 systems)
- Semidetached: 0.908 (234 systems)

## Feature Engineering

The pipeline extracts **52 features** per light curve:

1. **Eclipse Features (24 features)**
   - Primary/secondary eclipse depths, widths, asymmetries
   - Eclipse duration, flatness, sharpness
   - O'Connell effect

2. **Fourier Features (14 features)**
   - 10 harmonic amplitudes (A₁-A₁₀)
   - 3 harmonic phases (φ₁-φ₃)
   - Total Fourier power

3. **Phase-Binned Statistics (10 features)**
   - 10 phase bin means ([0.25, 1.25] range)

4. **Statistical Features (4 features)**
   - Mean, median, standard deviation, range

## Output Files

### Training Pipeline Outputs

- `processed_data/training_data.pkl` - Preprocessed light curves (1000 points each)
- `processed_data/training_features.pkl` - Extracted features (995 × 52)
- `models/models_rf/` - Random Forest models (5 folds × 6 tasks)
- `models/models_xgb/` - XGBoost models (5 folds × 6 tasks)
- `models/models_rf/rf_cv_summary.csv` - Cross-validation results (RF)
- `models/models_xgb/xgb_cv_summary.csv` - Cross-validation results (XGB)

### Prediction Outputs

- `predictions/kepler_predictions/kepler_predictions_with_confidence.csv` - Full Kepler catalog predictions
- `predictions/ogle_predictions/ogle_predictions_with_confidence.csv` - Full OGLE catalog predictions

**Prediction CSV columns:**
- System identifier (KIC/OGLE ID)
- Predicted parameters: `i_pred`, `q_pred`, `t2_t1_pred`, `p1_pred`, `p2_pred`
- Predicted morphology: `morphology_pred` (detached/contact/semidetached)
- Confidence scores: `confidence_rf_xgb`, `feature_quality`, `combined_confidence`

## Methodology

### Data Preprocessing
- Phase range normalized to [0.25, 1.25] (primary eclipse at phase 1.0)
- PCHIP interpolation to uniform 1000-point grid
- Flux normalized to maximum = 1.0

### Training Strategy
- 5-fold stratified cross-validation (by morphology)
- Feature standardization (StandardScaler) per fold
- Independent models for each parameter (no error propagation)
- Ensemble averaging across 5 folds for final predictions

### Algorithms
- **Random Forest:** 500 trees, max_depth=25, baseline model
- **XGBoost:** 500 trees, max_depth=8, learning_rate=0.05, L1/L2 regularization

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourlastname2026,
  title={Learning from the Best: A Multi-Task Machine Learning Framework for Eclipsing Binary Parameter Estimation Using Well-Characterized Systems},
  author={Your Name et al.},
  journal={Astronomy and Computing},
  year={2026},
  note={In preparation}
}
```

## Known Limitations

1. **Inclination bias:** Training data heavily biased toward edge-on systems (mean i = 82°, std = 7.6°). Predictions for low-inclination systems (i < 70°) may be unreliable.

2. **Temperature range:** Training covers 3,160-60,000 K. Extreme temperatures outside this range require caution.

3. **Data quality dependency:** Models assume well-sampled light curves with both eclipses visible. Single-eclipse or sparse data may produce poor results.

4. **Phase range assumption:** Input light curves must use phase range [0.25, 1.25] with primary eclipse at phase 1.0.

## Troubleshooting

**Import errors:**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- For GPU acceleration: Install CuPy matching your CUDA version

**File not found errors:**
- Check that `training_data/` directory contains `.dat` files
- Verify filename format matches expected pattern

**Memory errors:**
- Reduce batch size in prediction scripts
- For large catalogs, process in chunks

**Poor predictions:**
- Verify input phase range is [0.25, 1.25]
- Check that light curves have both eclipses visible
- Ensure parameter values within training range

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@domain.com].

## Acknowledgments

This work uses data from:
- CALEB (Catalog of Eclipsing Binaries)
- DEBCat (Detached Eclipsing Binary Catalog)
- Kepler Mission
- OGLE Survey

Models trained on 995 well-characterized eclipsing binary systems compiled from literature sources.

---

**Last Updated:** January 2026
