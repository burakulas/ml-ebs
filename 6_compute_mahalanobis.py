#!/usr/bin/env python3
"""
Compute Mahalanobis distance for OGLE and Kepler predictions
relative to the training feature distribution.

Measures how far each test system's features are from the training
distribution — systems with high distance are out-of-distribution
and predictions are less reliable.

Input files (all in WORK_DIR):
  - training_features.pkl  (dict: 'features' np.array, 'feature_names' list)
  - ogle_features.pkl      (dict: 'ids' list, 'features' DataFrame)  [from 5a]
  - kepler_features.pkl    (dict: 'ids' list, 'features' DataFrame)  [from 5b]
  - ogle_predictions.csv   (predictions with id column)
  - kepler_predictions.csv (predictions with id column)

Output files:
  - ogle_predictions_with_distance.csv
  - kepler_predictions_with_distance.csv
  - mahalanobis_summary.txt
"""

import os
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
WORK_DIR = "."

TRAINING_FEATURES_PATH = os.path.join(WORK_DIR, "processed_data/training_features.pkl")
OGLE_FEATURES_PATH = os.path.join(WORK_DIR, "ogle_features.pkl")
KEPLER_FEATURES_PATH = os.path.join(WORK_DIR, "kepler_features.pkl")
OGLE_PREDICTIONS_PATH = os.path.join(WORK_DIR, "predictions/ogle_predictions/ogle_predictions.csv")
KEPLER_PREDICTIONS_PATH = os.path.join(WORK_DIR, "predictions/kepler_predictions/kepler_predictions.csv")

# =============================================================================
# LOAD TRAINING FEATURES AND COMPUTE DISTRIBUTION
# =============================================================================
def load_training_distribution(path):
    """Load training features and compute mean + inverse covariance.

    Removes constant/near-zero-variance features before computing
    the covariance matrix to avoid singularity issues.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    features = data['features']
    feature_names = list(data['feature_names'])

    # Convert DataFrame to numpy array if needed
    if hasattr(features, 'values'):
        features = features.values

    print(f"Training set: {features.shape[0]} samples, {features.shape[1]} features")

    # Replace NaN/Inf with column medians
    features_clean = features.copy().astype(np.float64)
    for col in range(features_clean.shape[1]):
        col_data = features_clean[:, col]
        mask = np.isfinite(col_data)
        if mask.sum() > 0:
            median_val = np.median(col_data[mask])
            col_data[~mask] = median_val

    # Remove constant / near-zero-variance features
    # (e.g. flux_max is always 1.0 after normalization)
    stds = np.std(features_clean, axis=0)
    keep_mask = stds > 1e-10
    removed_features = [feature_names[i] for i in range(len(feature_names)) if not keep_mask[i]]
    kept_features = [feature_names[i] for i in range(len(feature_names)) if keep_mask[i]]

    if removed_features:
        print(f"Removed {len(removed_features)} constant feature(s): {removed_features}")

    features_clean = features_clean[:, keep_mask]
    print(f"Using {len(kept_features)} features for distance computation")

    # Compute mean and covariance
    mean = np.mean(features_clean, axis=0)
    cov = np.cov(features_clean, rowvar=False)

    # Regularize covariance: ridge = 1e-5 * max eigenvalue
    # Gives condition number ~1e5 — well-conditioned without over-regularizing
    eigvals = np.linalg.eigvalsh(cov)
    ridge = 1e-5 * eigvals.max()
    cov_reg = cov + ridge * np.eye(cov.shape[0])

    # Verify regularization worked
    eigvals_reg = np.linalg.eigvalsh(cov_reg)
    n_negative = (eigvals_reg < 0).sum()
    condition_number = eigvals_reg.max() / eigvals_reg.min()

    print(f"Covariance regularization: ridge = {ridge:.6e}")
    print(f"  Eigenvalue range: [{eigvals_reg.min():.4e}, {eigvals_reg.max():.4e}]")
    print(f"  Condition number: {condition_number:.2e}")
    print(f"  Negative eigenvalues: {n_negative}")

    if n_negative > 0:
        print("WARNING: Negative eigenvalues remain after regularization!")

    try:
        cov_inv = np.linalg.inv(cov_reg)
        print("Covariance matrix inverted successfully.")
    except np.linalg.LinAlgError:
        print("WARNING: Covariance inversion failed, using pseudo-inverse.")
        cov_inv = np.linalg.pinv(cov_reg)

    return mean, cov_inv, kept_features


def compute_mahalanobis(features, mean, cov_inv):
    """Compute Mahalanobis distance for each row in features array."""
    diff = features - mean  # (N, D)
    # d = sqrt( (x-mu)^T * C^-1 * (x-mu) )
    left = diff @ cov_inv  # (N, D)
    distances = np.sqrt(np.sum(left * diff, axis=1))  # (N,)
    return distances


def align_features(test_df, training_feature_names):
    """Align test features to training feature names, filling missing with 0."""
    aligned = test_df.reindex(columns=training_feature_names, fill_value=0.0)

    common = set(test_df.columns) & set(training_feature_names)
    missing = set(training_feature_names) - set(test_df.columns)
    extra = set(test_df.columns) - set(training_feature_names)

    print(f"  Common features: {len(common)}/{len(training_feature_names)}")
    if missing:
        print(f"  Missing (filled with 0): {sorted(missing)}")
    if extra:
        print(f"  Extra (ignored): {sorted(extra)}")

    # Replace NaN/Inf with column medians of finite values
    for col in aligned.columns:
        bad_mask = ~np.isfinite(aligned[col])
        if bad_mask.any():
            finite_vals = aligned.loc[~bad_mask, col]
            median_val = finite_vals.median() if len(finite_vals) > 0 else 0.0
            aligned.loc[bad_mask, col] = median_val if np.isfinite(median_val) else 0.0

    return aligned.values


def process_dataset(features_path, predictions_path, output_path, mean, cov_inv, feature_names, dataset_name):
    """Load features, compute distances, merge with predictions, save."""
    if not os.path.exists(features_path):
        print(f"\n{dataset_name}: features file not found ({features_path}), skipping.")
        return None

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")

    # Load features
    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    ids = data['ids']
    features_df = data['features']
    print(f"Loaded {len(ids)} systems with {features_df.shape[1]} features")

    # Align to training features
    features_aligned = align_features(features_df, feature_names)

    # Compute Mahalanobis distance
    distances = compute_mahalanobis(features_aligned, mean, cov_inv)
    print(f"Distances: min={distances.min():.2f}, median={np.median(distances):.2f}, "
          f"max={distances.max():.2f}, mean={distances.mean():.2f}")

    # Create distance DataFrame
    df_dist = pd.DataFrame({
        'id': ids,
        'mahal_distance': distances
    })

    # Merge with predictions (required — output must contain predictions + D_M)
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}\n"
            f"Place the predictions CSV in WORK_DIR before running."
        )

    df_pred = pd.read_csv(predictions_path)
    df_pred['id'] = df_pred['id'].astype(str)
    df_dist['id'] = df_dist['id'].astype(str)

    df_merged = df_pred.merge(df_dist, on='id', how='left')

    # Compute percentile rank (0-100, lower = closer to training)
    valid_mask = df_merged['mahal_distance'].notna()
    df_merged.loc[valid_mask, 'mahal_percentile'] = (
        df_merged.loc[valid_mask, 'mahal_distance'].rank(pct=True) * 100
    )

    df_merged.to_csv(output_path, index=False)
    print(f"Saved {len(df_merged)} predictions with distance to {output_path}")

    # Summary statistics by quartile
    if valid_mask.sum() > 0:
        df_valid = df_merged[valid_mask].copy()
        df_valid['quartile'] = pd.qcut(df_valid['mahal_distance'], 4,
                                       labels=['Q1 (closest)', 'Q2', 'Q3', 'Q4 (farthest)'])

        print(f"\n  Distance by quartile:")
        for q in ['Q1 (closest)', 'Q2', 'Q3', 'Q4 (farthest)']:
            subset = df_valid[df_valid['quartile'] == q]
            print(f"    {q}: n={len(subset)}, "
                  f"dist=[{subset['mahal_distance'].min():.1f} - {subset['mahal_distance'].max():.1f}], "
                  f"mean_APC={subset['overall_confidence'].mean():.4f}")

    return df_merged


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("MAHALANOBIS DISTANCE CALCULATION")
    print("="*60)

    # Step 1: Load training distribution
    print("\nLoading training feature distribution...")
    mean, cov_inv, feature_names = load_training_distribution(TRAINING_FEATURES_PATH)

    # Step 2: Process OGLE
    ogle_result = process_dataset(
        OGLE_FEATURES_PATH,
        OGLE_PREDICTIONS_PATH,
        os.path.join(WORK_DIR, "ogle_predictions_with_distance.csv"),
        mean, cov_inv, feature_names,
        "OGLE"
    )

    # Step 3: Process Kepler
    kepler_result = process_dataset(
        KEPLER_FEATURES_PATH,
        KEPLER_PREDICTIONS_PATH,
        os.path.join(WORK_DIR, "kepler_predictions_with_distance.csv"),
        mean, cov_inv, feature_names,
        "Kepler"
    )

    # Step 4: Summary report
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    summary_lines = []
    summary_lines.append("Mahalanobis Distance Summary")
    summary_lines.append("="*40)
    summary_lines.append(f"Training set: {mean.shape[0]} features")
    summary_lines.append(f"Features used: {feature_names}")
    summary_lines.append("")

    for name, result in [("OGLE", ogle_result), ("Kepler", kepler_result)]:
        if result is not None and 'mahal_distance' in result.columns:
            d = result['mahal_distance'].dropna()
            summary_lines.append(f"{name} ({len(d)} systems):")
            summary_lines.append(f"  Distance: min={d.min():.2f}, median={d.median():.2f}, "
                               f"max={d.max():.2f}, mean={d.mean():.2f}, std={d.std():.2f}")

            # Percentile thresholds
            for pct in [90, 95, 99]:
                thresh = np.percentile(d, pct)
                n_above = (d > thresh).sum()
                summary_lines.append(f"  {pct}th percentile threshold: {thresh:.2f} "
                                   f"({n_above} systems above)")
            summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(os.path.join(WORK_DIR, "mahalanobis_summary.txt"), 'w') as f:
        f.write(summary_text)
    print(f"\nSummary saved to mahalanobis_summary.txt")


if __name__ == "__main__":
    main()
