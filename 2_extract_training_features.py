#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pickle
from scipy.fft import fft
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR = "."
INPUT_FILE = os.path.join(WORK_DIR, "processed_data/training_data.pkl")
OUTPUT_FILE = os.path.join(WORK_DIR, "processed_data/training_features.pkl")

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(flux):
    """
    Extract features with DYNAMIC eclipse detection.
    Works regardless of phase convention - finds eclipses by depth.
    """

    features = {}

    # Implicit phase grid
    phase = np.linspace(0.25, 1.25, len(flux), endpoint=False)

    # Basic statistics
    baseline_flux = np.median(flux)
    features['baseline'] = baseline_flux
    features['flux_min'] = np.min(flux)
    features['flux_max'] = np.max(flux)
    features['flux_mean'] = np.mean(flux)
    features['flux_std'] = np.std(flux)
    features['flux_median'] = np.median(flux)
    features['flux_range'] = features['flux_max'] - features['flux_min']
    features['flux_skew'] = pd.Series(flux).skew()
    features['flux_kurtosis'] = pd.Series(flux).kurtosis()

    # DYNAMIC ECLIPSE DETECTION
    # Smooth flux to avoid noise spikes
    flux_smooth = uniform_filter1d(flux, size=5, mode='wrap')

    # Find first minimum (deepest point)
    min1_idx = np.argmin(flux_smooth)
    min1_phase = phase[min1_idx]
    min1_depth = baseline_flux - flux[min1_idx]

    # Mask out region around first minimum (±0.15 phase units)
    mask_width = int(0.15 * len(flux))  # 15% of light curve
    flux_masked = flux_smooth.copy()

    # Handle wrap-around masking
    for i in range(-mask_width, mask_width):
        idx = (min1_idx + i) % len(flux)
        flux_masked[idx] = np.nan

    # Find second minimum
    if not np.all(np.isnan(flux_masked)):
        min2_idx = np.nanargmin(flux_masked)
        min2_phase = phase[min2_idx]
        min2_depth = baseline_flux - flux[min2_idx]
    else:
        # If masking removed everything, no clear second eclipse
        min2_idx = min1_idx
        min2_phase = min1_phase
        min2_depth = 0

    # Primary = deeper eclipse, Secondary = shallower
    if min1_depth >= min2_depth:
        primary_idx, secondary_idx = min1_idx, min2_idx
        primary_depth, secondary_depth = min1_depth, min2_depth
        primary_phase, secondary_phase = min1_phase, min2_phase
    else:
        primary_idx, secondary_idx = min2_idx, min1_idx
        primary_depth, secondary_depth = min2_depth, min1_depth
        primary_phase, secondary_phase = min2_phase, min1_phase

    features['primary_depth'] = primary_depth
    features['secondary_depth'] = secondary_depth

    # Eclipse widths - fraction of curve below half-depth
    primary_half_depth = baseline_flux - 0.5 * primary_depth
    secondary_half_depth = baseline_flux - 0.5 * secondary_depth

    features['primary_width'] = np.sum(flux < primary_half_depth) / len(flux)
    features['secondary_width'] = np.sum(flux < secondary_half_depth) / len(flux)

    # O'Connell effect - compare maxima between eclipses
    # Find phase midpoint between eclipses
    if abs(primary_phase - secondary_phase) > 0.5:
        # Eclipses are ~0.5 apart (typical)
        mid1_phase = (primary_phase + secondary_phase) / 2
        mid2_phase = mid1_phase + 0.5
    else:
        # Use quadrature phases
        mid1_phase = primary_phase + 0.25
        mid2_phase = primary_phase + 0.75

    # Wrap to [0.25, 1.25]
    mid1_phase = ((mid1_phase - 0.25) % 1.0) + 0.25
    mid2_phase = ((mid2_phase - 0.25) % 1.0) + 0.25

    # Extract maxima near these phases (±0.1 phase units)
    window = 0.1
    max1_mask = np.abs(phase - mid1_phase) < window
    max2_mask = np.abs(phase - mid2_phase) < window

    if np.sum(max1_mask) > 0 and np.sum(max2_mask) > 0:
        max1 = np.max(flux[max1_mask])
        max2 = np.max(flux[max2_mask])
        features['oconnell_effect'] = abs(max1 - max2)
        features['ooe_range'] = max(max1, max2) - min(max1, max2)
        features['ooe_std'] = np.std([max1, max2])
    else:
        features['oconnell_effect'] = 0
        features['ooe_range'] = 0
        features['ooe_std'] = 0

    # Fourier
    n_harmonics = 10
    flux_centered = flux - np.mean(flux)
    fft_vals = fft(flux_centered)
    n = len(flux_centered)

    for k in range(1, n_harmonics + 1):
        amp = 2.0 * np.abs(fft_vals[k]) / n
        features[f'fourier_amp_{k}'] = amp

    for k in range(1, 4):
        features[f'fourier_phase_{k}'] = np.angle(fft_vals[k])

    if features['fourier_amp_1'] > 0:
        for k in [2, 3, 4]:
            features[f'fourier_ratio_{k}_1'] = features[f'fourier_amp_{k}'] / features['fourier_amp_1']
    else:
        for k in [2, 3, 4]:
            features[f'fourier_ratio_{k}_1'] = 0

    features['fourier_total_power'] = np.sum([features[f'fourier_amp_{k}']**2 for k in range(1, n_harmonics + 1)])

    # Phase binning
    n_bins = 10
    bin_edges = np.linspace(0.25, 1.25, n_bins + 1)
    bin_means = []

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(flux[mask]))
        else:
            bin_means.append(baseline_flux)

    for i in range(n_bins):
        features[f'phase_bin_{i+1}_mean'] = bin_means[i]

    for i in range(n_bins - 1):
        features[f'phase_bin_diff_{i+1}'] = bin_means[i+1] - bin_means[i]

    return features


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("SCRIPT 02: EXTRACT TRAINING FEATURES")
    print("=" * 80)

    # Load data
    print(f"\nLoading: {INPUT_FILE}")
    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)

    lcs = data['lcs']
    params_df = data['params']
    phase_range = data['phase_range']

    print(f"Loaded {len(lcs)} light curves")
    print(f"Phase range: {phase_range}")

    # Extract features
    print("\nExtracting features...")
    print("  Primary eclipse: ~1.0")
    print("  Secondary eclipse: ~0.5")

    features_list = []
    for flux in tqdm(lcs):
        features = extract_features(flux)
        features_list.append(features)

    X = pd.DataFrame(features_list)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Features (first 10): {list(X.columns[:10])}")

    # Save
    output_data = {
        'features': X,
        'params': params_df,
        'feature_names': list(X.columns),
        'phase_range': phase_range
    }

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n" + "=" * 80)
    print(f"SAVED: {OUTPUT_FILE}")
    print("=" * 80)

    # Statistics
    print(f"\nFeature statistics:")
    print(X[['primary_depth', 'secondary_depth', 'flux_mean', 'flux_std']].describe().T)


if __name__ == "__main__":
    main()
