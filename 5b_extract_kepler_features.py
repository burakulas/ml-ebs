#!/usr/bin/env python3
"""
Extract features from Kepler light curves and save as pickle.
Stripped from 4b_kepler_prediction.py — no model loading, no predictions.

Input:  Kepler light curve files (.csv/.dat) in KEPLER_LC_DIR
Output: kepler_features.pkl  (dict with 'ids' and 'features' DataFrame)
"""

import os
import numpy as np
import pandas as pd
import pickle
import warnings
from scipy.fft import fft
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

# =============================================================================
# CONFIGURATION — UPDATE THESE PATHS
# =============================================================================
WORK_DIR = "."
KEPLER_LC_DIR = os.path.join(WORK_DIR, "kepler_data")
OUTPUT_PATH = os.path.join(WORK_DIR, "kepler_features.pkl")

BATCH_SIZE = 1000

# =============================================================================
# ROBUST LOADER (identical to 4b — Kepler headerless version)
# =============================================================================
def load_and_bin_lc_robust(filepath, n_bins=1000):
    try:
        df = pd.read_csv(filepath, header=None)

        if df.shape[1] < 2:
            return None

        raw_phase = df.iloc[:, 0].values
        raw_flux = df.iloc[:, 1].values
        phase_folded = raw_phase % 1.0

        bin_edges = np.linspace(0, 1.0, n_bins + 1)
        flux_binned = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (phase_folded >= bin_edges[i]) & (phase_folded < bin_edges[i+1])
            flux_binned[i] = np.median(raw_flux[mask]) if np.any(mask) else np.nan

        s = pd.Series(flux_binned)
        flux_filled = s.interpolate(method='linear', limit_direction='both').values

        try:
            flux_smooth = savgol_filter(flux_filled, 31, 3, mode='wrap')
        except:
            flux_smooth = flux_filled

        phase_grid = np.linspace(0, 1, n_bins, endpoint=False)
        ext_phase = np.concatenate([phase_grid - 1.0, phase_grid, phase_grid + 1.0])
        ext_flux = np.concatenate([flux_smooth, flux_smooth, flux_smooth])

        interp_func = PchipInterpolator(ext_phase, ext_flux)
        target_grid = np.linspace(0.25, 1.25, n_bins)
        flux_final = interp_func(target_grid)

        f_max = np.nanmax(flux_final)
        if f_max <= 0 or np.isnan(f_max):
            return None

        return flux_final / f_max

    except Exception:
        return None

# =============================================================================
# FEATURE EXTRACTION (identical to 2_extract_training_features.py — 51 features)
# =============================================================================
def extract_features(flux):
    features = {}

    phase = np.linspace(0.25, 1.25, len(flux), endpoint=False)

    baseline_flux = np.median(flux)
    features['baseline'] = baseline_flux
    features['flux_min'] = np.min(flux)
    features['flux_max'] = np.max(flux)
    features['flux_mean'] = np.mean(flux)
    features['flux_std'] = np.std(flux)
    features['flux_range'] = features['flux_max'] - features['flux_min']
    features['flux_skew'] = pd.Series(flux).skew()
    features['flux_kurtosis'] = pd.Series(flux).kurtosis()

    flux_smooth = uniform_filter1d(flux, size=5, mode='wrap')
    min1_idx = np.argmin(flux_smooth)
    min1_phase = phase[min1_idx]
    min1_depth = baseline_flux - flux[min1_idx]

    mask_width = int(0.15 * len(flux))
    flux_masked = flux_smooth.copy()
    for ii in range(-mask_width, mask_width):
        idx = (min1_idx + ii) % len(flux)
        flux_masked[idx] = np.nan

    if not np.all(np.isnan(flux_masked)):
        min2_idx = np.nanargmin(flux_masked)
        min2_phase = phase[min2_idx]
        min2_depth = baseline_flux - flux[min2_idx]
    else:
        min2_idx = min1_idx
        min2_phase = min1_phase
        min2_depth = 0

    if min1_depth >= min2_depth:
        primary_depth, secondary_depth = min1_depth, min2_depth
        primary_phase, secondary_phase = min1_phase, min2_phase
    else:
        primary_depth, secondary_depth = min2_depth, min1_depth
        primary_phase, secondary_phase = min2_phase, min1_phase

    features['primary_depth'] = primary_depth
    features['secondary_depth'] = secondary_depth
    features['primary_width'] = np.sum(flux < (baseline_flux - 0.5 * primary_depth)) / len(flux)
    features['secondary_width'] = np.sum(flux < (baseline_flux - 0.5 * secondary_depth)) / len(flux)

    if abs(primary_phase - secondary_phase) > 0.5:
        mid1_phase = (primary_phase + secondary_phase) / 2
        mid2_phase = mid1_phase + 0.5
    else:
        mid1_phase = primary_phase + 0.25
        mid2_phase = primary_phase + 0.75

    mid1_phase = ((mid1_phase - 0.25) % 1.0) + 0.25
    mid2_phase = ((mid2_phase - 0.25) % 1.0) + 0.25

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

    n_harmonics = 10
    flux_centered = flux - np.mean(flux)
    fft_vals = fft(flux_centered)
    n = len(flux_centered)

    for k in range(1, n_harmonics + 1):
        features[f'fourier_amp_{k}'] = 2.0 * np.abs(fft_vals[k]) / n

    for k in range(1, 4):
        features[f'fourier_phase_{k}'] = np.angle(fft_vals[k])

    if features['fourier_amp_1'] > 0:
        for k in [2, 3, 4]:
            features[f'fourier_ratio_{k}_1'] = features[f'fourier_amp_{k}'] / features['fourier_amp_1']
    else:
        for k in [2, 3, 4]:
            features[f'fourier_ratio_{k}_1'] = 0

    features['fourier_total_power'] = np.sum([features[f'fourier_amp_{k}']**2 for k in range(1, n_harmonics + 1)])

    n_bins = 10
    bin_edges = np.linspace(0.25, 1.25, n_bins + 1)
    bin_means = []

    for ii in range(n_bins):
        mask = (phase >= bin_edges[ii]) & (phase < bin_edges[ii+1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(flux[mask]))
        else:
            bin_means.append(baseline_flux)

    for ii in range(n_bins):
        features[f'phase_bin_{ii+1}_mean'] = bin_means[ii]

    for ii in range(n_bins - 1):
        features[f'phase_bin_diff_{ii+1}'] = bin_means[ii+1] - bin_means[ii]

    return features

# =============================================================================
# MAIN
# =============================================================================
def main():
    all_files = [f for f in os.listdir(KEPLER_LC_DIR) if f.lower().endswith(('.csv', '.dat'))]
    print(f"Extracting features from {len(all_files)} Kepler files...")

    all_ids = []
    all_features = []

    for i in tqdm(range(0, len(all_files), BATCH_SIZE), desc="Batches"):
        batch_files = all_files[i:i + BATCH_SIZE]

        for f_name in batch_files:
            flux = load_and_bin_lc_robust(os.path.join(KEPLER_LC_DIR, f_name))
            if flux is not None:
                all_features.append(extract_features(flux))
                all_ids.append(os.path.splitext(f_name)[0])

    if all_features:
        df_features = pd.DataFrame(all_features)
        output = {
            'ids': all_ids,
            'features': df_features
        }
        with open(OUTPUT_PATH, 'wb') as f:
            pickle.dump(output, f)
        print(f"Saved {len(all_ids)} feature vectors ({df_features.shape[1]} features each) to {OUTPUT_PATH}")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()
