#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pickle
import warnings
from scipy.fft import fft
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from tqdm import tqdm

# GPU support: use cupy if available, otherwise fall back to numpy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU (cupy) detected — running on CUDA.")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("cupy not found — running on CPU.")

# Suppress noise
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
WORK_DIR = "."
OGLE_LC_DIR = os.path.join(WORK_DIR, "ogle_data")
MODEL_DIR = os.path.join(WORK_DIR, "models/models_xgb")
OUTPUT_DIR = os.path.join(WORK_DIR, "predictions/ogle_predictions")

PARAMS_TO_PREDICT = ['i', 't2_t1', 'q', 'p1', 'p2']
CLASSIFICATION_PARAM = 'morphology_classifier'

N_FOLDS = 5
BATCH_SIZE = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. PHYSICS HELPERS
# =============================================================================
def calculate_omega_in(q):
    def force_eq(x, q):
        if x <= 0 or x >= 1: return np.inf
        return (1.0 / x**2) - (q / (1.0 - x)**2) - ((1.0 + q) * x)
    try:
        x_L1 = brentq(force_eq, 1e-5, 1.0 - 1e-5, args=(q,))
        r1, r2 = x_L1, 1.0 - x_L1
        return (1.0 / r1) + (q / r2) + 0.5 * (1.0 + q) * (x_L1**2)
    except:
        return None

def apply_physics_constraints(star):
    m_code = star.get('morphology', -1)
    q = star.get('q', np.nan)
    if np.isnan(q) or q <= 0: return star

    omega_in = calculate_omega_in(q)
    if omega_in is None: return star

    if m_code == 3:  # CONTACT
        star['p1'] = omega_in
        star['p2'] = omega_in
        star['physics_note'] = "Contact: Forced p1=p2=Omega_in"
    elif m_code == 5:  # SEMI-DETACHED
        star['p2'] = omega_in
        if star['p1'] <= omega_in:
            star['p1'] = omega_in * 1.05
            star['physics_note'] = "Semi-Det: Fixed p2, Forced p1 detached"
        else:
            star['physics_note'] = "Semi-Det: Fixed p2"
    elif m_code == 2:  # DETACHED
        updated = False
        if star['p1'] <= omega_in:
            star['p1'] = omega_in * 1.05; updated = True
        if star['p2'] <= omega_in:
            star['p2'] = omega_in * 1.05; updated = True
        star['physics_note'] = "Detached: Constraints applied" if updated else "Detached: OK"

    return star

# =============================================================================
# 2. ROBUST LOADER
# =============================================================================
def load_and_bin_lc_robust(filepath, n_bins=1000):
    try:
        df = pd.read_csv(filepath, header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

        if df.shape[1] < 2:
            return None

        raw_phase = df.iloc[:, 0].values
        raw_flux = df.iloc[:, 1].values

        phase_folded = raw_phase % 1.0

        bin_edges = np.linspace(0, 1.0, n_bins + 1)
        flux_binned = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (phase_folded >= bin_edges[i]) & (phase_folded < bin_edges[i+1])
            if np.any(mask):
                flux_binned[i] = np.median(raw_flux[mask])
            else:
                flux_binned[i] = np.nan

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
        if f_max <= 0 or np.isnan(f_max): return None

        return flux_final / f_max

    except Exception:
        return None

# =============================================================================
# 3. FEATURE EXTRACTION
# =============================================================================
def extract_features(flux):
    """
    Extract 51 features — identical to 2_extract_training_features.py.
    """
    features = {}

    # Implicit phase grid (same as training)
    phase = np.linspace(0.25, 1.25, len(flux), endpoint=False)

    # Basic statistics
    baseline_flux = np.median(flux)
    features['baseline'] = baseline_flux
    features['flux_min'] = np.min(flux)
    features['flux_max'] = np.max(flux)
    features['flux_mean'] = np.mean(flux)
    features['flux_std'] = np.std(flux)
    features['flux_range'] = features['flux_max'] - features['flux_min']
    features['flux_skew'] = pd.Series(flux).skew()
    features['flux_kurtosis'] = pd.Series(flux).kurtosis()

    # DYNAMIC ECLIPSE DETECTION
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
        primary_idx, secondary_idx = min1_idx, min2_idx
        primary_depth, secondary_depth = min1_depth, min2_depth
        primary_phase, secondary_phase = min1_phase, min2_phase
    else:
        primary_idx, secondary_idx = min2_idx, min1_idx
        primary_depth, secondary_depth = min2_depth, min1_depth
        primary_phase, secondary_phase = min2_phase, min1_phase

    features['primary_depth'] = primary_depth
    features['secondary_depth'] = secondary_depth

    primary_half_depth = baseline_flux - 0.5 * primary_depth
    secondary_half_depth = baseline_flux - 0.5 * secondary_depth
    features['primary_width'] = np.sum(flux < primary_half_depth) / len(flux)
    features['secondary_width'] = np.sum(flux < secondary_half_depth) / len(flux)

    # O'Connell effect
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

    # Fourier
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

    # Phase binning
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
# 4. MAIN
# =============================================================================
def main():
    fold_assets = []
    print(f"Loading models from {MODEL_DIR}...")
    for f in range(N_FOLDS):
        m_path = os.path.join(MODEL_DIR, f"xgb_fold_{f}.pkl")
        if os.path.exists(m_path):
            with open(m_path, 'rb') as f_in:
                data = pickle.load(f_in)
                fold_assets.append({
                    'scaler': data['scaler'],
                    'models': data['models'],
                    'feature_names': data.get('feature_names', [])
                })
        else:
            print(f"Warning: {m_path} missing")

    if not fold_assets:
        print("Error: No models loaded.")
        return

    all_files = [f for f in os.listdir(OGLE_LC_DIR) if f.lower().endswith(('.csv', '.dat'))]
    print(f"Processing {len(all_files)} files...")
    results = []

    for i in tqdm(range(0, len(all_files), BATCH_SIZE), desc="Batches"):
        batch_files = all_files[i:i + BATCH_SIZE]
        batch_features, batch_ids = [], []

        for f_name in batch_files:
            flux = load_and_bin_lc_robust(os.path.join(OGLE_LC_DIR, f_name))
            if flux is not None:
                batch_features.append(extract_features(flux))
                batch_ids.append(os.path.splitext(f_name)[0])

        if not batch_features: continue

        X_df = pd.DataFrame(batch_features)
        fold_preds = {p: [] for p in PARAMS_TO_PREDICT + [CLASSIFICATION_PARAM]}

        for assets in fold_assets:
            scaler, models_dict = assets['scaler'], assets['models']

            if assets['feature_names']:
                X_fold = X_df.reindex(columns=assets['feature_names'], fill_value=0.0)
            else:
                X_fold = X_df

            try: X_scaled = scaler.transform(X_fold)
            except: X_scaled = scaler.transform(X_fold.values)

            for p in PARAMS_TO_PREDICT + [CLASSIFICATION_PARAM]:
                if p in models_dict:
                    if GPU_AVAILABLE:
                        models_dict[p].set_params(device="cuda:0")
                        X_input = cp.array(X_scaled, dtype='float32')
                        preds = cp.asnumpy(models_dict[p].predict(X_input))
                    else:
                        models_dict[p].set_params(device="cpu")
                        preds = models_dict[p].predict(X_scaled)
                    fold_preds[p].append(preds)

        for idx in range(len(batch_ids)):
            star = {'id': batch_ids[idx]}
            conf_vals = []

            for p in PARAMS_TO_PREDICT:
                if fold_preds[p]:
                    f_vals = [fold_preds[p][f][idx] for f in range(len(fold_preds[p]))]
                    avg = np.mean(f_vals)
                    c = max(0.0, 1.0 - (np.std(f_vals) / (abs(avg) + 1e-6)))
                    star[p] = avg
                    star[f"{p}_conf"] = c
                    conf_vals.append(c)
                else:
                    star[p], star[f"{p}_conf"] = np.nan, 0.0
                    conf_vals.append(0.0)

            if fold_preds[CLASSIFICATION_PARAM]:
                m_vals = [int(fold_preds[CLASSIFICATION_PARAM][f][idx]) for f in range(len(fold_preds[CLASSIFICATION_PARAM]))]
                final_idx = max(set(m_vals), key=m_vals.count)
                mapping = {0: 2, 1: 5, 2: 3}
                star['morphology'] = mapping.get(final_idx, final_idx)
                star['morph_conf'] = m_vals.count(final_idx) / len(m_vals)
            else:
                star['morphology'], star['morph_conf'] = -1, 0.0

            star['overall_confidence'] = np.mean(conf_vals) if conf_vals else 0.0
            star = apply_physics_constraints(star)
            results.append(star)

    if results:
        output_path = os.path.join(OUTPUT_DIR, "ogle_predictions.csv")
        df_res = pd.DataFrame(results)

        priority = ['id', 'morphology', 'morph_conf', 'overall_confidence', 'physics_note']
        params = []
        for p in PARAMS_TO_PREDICT:
            params.extend([p, f"{p}_conf"])

        final_cols = priority + params
        remaining = [c for c in df_res.columns if c not in final_cols]

        df_res[final_cols + remaining].to_csv(output_path, index=False)
        print(f"Saved {len(results)} predictions to {output_path}")

if __name__ == "__main__":
    main()
