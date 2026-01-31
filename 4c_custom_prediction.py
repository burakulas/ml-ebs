#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pickle
import warnings
import cupy as cp  
from scipy.fft import fft
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from tqdm import tqdm

# Suppress noise
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
WORK_DIR = "."
OGLE_LC_DIR = os.path.join(WORK_DIR, "custom_data")
MODEL_DIR = os.path.join(WORK_DIR, "models/models_xgb")
OUTPUT_DIR = os.path.join(WORK_DIR, "predictions/custom_predictions")

# EXACT KEYS FOUND IN YOUR MODEL
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
    
    if m_code == 3: # CONTACT
        star['p1'] = omega_in
        star['p2'] = omega_in
        star['physics_note'] = "Contact: Forced p1=p2=Omega_in"
    elif m_code == 5: # SEMI-DETACHED
        star['p2'] = omega_in
        if star['p1'] <= omega_in:
            star['p1'] = omega_in * 1.05
            star['physics_note'] = "Semi-Det: Fixed p2, Forced p1 detached"
        else:
            star['physics_note'] = "Semi-Det: Fixed p2"
    elif m_code == 2: # DETACHED
        updated = False
        if star['p1'] <= omega_in:
            star['p1'] = omega_in * 1.05; updated = True
        if star['p2'] <= omega_in:
            star['p2'] = omega_in * 1.05; updated = True
        star['physics_note'] = "Detached: Constraints applied" if updated else "Detached: OK"
        
    return star

# =============================================================================
# 2. ROBUST LOADER (FIXED INTERPOLATION)
# =============================================================================
def load_and_bin_lc_robust(filepath, n_bins=1000):
    try:
        # 1. Load Data
        df = pd.read_csv(filepath)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # 2. Validate Headers
        if 'phase' not in df.columns or 'flux' not in df.columns:
            # print(f"SKIP {os.path.basename(filepath)}: Missing columns. Found {df.columns}")
            return None
            
        raw_phase, raw_flux = df['phase'].values % 1.0, df['flux'].values
        
        # 3. Binning
        bin_edges = np.linspace(0, 1.0, n_bins + 1)
        flux_binned = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (raw_phase >= bin_edges[i]) & (raw_phase < bin_edges[i+1])
            flux_binned[i] = np.median(raw_flux[mask]) if np.any(mask) else np.nan

        # 4. Fill Gaps
        s = pd.Series(flux_binned)
        flux_filled = s.interpolate(method='linear', limit_direction='both').values
        
        # 5. Smooth
        try: flux_smooth = savgol_filter(flux_filled, 31, 3, mode='wrap')
        except: flux_smooth = flux_filled

        # 6. Interpolate (CRITICAL FIX: endpoint=False to avoid duplicates)
        # We use endpoint=False so 0.0 is included but 1.0 is not (it wraps to 0)
        phase_grid = np.linspace(0, 1, n_bins, endpoint=False)
        
        ext_phase = np.concatenate([phase_grid - 1.0, phase_grid, phase_grid + 1.0])
        ext_flux = np.concatenate([flux_smooth, flux_smooth, flux_smooth])
        
        # Now ext_phase is strictly increasing, PchipInterpolator will work
        interp_func = PchipInterpolator(ext_phase, ext_flux)
        
        # Target Grid (0.25 to 1.25)
        target_grid = np.linspace(0.25, 1.25, n_bins)
        flux_final = interp_func(target_grid)
        
        # 7. Normalize
        f_max = np.nanmax(flux_final)
        if f_max <= 0 or np.isnan(f_max): return None
        
        return flux_final / f_max

    except Exception as e:
        # print(f"Error processing {os.path.basename(filepath)}: {e}")
        return None

# =============================================================================
# 3. FEATURE EXTRACTION
# =============================================================================
def extract_features(flux):
    features = {}
    baseline_flux = np.nanmedian(flux)
    flux_min, flux_max = np.nanmin(flux), np.nanmax(flux)
    
    features.update({
        'baseline': baseline_flux, 'flux_min': flux_min, 'flux_max': flux_max,
        'flux_mean': np.nanmean(flux), 'flux_std': np.nanstd(flux),
        'flux_median': np.nanmedian(flux), 'flux_range': flux_max - flux_min,
        'flux_skew': pd.Series(flux).skew(), 'flux_kurtosis': pd.Series(flux).kurtosis()
    })

    flux_smooth = uniform_filter1d(np.nan_to_num(flux, nan=baseline_flux), size=5, mode='wrap')
    min1_idx = np.argmin(flux_smooth)
    min1_depth = baseline_flux - flux_smooth[min1_idx]
    
    mask_width = int(0.15 * len(flux))
    flux_masked = flux_smooth.copy()
    indices = (np.arange(len(flux)) - min1_idx) % len(flux)
    flux_masked[(indices < mask_width) | (indices > len(flux)-mask_width)] = np.nan
    
    min2_idx = np.nanargmin(flux_masked) if not np.all(np.isnan(flux_masked)) else min1_idx
    min2_depth = baseline_flux - flux_smooth[min2_idx]
    
    p_depth, s_depth = max(min1_depth, min2_depth), min(min1_depth, min2_depth)
    features['primary_depth'] = p_depth
    features['secondary_depth'] = s_depth
    features['primary_width'] = np.mean(flux < (baseline_flux - 0.5 * p_depth))
    features['secondary_width'] = np.mean(flux < (baseline_flux - 0.5 * s_depth))

    fft_vals = fft(np.nan_to_num(flux - np.mean(flux)))
    for k in range(1, 11):
        features[f'fourier_amp_{k}'] = 2.0 * np.abs(fft_vals[k]) / len(flux)
        if k <= 3: features[f'fourier_phase_{k}'] = np.angle(fft_vals[k])
    
    f1 = features.get('fourier_amp_1', 1e-9)
    for k in [2, 3, 4]: features[f'fourier_ratio_{k}_1'] = features.get(f'fourier_amp_{k}',0) / f1

    return features

# =============================================================================
# 4. MAIN (SEPARATED CONFIDENCE LOGIC)
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

    # DEBUG CHECK
    # print(f"DEBUG: Model Keys: {list(fold_assets[0]['models'].keys())}")
    # print(f"DEBUG: Predicting: {PARAMS_TO_PREDICT}")

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
            
            # Feature alignment
            if assets['feature_names']:
                X_fold = X_df.reindex(columns=assets['feature_names'], fill_value=0.0)
            else:
                X_fold = X_df

            try: X_scaled = scaler.transform(X_fold)
            except: X_scaled = scaler.transform(X_fold.values)
            
            X_gpu = cp.array(X_scaled, dtype='float32')
            
            # Predict
            for p in PARAMS_TO_PREDICT + [CLASSIFICATION_PARAM]:
                if p in models_dict:
                    models_dict[p].set_params(device="cuda:0")
                    fold_preds[p].append(cp.asnumpy(models_dict[p].predict(X_gpu)))

        # Aggregate
        for idx in range(len(batch_ids)):
            star = {'id': batch_ids[idx]}
            conf_vals = [] # Only stores parameter confidences
            
            # 1. Regression Params
            for p in PARAMS_TO_PREDICT:
                if fold_preds[p]:
                    f_vals = [fold_preds[p][f][idx] for f in range(len(fold_preds[p]))]
                    avg = np.mean(f_vals)
                    # Conf = 1 - CV (Coefficient of Variation)
                    c = max(0.0, 1.0 - (np.std(f_vals) / (abs(avg) + 1e-6)))
                    star[p] = avg
                    star[f"{p}_conf"] = c
                    conf_vals.append(c) # Add to Average
                else:
                    star[p], star[f"{p}_conf"] = np.nan, 0.0
                    conf_vals.append(0.0) # Penalize missing parameter

            # 2. Morphology (SEPARATE from overall average)
            if fold_preds[CLASSIFICATION_PARAM]:
                m_vals = [int(fold_preds[CLASSIFICATION_PARAM][f][idx]) for f in range(len(fold_preds[CLASSIFICATION_PARAM]))]
                final_idx = max(set(m_vals), key=m_vals.count)
                # Map 0->2(Det), 1->5(SD), 2->3(Con)
                mapping = {0: 2, 1: 5, 2: 3}
                star['morphology'] = mapping.get(final_idx, final_idx)
                
                m_conf = m_vals.count(final_idx) / len(m_vals)
                star['morph_conf'] = m_conf
                # conf_vals.append(m_conf) <--- EXCLUDED FROM AVERAGE
            else:
                star['morphology'], star['morph_conf'] = -1, 0.0
                # conf_vals.append(0.0) <--- EXCLUDED FROM AVERAGE

            # Overall Confidence = Average of PARAMETERS ONLY
            star['overall_confidence'] = np.mean(conf_vals) if conf_vals else 0.0
            
            # Physics Constraints
            star = apply_physics_constraints(star)
            results.append(star)

    if results:
        output_path = os.path.join(OUTPUT_DIR, "custom_predictions.csv")
        df_res = pd.DataFrame(results)
        
        # Nicer column order
        priority = ['id', 'morphology', 'morph_conf', 'overall_confidence', 'physics_note']
        params = []
        for p in PARAMS_TO_PREDICT:
            params.extend([p, f"{p}_conf"])
        
        final_cols = priority + params
        # Add whatever is left
        remaining = [c for c in df_res.columns if c not in final_cols]
        
        df_res[final_cols + remaining].to_csv(output_path, index=False)
        print(f"Saved {len(results)} predictions to {output_path}")

if __name__ == "__main__":
    main()