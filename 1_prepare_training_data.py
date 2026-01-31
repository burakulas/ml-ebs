#!/usr/bin/env python3

import os
import re
import numpy as np
import pandas as pd
import pickle
from glob import glob
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR = "."
DATA_DIR = os.path.join(WORK_DIR, "training_data")
OUTPUT_DIR = os.path.join(WORK_DIR, "processed_data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "training_data.pkl")

N_POINTS = 1000  # Uniform grid size

MORPHOLOGY_MAP = {
    2: 'detached',
    3: 'contact',
    5: 'semidetached'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_caleb_format(filename):
    """Detect Caleb format (has _e_ for eccentricity)."""
    return '_e_' in os.path.basename(filename)


def parse_scientific_notation(s):
    """Convert Fortran scientific notation to float."""
    s = s.rstrip('.')
    s = s.replace('d', 'e').replace('D', 'e')
    return float(s)


def parse_filename(filename):
    """Parse parameters from filename."""
    params = {}
    basename = os.path.basename(filename)
    caleb_fmt = is_caleb_format(filename)

    params['filename'] = basename
    params['is_caleb_format'] = caleb_fmt

    # Morphology
    m_match = re.search(r'_m_(\d+)', basename)
    if m_match:
        morph_class = int(m_match.group(1))
        params['morphology_class'] = morph_class
        params['morphology'] = MORPHOLOGY_MAP.get(morph_class, 'unknown')
    else:
        params['morphology_class'] = None
        params['morphology'] = 'unknown'

    # Inclination
    i_match = re.search(r'_i_([\d.]+)', basename)
    params['i'] = float(i_match.group(1)) if i_match else np.nan

    # q
    q_match = re.search(r'_q_([\d.]+(?:d[+-]?\d+)?)', basename)
    if q_match:
        try:
            params['q'] = parse_scientific_notation(q_match.group(1))
        except:
            params['q'] = np.nan
    else:
        params['q'] = np.nan

    # Temperatures (Caleb: divide by 10000, New: already in Kelvin)
    t1_match = re.search(r'_t1_([\d.]+)', basename)
    t2_match = re.search(r'_t2_([\d.]+)', basename)

    if t1_match:
        params['t1'] = float(t1_match.group(1)) * (10000 if caleb_fmt else 1)
    else:
        params['t1'] = np.nan

    if t2_match:
        params['t2'] = float(t2_match.group(1)) * (10000 if caleb_fmt else 1)
    else:
        params['t2'] = np.nan

    # Potentials
    if caleb_fmt:
        p1_match = re.search(r'_p1_([\d.]+(?:d[+-]?\d+)?)', basename)
        p2_match = re.search(r'_p2_([\d.]+(?:d[+-]?\d+)?)', basename)
    else:
        p1_match = re.search(r'_p1_([\d.]+)', basename)
        p2_match = re.search(r'_p2_([\d.]+)', basename)

    if p1_match:
        try:
            params['p1'] = parse_scientific_notation(p1_match.group(1))
        except:
            params['p1'] = np.nan
    else:
        params['p1'] = np.nan

    if p2_match:
        try:
            params['p2'] = parse_scientific_notation(p2_match.group(1))
        except:
            params['p2'] = np.nan
    else:
        params['p2'] = np.nan

    # FIX: Auto-correct temperatures if they're too small (Caleb format issue)
    # Some Caleb files may not have been multiplied correctly
    if not np.isnan(params['t1']) and params['t1'] < 10:
        params['t1'] = params['t1'] * 10000
    if not np.isnan(params['t2']) and params['t2'] < 10:
        params['t2'] = params['t2'] * 10000

    # Temperature ratio (calculate AFTER any corrections)
    if params['t1'] > 0 and not np.isnan(params['t1']) and not np.isnan(params['t2']):
        params['t2_t1'] = params['t2'] / params['t1']
    else:
        params['t2_t1'] = np.nan

    return params


def load_light_curve(filepath):
    """Load light curve from .dat file."""
    try:
        # Both Caleb and New format files are comma-separated
        delimiter = ','

        data = np.loadtxt(filepath, delimiter=delimiter)
        if data.ndim == 1:
            return None, None

        phase = data[:, 0]
        flux = data[:, 1]
        return phase, flux
    except:
        return None, None


def normalize_light_curve(phase, flux, n_points=1000):
    """
    Interpolate flux to uniform phase grid in [0.25, 1.25] using PCHIP.

    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial):
    - Preserves monotonicity (no overshoot at eclipses)
    - Smoother than linear interpolation
    - No oscillations like cubic splines

    CRITICAL: NO PHASE SHIFTING!
    Primary eclipse stays at ~1.0 (wraps around).
    """
    # Sort by phase
    sort_idx = np.argsort(phase)
    phase = phase[sort_idx]
    flux = flux[sort_idx]

    # Remove duplicates
    unique_idx = np.unique(phase, return_index=True)[1]
    phase = phase[unique_idx]
    flux = flux[unique_idx]

    if len(phase) < 10:
        return None

    # Create uniform grid [0.25, 1.25]
    phase_grid = np.linspace(0.25, 1.25, n_points, endpoint=False)

    # Handle wrap-around for periodic interpolation
    # Extend data to cover [phase_min - 1, phase_max + 1] for smooth wrapping
    phase_extended = np.concatenate([phase - 1, phase, phase + 1])
    flux_extended = np.concatenate([flux, flux, flux])

    # Sort extended data
    sort_idx = np.argsort(phase_extended)
    phase_extended = phase_extended[sort_idx]
    flux_extended = flux_extended[sort_idx]

    # Remove duplicates from extended arrays (PCHIP requires strictly increasing x)
    unique_idx = np.unique(phase_extended, return_index=True)[1]
    phase_extended = phase_extended[unique_idx]
    flux_extended = flux_extended[unique_idx]

    # PCHIP interpolation (shape-preserving, no overshoot)
    pchip = PchipInterpolator(phase_extended, flux_extended)
    flux_interp = pchip(phase_grid)

    # Normalize flux to max=1
    flux_interp = flux_interp / np.max(flux_interp)

    return flux_interp


def validate_params(params):
    """Validate parameters. Temperatures should already be corrected in parse_filename()."""
    if params['i'] > 90:
        return False, f"Invalid i={params['i']:.2f} > 90"

    required = ['i', 'q', 't1', 't2', 'p1', 'p2']
    for p in required:
        if np.isnan(params[p]):
            return False, f"Missing {p}"

    # Validate temperature range (after corrections in parsing)
    # Accept wider range to be more permissive
    if params['t1'] < 1000 or params['t1'] > 80000:
        return False, f"Unusual t1={params['t1']:.0f}K"
    if params['t2'] < 1000 or params['t2'] > 80000:
        return False, f"Unusual t2={params['t2']:.0f}K"

    return True, "OK"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("SCRIPT 01: PREPARE TRAINING DATA")
    print("=" * 80)
    print(f"\nPhase convention: [0.25, 1.25] (primary eclipse at ~1.0)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find files
    dat_files = glob(os.path.join(DATA_DIR, "*.dat"))
    print(f"\nFound {len(dat_files)} .dat files")

    if len(dat_files) == 0:
        print("ERROR: No files found!")
        return

    # Process
    all_lcs = []
    all_params = []
    skipped = []

    print("\nProcessing light curves...")
    for filepath in tqdm(dat_files):
        params = parse_filename(filepath)

        is_valid, reason = validate_params(params)
        if not is_valid:
            skipped.append((params['filename'], reason))
            continue

        phase, flux = load_light_curve(filepath)
        if phase is None:
            skipped.append((params['filename'], "Failed to load"))
            continue

        flux_norm = normalize_light_curve(phase, flux, N_POINTS)
        if flux_norm is None:
            skipped.append((params['filename'], "Too few points"))
            continue

        all_lcs.append(flux_norm)
        all_params.append(params)

    print(f"\nProcessed: {len(all_lcs)}")
    print(f"Skipped: {len(skipped)}")

    # Create DataFrame
    params_df = pd.DataFrame(all_params)

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Total samples: {len(all_lcs)}")
    print(f"\nMorphology:")
    print(params_df['morphology'].value_counts())

    print(f"\nParameter ranges:")
    for param in ['i', 'q', 't1', 't2', 'p1', 'p2', 't2_t1']:
        vals = params_df[param].dropna()
        if len(vals) > 0:
            print(f"  {param:6s}: [{vals.min():7.2f}, {vals.max():7.2f}]  "
                  f"mean={vals.mean():6.2f}  std={vals.std():5.2f}")

    # Save
    output_data = {
        'lcs': all_lcs,
        'params': params_df,
        'n_points': N_POINTS,
        'phase_range': [0.25, 1.25]
    }

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n" + "=" * 80)
    print(f"SAVED: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
